#!/usr/bin/env python3
"""train_tiny_hetmoe.py — DDP trainer for TinyHetMoE.

Same recipe as production Highway-B (`train_highway_b.py`) at small
scale:
  - Highway-B model (132 meaning + 132 intuition + Highway 2×, 4 layers)
  - QAT-from-step-0 (ternary forward + vote-backward)
  - QK-Norm
  - Plateau-detect LR halving (patience=8 vals × 500 steps)
  - lr_scale persisted in checkpoint for safe resume
  - chunked CE for memory efficiency
  - DDP — runs on both GPUs, ~1.8× throughput. HetMoE top-2 needs
    find_unused_parameters=True (Day 6 lesson, ~18% overhead).
    Falls back cleanly to single-GPU when launched without torchrun.

What's stripped vs production:
  - No multi-corpus weighted sampling — TinyStories is one source
  - No variable-length training — TinyStories is short, fixed seq_len 512
  - No grad_checkpointing — model is small enough that activations fit

Usage:
    # Two-GPU (preferred):
    torchrun --nproc_per_node=2 training/train_tiny_hetmoe.py \\
        --config training/configs/tiny_hetmoe.json

    # Single-GPU:
    python training/train_tiny_hetmoe.py \\
        --config training/configs/tiny_hetmoe.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.tiny_hetmoe import (  # noqa: E402
    TinyHetMoE, TinyHetMoEConfig, set_quantize_mode, set_qat_backward_mode,
)


def init_ddp():
    """Returns (is_ddp, local_rank, world_size). Falls back to single-GPU
    if torchrun env vars are absent."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        return True, local_rank, dist.get_world_size()
    return False, 0, 1


def cleanup_ddp(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Config
# ============================================================================

@dataclass
class TrainCfg:
    run_name: str = "tiny_hetmoe"
    out_dir: str = "runs/tiny_hetmoe"
    resume_from: str = ""

    # Data
    train_file: str = "data/train.bin"
    val_file: str = "data/val.bin"
    meaning_emb_path: str = "data/meaning_axes_132.npy"

    # Training shape
    seq_len: int = 512
    micro_batch: int = 16
    grad_accum: int = 2

    # Variable-length training. If non-empty, each optimizer step samples
    # one (seq_len, micro_batch) from this list by weight. Critical for
    # long-context extrapolation since NoPE alone collapses past the
    # fixed training length. Production Highway-B uses this. Example:
    #   [{"seq_len": 256, "micro_batch": 24, "weight": 0.3},
    #    {"seq_len": 512, "micro_batch": 12, "weight": 0.4},
    #    {"seq_len": 1024, "micro_batch": 6, "weight": 0.3}]
    # When empty, falls back to fixed seq_len/micro_batch above.
    var_lengths: list = field(default_factory=list)
    max_steps: int = 60000
    warmup_steps: int = 1000
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Logging
    log_interval: int = 20
    val_interval: int = 500
    save_interval: int = 2000
    val_seqs: int = 50
    keep_recent_ckpts: int = 3
    seed: int = 1337
    dtype: str = "bf16"

    # Plateau detector
    lr_plateau_patience: int = 8
    lr_floor: float = 1e-5
    lr_decay_factor: float = 0.5

    # Recipe knobs
    freeze_meaning: bool = False     # B = trainable. Default False.
    qat_from_zero: bool = True

    # Backward mode for the ternary autograd function. Production findings:
    #   - "ste"       (production 130M PoC default; recommended for ≤200M scale)
    #   - "vote"      (our v1-v3; grad × sign(w) with 0.1× zero rescue)
    #   - "vote_sign" (production 2B; aggressive integer routing — recovered in 4 vals)
    qat_backward_mode: str = "ste"

    # bf16-then-QAT scheduling. When `qat_start_step > 0`, model trains
    # in FP for [0, qat_start_step) and switches to ternary for the
    # remainder. Set qat_from_zero=False AND qat_start_step>0 for the
    # bf16-then-QAT recipe. If qat_from_zero=True, qat_start_step is
    # ignored (QAT is always on).
    qat_start_step: int = 0

    # Trigger QAT on validation plateau (instead of fixed step). When
    # True, the FIRST plateau detected during the bf16 phase enables
    # QAT instead of halving LR. After QAT enables, plateau detection
    # reverts to normal LR-halving behavior. Cleaner than guessing
    # `qat_start_step` upfront. Requires qat_from_zero=False.
    qat_on_plateau: bool = False

    # Load entire corpus into GPU memory at startup (vs per-batch
    # memmap+transfer). Big throughput win when GPU has headroom.
    # Caller is responsible for ensuring the corpus fits.
    gpu_resident_corpus: bool = False

    # Token storage dtype on disk. v6 used uint16 (vocab ≤ 65535); v7
    # uses uint32 (Qwen full vocab 151665). Mismatch silently corrupts
    # every batch — must match what prepare_data_v* wrote.
    token_dtype: str = "uint16"


# ============================================================================
# Dataset
# ============================================================================

class TinyStoriesDataset:
    """Random-window sampler over a tokenized corpus.

    `token_dtype` is "uint16" for vocab ≤ 65535 (v6) and "uint32" for
    larger vocabs (v7+ uses Qwen full vocab 151665). Mismatch silently
    corrupts every loaded window — always set this from the data meta.

    Two modes:
      - **memmap (default)**: data lives on disk via `np.memmap`. Each
        `sample_batch` reads the requested windows, copies into a numpy
        buffer, then to a torch tensor. The trainer moves it to GPU.
      - **GPU-resident** (`gpu_device != None`): the entire corpus is
        loaded once into a torch tensor on `gpu_device` at construction.
        `sample_batch` then just indexes into that GPU tensor — no
        host→device transfer per batch."""

    def __init__(self, path: str, seq_len: int, seed: int = 1337,
                 gpu_device: str | None = None,
                 token_dtype: str = "uint16"):
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.gpu_device = gpu_device
        np_dtype = {"uint16": np.uint16, "uint32": np.uint32}[token_dtype]

        if gpu_device is not None:
            # Load whole file into a GPU int64 tensor (training expects int64
            # ids; since we'll index from this tensor directly we may as well
            # pre-cast to skip the per-batch dtype conversion).
            arr = np.fromfile(path, dtype=np_dtype).astype(np.int64)
            self.gpu_data = torch.from_numpy(arr).to(gpu_device, non_blocking=True)
            self.n_tokens = self.gpu_data.shape[0]
            self.cpu_data = None
        else:
            self.cpu_data = np.memmap(path, dtype=np_dtype, mode="r")
            self.n_tokens = len(self.cpu_data)
            self.gpu_data = None

    def sample_batch(self, batch_size: int, seq_len: int | None = None):
        # Variable-length training: caller may override seq_len per step.
        # Falls back to construction-time seq_len if not specified.
        sl = seq_len if seq_len is not None else self.seq_len
        max_start = self.n_tokens - sl - 2
        # Generate batch_size random start positions
        starts = self.rng.integers(0, max_start, size=batch_size)

        if self.gpu_data is not None:
            # GPU-resident path: gather batch_size windows directly on GPU.
            # Build an index tensor (B, T+1) of absolute positions, then
            # gather. Index tensor is built on CPU then transferred (small).
            offsets = torch.arange(sl + 1, dtype=torch.long)  # (T+1,)
            starts_t = torch.from_numpy(starts).long()                   # (B,)
            idx = starts_t.unsqueeze(1) + offsets.unsqueeze(0)            # (B, T+1)
            idx = idx.to(self.gpu_device, non_blocking=True)
            chunk = self.gpu_data[idx]                                    # (B, T+1) on GPU
            return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()

        # CPU memmap path
        ids_list = []
        tgt_list = []
        for s in starts:
            chunk = np.asarray(
                self.cpu_data[int(s):int(s) + sl + 1], dtype=np.int64,
            )
            ids_list.append(chunk[:-1])
            tgt_list.append(chunk[1:])
        ids = torch.from_numpy(np.stack(ids_list))
        tgt = torch.from_numpy(np.stack(tgt_list))
        return ids, tgt


# ============================================================================
# Helpers
# ============================================================================

def cosine_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    decay_ratio = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(decay_ratio, 1.0)))
    return min_lr + coeff * (max_lr - min_lr)


def rotate_checkpoints(ckpt_dir: Path, keep: int):
    """Keep only the `keep` most recent ckpt_NNNN.pt files. best.pt and
    ckpt_final_NNNN.pt are not rotated."""
    ckpts = sorted(
        [p for p in ckpt_dir.glob("ckpt_*.pt")
         if not p.name.startswith("ckpt_final")],
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    while len(ckpts) > keep:
        old = ckpts.pop(0)
        old.unlink()
        print(f"  [rotate] deleted {old.name} (kept {keep} most recent)", flush=True)


@torch.no_grad()
def evaluate(model, val_ds: TinyStoriesDataset, device: str,
             n_seqs: int, micro_batch: int, dtype, world_size: int):
    """Each rank evaluates `n_seqs` independently; we all-reduce the
    summed loss + token count across ranks for a fair average.
    `n_seqs` is total sequences divided across ranks."""
    model.eval()
    per_rank = max(1, n_seqs // world_size)
    total = 0.0
    count = 0
    n_done = 0
    while n_done < per_rank:
        bs = min(micro_batch, per_rank - n_done)
        ids, tgt = val_ds.sample_batch(bs)
        ids = ids.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            _, loss = model(ids, tgt)
        total += loss.item() * ids.numel()
        count += ids.numel()
        n_done += bs
    if world_size > 1 and dist.is_initialized():
        t = torch.tensor([total, count], dtype=torch.float64, device=device)
        dist.all_reduce(t)
        total, count = t[0].item(), t[1].item()
    model.train()
    return total / max(1, count)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-optimizer", action="store_true",
                    help="Resume model weights but reset optimizer (for "
                         "corruption recovery)")
    args = ap.parse_args()

    is_ddp, local_rank, world_size = init_ddp()
    is_main = (local_rank == 0)

    with open(args.config) as f:
        raw = json.load(f)

    mc_fields = {f.name for f in fields(TinyHetMoEConfig)}
    tr_fields = {f.name for f in fields(TrainCfg)}
    mc_kwargs = {k: v for k, v in raw.items() if k in mc_fields}
    tr_kwargs = {k: v for k, v in raw.items() if k in tr_fields}

    mcfg = TinyHetMoEConfig(**mc_kwargs)
    tcfg = TrainCfg(**tr_kwargs)

    torch.manual_seed(tcfg.seed + local_rank)
    np.random.seed(tcfg.seed + local_rank)

    # Resolve relative paths against the repo root
    def resolve(p: str) -> str:
        return p if Path(p).is_absolute() else str(REPO / p)

    out_dir = Path(resolve(tcfg.out_dir))
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True)
        (out_dir / "logs").mkdir(exist_ok=True)
        with open(out_dir / "config.json", "w") as f:
            full = {**asdict(mcfg), **asdict(tcfg)}
            json.dump(full, f, indent=2, default=str)
    if is_ddp:
        dist.barrier()

    device = f"cuda:{local_rank}"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[tcfg.dtype]

    if is_main:
        print(f"[tiny] DDP={is_ddp} world_size={world_size}", flush=True)
        print(f"[tiny] building TinyHetMoE", flush=True)
        print(f"[tiny] cfg: vocab={mcfg.vocab_size} hidden={mcfg.input_dim} "
              f"internal={mcfg.internal_dim} L={mcfg.num_layers} "
              f"H={mcfg.num_heads} E={mcfg.num_experts}/{mcfg.top_k_experts}",
              flush=True)
        print(f"[tiny] B recipe: freeze_meaning={tcfg.freeze_meaning}",
              flush=True)
        print(f"[tiny] QAT from zero: {tcfg.qat_from_zero}", flush=True)

    set_qat_backward_mode(tcfg.qat_backward_mode)
    if is_main:
        print(f"[tiny] QAT backward mode: {tcfg.qat_backward_mode}", flush=True)

    model = TinyHetMoE(mcfg)
    model.load_meaning_embeddings(resolve(tcfg.meaning_emb_path),
                                   freeze=tcfg.freeze_meaning)
    model = model.to(device)

    if is_main:
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[tiny] params: total={n_total/1e6:.2f}M  "
              f"trainable={n_train/1e6:.2f}M", flush=True)

    # ── Resume ──
    start_step = 0
    best_val = float("inf")
    # QAT-mode best is tracked separately. With bf16-then-QAT, bf16's
    # val numbers are always lower than QAT's (FP > ternary capacity),
    # so `best.pt` always lands on bf16 and we never save the actual
    # ternary deployment artifact. `best_qat.pt` fixes this.
    best_qat_val = float("inf")
    resume_lr_scale = 1.0
    resume_state = None
    resumed_val_history = None  # populated by resume block if present
    if tcfg.resume_from:
        resume_path = Path(resolve(tcfg.resume_from))
        if resume_path.exists():
            if is_main:
                print(f"[tiny] resuming from {resume_path}", flush=True)
            resume_state = torch.load(resume_path, map_location=device,
                                       weights_only=False)
            sd = resume_state["model"]
            sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if is_main:
                if missing:
                    print(f"[tiny] WARNING missing keys on resume: {missing[:5]}",
                          flush=True)
                if unexpected:
                    print(f"[tiny] WARNING unexpected keys: {unexpected[:5]}",
                          flush=True)
            start_step = int(resume_state.get("step", 0))
            best_val = float(resume_state.get("best_val", float("inf")))
            best_qat_val = float(resume_state.get("best_qat_val", float("inf")))
            resume_lr_scale = float(resume_state.get("lr_scale", 1.0))
            # Persisted training-loop state (added post-audit). Lets resume
            # pick up plateau detection, QAT phase, and val history exactly
            # where the previous run left off.
            resumed_qat_on = bool(resume_state.get("qat_currently_on", False))
            resumed_val_history = resume_state.get("val_history", None)
            if is_main:
                print(f"[tiny] resumed step {start_step}, best_val "
                      f"{best_val:.4f}, lr_scale {resume_lr_scale:.3f}, "
                      f"qat_currently_on={resumed_qat_on}",
                      flush=True)
        elif is_main:
            print(f"[tiny] resume_from path does not exist: {resume_path}",
                  flush=True)

    # QAT mode setup. Four regimes (last one added post-audit):
    #   1. qat_from_zero=True: ternary forward from step 0 (default, current).
    #   2. qat_from_zero=False, qat_start_step=0: pure FP run, no ternary ever.
    #   3. qat_from_zero=False, qat_start_step>0: bf16-then-QAT — start FP,
    #      switch to ternary at qat_start_step.
    #   4. Resume from a checkpoint where qat_currently_on=True (e.g. via
    #      qat_on_plateau): respect the persisted flag.
    qat_active_at_resume = (tcfg.qat_from_zero
                             or (tcfg.qat_start_step > 0 and start_step >= tcfg.qat_start_step)
                             or (resume_state is not None
                                 and bool(resume_state.get("qat_currently_on", False))))
    if qat_active_at_resume:
        n_qat = set_quantize_mode(model, on=True)
        if is_main:
            print(f"[tiny] QAT mode ON at startup ({n_qat} modules)", flush=True)
    else:
        if is_main:
            if tcfg.qat_start_step > 0:
                print(f"[tiny] QAT mode OFF — bf16 phase. Will switch ON at "
                      f"step {tcfg.qat_start_step}.", flush=True)
            else:
                print(f"[tiny] QAT mode OFF — pure FP run.", flush=True)

    if is_ddp:
        # HetMoE top-2 routing varies per rank → use find_unused_parameters=True
        # (Day 6 lesson; ~18% throughput cost is much less than the ~80%
        # we gain from a second GPU).
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        unwrapped = model.module
    else:
        unwrapped = model

    # ── Datasets ──
    # Different RNG per rank so each GPU draws independent batches.
    # GPU-resident mode: each rank loads the corpus to its own GPU.
    ds_device = device if tcfg.gpu_resident_corpus else None
    if is_main and tcfg.gpu_resident_corpus:
        print(f"[tiny] GPU-resident corpus: loading train + val to {device}",
              flush=True)
    train_ds = TinyStoriesDataset(resolve(tcfg.train_file), tcfg.seq_len,
                                   seed=tcfg.seed + local_rank * 1000,
                                   gpu_device=ds_device,
                                   token_dtype=tcfg.token_dtype)
    val_ds = TinyStoriesDataset(resolve(tcfg.val_file), tcfg.seq_len,
                                 seed=tcfg.seed + 1 + local_rank * 1000,
                                 gpu_device=ds_device,
                                 token_dtype=tcfg.token_dtype)
    bytes_per_tok = {"uint16": 2, "uint32": 4}[tcfg.token_dtype]
    if is_main:
        train_mb = train_ds.n_tokens * 8 / 1e6 if tcfg.gpu_resident_corpus else train_ds.n_tokens * bytes_per_tok / 1e6
        val_mb = val_ds.n_tokens * 8 / 1e6 if tcfg.gpu_resident_corpus else val_ds.n_tokens * bytes_per_tok / 1e6
        loc = "GPU" if tcfg.gpu_resident_corpus else "memmap"
        print(f"[tiny] train: {train_ds.n_tokens:,} tokens, "
              f"{train_mb:.0f} MB on {loc}", flush=True)
        print(f"[tiny]   val: {val_ds.n_tokens:,} tokens, "
              f"{val_mb:.0f} MB on {loc}", flush=True)

    # ── Optimizer ──
    opt = torch.optim.AdamW(
        [p for p in unwrapped.parameters() if p.requires_grad],
        lr=tcfg.lr, betas=(0.9, 0.95), weight_decay=tcfg.weight_decay,
    )
    if (resume_state is not None and "optimizer" in resume_state
            and not args.skip_optimizer):
        try:
            opt.load_state_dict(resume_state["optimizer"])
            if is_main:
                print(f"[tiny] resumed optimizer state", flush=True)
        except (ValueError, KeyError) as e:
            if is_main:
                print(f"[tiny] could not load optimizer state: {e}", flush=True)
    elif args.skip_optimizer and is_main:
        print(f"[tiny] --skip-optimizer: AdamW starts fresh", flush=True)
    resume_state = None
    torch.cuda.empty_cache()

    log_f = open(out_dir / "logs" / "train.jsonl", "a") if is_main else None

    # ── Train ──
    t_start = time.time()
    step = start_step
    opt.zero_grad()
    model.train()
    ema_loss = None
    # Restore val_history from resume if available — otherwise plateau
    # detection re-arms from zero on every restart, which breaks
    # qat_on_plateau / LR-halving for multi-day-with-resume runs.
    if resumed_val_history:
        val_history = list(resumed_val_history)
        if is_main:
            print(f"[tiny] restored val_history with {len(val_history)} entries",
                  flush=True)
    else:
        val_history = []
    current_lr_scale = resume_lr_scale

    qat_currently_on = qat_active_at_resume
    # Set by the plateau-detection block (rank 0 only) when qat_on_plateau
    # is configured; consumed at the top of the next iteration.
    qat_trigger_pending = False

    # Variable-length training setup. If `var_lengths` is configured, each
    # optimizer step samples one (seq_len, micro_batch) pair from the list
    # by weight. Critical for long-context extrapolation. Falls back to
    # fixed (tcfg.seq_len, tcfg.micro_batch) if var_lengths is empty.
    if tcfg.var_lengths:
        var_lens = tcfg.var_lengths
        var_len_weights = np.array([v["weight"] for v in var_lens], dtype=np.float64)
        var_len_weights = var_len_weights / var_len_weights.sum()
        if is_main:
            print(f"[tiny] variable-length training:", flush=True)
            for v, w in zip(var_lens, var_len_weights):
                print(f"        seq_len={v['seq_len']:>5}  "
                      f"micro_batch={v['micro_batch']}  weight={w:.3f}",
                      flush=True)
    else:
        var_lens = [{"seq_len": tcfg.seq_len, "micro_batch": tcfg.micro_batch,
                     "weight": 1.0}]
        var_len_weights = np.array([1.0])
        if is_main:
            print(f"[tiny] fixed-length training: seq_len={tcfg.seq_len}, "
                  f"micro_batch={tcfg.micro_batch}", flush=True)
    var_len_rng = np.random.default_rng(tcfg.seed + local_rank * 1000 + 7)

    def enable_qat(reason: str):
        nonlocal qat_currently_on
        n_qat = set_quantize_mode(unwrapped, on=True)
        qat_currently_on = True
        if is_main:
            print(f"[tiny] ★★★ QAT mode ENABLED at step {step} ({reason}, "
                  f"{n_qat} modules ternarized). bf16 phase done.", flush=True)
            if log_f is not None:
                log_f.write(json.dumps({
                    "step": step, "event": "qat_enabled", "reason": reason,
                    "wall": time.time() - t_start,
                }) + "\n")
                log_f.flush()

    while step < tcfg.max_steps:
        # ── QAT enable: fixed-step, plateau, or 90% backstop ───────────
        # Trigger logic: (qat_start_step) OR (plateau detected) OR
        # (step >= 90% of max_steps) — the backstop ensures QAT always
        # fires before the run ends, even if the bf16 phase never plateaued.
        if not qat_currently_on:
            backstop_trigger = (tcfg.qat_on_plateau
                                 and not tcfg.qat_from_zero
                                 and step >= int(0.9 * tcfg.max_steps))
            if tcfg.qat_start_step > 0 and step >= tcfg.qat_start_step:
                enable_qat(f"qat_start_step={tcfg.qat_start_step}")
                val_history = []
            elif qat_trigger_pending or backstop_trigger:
                # Broadcast across ranks first (rank 0 sets the flag from
                # validation; other ranks need to know). Backstop is
                # step-based and identical on all ranks, no broadcast.
                if is_ddp and qat_trigger_pending:
                    flag = torch.tensor([1.0 if qat_trigger_pending else 0.0],
                                         dtype=torch.float64, device=device)
                    dist.broadcast(flag, src=0)
                    qat_trigger_pending = bool(flag.item())
                reason = ("plateau detected" if qat_trigger_pending
                          else f"90% backstop (step {step}/{tcfg.max_steps})")
                enable_qat(reason)
                val_history = []
                qat_trigger_pending = False

        # Pick a (seq_len, micro_batch) pair for this entire optimizer step.
        var_idx = int(var_len_rng.choice(len(var_lens), p=var_len_weights))
        v = var_lens[var_idx]
        current_seq_len = v["seq_len"]
        current_micro_batch = v["micro_batch"]

        loss_sum_for_step = 0.0
        nan_seen = False
        for _ in range(tcfg.grad_accum):
            ids, tgt = train_ds.sample_batch(current_micro_batch, seq_len=current_seq_len)
            ids = ids.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=dtype,
                                 enabled=(dtype != torch.float32)):
                _, loss = model(ids, tgt)
            if not torch.isfinite(loss):
                print(f"[tiny] NaN/Inf at step {step}; aborting", flush=True)
                nan_seen = True
                break
            (loss / tcfg.grad_accum).backward()
            loss_sum_for_step += loss.item()

        if nan_seen:
            break

        base_lr = cosine_lr(step, tcfg.warmup_steps, tcfg.max_steps,
                             tcfg.lr, tcfg.min_lr)
        scaled_lr = max(tcfg.lr_floor, base_lr * current_lr_scale)
        for g in opt.param_groups:
            g["lr"] = scaled_lr
        torch.nn.utils.clip_grad_norm_(
            [p for p in unwrapped.parameters() if p.requires_grad],
            tcfg.max_grad_norm,
        )
        opt.step()
        opt.zero_grad()
        step += 1

        li = loss_sum_for_step / tcfg.grad_accum
        ema_loss = li if ema_loss is None else 0.98 * ema_loss + 0.02 * li

        if is_main and step % tcfg.log_interval == 0:
            toks_per_step = (current_micro_batch * current_seq_len *
                              tcfg.grad_accum * world_size)
            dt_step = (time.time() - t_start) / max(1, step - start_step)
            tok_s = toks_per_step / dt_step if dt_step else 0
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"step {step:>6} | loss {li:6.4f} | ema {ema_loss:6.4f} "
                  f"| lr {scaled_lr:.2e} (×{current_lr_scale:.3f}) "
                  f"| {tok_s:.0f} tok/s | mem {mem:.1f}GB", flush=True)
            log_f.write(json.dumps({
                "step": step, "loss": li, "ema": ema_loss,
                "lr": scaled_lr, "lr_scale": current_lr_scale,
                "tok_s": tok_s, "mem_gb": mem,
                "wall": time.time() - t_start,
            }) + "\n")
            log_f.flush()

        if step % tcfg.val_interval == 0 or step == tcfg.max_steps:
            vloss = evaluate(model, val_ds, device, tcfg.val_seqs,
                              tcfg.micro_batch, dtype, world_size)
            if is_main:
                print(f"  ── val @ {step}: {vloss:.4f}  "
                      f"(best {best_val:.4f}) ──", flush=True)
                log_f.write(json.dumps({
                    "step": step, "val": vloss,
                    "lr": scaled_lr, "lr_scale": current_lr_scale,
                    "wall": time.time() - t_start,
                }) + "\n")
                log_f.flush()
                val_history.append((step, vloss))

                # When in QAT phase, plateau detection compares against
                # the QAT-best, not the lifetime best (bf16's val is
                # always lower and would freeze the detector). Track which
                # one we're comparing against:
                phase_best = best_qat_val if qat_currently_on else best_val
                beat_phase_best = vloss < phase_best

                # Always-on best.pt (lowest val ever). For bf16-then-QAT
                # this lands on the bf16 phase since bf16 vals < QAT vals.
                if vloss < best_val:
                    best_val = vloss
                    torch.save({
                        "model": unwrapped.state_dict(),
                        "optimizer": opt.state_dict(),
                        "step": step, "best_val": best_val,
                        "best_qat_val": best_qat_val,
                        "qat_currently_on": qat_currently_on,
                        "val_history": val_history,
                        "config": {**asdict(mcfg), **asdict(tcfg)},
                        "lr_scale": current_lr_scale,
                    }, out_dir / "checkpoints" / "best.pt")
                    print(f"  saved best.pt (val={best_val:.4f}, "
                          f"lr_scale={current_lr_scale:.3f})", flush=True)

                # QAT-mode best — saved separately while QAT is on.
                # This is the ternary deployment artifact.
                if qat_currently_on and vloss < best_qat_val:
                    best_qat_val = vloss
                    torch.save({
                        "model": unwrapped.state_dict(),
                        "optimizer": opt.state_dict(),
                        "step": step, "best_val": best_val,
                        "best_qat_val": best_qat_val,
                        "qat_currently_on": qat_currently_on,
                        "val_history": val_history,
                        "config": {**asdict(mcfg), **asdict(tcfg)},
                        "lr_scale": current_lr_scale,
                    }, out_dir / "checkpoints" / "best_qat.pt")
                    print(f"  saved best_qat.pt (val={best_qat_val:.4f}, "
                          f"PPL {math.exp(best_qat_val):.2f}) ←★ deployment artifact",
                          flush=True)

                # Plateau detection — count consecutive non-improving vals
                # against the **phase-appropriate all-time best**, not against
                # a rolling window. The rolling-window baseline let oscillating
                # vals (e.g. 1.7 / 1.8 / 1.7 / 1.8 ...) read as "improving"
                # within each 9-val slice while never beating the actual best.
                if beat_phase_best:
                    val_history = [(step, vloss)]
                else:
                    # vals_since_best counts entries we've seen without beating
                    # the phase best (val_history was reset to [best] at the
                    # last beat, so its length minus 1 = consecutive misses).
                    vals_since_best = len(val_history) - 1
                    if vals_since_best >= tcfg.lr_plateau_patience:
                        # First plateau in bf16 phase + qat_on_plateau:
                        # enable QAT instead of halving LR. After this
                        # the detector reverts to normal LR halving.
                        if (tcfg.qat_on_plateau
                                and not qat_currently_on
                                and not tcfg.qat_from_zero):
                            qat_trigger_pending = True
                        elif scaled_lr > tcfg.lr_floor * 1.5:
                            old = scaled_lr
                            current_lr_scale *= tcfg.lr_decay_factor
                            new_lr = max(tcfg.lr_floor,
                                          base_lr * current_lr_scale)
                            print(f"  ⚠ plateau ({vals_since_best} vals "
                                  f"since best): halving LR {old:.2e} → "
                                  f"{new_lr:.2e} (scale "
                                  f"×{current_lr_scale:.3f})", flush=True)
                            val_history = [(step, vloss)]
            # All ranks need the new lr_scale if it changed; broadcast.
            if is_ddp:
                ls = torch.tensor([current_lr_scale], dtype=torch.float64,
                                   device=device)
                dist.broadcast(ls, src=0)
                current_lr_scale = ls.item()

        if is_main and (step % tcfg.save_interval == 0 or step == tcfg.max_steps):
            rotate_checkpoints(
                out_dir / "checkpoints",
                keep=max(1, tcfg.keep_recent_ckpts - 1),
            )
            ckpt_path = out_dir / "checkpoints" / f"ckpt_{step}.pt"
            torch.save({
                "model": unwrapped.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step, "best_val": best_val,
                "best_qat_val": best_qat_val,
                "qat_currently_on": qat_currently_on,
                "val_history": val_history,
                "config": {**asdict(mcfg), **asdict(tcfg)},
                "lr_scale": current_lr_scale,
            }, ckpt_path)

    if is_main:
        final = out_dir / "checkpoints" / f"ckpt_final_{step}.pt"
        torch.save({
            "model": unwrapped.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step, "best_val": best_val,
            "best_qat_val": best_qat_val,
            "qat_currently_on": qat_currently_on,
            "val_history": val_history,
            "config": {**asdict(mcfg), **asdict(tcfg)},
            "lr_scale": current_lr_scale,
        }, final)
        print(f"[tiny] done. final: {final}, best_val {best_val:.4f}",
              flush=True)
        if log_f is not None:
            log_f.close()

    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
