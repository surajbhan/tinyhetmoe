#!/usr/bin/env python3
"""eval_extrapolation.py — measure long-context generalization.

For a given checkpoint, runs a long val sequence through the model
token-by-token and reports per-position-bucket cross-entropy. The
diagnostic is whether PPL stays ~stable past the training seq_len.

Runs on CPU by default (so it can run alongside live GPU training
without contention). Pass --device cuda:N to use a GPU.

Usage:
    python scripts/eval_extrapolation.py \\
        --ckpt runs/tiny_hetmoe_v5/checkpoints/ckpt_5000.pt \\
        --seq 2048
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.tiny_hetmoe import TinyHetMoE, TinyHetMoEConfig, set_quantize_mode  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val_bin", default="data/val.bin")
    p.add_argument("--seq", type=int, default=2048,
                   help="Total sequence length to evaluate")
    p.add_argument("--start", type=int, default=10000,
                   help="Token offset into val.bin")
    p.add_argument("--device", default="cpu")
    p.add_argument("--qat", action="store_true",
                   help="Force QAT mode on (default: detect from ckpt)")
    args = p.parse_args()

    print(f"[eval] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = TinyHetMoEConfig(**{k: cfg_dict[k] for k in [
        "vocab_size", "meaning_dim", "intuition_dim", "input_dim",
        "internal_dim", "new_intuition", "num_layers", "num_heads",
        "num_experts", "top_k_experts", "ffn_mult", "max_seq_len",
        "load_balance_weight",
    ]})
    print(f"[eval] cfg: vocab={cfg.vocab_size} hidden={cfg.input_dim} "
          f"L={cfg.num_layers} H={cfg.num_heads} max_seq_len={cfg.max_seq_len}")
    print(f"[eval] ckpt step={ckpt.get('step')} "
          f"best_val={ckpt.get('best_val', 'n/a')} "
          f"best_qat_val={ckpt.get('best_qat_val', 'n/a')}")

    model = TinyHetMoE(cfg)
    sd = ckpt["model"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[eval] missing keys: {missing[:3]}")
    model.eval().to(args.device)

    # QAT mode: detect whether the ckpt was trained in QAT phase. Heuristic:
    # if best_qat_val is set and the ckpt's step is >= the QAT-enable step,
    # we're in QAT. Easier: look at config's qat_from_zero or qat_start_step.
    cfg_train = ckpt.get("config", {})
    qat_active = (
        args.qat
        or cfg_train.get("qat_from_zero", False)
        or (cfg_train.get("qat_start_step", 0) > 0
            and ckpt.get("step", 0) >= cfg_train["qat_start_step"])
        or ckpt.get("best_qat_val", float("inf")) < float("inf")
    )
    if qat_active:
        n_qat = set_quantize_mode(model, on=True)
        print(f"[eval] QAT mode ON ({n_qat} modules ternarized)")
    else:
        print(f"[eval] QAT mode OFF (FP eval)")

    # Load val.bin slice
    val_path = REPO / args.val_bin if not Path(args.val_bin).is_absolute() else Path(args.val_bin)
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    end = args.start + args.seq + 1
    if end > len(val_data):
        raise RuntimeError(f"val.bin only has {len(val_data)} tokens, need {end}")
    chunk = np.asarray(val_data[args.start:end], dtype=np.int64)
    ids = torch.from_numpy(chunk[:-1]).unsqueeze(0).to(args.device)  # (1, T)
    targets = torch.from_numpy(chunk[1:]).unsqueeze(0).to(args.device)
    T = ids.shape[1]
    print(f"[eval] sequence: {T} tokens (offset {args.start} into val.bin)")

    # Forward pass, get per-position CE.
    print(f"[eval] running forward...")
    t0 = time.time()
    with torch.no_grad():
        logits, _ = model(ids)   # (1, T, vocab)
    print(f"[eval] forward: {time.time() - t0:.1f}s")

    # Per-position cross-entropy
    log_probs = F.log_softmax(logits.float(), dim=-1)   # (1, T, V)
    target_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)  # (T,)
    losses = -target_lp.cpu().numpy()  # (T,)

    # Bucket
    buckets = [
        (1, 32, "    1-32 (warmup)"),
        (32, 128, "   32-128"),
        (128, 256, "  128-256"),
        (256, 512, "  256-512"),
        (512, 1024, "  512-1024"),
        (1024, 2048, " 1024-2048"),
        (2048, 3072, " 2048-3072"),
        (3072, 4096, " 3072-4096"),
    ]
    print(f"\nMean cross-entropy by position bucket:")
    print(f"(if extrapolation works, post-512 buckets should match training-range buckets)\n")
    train_range_loss = None
    far_loss = None
    for lo, hi, name in buckets:
        if hi > T: continue
        mean = float(losses[lo:hi].mean())
        ppl = math.exp(mean)
        bar = "▓" * round(mean * 10)
        print(f"  {name.ljust(28)} mean={mean:.3f}  PPL={ppl:>7.2f}  {bar}")
        if lo == 256 and hi == 512:
            train_range_loss = mean
        if hi == T:
            far_loss = mean

    if train_range_loss is not None and far_loss is not None and far_loss != train_range_loss:
        drift = far_loss - train_range_loss
        print(f"\n  in-training-range (256-512):  mean {train_range_loss:.3f}  PPL {math.exp(train_range_loss):.2f}")
        far_lo = max(b[0] for b in buckets if b[1] <= T)
        print(f"  far extrapolation:            mean {far_loss:.3f}  PPL {math.exp(far_loss):.2f}")
        print(f"  drift: {drift:+.3f} nats ({drift / max(0.01, train_range_loss) * 100:+.1f}%)")
        if abs(drift) < 0.20:
            print(f"  ✓ extrapolation holds")
        elif drift > 0 and drift < 0.6:
            print(f"  ~ mild degradation but no collapse")
        else:
            print(f"  ✗ significant degradation past training length")


if __name__ == "__main__":
    main()
