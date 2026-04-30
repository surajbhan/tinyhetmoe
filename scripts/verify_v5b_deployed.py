#!/usr/bin/env python3
"""verify_v5b_deployed.py — measure the true deployed val of v5b.

Training reported v5b at val=1.5977 (PPL 4.94) — but that was computed
with the buggy loss path that uses FP lm_head even when QAT is on. The
actual deployed model (docs/tiny.bin in the published browser demo)
runs with TERNARY lm_head, which the training never saw.

This script measures THREE numbers on the v5b best_qat.pt:

  1. FP forward (lm_head FP, no QAT anywhere) — matches v5b's bf16-best
     reported val of 1.6056
  2. Buggy-training-loss forward (QAT on, but FP lm_head shortcut in
     loss path) — matches v5b's reported best_qat.pt val of 1.5977
  3. **Deployed forward** (QAT on, ternary lm_head via QuantizedLinear)
     — what the published artifact actually does. This is the number
     we should be publishing.

The gap between (2) and (3) is the size of the publishing error.

TinyStories vocab=5967, val.bin lives in archive after the disk cleanup.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F

from model.tiny_hetmoe import (
    TinyHetMoE, TinyHetMoEConfig, set_quantize_mode, ternary_quantize,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_TOKENS = 100_000
SEQ_LEN = 512


def load_v5b(ckpt_path: Path, qat: bool):
    cfg_path = REPO / "runs/tiny_hetmoe_v5b/config.json"
    cfg_dict = json.load(cfg_path.open())
    mc_fields = {
        "vocab_size", "meaning_dim", "intuition_dim", "input_dim",
        "internal_dim", "new_intuition", "num_layers", "num_heads",
        "num_experts", "top_k_experts", "ffn_mult", "max_seq_len",
        "load_balance_weight",
    }
    mc_kwargs = {k: v for k, v in cfg_dict.items() if k in mc_fields}
    mcfg = TinyHetMoEConfig(**mc_kwargs)
    model = TinyHetMoE(mcfg)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob.get("state_dict", blob))
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    if qat:
        n = set_quantize_mode(model, on=True)
        print(f"  enabled QAT forward on {n} modules")
    model.eval()
    model.to(DEVICE)
    return model


@torch.no_grad()
def eval_loss_path(model, val_seqs: np.ndarray, label: str) -> float:
    """Use the targets-CE path via model(ids, tgt). This is what training
    reported. With QAT on, it ternarizes everything EXCEPT lm_head — that
    matmul shortcut at line 465 of the model uses FP weights.

    NOTE: run this with the OLD buggy model.py to reproduce v5b's
    reported numbers exactly. The fixed model.py (today's lm_head fix)
    will produce slightly higher loss because it now correctly
    ternarizes lm_head in the loss path too — that's the bug being
    fixed. The DEPLOYED path (eval_inference_path) is unaffected
    by the fix.
    """
    total_loss = 0.0
    total_count = 0
    BATCH = 8
    for i in range(0, val_seqs.shape[0], BATCH):
        chunk = torch.from_numpy(val_seqs[i:i + BATCH].astype(np.int64)).to(DEVICE)
        ids = chunk[:, :-1].contiguous()
        tgt = chunk[:, 1:].contiguous()
        _, loss = model(ids, tgt)
        n_tok = tgt.numel()
        total_loss += loss.item() * n_tok
        total_count += n_tok
    avg = total_loss / total_count
    print(f"  [{label}] val (loss path): {avg:.4f}  ppl {np.exp(avg):.2f}  "
          f"({total_count:,} tokens)")
    return avg


@torch.no_grad()
def eval_inference_path(model, val_seqs: np.ndarray, label: str) -> float:
    """Use the inference path: model(ids) returns logits, then we compute
    CE manually. This goes through QuantizedLinear.forward for lm_head
    which DOES apply ternary_quantize when quantize=True. This is what
    the deployed engine does (and what tiny.bin in the browser does).
    """
    total_loss = 0.0
    total_count = 0
    BATCH = 8
    for i in range(0, val_seqs.shape[0], BATCH):
        chunk = torch.from_numpy(val_seqs[i:i + BATCH].astype(np.int64)).to(DEVICE)
        ids = chunk[:, :-1].contiguous()
        tgt = chunk[:, 1:].contiguous()
        # logits via the inference path (calls self.lm_head(hidden) which
        # goes through QuantizedLinear.forward → ternary_quantize when
        # quantize=True)
        logits, _ = model(ids)  # (B, T, V)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            reduction="sum",
        )
        n_tok = tgt.numel()
        total_loss += ce.item()
        total_count += n_tok
    avg = total_loss / total_count
    print(f"  [{label}] val (deployed):  {avg:.4f}  ppl {np.exp(avg):.2f}  "
          f"({total_count:,} tokens)")
    return avg


def main():
    print(f"[verify-v5b] device {DEVICE}")
    val_path = Path("/home/suraj/tinyhetmoe_archive/old_tinystories_val.bin")
    raw = np.fromfile(val_path, dtype=np.uint16)
    print(f"[verify-v5b] val tokens: {len(raw):,}")
    # Match training's eval: random 512-token windows from the val stream,
    # not sequential. Use a large enough sample (500 windows = 256K tokens)
    # to get stable averages — the published 1.5977 was 50 windows so had
    # higher variance.
    rng = np.random.default_rng(42)
    n_seqs = 500
    max_start = len(raw) - SEQ_LEN - 2
    starts = rng.integers(0, max_start, size=n_seqs)
    seqs = np.stack([raw[s:s + SEQ_LEN + 1] for s in starts])
    print(f"[verify-v5b] using {seqs.shape[0]} random {SEQ_LEN+1}-token windows")

    print(f"\n=== v5b best.pt (bf16, FP everywhere) ===")
    print(f"    expected: ~1.6056 (training-reported val)")
    model = load_v5b(REPO / "runs/tiny_hetmoe_v5b/checkpoints/best.pt", qat=False)
    eval_loss_path(model, seqs, "bf16/loss-path")
    eval_inference_path(model, seqs, "bf16/inference")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n=== v5b best_qat.pt — TWO measurements ===")
    print(f"    training-reported val (loss path, FP lm_head shortcut): 1.5977")
    print(f"    DEPLOYED inference (ternary lm_head): unknown — that's what we measure")
    model = load_v5b(REPO / "runs/tiny_hetmoe_v5b/checkpoints/best_qat.pt", qat=True)
    print(f"\n  --- Loss path (matches training-reported number) ---")
    loss_path_val = eval_loss_path(model, seqs, "qat/loss-path")
    print(f"\n  --- Inference path (matches what docs/tiny.bin actually does) ---")
    inf_path_val = eval_inference_path(model, seqs, "qat/inference")

    print(f"\n[verify-v5b] SUMMARY")
    print(f"  Reported in journal/blog: val 1.5977, PPL 4.94")
    print(f"  Loss path measured:       val {loss_path_val:.4f}, PPL {np.exp(loss_path_val):.2f}")
    print(f"  DEPLOYED measured:        val {inf_path_val:.4f}, PPL {np.exp(inf_path_val):.2f}")
    gap = inf_path_val - loss_path_val
    print(f"  Publishing-error gap:     +{gap:.4f} nats  ({np.exp(gap):.2f}x PPL)")


if __name__ == "__main__":
    main()
