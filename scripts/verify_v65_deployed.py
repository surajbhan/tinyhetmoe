#!/usr/bin/env python3
"""verify_v65_deployed.py — measure the true deployed val of v6.5 QAT
checkpoints (apples-to-apples with v6.6 numbers).

v6.5 was trained with the buggy lm_head loss-path (FP shortcut for the
matmul). Reported numbers were:
  PG-19 best_qat: 4.18 (PPL 65.7)
  Wiki  best_qat: 3.75 (PPL 42.4)

But those were measured with the FP shortcut — what users get in the
browser is ternary lm_head. This script measures both paths on the
same val sample so we can see how much of the v6.5 → v6.6 "gap" is
the bug correction vs. real model change.
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
    TinyHetMoE, TinyHetMoEConfig, set_quantize_mode,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 512


def load_v65(ckpt_path: Path, cfg_path: Path):
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
    n = set_quantize_mode(model, on=True)
    print(f"  enabled QAT forward on {n} modules")
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def eval_both_paths(model, val_seqs: np.ndarray, label: str):
    BATCH = 8
    total_loss_path = 0.0
    total_inf_path = 0.0
    total_count = 0
    for i in range(0, val_seqs.shape[0], BATCH):
        chunk = torch.from_numpy(val_seqs[i:i + BATCH].astype(np.int64)).to(DEVICE)
        ids = chunk[:, :-1].contiguous()
        tgt = chunk[:, 1:].contiguous()

        # Loss path (with the FIX now applied — ternary lm_head)
        _, loss = model(ids, tgt)
        total_loss_path += loss.item() * tgt.numel()

        # Inference path (always ternary lm_head — what wasm engine does)
        logits, _ = model(ids)
        ce_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            reduction="sum",
        )
        total_inf_path += ce_sum.item()

        total_count += tgt.numel()

    avg_loss = total_loss_path / total_count
    avg_inf = total_inf_path / total_count
    print(f"  [{label}] loss path:      val={avg_loss:.4f}  PPL={np.exp(avg_loss):.2f}")
    print(f"  [{label}] inference path: val={avg_inf:.4f}  PPL={np.exp(avg_inf):.2f}")
    return avg_loss, avg_inf


def main():
    print(f"[verify-v65] device {DEVICE}")

    cases = [
        ("pg19", "runs/tiny_hetmoe_v6_5_pg19/checkpoints/best_qat.pt",
                  "training/configs/tiny_hetmoe_v6_5_pg19.json",
                  "data/unified_val_pg19.bin", 4.1848),
        ("wiki", "runs/tiny_hetmoe_v6_5_wiki/checkpoints/best_qat.pt",
                  "training/configs/tiny_hetmoe_v6_5_wiki.json",
                  "data/unified_val_wiki.bin", 3.7471),
    ]

    rng = np.random.default_rng(42)
    n_seqs = 200

    for name, ckpt, cfg, val_bin, reported in cases:
        print(f"\n=== {name} (reported best_qat val={reported:.4f}) ===")
        val = np.fromfile(REPO / val_bin, dtype=np.uint16)
        max_start = len(val) - SEQ_LEN - 2
        starts = rng.integers(0, max_start, size=n_seqs)
        seqs = np.stack([val[s:s + SEQ_LEN + 1] for s in starts])

        model = load_v65(REPO / ckpt, REPO / cfg)
        loss_path, inf_path = eval_both_paths(model, seqs, name)

        gap = inf_path - loss_path
        print(f"  reported - inference = {reported - inf_path:+.4f} nats "
              f"(reported was {'too low' if reported < inf_path else 'too high'})")
        print(f"  loss path - inference path = {gap:+.4f} nats "
              f"(bug-fix effect on this checkpoint)")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
