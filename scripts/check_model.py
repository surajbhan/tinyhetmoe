#!/usr/bin/env python3
"""check_model.py — instantiate the model, check param count, run one
forward pass on dummy data + one with return_trace=True to verify shapes.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch

from model.tiny_hetmoe import TinyHetMoE, TinyHetMoEConfig, count_params, set_quantize_mode


def main():
    cfg = TinyHetMoEConfig()  # defaults: vocab 5967, hidden 264, 4 layers
    model = TinyHetMoE(cfg)

    # Load 132-axis meaning embeddings
    meaning_path = REPO / "data" / "meaning_axes_132.npy"
    if meaning_path.exists():
        model.load_meaning_embeddings(str(meaning_path), freeze=False)
        print(f"[check] loaded meaning embeddings from {meaning_path.name}")
    else:
        print(f"[check] WARNING: {meaning_path} not found, using random init")

    bd = count_params(model)
    print("\n[check] param breakdown:")
    for name, n in sorted(bd.items(), key=lambda kv: -kv[1]):
        print(f"  {name:24s} {n:>12,}  ({n/1e6:.2f}M)")

    print(f"\n[check] FP forward pass test")
    set_quantize_mode(model, on=False)
    ids = torch.randint(0, cfg.vocab_size, (2, 64))
    targets = torch.randint(0, cfg.vocab_size, (2, 64))
    model.eval()
    with torch.no_grad():
        logits, _ = model(ids)
        print(f"  logits shape: {tuple(logits.shape)}")
    model.train()
    _, loss = model(ids, targets)
    print(f"  train loss (FP, untrained): {loss.item():.4f} "
          f"(random ≈ {torch.log(torch.tensor(cfg.vocab_size, dtype=torch.float)).item():.4f})")
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  backward ok, max grad norm: {max(grad_norms):.4f}")
    model.zero_grad()

    print(f"\n[check] QAT forward pass test")
    n_qat = set_quantize_mode(model, on=True)
    print(f"  toggled {n_qat} QuantizedLinear modules to ternary")
    _, loss = model(ids, targets)
    print(f"  train loss (QAT, untrained): {loss.item():.4f}")
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  backward ok, max grad norm: {max(grad_norms):.4f}")
    model.zero_grad()

    print(f"\n[check] trace mode test")
    model.eval()
    with torch.no_grad():
        logits, trace = model(ids, return_trace=True)
    print(f"  logits: {tuple(logits.shape)}")
    print(f"  trace keys: {list(trace.keys())}")
    print(f"  meaning: {tuple(trace['meaning'].shape)}")
    print(f"  attn_per_layer[0]: {tuple(trace['attn_per_layer'][0].shape)}")
    print(f"  route_per_layer[0]: {tuple(trace['route_per_layer'][0].shape)}")
    print(f"  hidden_out: {tuple(trace['hidden_out'].shape)}")

    print(f"\n[check] all good. ~{bd['TOTAL']/1e6:.1f}M params.")


if __name__ == "__main__":
    main()
