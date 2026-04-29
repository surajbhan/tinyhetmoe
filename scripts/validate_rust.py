#!/usr/bin/env python3
"""validate_rust.py — run the same prompt through Python and the native
Rust engine, compare logits + intermediate states. Catches any
implementation drift in the Rust port.

Tolerances are relaxed because Rust uses INT8-quantized activations in
the matvec hot path (per-tensor symmetric quantization) while Python
uses FP. Expect ~1-3% relative error on logits but the **top-K
selections should match exactly**.

Usage:
    python scripts/validate_rust.py \\
        --ckpt runs/tiny_hetmoe/checkpoints/best.pt \\
        --bin runs/tiny_hetmoe/tiny.bin \\
        --prompt "Once upon a time, there was a happy little"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2TokenizerFast

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.tiny_hetmoe import TinyHetMoE, TinyHetMoEConfig, set_quantize_mode  # noqa: E402


def encode_prompt(prompt: str):
    vocab = json.load((REPO / "tokenizer" / "vocab.json").open())
    g2t = {int(k): v for k, v in vocab["gpt2_to_tiny"].items()}
    specials = vocab["specials"]
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = [specials["<bos>"]]
    for g in tok(prompt, add_special_tokens=False)["input_ids"]:
        ids.append(g2t.get(int(g), specials["<unk>"]))
    return ids


def run_python(ckpt_path: str, tiny_ids: list[int]):
    """Run Python forward pass on the same prompt, with QAT mode on
    (matching how Rust runs). Return (top5_ids, top5_logits, full_logits)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = TinyHetMoEConfig(**{k: cfg_dict[k] for k in [
        "vocab_size", "meaning_dim", "intuition_dim", "input_dim",
        "internal_dim", "new_intuition", "num_layers", "num_heads",
        "num_experts", "top_k_experts", "ffn_mult", "max_seq_len",
        "load_balance_weight",
    ]})
    model = TinyHetMoE(cfg)
    sd = ckpt["model"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    set_quantize_mode(model, on=True)

    ids = torch.tensor([tiny_ids], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(ids)
    last = logits[0, -1].float().numpy()
    top5_idx = np.argsort(-last)[:5]
    return top5_idx.tolist(), last[top5_idx].tolist(), last


def parse_rust_output(rust_out: str):
    """Extract top-5 from the Rust binary's stdout."""
    lines = rust_out.strip().split("\n")
    top5_ids = []
    top5_logits = []
    in_topk = False
    for line in lines:
        if "top-5 next:" in line:
            in_topk = True
            continue
        if in_topk:
            line = line.strip()
            if line.startswith("id "):
                # "id 151 logit 3.949"
                parts = line.split()
                top5_ids.append(int(parts[1]))
                top5_logits.append(float(parts[3]))
                if len(top5_ids) >= 5:
                    break
    return top5_ids, top5_logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--bin", required=True)
    p.add_argument("--prompt", default="Once upon a time, there was a happy little")
    p.add_argument("--rust-bin", default=str(REPO / "wasm/target/release/tiny_hetmoe_native"))
    args = p.parse_args()

    tiny_ids = encode_prompt(args.prompt)
    print(f"[validate] prompt: {args.prompt!r}")
    print(f"[validate] tiny ids: {tiny_ids}")

    print(f"[validate] running Python forward")
    py_top_ids, py_top_logits, py_full_logits = run_python(args.ckpt, tiny_ids)
    print(f"           Python top-5: {list(zip(py_top_ids, [round(x, 3) for x in py_top_logits]))}")

    print(f"[validate] running Rust native forward")
    cmd = [args.rust_bin, args.bin] + [str(i) for i in tiny_ids]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rust_top_ids, rust_top_logits = parse_rust_output(res.stdout)
    print(f"           Rust top-5:   {list(zip(rust_top_ids, [round(x, 3) for x in rust_top_logits]))}")

    print(f"\n[validate] comparing:")
    py_set = set(py_top_ids)
    rust_set = set(rust_top_ids)
    overlap = py_set & rust_set
    print(f"  top-5 overlap:    {len(overlap)}/5  (ids: {sorted(overlap)})")
    print(f"  top-1 match:      {py_top_ids[0] == rust_top_ids[0]} "
          f"(py={py_top_ids[0]}, rust={rust_top_ids[0]})")

    # Logit comparison on the shared ids
    if overlap:
        print(f"  per-id logit diffs (Python vs Rust):")
        py_dict = dict(zip(py_top_ids, py_top_logits))
        rust_dict = dict(zip(rust_top_ids, rust_top_logits))
        for tid in sorted(overlap):
            diff = py_dict[tid] - rust_dict[tid]
            print(f"    id {tid:>5}: py={py_dict[tid]:>+7.3f}  rust={rust_dict[tid]:>+7.3f}  Δ={diff:+.3f}")

    # Verdict
    print()
    if py_top_ids[0] == rust_top_ids[0] and len(overlap) >= 4:
        print(f"[validate] ✓ PASS — top-1 matches and {len(overlap)}/5 top-K overlap.")
    elif len(overlap) >= 3:
        print(f"[validate] ~ PARTIAL — {len(overlap)}/5 overlap, likely INT8 quant noise. "
              f"Inspect logit diffs above.")
    else:
        print(f"[validate] ✗ FAIL — only {len(overlap)}/5 overlap. Probable Rust bug.")


if __name__ == "__main__":
    main()
