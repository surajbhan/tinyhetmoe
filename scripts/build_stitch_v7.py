#!/usr/bin/env python3
"""build_stitch_v7.py — assemble v7 stitch.json for the browser.

Inputs:
  - docs/stitch_v7/<domain>.bin     (6 expert files, already exported)
  - docs/stitch_v7/meaning_shared.bin (shared meaning_embed)
  - data_v7/domain_classifier_v7.pt (trained 6-way classifier)

Output:
  docs/stitch_v7/stitch.json — bundle manifest with classifier weights

The browser loads stitch.json at startup, downloads meaning_shared.bin
once, then lazily fetches per-expert .bin files based on classifier
routing decisions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent

DOMAINS = ["general", "thinker", "code_py", "code_js", "medical", "legal"]


def _safe_float(v):
    """Coerce NaN/Inf/None to None for JSON safety."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f == float("inf") or f == float("-inf"):  # NaN check
        return None
    return f


def main():
    out_dir = REPO / "docs" / "stitch_v7"
    if not out_dir.exists():
        raise SystemExit(f"out_dir {out_dir} does not exist; run export_model first")

    # Verify all 6 expert .bin files exist
    bin_files = []
    for d in DOMAINS:
        p = out_dir / f"{d}.bin"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        meta_p = p.with_suffix(".meta.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        bin_files.append((d, p.name, p.stat().st_size, meta))

    # Verify shared meaning
    meaning_p = out_dir / "meaning_shared.bin"
    if not meaning_p.exists():
        raise SystemExit(f"missing {meaning_p}")
    meaning_size = meaning_p.stat().st_size

    # Load classifier
    clf_p = REPO / "data_v7" / "domain_classifier_v7.pt"
    print(f"[stitch] loading classifier {clf_p}")
    clf = torch.load(clf_p, map_location="cpu", weights_only=False)
    sd = clf["state_dict"]

    # Sequential layout: net.0 = Linear(132, 64), net.3 = Linear(64, 6)
    w1 = sd["net.0.weight"].cpu().numpy()  # (64, 132)
    b1 = sd["net.0.bias"].cpu().numpy()    # (64,)
    w2 = sd["net.3.weight"].cpu().numpy()  # (6, 64)
    b2 = sd["net.3.bias"].cpu().numpy()    # (6,)
    feat_mu = clf["feat_mean"].squeeze()   # (132,)
    feat_sd = clf["feat_std"].squeeze()    # (132,)
    domains_in_clf = clf.get("domains", DOMAINS)
    assert domains_in_clf == DOMAINS, \
        f"classifier domain order mismatch: {domains_in_clf} vs {DOMAINS}"
    ema_alpha = clf.get("ema_alpha", 0.05)

    # Build stitch.json
    stitch = {
        "format_version": "STITCHV7_001",
        "ema_alpha": ema_alpha,
        "meaning_dim": 132,
        "intuition_dim": 132,
        "vocab_size": 151665,
        "tokenizer": "Qwen/Qwen2.5-Coder-0.5B",
        "shared_meaning": {
            "url": "meaning_shared.bin",
            "format": "MNGSHR04",
            "size_bytes": meaning_size,
        },
        "experts": [
            {
                "name": d,
                "url": fname,
                "format": meta.get("format", "HTMOE004"),
                "size_bytes": size,
                # NaN → null so JS JSON.parse doesn't choke (Python's
                # json.dumps emits literal "NaN" which is non-standard).
                "best_val": _safe_float(meta.get("best_val")),
                "training_step": meta.get("training_step"),
            }
            for (d, fname, size, meta) in bin_files
        ],
        "classifier": {
            "input_dim": 132,
            "hidden": int(b1.shape[0]),
            "output_dim": int(b2.shape[0]),
            "domains": DOMAINS,
            "best_val_acc": float(clf.get("best_val_acc", 0.0)),
            "w1": w1.astype(np.float32).flatten().tolist(),
            "b1": b1.astype(np.float32).tolist(),
            "w2": w2.astype(np.float32).flatten().tolist(),
            "b2": b2.astype(np.float32).tolist(),
            "feat_mu": feat_mu.astype(np.float32).tolist(),
            "feat_sd": feat_sd.astype(np.float32).tolist(),
        },
    }

    out_path = out_dir / "stitch.json"
    # allow_nan=False makes json.dumps raise on NaN/Inf instead of silently
    # emitting non-standard "NaN" / "Infinity" tokens that fail in browsers.
    out_path.write_text(json.dumps(stitch, indent=2, allow_nan=False))
    print(f"[stitch] wrote {out_path}: {out_path.stat().st_size/1e6:.2f} MB")

    # Bundle summary
    print(f"\n[stitch] BUNDLE INVENTORY:")
    print(f"  shared meaning:    {meaning_size/1e6:.1f} MB  (downloaded once)")
    for d, fname, size, _ in bin_files:
        print(f"  {d:<10}:        {size/1e6:.1f} MB  (lazy)")
    print(f"  classifier:        {out_path.stat().st_size/1e6:.2f} MB  (in stitch.json)")
    total = meaning_size + sum(s for _,_,s,_ in bin_files) + out_path.stat().st_size
    print(f"  TOTAL (full):      {total/1e6:.1f} MB")
    print(f"  TYPICAL (1 expert): {(meaning_size + bin_files[0][2] + out_path.stat().st_size)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
