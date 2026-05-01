#!/usr/bin/env python3
"""build_stitch_bundle.py — produce a deployable bundle for the wasm
stitched-MoE engine.

Inputs:
  - One or more PyTorch best_qat.pt checkpoints (one per expert)
  - A trained classifier (132→hidden→K MLP)
  - Per-expert masked_tids list (logit-mask for tokens that expert
    should never emit; e.g. ChatML specials for prose experts)

Outputs into a target directory:
  - <expert_name>.bin       (HTMOE003 packed ternary, one per expert)
  - stitch.json             (classifier weights + per-expert metadata)

The wasm engine reads `stitch.json` plus the .bin files at load time.

Usage example:
  python3 scripts/build_stitch_bundle.py \\
      --out-dir docs/stitch_v66 \\
      --expert pg19=runs/.../best_qat_padded.pt \\
      --expert wiki=runs/.../best_qat_padded.pt \\
      --classifier data/domain_classifier_pg19_wiki_v66.pt \\
      --masked-tids 32768,32769,32770,32771 \\
      --ema-alpha 0.05
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from model.tiny_hetmoe import (
    TinyHetMoE, TinyHetMoEConfig, set_quantize_mode,
)


def export_expert_bin(ckpt_path: Path, out_bin: Path):
    """Run the standard export_model.py logic on a checkpoint, writing
    HTMOE003 packed ternary to out_bin. The exporter reads vocab/arch
    config from the checkpoint's saved `config` field, so the padded
    checkpoints (which now have config.vocab_size=32772) export at the
    right size."""
    import subprocess
    cmd = [
        "python3", str(REPO / "scripts/export_model.py"),
        "--ckpt", str(ckpt_path),
        "--out", str(out_bin),
    ]
    print(f"[export] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_classifier(path: Path) -> dict:
    """Load saved classifier (from train_domain_classifier.py output) and
    return weights + normalization stats as numpy arrays."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["state_dict"]
    # Sequential layout: net.0 = Linear(132,64), net.3 = Linear(64,K)
    w1 = sd["net.0.weight"].cpu().numpy()  # (hidden, 132)
    b1 = sd["net.0.bias"].cpu().numpy()    # (hidden,)
    w2 = sd["net.3.weight"].cpu().numpy()  # (K, hidden)
    b2 = sd["net.3.bias"].cpu().numpy()    # (K,)
    feat_mu = blob["feat_mean"].squeeze()  # (132,)
    feat_sd = blob["feat_std"].squeeze()   # (132,)
    domains = blob.get("domains", [])
    return dict(
        w1=w1, b1=b1, w2=w2, b2=b2,
        feat_mu=feat_mu, feat_sd=feat_sd,
        domains=domains,
        hidden=int(b1.shape[0]),
        k=int(b2.shape[0]),
    )


def parse_expert_arg(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise ValueError(f"--expert expects NAME=PATH, got {s!r}")
    name, p = s.split("=", 1)
    return name.strip(), Path(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Where to write bundle")
    ap.add_argument("--expert", action="append", required=True,
                    help="NAME=PATH/to/best_qat_padded.pt; pass multiple")
    ap.add_argument("--classifier", required=True,
                    help="Path to domain_classifier_*.pt")
    ap.add_argument("--masked-tids", default="",
                    help="Comma-separated tids to mask in PROSE experts' "
                         "logits (typically ChatML specials). Applied per "
                         "--mask-experts comma-separated names.")
    ap.add_argument("--mask-experts", default="",
                    help="Comma-separated names of experts that should "
                         "have masked_tids applied. Others get empty mask.")
    ap.add_argument("--ema-alpha", type=float, default=0.05,
                    help="EMA alpha used at runtime; must match training")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build] out: {out_dir}")

    # Parse experts
    experts = [parse_expert_arg(s) for s in args.expert]
    print(f"[build] experts: {[(n, str(p)) for n, p in experts]}")

    # Determine which get the mask
    masked_set = set(s.strip() for s in args.mask_experts.split(",") if s.strip())
    masked_tids = [int(s) for s in args.masked_tids.split(",") if s.strip()]
    print(f"[build] masked_experts: {sorted(masked_set)}")
    print(f"[build] masked_tids:    {masked_tids}")

    # Export each expert .bin
    bin_files = []
    for name, ckpt in experts:
        out_bin = out_dir / f"{name}.bin"
        export_expert_bin(ckpt, out_bin)
        bin_files.append((name, out_bin.name))

    # Load classifier
    print(f"[build] loading classifier {args.classifier}")
    clf = load_classifier(Path(args.classifier))
    print(f"[build] classifier: hidden={clf['hidden']}, k={clf['k']}, "
          f"domains={clf['domains']}")

    # Build stitch.json
    stitch = {
        "format_version": "STITCH001",
        "ema_alpha": args.ema_alpha,
        "meaning_dim": 132,
        "experts": [
            {
                "name": name,
                "bin": fname,
                "masked_tids": masked_tids if name in masked_set else [],
            }
            for (name, _), (_, fname) in zip(experts, bin_files)
        ],
        "classifier": {
            "hidden": clf["hidden"],
            "k": clf["k"],
            "domains": clf["domains"],
            # Weights are flattened row-major.
            "w1": clf["w1"].astype(np.float32).flatten().tolist(),
            "b1": clf["b1"].astype(np.float32).tolist(),
            "w2": clf["w2"].astype(np.float32).flatten().tolist(),
            "b2": clf["b2"].astype(np.float32).tolist(),
            "feat_mu": clf["feat_mu"].astype(np.float32).tolist(),
            "feat_sd": clf["feat_sd"].astype(np.float32).tolist(),
        },
    }

    out_stitch = out_dir / "stitch.json"
    out_stitch.write_text(json.dumps(stitch, indent=2))
    print(f"[build] wrote {out_stitch} ({out_stitch.stat().st_size/1e6:.2f} MB)")

    # Gzip each .bin so the browser can stream + decompress on download.
    # GitHub Pages doesn't auto-gzip .bin files; serving .bin.gz with the
    # JS-side DecompressionStream avoids that problem and cuts wire size
    # ~12% on top of our already-packed ternary format.
    import gzip, shutil
    print(f"\n[build] gzipping expert .bin files…")
    for name, fname in bin_files:
        bin_path = out_dir / fname
        gz_path = bin_path.with_suffix(".bin.gz")
        with bin_path.open("rb") as f_in, gzip.open(gz_path, "wb",
                                                     compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_mb = gz_path.stat().st_size / 1e6
        bin_mb = bin_path.stat().st_size / 1e6
        print(f"  {fname}: {bin_mb:.1f} MB → {fname}.gz {gz_mb:.1f} MB "
              f"({gz_mb/bin_mb*100:.0f}%)")

    # Size summary (post-gzip).
    print(f"\n[build] bundle contents:")
    total_bin_gz = sum((out_dir / f"{f}.gz").stat().st_size for _, f in bin_files)
    print(f"  experts (.bin.gz): {total_bin_gz/1e6:.1f} MB total "
          f"({len(bin_files)} files)")
    print(f"  stitch.json:       {out_stitch.stat().st_size/1e6:.2f} MB")
    if (out_dir / "encode.json").exists():
        print(f"  encode.json:       {(out_dir/'encode.json').stat().st_size/1e6:.2f} MB")
    if (out_dir / "decode.json").exists():
        print(f"  decode.json:       {(out_dir/'decode.json').stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
