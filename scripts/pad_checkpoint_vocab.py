#!/usr/bin/env python3
"""pad_checkpoint_vocab.py — grow embed/lm_head vocab dim by appending zeros.

After the ChatML vocab patch (32768 → 32772), pg19/wiki checkpoints
trained at the old size have 4 fewer rows in:
  - meaning_embed.weight       (vocab_size, meaning_dim)
  - intuition_embed.weight     (vocab_size, intuition_dim)
  - lm_head.weight             (vocab_size, input_dim)

Append 4 zero rows to each so the checkpoint loads with the new vocab=32772
config. The 4 new rows correspond to ChatML special tokens that pg19/wiki
training data never contained — model never learned them, won't ever
generate them, and will see them as <unk>-equivalent at inference.

That's correct behavior: pg19/wiki experts are continuation models for raw
prose, not chat models. They'd never produce <|im_start|> on their own.
The hierarchical router won't route ChatML formatting to them either.

For meaning_embed specifically, we initialize the 4 new rows from the
production Qwen contextual file (the meaning axes for those qids) — that
way the meaning vectors for ChatML tokens are correct even though the
intuition_embed and lm_head rows stay zero.

Usage:
  python3 scripts/pad_checkpoint_vocab.py \\
    --in runs/tiny_hetmoe_v6_5_pg19/checkpoints/best.pt \\
    --out runs/tiny_hetmoe_v6_5_pg19/checkpoints/best_padded.pt \\
    --new-vocab 32772
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ckpt_in", required=True,
                    help="Path to old checkpoint (.pt)")
    ap.add_argument("--out", dest="ckpt_out", required=True,
                    help="Path for padded checkpoint (.pt)")
    ap.add_argument("--new-vocab", type=int, default=32772,
                    help="Target vocab_size after padding")
    ap.add_argument("--meaning-emb",
                    default=str(DATA_DIR / "unified_meaning_axes_132.npy"),
                    help="Path to (new-vocab, 132) meaning matrix; the new "
                         "rows in meaning_embed are copied from here")
    args = ap.parse_args()

    ckpt_in = Path(args.ckpt_in)
    ckpt_out = Path(args.ckpt_out)
    print(f"[pad] loading {ckpt_in}")
    blob = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob.get("state_dict", blob))

    # Detect old vocab from any of the embedding tensors
    old_vocab = None
    for key in ("meaning_embed.weight", "intuition_embed.weight", "lm_head.weight"):
        if key in sd:
            old_vocab = sd[key].shape[0]
            break
    if old_vocab is None:
        raise RuntimeError(f"Could not find embed/lm_head in {ckpt_in}")

    n_new = args.new_vocab - old_vocab
    if n_new == 0:
        print(f"[pad] checkpoint already at vocab={args.new_vocab}, nothing to do")
        return
    if n_new < 0:
        raise RuntimeError(f"Target vocab {args.new_vocab} < checkpoint vocab {old_vocab}")
    print(f"[pad] old vocab={old_vocab}, new vocab={args.new_vocab}, "
          f"appending {n_new} rows")

    # Load meaning axes for new rows
    print(f"[pad] loading meaning axes from {args.meaning_emb}")
    axes = np.load(args.meaning_emb).astype(np.float32)
    if axes.shape[0] != args.new_vocab:
        raise RuntimeError(
            f"Meaning axes shape {axes.shape} doesn't match new vocab "
            f"{args.new_vocab}")

    # All padded rows are zeros. Consumers (eval, wasm engine) must mask
    # logits for tids ≥ old_vocab on pg19/wiki/code experts to suppress
    # spurious emission. The zero row is ARCHITECTURALLY required (so all
    # 4 experts share vocab=32772 and can be served by one engine), but
    # SEMANTICALLY meaningless — these experts never emit ChatML tokens.
    targets = [
        ("meaning_embed.weight",
         torch.from_numpy(axes[old_vocab:args.new_vocab, :])),
        ("intuition_embed.weight", None),  # zeros — never looked up
        ("lm_head.weight",         None),  # zeros — masked at inference
    ]

    for key, init_block in targets:
        if key not in sd:
            print(f"[pad]   skipping {key} (not in checkpoint)")
            continue
        old = sd[key]
        cols = old.shape[1]
        if init_block is None:
            block = torch.zeros((n_new, cols), dtype=old.dtype)
        else:
            block = init_block.to(old.dtype)
            if block.shape != (n_new, cols):
                raise RuntimeError(
                    f"Init block for {key} has shape {block.shape}, "
                    f"expected ({n_new}, {cols})")
        new = torch.cat([old, block], dim=0)
        sd[key] = new
        print(f"[pad]   {key}: {tuple(old.shape)} -> {tuple(new.shape)}")

    # Reassemble blob, preserving any other top-level fields (best_val,
    # step, optimizer state, etc.)
    if "model" in blob:
        blob["model"] = sd
    elif "state_dict" in blob:
        blob["state_dict"] = sd
    else:
        blob = sd

    # Strip optimizer state — likely shape-mismatched against new vocab
    # and we don't intend to resume training from this checkpoint.
    if isinstance(blob, dict):
        for k in ("optimizer", "optim_state"):
            if k in blob:
                del blob[k]
                print(f"[pad]   stripped {k} (shape mismatch on resume)")

        # Update saved config's vocab_size so downstream tooling
        # (scripts/export_model.py, etc.) sees the real shape.
        if "config" in blob and isinstance(blob["config"], dict):
            old_v = blob["config"].get("vocab_size")
            if old_v != args.new_vocab:
                blob["config"]["vocab_size"] = args.new_vocab
                print(f"[pad]   updated config.vocab_size: {old_v} -> "
                      f"{args.new_vocab}")

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, ckpt_out)
    print(f"[pad] wrote {ckpt_out} ({ckpt_out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
