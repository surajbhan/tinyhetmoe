#!/usr/bin/env python3
"""make_unified_meaning_axes.py — build (vocab_size, 132) meaning embedding
for the unified-vocab models (v6+).

Same approach as `make_meaning_axes.py` but operates on the unified
Qwen-vocab vocab.json instead of the GPT-2-derived TinyStories vocab.

The 132-axis production matrix is keyed on Qwen vocab. Since our unified
vocab is also Qwen-based (just pruned), every tiny_id maps directly to
exactly one Qwen ID — no remap-and-average needed. Pure 1-to-1 lookup.

Output:
  data/unified_meaning_axes_132.npy    (vocab_size, 132) float32
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
TOK_DIR = REPO / "tokenizer"

PROD_AXES = Path("/data/sematic-embedding/contextual_embeddings/contextual_axis_embeddings.npy")
PROD_NAMES = Path("/data/sematic-embedding/contextual_embeddings/contextual_axis_names.json")


def main():
    print(f"[uni-axes] loading production 132-axis file: {PROD_AXES}")
    qwen_axes = np.load(PROD_AXES).astype(np.float32)
    print(f"[uni-axes] qwen_axes shape: {qwen_axes.shape}")
    assert qwen_axes.shape[1] == 132

    axis_names = json.load(PROD_NAMES.open())

    print(f"[uni-axes] loading unified vocab")
    vocab = json.load((TOK_DIR / "unified_vocab.json").open())
    tiny_to_qwen = vocab["tiny_to_qwen"]
    vocab_size = vocab["vocab_size"]
    n_specials = len(vocab["specials"])
    print(f"[uni-axes] vocab_size: {vocab_size}, n_specials: {n_specials}")

    # Compute fallback (corpus mean of seen Qwen tokens)
    seen_mask = (qwen_axes.sum(axis=1) != 0)
    fallback = qwen_axes[seen_mask].mean(axis=0).astype(np.float32)
    print(f"[uni-axes] fallback norm: {np.linalg.norm(fallback):.4f} "
          f"({seen_mask.sum()} seen qwen tokens)")

    tiny_axes = np.zeros((vocab_size, 132), dtype=np.float32)
    n_direct = 0
    n_fallback = 0
    n_special = 0

    for tid in range(vocab_size):
        qid = tiny_to_qwen[tid]
        if qid is None:
            n_special += 1
            continue
        if seen_mask[qid]:
            tiny_axes[tid] = qwen_axes[qid]
            n_direct += 1
        else:
            tiny_axes[tid] = fallback
            n_fallback += 1

    out_path = DATA_DIR / "unified_meaning_axes_132.npy"
    np.save(out_path, tiny_axes)
    print(f"[uni-axes] wrote {out_path} ({tiny_axes.shape}, "
          f"{tiny_axes.nbytes/1e6:.1f} MB)")
    print(f"  direct Qwen match: {n_direct} ({n_direct/vocab_size*100:.1f}%)")
    print(f"  fallback:          {n_fallback} ({n_fallback/vocab_size*100:.1f}%)")
    print(f"  specials:          {n_special}")


if __name__ == "__main__":
    main()
