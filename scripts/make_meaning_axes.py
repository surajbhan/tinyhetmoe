#!/usr/bin/env python3
"""make_meaning_axes.py — produce a (vocab_size, 132) meaning embedding
matrix for the pruned-GPT-2 TinyHetMoE vocab.

We **reuse** the production 132-axis file:
    /data/sematic-embedding/contextual_embeddings/contextual_axis_embeddings.npy
which is keyed on the Qwen2.5-Coder-0.5B vocab (151665 tokens). The
production model uses these same axes — TinyHetMoE shares the recipe at
small scale.

Approach: for each tiny token id (in the pruned GPT-2 space), decode to
a UTF-8 string, re-encode with Qwen, and copy the corresponding axis
vector. Multi-subtoken Qwen encodings are averaged. Specials and
no-match tokens get the corpus-mean axis vector (zero-centered, so
≈0).

Output:
  data/meaning_axes_132.npy        (vocab_size, 132) float32
  data/meaning_axis_names.json     ["I", "YOU", "SOMEONE", ...]  (132 names)

No GPU needed. ~30 seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, GPT2TokenizerFast

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
TOK_DIR = REPO / "tokenizer"
DATA_DIR.mkdir(exist_ok=True)

# Production 132-axis file, keyed on Qwen vocab
PROD_AXES = Path("/data/sematic-embedding/contextual_embeddings/contextual_axis_embeddings.npy")
PROD_NAMES = Path("/data/sematic-embedding/contextual_embeddings/contextual_axis_names.json")


def main():
    print(f"[axes] loading production axes from {PROD_AXES}")
    qwen_axes = np.load(PROD_AXES).astype(np.float32)
    axis_names = json.load(PROD_NAMES.open())
    n_axes = qwen_axes.shape[1]
    print(f"[axes] qwen_axes shape: {qwen_axes.shape}, n_axes: {n_axes}")
    assert n_axes == len(axis_names) == 132, "expected 132 axes"

    print("[axes] loading tiny vocab")
    vocab_path = TOK_DIR / "vocab.json"
    vocab = json.load(vocab_path.open())
    tiny_vocab_size = vocab["vocab_size"]
    tiny_to_gpt2: list[int | None] = vocab["tiny_to_gpt2"]
    n_specials = len(vocab["specials"])
    print(f"[axes] tiny vocab size: {tiny_vocab_size} (specials: {n_specials})")

    print("[axes] loading GPT-2 + Qwen tokenizers")
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
    qwen_tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True,
    )

    # Compute corpus mean from non-zero qwen tokens — use as fallback
    seen_mask = (qwen_axes.sum(axis=1) != 0)
    fallback = qwen_axes[seen_mask].mean(axis=0).astype(np.float32)
    print(f"[axes] fallback axis vector norm: {np.linalg.norm(fallback):.4f} "
          f"(across {seen_mask.sum()} qwen tokens with data)")

    tiny_axes = np.zeros((tiny_vocab_size, n_axes), dtype=np.float32)
    n_single = 0   # one-to-one Qwen match
    n_multi = 0    # multi-Qwen-subtoken, averaged
    n_fallback = 0
    n_special = 0

    for tiny_id in range(tiny_vocab_size):
        gpt2_id = tiny_to_gpt2[tiny_id]
        if gpt2_id is None:
            # Special token
            n_special += 1
            continue
        # Decode GPT-2 token → string
        text = gpt2_tok.decode([gpt2_id])
        # Re-encode with Qwen
        qwen_ids = qwen_tok(text, add_special_tokens=False)["input_ids"]
        if len(qwen_ids) == 0:
            tiny_axes[tiny_id] = fallback
            n_fallback += 1
            continue
        if len(qwen_ids) == 1:
            qid = qwen_ids[0]
            if seen_mask[qid]:
                tiny_axes[tiny_id] = qwen_axes[qid]
                n_single += 1
            else:
                tiny_axes[tiny_id] = fallback
                n_fallback += 1
        else:
            # Average over Qwen subtokens (only seen ones)
            subseen = [qid for qid in qwen_ids if seen_mask[qid]]
            if subseen:
                tiny_axes[tiny_id] = qwen_axes[subseen].mean(axis=0)
                n_multi += 1
            else:
                tiny_axes[tiny_id] = fallback
                n_fallback += 1

    out_path = DATA_DIR / "meaning_axes_132.npy"
    np.save(out_path, tiny_axes)
    print(f"\n[axes] wrote {out_path} ({tiny_axes.shape}, "
          f"{tiny_axes.nbytes/1e6:.1f} MB)")

    names_path = DATA_DIR / "meaning_axis_names.json"
    json.dump(axis_names, names_path.open("w"), indent=2)
    print(f"[axes] wrote {names_path} ({len(axis_names)} names)")

    print(f"\n[axes] coverage:")
    print(f"  single-token Qwen match: {n_single} ({n_single/tiny_vocab_size*100:.1f}%)")
    print(f"  multi-token avg:         {n_multi} ({n_multi/tiny_vocab_size*100:.1f}%)")
    print(f"  fallback (no/empty):     {n_fallback} ({n_fallback/tiny_vocab_size*100:.1f}%)")
    print(f"  specials:                {n_special}")

    sanity_check(tiny_axes, vocab, axis_names, gpt2_tok)


def sanity_check(tiny_axes, vocab, axis_names, gpt2_tok):
    """Show top axes for a few interpretable words."""
    print("\n[sanity] axis activations for anchor words")
    gpt2_to_tiny = {int(k): v for k, v in vocab["gpt2_to_tiny"].items()}
    test_words = ["happy", "sad", "I", "you", "good", "bad", "saw",
                  "wanted", "scared", "loved", "small", "big",
                  "mother", "child", "dog", "cat"]
    for w in test_words:
        for variant in [" " + w, w]:
            ids = gpt2_tok(variant, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and ids[0] in gpt2_to_tiny:
                tiny_id = gpt2_to_tiny[ids[0]]
                vec = tiny_axes[tiny_id]
                top3 = np.argsort(np.abs(vec))[-3:][::-1]
                top_str = ", ".join(
                    f"{axis_names[i]:>15s} {vec[i]:+.2f}" for i in top3
                )
                print(f"  '{variant:>10s}' (tiny_id={tiny_id:>4d}): {top_str}")
                break
        else:
            print(f"  '{w}': not in tiny vocab as single token")


if __name__ == "__main__":
    main()
