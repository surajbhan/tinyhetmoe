#!/usr/bin/env python3
"""prepare_data.py — download TinyStories, prune GPT-2 tokenizer to ~7K
vocab on actually-used tokens, tokenize, write train.bin / val.bin.

Output:
  data/train.bin           uint16 token ids for the pruned vocab
  data/val.bin             uint16 token ids
  tokenizer/vocab.json     {"gpt2_to_tiny": {gpt2_id: tiny_id},
                            "tiny_to_gpt2": [gpt2_id, ...],
                            "vocab_size": int,
                            "specials": {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3}}

Vocab structure:
  - tiny_id 0..3: <unk>, <bos>, <eos>, <pad>
  - tiny_id 4..N: top-K GPT-2 tokens that appear in TinyStories
                  (K chosen so coverage > 99.5% of corpus tokens)

Run once. ~10-15 min: HF download (~1 GB), counting (~3 min CPU),
tokenize+write (~5 min).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
TOK_DIR = REPO / "tokenizer"
DATA_DIR.mkdir(exist_ok=True)
TOK_DIR.mkdir(exist_ok=True)

# Special tokens (we add these; they don't exist in pure GPT-2)
SPECIALS = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3}
N_SPECIALS = len(SPECIALS)

# Coverage target — keep enough GPT-2 tokens to cover this fraction
# of all token occurrences. 99.5% means ~0.5% of tokens get <unk>'d,
# which is plenty for TinyStories' simple vocab.
COVERAGE_TARGET = 0.995

# Hard cap on vocab size — keeps embeddings small even if coverage
# would prefer more. 8192 is a clean round number.
MAX_VOCAB = 8192


def main():
    print("[prep] loading GPT-2 tokenizer")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"       gpt2 vocab size: {tok.vocab_size}")

    print("[prep] loading TinyStories from HuggingFace")
    ds_train = load_dataset("roneneldan/TinyStories", split="train")
    ds_val = load_dataset("roneneldan/TinyStories", split="validation")
    print(f"       train: {len(ds_train):,} stories")
    print(f"       val:   {len(ds_val):,} stories")

    print("[prep] tokenizing train (counting GPT-2 token frequencies)")
    counter = Counter()
    n_tokens_total = 0
    BATCH = 1000
    for i in range(0, len(ds_train), BATCH):
        batch = ds_train[i:i + BATCH]["text"]
        # Encode each story; GPT-2 doesn't insert BOS/EOS by default
        enc = tok(batch, add_special_tokens=False)["input_ids"]
        for ids in enc:
            counter.update(ids)
            n_tokens_total += len(ids)
        if i % 50000 == 0:
            print(f"       {i:>7}/{len(ds_train)} stories  "
                  f"({n_tokens_total/1e6:.1f}M tokens, "
                  f"{len(counter)} unique gpt2 ids seen)")

    print(f"[prep] total train tokens: {n_tokens_total:,}")
    print(f"[prep] unique gpt2 ids in train: {len(counter):,}")

    # Sort by frequency desc, pick top-K to hit coverage target
    sorted_ids = counter.most_common()
    cum = 0
    cutoff = len(sorted_ids)
    for i, (_, c) in enumerate(sorted_ids):
        cum += c
        if cum / n_tokens_total >= COVERAGE_TARGET:
            cutoff = i + 1
            break
    cutoff = min(cutoff, MAX_VOCAB - N_SPECIALS)
    print(f"[prep] keeping top {cutoff} gpt2 ids "
          f"(coverage {cum/n_tokens_total*100:.2f}%, cap {MAX_VOCAB})")

    kept_gpt2_ids = [gid for gid, _ in sorted_ids[:cutoff]]

    # Build remap: gpt2_id -> tiny_id
    # tiny ids 0..N_SPECIALS-1 are reserved for specials.
    gpt2_to_tiny: dict[int, int] = {}
    tiny_to_gpt2: list[int | None] = [None] * (N_SPECIALS + cutoff)
    for tiny_id, name in [(0, "<unk>"), (1, "<bos>"), (2, "<eos>"), (3, "<pad>")]:
        tiny_to_gpt2[tiny_id] = None  # specials don't map to a gpt2 id
    for i, gid in enumerate(kept_gpt2_ids):
        tiny_id = N_SPECIALS + i
        gpt2_to_tiny[gid] = tiny_id
        tiny_to_gpt2[tiny_id] = gid

    vocab_size = N_SPECIALS + cutoff
    print(f"[prep] final tiny vocab size: {vocab_size}")
    assert vocab_size <= 65535, "vocab must fit in uint16"

    # Save vocab mapping
    vocab_path = TOK_DIR / "vocab.json"
    with vocab_path.open("w") as f:
        json.dump({
            "gpt2_to_tiny": {str(k): v for k, v in gpt2_to_tiny.items()},
            "tiny_to_gpt2": tiny_to_gpt2,
            "vocab_size": vocab_size,
            "specials": SPECIALS,
            "coverage_target": COVERAGE_TARGET,
            "actual_coverage": cum / n_tokens_total,
            "n_train_tokens": n_tokens_total,
        }, f, indent=2)
    print(f"[prep] wrote {vocab_path}")

    def remap_and_write(ds, out_path: Path, label: str):
        """Tokenize ds, remap to tiny ids, write uint16 .bin."""
        unk = SPECIALS["<unk>"]
        bos = SPECIALS["<bos>"]
        eos = SPECIALS["<eos>"]
        # Pre-allocate generously — we'll trim. ~1.4x avg story length
        # gives plenty of headroom.
        chunks: list[np.ndarray] = []
        n_unk = 0
        n_total = 0
        for i in range(0, len(ds), BATCH):
            batch = ds[i:i + BATCH]["text"]
            enc = tok(batch, add_special_tokens=False)["input_ids"]
            for ids in enc:
                # remap, prepend bos, append eos
                arr = np.empty(len(ids) + 2, dtype=np.uint16)
                arr[0] = bos
                arr[-1] = eos
                for j, gid in enumerate(ids):
                    tid = gpt2_to_tiny.get(gid, unk)
                    arr[j + 1] = tid
                    if tid == unk:
                        n_unk += 1
                    n_total += 1
                chunks.append(arr)
            if i % 50000 == 0:
                so_far = sum(c.size for c in chunks)
                print(f"       {label} {i:>7}/{len(ds)}  "
                      f"({so_far/1e6:.1f}M tiny tokens written)")
        all_arr = np.concatenate(chunks)
        all_arr.tofile(out_path)
        print(f"[prep] wrote {out_path} ({all_arr.size:,} uint16 tokens, "
              f"{all_arr.nbytes/1e6:.1f} MB)")
        print(f"       unk rate: {n_unk/n_total*100:.3f}%")

    print("[prep] writing train.bin")
    remap_and_write(ds_train, DATA_DIR / "train.bin", "train")

    print("[prep] writing val.bin")
    remap_and_write(ds_val, DATA_DIR / "val.bin", "val")

    print("[prep] done.")


if __name__ == "__main__":
    main()
