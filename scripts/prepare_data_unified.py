#!/usr/bin/env python3
"""prepare_data_unified.py — unified Qwen-vocab dataprep for multi-domain MoE.

Builds a single trimmed Qwen2.5 tokenizer that covers TinyStories +
WikiText-103 (and later: tool-calling) with high coverage. Tokenizes
each corpus into its own .bin for parallel training.

Why Qwen2.5 tokenizer:
  - Matches our meaning-axis extraction pipeline (Qwen2.5-Coder-0.5B
    contextual embeddings already keyed on Qwen vocab)
  - Public, well-tested, handles code + prose + symbols
  - 151K tokens; we prune to ~16K covering both corpora at 99%+

Output:
  data/unified_train_stories.bin     uint16 token ids (TinyStories train)
  data/unified_val_stories.bin
  data/unified_train_wiki.bin        uint16 token ids (WikiText-103 train)
  data/unified_val_wiki.bin
  tokenizer/unified_vocab.json       {qwen_to_tiny, tiny_to_qwen, vocab_size, ...}
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
TOK_DIR = REPO / "tokenizer"
DATA_DIR.mkdir(exist_ok=True)
TOK_DIR.mkdir(exist_ok=True)

SPECIALS = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3}
N_SPECIALS = len(SPECIALS)
COVERAGE_TARGET = 0.995
MAX_VOCAB = 16384  # 14-bit, comfortable in uint16


def main():
    print("[uni] loading Qwen2.5-Coder-0.5B tokenizer")
    qwen_tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True,
    )
    print(f"     qwen_vocab_size: {len(qwen_tok)}")

    print("\n[uni] loading TinyStories")
    ts_train = load_dataset("roneneldan/TinyStories", split="train")
    ts_val = load_dataset("roneneldan/TinyStories", split="validation")
    print(f"     stories train: {len(ts_train):,}, val: {len(ts_val):,}")

    print("\n[uni] loading WikiText-103")
    wt_train = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
    )
    wt_val = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="validation",
    )
    print(f"     wiki train: {len(wt_train):,}, val: {len(wt_val):,}")

    # ── Pass 1: count token frequencies on union of train sets ──────
    print("\n[uni] counting token frequencies (Qwen tokenization)")
    counter: Counter[int] = Counter()
    n_total = 0
    BATCH = 1000

    def tally(ds, label):
        nonlocal n_total
        n_local = 0
        for i in range(0, len(ds), BATCH):
            batch = [t for t in ds[i:i + BATCH]["text"] if t and t.strip()]
            if not batch: continue
            enc = qwen_tok(batch, add_special_tokens=False)["input_ids"]
            for ids in enc:
                counter.update(ids)
                n_local += len(ids)
            if i and i % 100000 == 0:
                print(f"     {label} {i:>7}/{len(ds)} ({n_local/1e6:.1f}M tokens, "
                      f"{len(counter):,} unique)")
        n_total += n_local
        print(f"     {label} done: {n_local/1e6:.1f}M tokens")

    tally(ts_train, "stories")
    tally(wt_train, "wiki")

    print(f"\n[uni] total tokens across both: {n_total:,}")
    print(f"[uni] unique qwen ids in union:  {len(counter):,}")

    # Pick top-K covering target fraction
    sorted_ids = counter.most_common()
    cum = 0
    cutoff = len(sorted_ids)
    for i, (_, c) in enumerate(sorted_ids):
        cum += c
        if cum / n_total >= COVERAGE_TARGET:
            cutoff = i + 1
            break
    cutoff = min(cutoff, MAX_VOCAB - N_SPECIALS)
    print(f"[uni] keeping top {cutoff} qwen ids "
          f"(coverage {cum/n_total*100:.2f}%, cap {MAX_VOCAB})")

    kept_qwen_ids = [qid for qid, _ in sorted_ids[:cutoff]]
    qwen_to_tiny: dict[int, int] = {}
    tiny_to_qwen: list[int | None] = [None] * (N_SPECIALS + cutoff)
    for i, qid in enumerate(kept_qwen_ids):
        tid = N_SPECIALS + i
        qwen_to_tiny[qid] = tid
        tiny_to_qwen[tid] = qid

    vocab_size = N_SPECIALS + cutoff
    print(f"[uni] unified vocab_size: {vocab_size}")
    assert vocab_size <= 65535

    # Save vocab map
    vocab_path = TOK_DIR / "unified_vocab.json"
    json.dump({
        "tokenizer_source": "Qwen/Qwen2.5-Coder-0.5B",
        "qwen_to_tiny": {str(k): v for k, v in qwen_to_tiny.items()},
        "tiny_to_qwen": tiny_to_qwen,
        "vocab_size": vocab_size,
        "specials": SPECIALS,
        "coverage_target": COVERAGE_TARGET,
        "actual_coverage": cum / n_total,
        "n_train_tokens": n_total,
        "domains": ["tinystories", "wikitext-103"],
    }, vocab_path.open("w"), indent=2)
    print(f"[uni] wrote {vocab_path}")

    # ── Pass 2: write .bin files for each (corpus, split) ────────────
    def write_bin(ds, out_path: Path, label: str):
        unk = SPECIALS["<unk>"]
        bos = SPECIALS["<bos>"]
        eos = SPECIALS["<eos>"]
        chunks: list[np.ndarray] = []
        n_unk = 0
        n_local = 0
        chunks.append(np.array([bos], dtype=np.uint16))
        for i in range(0, len(ds), BATCH):
            batch = [t for t in ds[i:i + BATCH]["text"] if t and t.strip()]
            if not batch: continue
            enc = qwen_tok(batch, add_special_tokens=False)["input_ids"]
            for ids in enc:
                arr = np.empty(len(ids) + 1, dtype=np.uint16)
                arr[-1] = eos  # eos between docs
                for j, qid in enumerate(ids):
                    tid = qwen_to_tiny.get(int(qid), unk)
                    arr[j] = tid
                    if tid == unk:
                        n_unk += 1
                    n_local += 1
                chunks.append(arr)
            if i and i % 100000 == 0:
                so_far = sum(c.size for c in chunks)
                print(f"     {label} {i:>7}/{len(ds)} ({so_far/1e6:.1f}M)")
        all_arr = np.concatenate(chunks)
        all_arr.tofile(out_path)
        print(f"[uni] wrote {out_path} ({all_arr.size:,} tokens, "
              f"{all_arr.nbytes/1e6:.1f} MB), unk rate {n_unk/n_local*100:.3f}%")

    print("\n[uni] writing TinyStories")
    write_bin(ts_train, DATA_DIR / "unified_train_stories.bin", "ts_train")
    write_bin(ts_val,   DATA_DIR / "unified_val_stories.bin",   "ts_val")

    print("\n[uni] writing WikiText-103")
    write_bin(wt_train, DATA_DIR / "unified_train_wiki.bin",    "wt_train")
    write_bin(wt_val,   DATA_DIR / "unified_val_wiki.bin",      "wt_val")

    print("\n[uni] done.")


if __name__ == "__main__":
    main()
