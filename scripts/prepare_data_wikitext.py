#!/usr/bin/env python3
"""prepare_data_wikitext.py — WikiText-103 → pruned GPT-2 vocab → .bin.

Same structure as prepare_data.py (TinyStories) but for WikiText-103.

Why a separate vocab: WikiText-103 has different token frequencies than
TinyStories (encyclopedia vs simple narratives). Reusing the TinyStories
vocab would <unk>-out 5-15% of tokens depending on subject matter.

Output:
  data/wikitext_train.bin       uint16 token ids, pruned vocab
  data/wikitext_val.bin
  tokenizer/wikitext_vocab.json  vocab map (separate file from TinyStories)

~10-15 min CPU. Run once.
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

SPECIALS = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3}
N_SPECIALS = len(SPECIALS)

# WikiText has more vocab diversity than TinyStories — give it more headroom.
COVERAGE_TARGET = 0.997
MAX_VOCAB = 16384  # 2^14, fits in uint16


def main():
    print("[wt] loading GPT-2 tokenizer")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    print("[wt] loading WikiText-103 (Salesforce mirror)")
    # The wikitext dataset on HF has multiple configurations. wikitext-103-raw-v1
    # is the canonical "raw" (untokenized) version.
    ds_train = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    ds_val = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    print(f"     train: {len(ds_train):,} rows")
    print(f"     val:   {len(ds_val):,} rows")

    # WikiText rows are paragraph-ish chunks separated by blank lines.
    # We want to glue an article into a single sequence. The dataset's
    # convention: "= title =" (single =) starts an article, "== section =="
    # is a section. Simpler: just concatenate everything; the model will
    # see article boundaries via the headings.

    print("[wt] counting token frequencies on train")
    counter: Counter[int] = Counter()
    n_total = 0
    BATCH = 1000
    for i in range(0, len(ds_train), BATCH):
        batch = ds_train[i:i + BATCH]["text"]
        # Filter empty rows
        batch = [t for t in batch if t.strip()]
        if not batch:
            continue
        enc = tok(batch, add_special_tokens=False)["input_ids"]
        for ids in enc:
            counter.update(ids)
            n_total += len(ids)
        if i % 100000 == 0:
            print(f"     {i:>7}/{len(ds_train)} rows, "
                  f"{n_total/1e6:.1f}M tokens, {len(counter)} unique gpt2 ids")

    print(f"[wt] total train tokens: {n_total:,}, unique gpt2 ids: {len(counter):,}")

    sorted_ids = counter.most_common()
    cum = 0
    cutoff = len(sorted_ids)
    for i, (_, c) in enumerate(sorted_ids):
        cum += c
        if cum / n_total >= COVERAGE_TARGET:
            cutoff = i + 1
            break
    cutoff = min(cutoff, MAX_VOCAB - N_SPECIALS)
    print(f"[wt] keeping top {cutoff} gpt2 ids "
          f"(coverage {cum/n_total*100:.2f}%, cap {MAX_VOCAB})")

    kept_gpt2_ids = [gid for gid, _ in sorted_ids[:cutoff]]

    gpt2_to_tiny: dict[int, int] = {}
    tiny_to_gpt2: list[int | None] = [None] * (N_SPECIALS + cutoff)
    for i, gid in enumerate(kept_gpt2_ids):
        tiny_id = N_SPECIALS + i
        gpt2_to_tiny[gid] = tiny_id
        tiny_to_gpt2[tiny_id] = gid

    vocab_size = N_SPECIALS + cutoff
    print(f"[wt] vocab_size: {vocab_size}")
    assert vocab_size <= 65535

    vocab_path = TOK_DIR / "wikitext_vocab.json"
    with vocab_path.open("w") as f:
        json.dump({
            "gpt2_to_tiny": {str(k): v for k, v in gpt2_to_tiny.items()},
            "tiny_to_gpt2": tiny_to_gpt2,
            "vocab_size": vocab_size,
            "specials": SPECIALS,
            "coverage_target": COVERAGE_TARGET,
            "actual_coverage": cum / n_total,
            "n_train_tokens": n_total,
            "dataset": "wikitext-103-raw-v1",
        }, f, indent=2)
    print(f"[wt] wrote {vocab_path}")

    def remap_and_write(ds, out_path: Path, label: str, glue_articles=True):
        """Tokenize, remap, write uint16 .bin.

        Adds <bos> at the start of each article-ish chunk. WikiText doesn't
        have explicit document boundaries, but we treat empty-line gaps as
        soft boundaries. For training data this matters less since we
        sample random windows; for val data it gives clean perplexity
        measurement on contiguous context."""
        unk = SPECIALS["<unk>"]
        bos = SPECIALS["<bos>"]
        eos = SPECIALS["<eos>"]
        chunks: list[np.ndarray] = []
        n_unk = 0
        n_total_local = 0

        # Stream through, prepending <bos> only at the very start (treat
        # the whole split as one long stream with article-heading markers
        # already embedded as = Title = lines).
        chunks.append(np.array([bos], dtype=np.uint16))

        for i in range(0, len(ds), BATCH):
            batch = ds[i:i + BATCH]["text"]
            batch = [t for t in batch if t.strip()]
            if not batch:
                continue
            enc = tok(batch, add_special_tokens=False)["input_ids"]
            for ids in enc:
                arr = np.empty(len(ids), dtype=np.uint16)
                for j, gid in enumerate(ids):
                    tid = gpt2_to_tiny.get(int(gid), unk)
                    arr[j] = tid
                    if tid == unk:
                        n_unk += 1
                    n_total_local += 1
                chunks.append(arr)
            if i % 100000 == 0:
                so_far = sum(c.size for c in chunks)
                print(f"     {label} {i:>7}/{len(ds)}, "
                      f"{so_far/1e6:.1f}M tiny tokens written")

        chunks.append(np.array([eos], dtype=np.uint16))
        all_arr = np.concatenate(chunks)
        all_arr.tofile(out_path)
        print(f"[wt] wrote {out_path} ({all_arr.size:,} tokens, "
              f"{all_arr.nbytes/1e6:.1f} MB), unk rate {n_unk/n_total_local*100:.3f}%")

    print("[wt] writing wikitext_train.bin")
    remap_and_write(ds_train, DATA_DIR / "wikitext_train.bin", "train")
    print("[wt] writing wikitext_val.bin")
    remap_and_write(ds_val, DATA_DIR / "wikitext_val.bin", "val")
    print("[wt] done.")


if __name__ == "__main__":
    main()
