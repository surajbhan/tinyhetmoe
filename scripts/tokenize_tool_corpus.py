#!/usr/bin/env python3
"""tokenize_tool_corpus.py — tokenize the unified ChatML tool corpus into
.bin files using the existing unified vocab map.

Replaces the glaive-only tool corpus produced by prepare_data_unified.py
with the ChatML-unified version (glaive + xLAM + Hermes).

Input:
  data/tool_corpus_chatml.txt        — produced by build_tool_corpus.py
  tokenizer/unified_vocab.json       — qwen→tiny id mapping

Output:
  data/unified_train_tool.bin
  data/unified_val_tool.bin

Each conversation ends with <|endoftext|>; we tokenize the whole stream
through Qwen and remap to the unified vocab.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
TOK_DIR = REPO / "tokenizer"

VAL_FRAC = 0.01
SEED = 7
BATCH = 200  # conversations per tokenize call
SPECIALS = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3}


def main():
    print("[tok-tool] loading Qwen tokenizer + unified vocab")
    qwen = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True,
    )
    vocab = json.load((TOK_DIR / "unified_vocab.json").open())
    qwen_to_tiny = {int(k): v for k, v in vocab["qwen_to_tiny"].items()}
    vocab_size = vocab["vocab_size"]
    print(f"     qwen_vocab={len(qwen)}, unified_vocab={vocab_size}")

    src = DATA_DIR / "tool_corpus_chatml.txt"
    print(f"[tok-tool] reading {src}")
    text = src.read_text()
    # Conversations are <|endoftext|>-separated. Each conversation is a
    # complete ChatML turn block. Treat each one as a "document."
    convs = [c.strip() for c in text.split("<|endoftext|>")]
    convs = [c for c in convs if c]
    print(f"     {len(convs)} conversations, {len(text)/1e6:.1f} MB raw")

    rng = random.Random(SEED)
    rng.shuffle(convs)
    n_val = max(1, int(len(convs) * VAL_FRAC))
    val_convs = convs[:n_val]
    train_convs = convs[n_val:]
    print(f"     train {len(train_convs)}, val {len(val_convs)}")

    unk = SPECIALS["<unk>"]
    bos = SPECIALS["<bos>"]
    eos = SPECIALS["<eos>"]

    def write_split(convs_split: list[str], out_path: Path, label: str):
        chunks: list[np.ndarray] = [np.array([bos], dtype=np.uint16)]
        n_unk = 0
        n_local = 0
        for i in range(0, len(convs_split), BATCH):
            batch = convs_split[i:i + BATCH]
            enc = qwen(batch, add_special_tokens=False)["input_ids"]
            for ids in enc:
                arr = np.empty(len(ids) + 1, dtype=np.uint16)
                arr[-1] = eos
                for j, qid in enumerate(ids):
                    tid = qwen_to_tiny.get(int(qid), unk)
                    arr[j] = tid
                    if tid == unk:
                        n_unk += 1
                    n_local += 1
                chunks.append(arr)
            if i and i % 2000 == 0:
                so_far = sum(c.size for c in chunks)
                print(f"     {label} {i:>6}/{len(convs_split)} "
                      f"({so_far/1e6:.1f}M)")
        all_arr = np.concatenate(chunks)
        all_arr.tofile(out_path)
        print(f"[bin] wrote {out_path} ({all_arr.size:,} tokens, "
              f"{all_arr.nbytes/1e6:.1f} MB), unk rate "
              f"{n_unk/max(n_local,1)*100:.3f}%")

    write_split(train_convs, DATA_DIR / "unified_train_tool.bin", "train")
    write_split(val_convs,   DATA_DIR / "unified_val_tool.bin",   "val")
    print("[tok-tool] done.")


if __name__ == "__main__":
    main()
