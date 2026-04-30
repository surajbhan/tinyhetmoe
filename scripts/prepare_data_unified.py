#!/usr/bin/env python3
"""prepare_data_unified.py — unified Qwen-vocab dataprep for 4-domain MoE.

Builds a single trimmed Qwen2.5 tokenizer covering all 4 domains:
  - PG-19            (literary / narrative)
  - WikiText-103     (encyclopedic / factual)
  - glaive func-call (structured JSON / tool-calling)
  - StarCoderData    (Python code)

TinyStories is intentionally dropped — its vocab is a strict subset of PG-19
and adding it would give the per-token outer router redundant domains.

Each corpus is capped to ~200 MB of raw UTF-8 text via streaming so we never
download a full multi-GB dataset.

Output:
  data/unified_train_pg19.bin       uint16 token ids
  data/unified_val_pg19.bin
  data/unified_train_wiki.bin
  data/unified_val_wiki.bin
  data/unified_train_tool.bin
  data/unified_val_tool.bin
  data/unified_train_code.bin
  data/unified_val_code.bin
  tokenizer/unified_vocab.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

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
MAX_VOCAB = 32768  # 4 corpora incl. code/JSON; 24K hit cap with code at 8% unk
DEFAULT_PER_CORPUS_BYTES = 200 * 1000 * 1000  # 200 MB; CLI overridable
VAL_FRAC = 0.01  # held-out fraction per corpus
SEED = 1234

GLAIVE_JSON = Path(
    "/home/suraj/.cache/huggingface/hub/datasets--glaiveai--glaive-function-calling-v2"
    "/snapshots/e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac/glaive-function-calling-v2.json"
)


def stream_pg19(byte_cap: int) -> list[str]:
    """PG-19 streamed; pulls books until raw text reaches byte_cap.

    Uses emozilla/pg19 (parquet-native mirror) instead of deepmind/pg19,
    which is script-format and breaks on newer `datasets` versions.
    """
    print(f"[pg19] streaming up to {byte_cap/1e6:.0f} MB")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    out, total = [], 0
    for ex in ds:
        t = ex.get("text") or ""
        if not t.strip():
            continue
        out.append(t)
        total += len(t.encode("utf-8"))
        if total >= byte_cap:
            break
    print(f"[pg19] got {len(out)} books, {total/1e6:.1f} MB")
    return out


def load_wiki(byte_cap: int) -> list[str]:
    """WikiText-103: load full train, sample articles up to byte_cap."""
    print(f"[wiki] loading WikiText-103, will sample to {byte_cap/1e6:.0f} MB")
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
    )
    rng = random.Random(SEED)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out, total = [], 0
    for i in idx:
        t = ds[i]["text"]
        if not t or not t.strip():
            continue
        out.append(t)
        total += len(t.encode("utf-8"))
        if total >= byte_cap:
            break
    print(f"[wiki] sampled {len(out)} lines, {total/1e6:.1f} MB")
    return out


def load_glaive(byte_cap: int) -> list[str]:
    """Glaive function-calling: local JSON, sample conversations."""
    print(f"[tool] loading glaive function-calling")
    raw = json.loads(GLAIVE_JSON.read_text())
    print(f"[tool] {len(raw)} conversations available")
    rng = random.Random(SEED + 1)
    idx = list(range(len(raw)))
    rng.shuffle(idx)
    out, total = [], 0
    for i in idx:
        ex = raw[i]
        # glaive entries are dicts with 'system' + 'chat' string fields
        parts = []
        if isinstance(ex, dict):
            for k in ("system", "chat"):
                v = ex.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v)
        else:
            continue
        if not parts:
            continue
        t = "\n".join(parts)
        out.append(t)
        total += len(t.encode("utf-8"))
        if total >= byte_cap:
            break
    print(f"[tool] sampled {len(out)} conversations, {total/1e6:.1f} MB")
    return out


def stream_code(byte_cap: int) -> list[str]:
    """StarCoderData Python subset, streamed."""
    print(f"[code] streaming starcoderdata python up to {byte_cap/1e6:.0f} MB")
    ds = load_dataset(
        "bigcode/starcoderdata", data_dir="python",
        split="train", streaming=True,
    )
    out, total = [], 0
    for ex in ds:
        t = ex.get("content") or ex.get("text") or ""
        if not t.strip():
            continue
        out.append(t)
        total += len(t.encode("utf-8"))
        if total >= byte_cap:
            break
    print(f"[code] got {len(out)} files, {total/1e6:.1f} MB")
    return out


def split_train_val(texts: list[str], val_frac: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    n_val = max(1, int(len(texts) * val_frac))
    val_idx = set(idx[:n_val])
    train = [t for i, t in enumerate(texts) if i not in val_idx]
    val = [t for i, t in enumerate(texts) if i in val_idx]
    return train, val


def tally(qwen_tok, texts: Iterable[str], counter: Counter, label: str):
    BATCH = 1000
    n_local = 0
    buf: list[str] = []
    n_seen = 0
    for t in texts:
        buf.append(t)
        if len(buf) >= BATCH:
            enc = qwen_tok(buf, add_special_tokens=False)["input_ids"]
            for ids in enc:
                counter.update(ids)
                n_local += len(ids)
            n_seen += len(buf)
            buf.clear()
            if n_seen % 5000 == 0:
                print(f"     {label} {n_seen} ({n_local/1e6:.1f}M tokens, "
                      f"{len(counter):,} unique)")
    if buf:
        enc = qwen_tok(buf, add_special_tokens=False)["input_ids"]
        for ids in enc:
            counter.update(ids)
            n_local += len(ids)
    print(f"     {label} done: {n_local/1e6:.1f}M tokens")
    return n_local


def write_bin(qwen_tok, texts: list[str], qwen_to_tiny: dict, out_path: Path, label: str):
    BATCH = 1000
    unk = SPECIALS["<unk>"]
    bos = SPECIALS["<bos>"]
    eos = SPECIALS["<eos>"]
    chunks: list[np.ndarray] = [np.array([bos], dtype=np.uint16)]
    n_unk = 0
    n_local = 0
    for i in range(0, len(texts), BATCH):
        batch = [t for t in texts[i:i + BATCH] if t and t.strip()]
        if not batch:
            continue
        enc = qwen_tok(batch, add_special_tokens=False)["input_ids"]
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
        if i and i % 5000 == 0:
            so_far = sum(c.size for c in chunks)
            print(f"     {label} {i:>6}/{len(texts)} ({so_far/1e6:.1f}M)")
    all_arr = np.concatenate(chunks)
    all_arr.tofile(out_path)
    print(f"[bin] wrote {out_path} ({all_arr.size:,} tokens, "
          f"{all_arr.nbytes/1e6:.1f} MB), unk rate "
          f"{n_unk/max(n_local,1)*100:.3f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-corpus-mb", type=int,
                    default=DEFAULT_PER_CORPUS_BYTES // 1_000_000,
                    help="MB of raw text per corpus (decimal MB)")
    args = ap.parse_args()
    per_corpus_bytes = args.per_corpus_mb * 1_000_000
    print(f"[uni] per-corpus cap: {args.per_corpus_mb} MB "
          f"({per_corpus_bytes/1e9:.2f} GB)")

    print("[uni] loading Qwen2.5-Coder-0.5B tokenizer")
    qwen_tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True,
    )
    print(f"     qwen_vocab_size: {len(qwen_tok)}")

    # Load each corpus capped to per_corpus_bytes
    corpora_raw = {
        "pg19": stream_pg19(per_corpus_bytes),
        "wiki": load_wiki(per_corpus_bytes),
        "tool": load_glaive(per_corpus_bytes),
        "code": stream_code(per_corpus_bytes),
    }

    # Split train/val per corpus
    splits: dict[str, tuple[list[str], list[str]]] = {}
    for name, texts in corpora_raw.items():
        tr, va = split_train_val(texts, VAL_FRAC, seed=SEED + hash(name) % 1000)
        splits[name] = (tr, va)
        print(f"[split] {name}: train={len(tr)}, val={len(va)}")

    # ── Pass 1: count token frequencies on union of TRAIN sets ──────
    print("\n[uni] counting token frequencies (Qwen tokenization, train only)")
    counter: Counter[int] = Counter()
    n_total = 0
    for name, (tr, _) in splits.items():
        n_total += tally(qwen_tok, tr, counter, name)

    print(f"\n[uni] total train tokens across all 4: {n_total:,}")
    print(f"[uni] unique qwen ids in union:        {len(counter):,}")

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
        "per_corpus_bytes_cap": per_corpus_bytes,
        "domains": list(splits.keys()),
    }, vocab_path.open("w"), indent=2)
    print(f"[uni] wrote {vocab_path}")

    # ── Pass 2: write .bin files for each (corpus, split) ────────────
    for name, (tr, va) in splits.items():
        print(f"\n[uni] writing {name}")
        write_bin(qwen_tok, tr, qwen_to_tiny,
                  DATA_DIR / f"unified_train_{name}.bin", f"{name}_train")
        write_bin(qwen_tok, va, qwen_to_tiny,
                  DATA_DIR / f"unified_val_{name}.bin",   f"{name}_val")

    print("\n[uni] done.")


if __name__ == "__main__":
    main()
