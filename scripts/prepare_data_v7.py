#!/usr/bin/env python3
"""prepare_data_v7.py — full-Qwen-vocab tokenization for 6-domain v7 model.

Differences from prepare_data_unified.py:
  - **No vocab trimming.** Full 151,665 Qwen2.5 tokens. Eliminates <unk>.
  - **uint32 .bin files.** Qwen tokens exceed 65535 so we can't use uint16.
    (Matches the experiments_may convention.)
  - **6 domain corpora.** general / thinker / code_py / code_js / medical / legal.
  - **No remap.** Tokens ARE Qwen IDs directly. Simpler downstream.

Output (in --out-dir):
  v7_train_<name>.bin   uint32 token ids
  v7_val_<name>.bin     uint32 token ids
  v7_meta.json          per-corpus stats (token counts, byte sizes)

The tokenizer is Qwen2.5-Coder-0.5B (same as v6, since the meaning_axes
are keyed on it). Each corpus runs through one source-loader; loaders
stream up to --per-corpus-mb of raw text per domain.

Usage:
  python3 scripts/prepare_data_v7.py --out-dir data_v7 \\
      --domains general,thinker,code_py,code_js,medical,legal \\
      --per-corpus-mb 1500 \\
      --val-frac 0.005

Single-domain mode is supported: --domains code_py only prepares one.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent

# ─── Per-domain stream loaders ──────────────────────────────────────────


def stream_general(byte_cap: int) -> Iterator[str]:
    """SlimOrca + UltraChat — instruct-style chat conversations.
    Render each conversation as plain ChatML so the model sees the format."""
    print(f"[general] streaming SlimOrca up to half cap")
    half = byte_cap // 2

    # SlimOrca: list of {conversations: [{from, value}, ...]}
    ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    total = 0
    for ex in ds:
        convs = ex.get("conversations", [])
        if not convs:
            continue
        out = ""
        for turn in convs:
            role_raw = (turn.get("from") or "").lower()
            role = {"system": "system", "human": "user",
                    "user": "user", "gpt": "assistant",
                    "assistant": "assistant"}.get(role_raw, role_raw)
            content = turn.get("value", "")
            if not content:
                continue
            out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if not out:
            continue
        out += "<|endoftext|>\n"
        total += len(out.encode("utf-8"))
        yield out
        if total >= half:
            break
    print(f"[general] SlimOrca: {total/1e6:.1f} MB")

    print(f"[general] streaming UltraChat for second half")
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True,
    )
    n_uc = 0
    for ex in ds:
        msgs = ex.get("messages", [])
        if not msgs:
            continue
        out = ""
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if not content:
                continue
            out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if not out:
            continue
        out += "<|endoftext|>\n"
        total += len(out.encode("utf-8"))
        n_uc += 1
        yield out
        if total >= byte_cap:
            break
    print(f"[general] total {total/1e6:.1f} MB ({n_uc} UltraChat convs)")


def stream_thinker(byte_cap: int) -> Iterator[str]:
    """orca-math-word-problems — reasoning traces with step-by-step solutions."""
    print(f"[thinker] streaming orca-math up to {byte_cap/1e6:.0f} MB")
    ds = load_dataset(
        "microsoft/orca-math-word-problems-200k",
        split="train", streaming=True,
    )
    total = 0
    n = 0
    for ex in ds:
        q = ex.get("question") or ""
        a = ex.get("answer") or ""
        if not q.strip() or not a.strip():
            continue
        out = (f"<|im_start|>user\n{q}<|im_end|>\n"
               f"<|im_start|>assistant\n{a}<|im_end|>\n<|endoftext|>\n")
        total += len(out.encode("utf-8"))
        n += 1
        yield out
        if total >= byte_cap:
            break
    print(f"[thinker] {n} problems, {total/1e6:.1f} MB")


def _stream_starcoder_subset(lang: str, byte_cap: int) -> Iterator[str]:
    print(f"[code_{lang}] streaming starcoderdata/{lang} up to {byte_cap/1e6:.0f} MB")
    ds = load_dataset(
        "bigcode/starcoderdata", data_dir=lang,
        split="train", streaming=True,
    )
    total = 0
    n = 0
    for ex in ds:
        t = ex.get("content") or ex.get("text") or ""
        if not t.strip():
            continue
        # Code files separated by <|endoftext|>
        out = t + "\n<|endoftext|>\n"
        total += len(out.encode("utf-8"))
        n += 1
        yield out
        if total >= byte_cap:
            break
    print(f"[code_{lang}] {n} files, {total/1e6:.1f} MB")


def stream_code_py(byte_cap: int) -> Iterator[str]:
    return _stream_starcoder_subset("python", byte_cap)


def stream_code_js(byte_cap: int) -> Iterator[str]:
    return _stream_starcoder_subset("javascript", byte_cap)


def stream_medical(byte_cap: int) -> Iterator[str]:
    """Multiple medical sources: medalpaca/medical_meadow_medqa, BI55/MedText,
    lavita/medical-qa-datasets, and PubMedQA. We hit each up to ~25% of the
    cap and rely on early-exit if any one source can fully cover."""
    print(f"[medical] streaming 4 sources, ~25% cap each")
    quarter = byte_cap // 4
    total = 0

    # 1. medalpaca/medical_meadow_medqa
    try:
        ds = load_dataset(
            "medalpaca/medical_meadow_medqa", split="train", streaming=True,
        )
        n = 0
        target = total + quarter
        for ex in ds:
            q = ex.get("input") or ex.get("instruction") or ""
            a = ex.get("output") or ""
            if not q.strip() or not a.strip():
                continue
            out = (f"<|im_start|>user\n{q}<|im_end|>\n"
                   f"<|im_start|>assistant\n{a}<|im_end|>\n<|endoftext|>\n")
            total += len(out.encode("utf-8"))
            n += 1
            yield out
            if total >= target:
                break
        print(f"[medical] medqa: {total/1e6:.1f} MB ({n} examples)")
    except Exception as e:
        print(f"[medical] medqa failed: {e}")

    # 2. BI55/MedText — clinical Q&A
    try:
        ds = load_dataset("BI55/MedText", split="train", streaming=True)
        n = 0
        target = total + quarter
        for ex in ds:
            q = ex.get("Prompt") or ""
            a = ex.get("Completion") or ""
            if not q.strip() or not a.strip():
                continue
            out = (f"<|im_start|>user\n{q}<|im_end|>\n"
                   f"<|im_start|>assistant\n{a}<|im_end|>\n<|endoftext|>\n")
            total += len(out.encode("utf-8"))
            n += 1
            yield out
            if total >= target:
                break
        print(f"[medical] MedText: {total/1e6:.1f} MB total ({n} examples)")
    except Exception as e:
        print(f"[medical] MedText failed: {e}")

    # 3. lavita/medical-qa-datasets / all-processed
    try:
        ds = load_dataset(
            "lavita/medical-qa-datasets", "all-processed",
            split="train", streaming=True,
        )
        n = 0
        target = total + quarter
        for ex in ds:
            # Various fields — try several
            q = (ex.get("input") or ex.get("question")
                 or ex.get("instruction") or "")
            a = (ex.get("output") or ex.get("answer") or "")
            if not q.strip() or not a.strip():
                continue
            out = (f"<|im_start|>user\n{q}<|im_end|>\n"
                   f"<|im_start|>assistant\n{a}<|im_end|>\n<|endoftext|>\n")
            total += len(out.encode("utf-8"))
            n += 1
            yield out
            if total >= target:
                break
        print(f"[medical] medical-qa-datasets: {total/1e6:.1f} MB total ({n})")
    except Exception as e:
        print(f"[medical] medical-qa-datasets failed: {e}")

    # 4. PubMedQA
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial",
                          split="train", streaming=True)
        n = 0
        for ex in ds:
            q = ex.get("question") or ""
            a = ex.get("long_answer") or ""
            if not q.strip() or not a.strip():
                continue
            out = (f"<|im_start|>user\n{q}<|im_end|>\n"
                   f"<|im_start|>assistant\n{a}<|im_end|>\n<|endoftext|>\n")
            total += len(out.encode("utf-8"))
            n += 1
            yield out
            if total >= byte_cap:
                break
        print(f"[medical] PubMedQA: {total/1e6:.1f} MB total ({n})")
    except Exception as e:
        print(f"[medical] PubMedQA failed: {e}")

    print(f"[medical] FINAL total {total/1e6:.1f} MB")


def stream_legal(byte_cap: int) -> Iterator[str]:
    """SCOTUS opinions (long substantive legal text) + lex_glue case_hold +
    legal_case_document_summarization. SCOTUS provides the bulk; lex_glue
    and the summarization corpus add diversity."""
    print(f"[legal] streaming 3 sources")
    total = 0

    # 1. SCOTUS opinions — full case text (this is the bulk)
    try:
        ds = load_dataset(
            "coastalcph/lex_glue", "scotus",
            split="train", streaming=True,
        )
        n = 0
        target = byte_cap * 2 // 3  # SCOTUS gets 2/3 of the cap
        for ex in ds:
            text = ex.get("text", "")
            if not text or len(text) < 200:
                continue
            out = text + "\n<|endoftext|>\n"
            total += len(out.encode("utf-8"))
            n += 1
            yield out
            if total >= target:
                break
        print(f"[legal] scotus: {total/1e6:.1f} MB ({n} opinions)")
    except Exception as e:
        print(f"[legal] scotus failed: {e}")

    # 2. case_hold (the original lex_glue source)
    try:
        ds = load_dataset(
            "coastalcph/lex_glue", "case_hold",
            split="train", streaming=True,
        )
        n = 0
        target = total + byte_cap // 6
        for ex in ds:
            ctx = ex.get("context") or ex.get("text") or ""
            endings = ex.get("endings") or []
            label = ex.get("label", -1)
            if not ctx.strip():
                continue
            out = ctx
            if isinstance(endings, list) and 0 <= label < len(endings):
                out += "\nHolding: " + endings[label]
            out += "\n<|endoftext|>\n"
            total += len(out.encode("utf-8"))
            n += 1
            yield out
            if total >= target:
                break
        print(f"[legal] case_hold: {total/1e6:.1f} MB total ({n} cases)")
    except Exception as e:
        print(f"[legal] case_hold failed: {e}")

    # 3. legal_case_document_summarization
    try:
        ds = load_dataset(
            "joelniklaus/legal_case_document_summarization",
            split="train", streaming=True,
        )
        n = 0
        for ex in ds:
            judgement = ex.get("judgement", "")
            summary = ex.get("summary", "")
            if not judgement:
                continue
            out = judgement + "\nSummary: " + summary + "\n<|endoftext|>\n"
            total += len(out.encode("utf-8"))
            n += 1
            yield out
            if total >= byte_cap:
                break
        print(f"[legal] case_doc_summarization: {total/1e6:.1f} MB total ({n})")
    except Exception as e:
        print(f"[legal] case_doc_summarization failed: {e}")

    print(f"[legal] FINAL total {total/1e6:.1f} MB")


DOMAIN_LOADERS = {
    "general":  stream_general,
    "thinker":  stream_thinker,
    "code_py":  stream_code_py,
    "code_js":  stream_code_js,
    "medical":  stream_medical,
    "legal":    stream_legal,
}


# ─── Tokenize + write ──────────────────────────────────────────────────


def write_corpus(qwen_tok, name: str, loader, out_dir: Path,
                 byte_cap: int, val_frac: float, seed: int):
    train_path = out_dir / f"v7_train_{name}.bin"
    val_path = out_dir / f"v7_val_{name}.bin"

    # Drain loader into in-memory string list (we need to split train/val
    # before tokenization, and corpus sizes are 1-2 GB which fits)
    rng = random.Random(seed)
    chunks = list(loader(byte_cap))
    rng.shuffle(chunks)
    n_val = max(1, int(len(chunks) * val_frac))
    val_chunks = chunks[:n_val]
    train_chunks = chunks[n_val:]
    print(f"[{name}] split: {len(train_chunks)} train, {len(val_chunks)} val")

    BATCH = 200
    EOT_TID = 151643  # <|endoftext|>

    def tokenize_split(split_chunks: list[str], out_path: Path):
        all_ids: list[int] = []
        for i in range(0, len(split_chunks), BATCH):
            batch = split_chunks[i:i + BATCH]
            enc = qwen_tok(batch, add_special_tokens=False)["input_ids"]
            for ids in enc:
                all_ids.extend(ids)
            if i % (BATCH * 10) == 0:
                print(f"  [{name}/{out_path.name}] {i}/{len(split_chunks)} chunks "
                      f"({len(all_ids)/1e6:.1f}M tokens)")
        arr = np.asarray(all_ids, dtype=np.uint32)
        arr.tofile(out_path)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  wrote {out_path} ({len(arr):,} tokens, {size_mb:.1f} MB)")
        return len(arr)

    n_train = tokenize_split(train_chunks, train_path)
    n_val = tokenize_split(val_chunks, val_path)
    return {"train_tokens": n_train, "val_tokens": n_val,
            "train_mb": train_path.stat().st_size / 1e6,
            "val_mb": val_path.stat().st_size / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_v7",
                    help="Output directory under repo root")
    ap.add_argument("--domains", default="general,thinker,code_py,code_js,medical,legal",
                    help="Comma-separated domain names")
    ap.add_argument("--per-corpus-mb", type=int, default=1500,
                    help="Raw-text byte cap per corpus (MB, decimal)")
    ap.add_argument("--val-frac", type=float, default=0.005,
                    help="Held-out val fraction (0.5%% default; small bc"
                         " corpora are large)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    for d in domains:
        if d not in DOMAIN_LOADERS:
            raise SystemExit(f"unknown domain: {d!r}; "
                             f"available: {list(DOMAIN_LOADERS)}")
    print(f"[v7] preparing {len(domains)} domains: {domains}")
    print(f"[v7] per-corpus byte cap: {args.per_corpus_mb} MB")
    print(f"[v7] val frac: {args.val_frac}")
    print(f"[v7] out dir: {out_dir}")

    print("[v7] loading Qwen2.5-Coder-0.5B tokenizer (full vocab, no trim)")
    qwen_tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True,
    )
    print(f"[v7] vocab_size: {len(qwen_tok)}")

    byte_cap = args.per_corpus_mb * 1_000_000
    stats = {}
    for d in domains:
        print(f"\n=== {d} ===")
        stats[d] = write_corpus(
            qwen_tok, d, DOMAIN_LOADERS[d],
            out_dir, byte_cap, args.val_frac, args.seed,
        )

    meta_path = out_dir / "v7_meta.json"
    meta = {
        "vocab_size": len(qwen_tok),
        "tokenizer_source": "Qwen/Qwen2.5-Coder-0.5B",
        "dtype": "uint32",
        "per_corpus_mb_cap": args.per_corpus_mb,
        "val_frac": args.val_frac,
        "domains": stats,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n[v7] wrote {meta_path}")
    total_train = sum(s["train_tokens"] for s in stats.values())
    total_val = sum(s["val_tokens"] for s in stats.values())
    print(f"[v7] total: {total_train:,} train tokens, {total_val:,} val")


if __name__ == "__main__":
    main()
