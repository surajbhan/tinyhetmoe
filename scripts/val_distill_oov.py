#!/usr/bin/env python3
"""val_distill_oov.py — out-of-training-distribution val. Measures CE
on text the student never saw during base pretraining or distillation.

Test sources (independent of v7 base + distill corpora):
  - Wikipedia (factual encyclopedic — never in our base)
  - The Pile validation slice (web text, news, books)
  - HumanEval prompts (for code experts)

If distillation generalizes (and isn't just overfitting to the streamed
SlimOrca distribution), val CE on these OOV sources should ALSO drop
post-distillation, not just on the in-domain val.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.tiny_hetmoe import (  # noqa: E402
    TinyHetMoE, TinyHetMoEConfig, set_quantize_mode, set_qat_backward_mode,
)


# ──────────────────────────────────────────────────────────────────────
# OOV val sources — totally outside training distribution
# ──────────────────────────────────────────────────────────────────────

def stream_wiki():
    """Wikipedia plain text — encyclopedic, factual."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)
    for ex in ds:
        text = (ex.get("text") or "").strip()
        if len(text) > 200:
            yield text + "\n\n"


def stream_pile():
    """Common Crawl-like web text from Pile-uncopyrighted."""
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted",
                      split="train", streaming=True)
    for ex in ds:
        text = (ex.get("text") or "").strip()
        if len(text) > 200:
            yield text + "\n\n"


def stream_humaneval():
    """HumanEval prompts — for code experts."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    for ex in ds:
        prompt = ex.get("prompt") or ""
        canonical = ex.get("canonical_solution") or ""
        if prompt and canonical:
            yield prompt + canonical + "\n\n"


def stream_gsm8k():
    """GSM8K test set — for thinker (math)."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    for ex in ds:
        q = (ex.get("question") or "").strip()
        a = (ex.get("answer") or "").strip()
        if q and a:
            yield f"Question: {q}\n\nAnswer: {a}\n\n"


SOURCES = {
    "wiki": stream_wiki,
    "pile": stream_pile,
    "humaneval": stream_humaneval,
    "gsm8k": stream_gsm8k,
}


def _token_buffer(stream, tok, eos_id):
    for text in stream:
        ids = tok.encode(text, add_special_tokens=False)
        for tid in ids:
            yield tid
        yield eos_id


def make_batch(buf, batch_size, seq_len):
    flat = [next(buf) for _ in range(batch_size * seq_len)]
    return torch.tensor(flat, dtype=torch.long).view(batch_size, seq_len)


def load_student(path, vocab, qat, device):
    mcfg = TinyHetMoEConfig(
        vocab_size=vocab, meaning_dim=132, intuition_dim=132,
        input_dim=264, internal_dim=528, new_intuition=264,
        num_layers=4, num_heads=4, num_experts=4, top_k_experts=2,
        ffn_mult=2.0, max_seq_len=2048,
    )
    m = TinyHetMoE(mcfg).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in ck.get("model", ck).items()}
    m.load_state_dict(sd, strict=False)
    m.eval()
    set_qat_backward_mode("ste")
    set_quantize_mode(m, on=qat)
    return m


@torch.no_grad()
def measure_ce(model, val_batches, device, is_teacher=False):
    total_ce, total_n = 0.0, 0
    for ids in val_batches:
        ids = ids.to(device)
        if is_teacher:
            logits = model(ids).logits
        else:
            logits, _ = model(ids)
        s = logits[:, :-1, :]
        targets = ids[:, 1:]
        ce = F.cross_entropy(s.reshape(-1, s.shape[-1]).float(),
                             targets.reshape(-1), reduction="sum")
        total_ce += float(ce.detach().cpu())
        total_n += targets.numel()
    return total_ce / total_n, float(torch.tensor(total_ce / total_n).exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-ckpts", nargs="+", required=True,
                    help="name=path pairs")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--sources", nargs="+", default=["wiki"],
                    choices=list(SOURCES.keys()))
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=151665)
    args = ap.parse_args()

    student_dev = "cuda:0"
    teacher_dev = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.teacher)
    eos = tok.eos_token_id

    # Build val batches per source (deterministic — same seq for all models)
    print(f"[val-oov] building val batches from sources: {args.sources}", flush=True)
    val_per_source = {}
    for src in args.sources:
        print(f"  [{src}] streaming...", flush=True)
        stream = SOURCES[src]()
        buf = _token_buffer(stream, tok, eos)
        batches = []
        try:
            for _ in range(args.n_batches):
                batches.append(make_batch(buf, args.batch_size, args.seq_len))
        except StopIteration:
            pass
        n_tok = sum(b.numel() for b in batches)
        print(f"  [{src}] {len(batches)} batches, {n_tok:,} tokens", flush=True)
        val_per_source[src] = batches

    # Teacher CE per source
    print(f"\n[val-oov] loading teacher {args.teacher} on {teacher_dev}", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map={"": teacher_dev},
    )
    teacher.eval()
    teacher_ce = {}
    for src, batches in val_per_source.items():
        ce, ppl = measure_ce(teacher, batches, teacher_dev, is_teacher=True)
        teacher_ce[src] = (ce, ppl)
        print(f"  TEACHER on {src}: CE={ce:.4f} PPL={ppl:.2f}", flush=True)
    del teacher
    torch.cuda.empty_cache()

    # Student CE per (ckpt, source)
    print()
    rows = []
    for spec in args.student_ckpts:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        qat = bool(ck.get("qat_currently_on", False))
        m = load_student(path, args.vocab, qat, student_dev)
        for src, batches in val_per_source.items():
            ce, ppl = measure_ce(m, batches, student_dev)
            t_ce, t_ppl = teacher_ce[src]
            rows.append((name, qat, src, ce, ppl, ce - t_ce))
            print(f"  {name} (qat={qat}) on {src}: CE={ce:.4f} PPL={ppl:.2f} gap={ce-t_ce:+.4f}",
                  flush=True)
        del m
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*84}\n  OOV SUMMARY\n{'='*84}")
    print(f"  {'name':<30} {'qat':<5} {'src':<10} {'CE':>8} {'PPL':>10} {'gap':>10}")
    for src in args.sources:
        t_ce, t_ppl = teacher_ce[src]
        print(f"  {'TEACHER':<30} {'-':<5} {src:<10} {t_ce:>8.4f} {t_ppl:>10.2f} {0:>+10.4f}")
        for name, qat, ssrc, ce, ppl, gap in rows:
            if ssrc == src:
                print(f"  {name:<30} {str(qat):<5} {ssrc:<10} {ce:>8.4f} {ppl:>10.2f} {gap:>+10.4f}")
        print()


if __name__ == "__main__":
    main()
