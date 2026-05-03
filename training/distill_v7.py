#!/usr/bin/env python3
"""distill_v7.py — online reverse-KL distillation from a Qwen 2.5 teacher
to a TinyHetMoE student, polishing the QAT deploy artifact.

Architecture:
  - GPU 0: student (TinyHetMoE 114M, ternary QAT mode), being trained
  - GPU 1: teacher (Qwen 2.5 0.5B / 1.5B / Coder), eval-only

Per training step:
  1. Sample (input_ids, response_mask) from a SFT-style instruct corpus
     (ChatML formatted prompts + responses)
  2. Forward through teacher (GPU 1, no grad), get top-K=50 logits per pos
  3. Forward through student (GPU 0, grad), get full-vocab logits
  4. Loss = reverse-KL = E_{P_student}[log P_student - log P_teacher]
     restricted to teacher's top-K (closed-form, no sampling)
  5. Backward + step on student only

Reverse-KL ("Asian parent") is mode-seeking: student concentrates
probability on teacher's peak token, instead of spreading mass to cover
teacher's distribution like forward-KL would. Empirically lines up with
MiniLLM (NeurIPS 2024) findings for tiny student / dense teacher.

Resumes from a TinyHetMoE best_qat.pt and stays in QAT mode throughout.
Saves a new best_distill_qat.pt when val_kl improves.

Usage:
  python3 training/distill_v7.py \\
      --student-ckpt runs/tiny_hetmoe_v7_general/checkpoints/best_qat.pt \\
      --teacher Qwen/Qwen2.5-0.5B-Instruct \\
      --domain general \\
      --out-dir runs/tiny_hetmoe_v7_general_distill \\
      --max-steps 5000 \\
      --lr 5e-5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.tiny_hetmoe import (  # noqa: E402
    TinyHetMoE, TinyHetMoEConfig, set_quantize_mode, set_qat_backward_mode,
)


# ──────────────────────────────────────────────────────────────────────
# Raw-text corpus streamers — yield plain text strings (not prompt/response
# pairs). Both teacher (base model) and student see the same sequence and
# do next-token prediction at every position. No ChatML, no loss masks.
# Mirrors the v7 base training distribution so we polish the same surface
# the student already learned.
# ──────────────────────────────────────────────────────────────────────

def stream_general() -> Iterator[str]:
    """SlimOrca + UltraChat — render conversations as plain text without
    ChatML markers. Teacher + student see raw dialogue."""
    from datasets import load_dataset
    ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    for ex in ds:
        convs = ex.get("conversations", [])
        if len(convs) < 2:
            continue
        out = ""
        for turn in convs:
            role_raw = (turn.get("from") or "").lower()
            role = {"system": "System", "human": "User", "user": "User",
                    "gpt": "Assistant", "assistant": "Assistant"}.get(role_raw, "")
            content = (turn.get("value") or "").strip()
            if role and content:
                out += f"{role}: {content}\n\n"
        if out.strip():
            yield out


def stream_thinker() -> Iterator[str]:
    """orca-math-word-problems — math Q/A as plain text."""
    from datasets import load_dataset
    ds = load_dataset(
        "microsoft/orca-math-word-problems-200k",
        split="train", streaming=True,
    )
    for ex in ds:
        q = (ex.get("question") or "").strip()
        a = (ex.get("answer") or "").strip()
        if not q or not a:
            continue
        yield f"Question: {q}\n\nAnswer: {a}\n\n"


def stream_code_py() -> Iterator[str]:
    """Magicoder OSS-Instruct, Python solutions — plain code blocks."""
    from datasets import load_dataset
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)
    for ex in ds:
        prob = (ex.get("problem") or "").strip()
        sol = (ex.get("solution") or "").strip()
        lang = (ex.get("lang") or "").lower()
        if lang and lang != "python":
            continue
        if not prob or not sol:
            continue
        yield f"# Problem\n{prob}\n\n# Solution\n{sol}\n\n"


def stream_code_js() -> Iterator[str]:
    """Magicoder OSS-Instruct, JS solutions."""
    from datasets import load_dataset
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)
    for ex in ds:
        prob = (ex.get("problem") or "").strip()
        sol = (ex.get("solution") or "").strip()
        lang = (ex.get("lang") or "").lower()
        if lang not in ("javascript", "js", "typescript", "ts"):
            continue
        if not prob or not sol:
            continue
        yield f"// Problem\n{prob}\n\n// Solution\n{sol}\n\n"


def stream_medical() -> Iterator[str]:
    """Medical Q&A as plain text."""
    from datasets import load_dataset
    ds = load_dataset("lavita/medical-qa-datasets", "all-processed",
                      split="train", streaming=True)
    for ex in ds:
        q = (ex.get("input") or ex.get("instruction") or ex.get("question") or "").strip()
        a = (ex.get("output") or ex.get("answer") or "").strip()
        if not q or not a:
            continue
        yield f"Question: {q}\n\nAnswer: {a}\n\n"


def stream_legal() -> Iterator[str]:
    """Legal Q&A as plain text. Lawyer-Instruct on HF currently has malformed
    JSON (column type changed mid-row); use Australian legal Q&A as the
    primary source, fall back to legalbench's consumer_contracts_qa."""
    from datasets import load_dataset
    try:
        ds = load_dataset("isaacus/open-australian-legal-qa",
                          split="train", streaming=True)
        for ex in ds:
            q = (ex.get("question") or "").strip()
            a = (ex.get("answer") or "").strip()
            if not q or not a:
                continue
            yield f"Question: {q}\n\nAnswer: {a}\n\n"
    except Exception as e:
        print(f"[legal] Australian QA stream failed ({e}); using legalbench")
        ds = load_dataset("nguha/legalbench", "consumer_contracts_qa",
                          split="train", streaming=True)
        for ex in ds:
            q = (ex.get("question") or ex.get("input") or "").strip()
            a = (ex.get("answer") or ex.get("response") or "").strip()
            if not q or not a:
                continue
            yield f"Question: {q}\n\nAnswer: {a}\n\n"


STREAMERS = {
    "general": stream_general,
    "thinker": stream_thinker,
    "code_py": stream_code_py,
    "code_js": stream_code_js,
    "medical": stream_medical,
    "legal": stream_legal,
}


# ──────────────────────────────────────────────────────────────────────
# Batching
# ──────────────────────────────────────────────────────────────────────

def _token_buffer(stream, tok, eos_id: int) -> Iterator[int]:
    """Tokenize streamed text segments and yield individual token ids,
    inserting an eos_id between segments so the model has a boundary."""
    for text in stream:
        ids = tok.encode(text, add_special_tokens=False)
        for tid in ids:
            yield tid
        yield eos_id


def make_batch(buf, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull batch_size × (seq_len+1) tokens from `buf` and split into
    (input_ids, attn_mask). seq_len+1 because we want input_ids of length
    seq_len with predictions over seq_len positions and ground truth at
    each position being the next token. We pack contiguous text — no
    padding, no prompt/response distinction, every position fires loss."""
    needed = batch_size * seq_len
    flat = []
    for _ in range(needed):
        flat.append(next(buf))
    ids = torch.tensor(flat, dtype=torch.long).view(batch_size, seq_len)
    attn = torch.ones_like(ids, dtype=torch.bool)
    return ids, attn


# ──────────────────────────────────────────────────────────────────────
# Reverse-KL loss with top-k closed-form expansion
# ──────────────────────────────────────────────────────────────────────

def topk_ce_distill(student_logits: torch.Tensor,
                    teacher_logits: torch.Tensor,
                    target_ids: torch.Tensor,
                    top_k: int = 50,
                    sft_anchor_weight: float = 0.1) -> tuple[torch.Tensor, dict]:
    """Top-K cross-entropy distillation. Teacher's top-K tokens at each
    position are ALL treated as positive targets, weighted by their
    teacher-softmax probability (renormalized over top-K).

    Loss per position = -Σ_{v in teacher_top_K} P_teacher(v) · log P_student(v)
    where P_student is over the FULL vocab (NOT renormalized over top-K).

    Why this is strict / "Asian parent" style: gradient at every teacher-
    top-K token v is proportional to P_teacher(v) — the model gets pushed
    HARD on tokens teacher likes, regardless of where student currently
    has its mass. If P_student(v) is near zero, log P_student(v) is very
    negative, the loss is large, and gradient at v's logit is large
    positive. The student cannot ignore teacher's top tokens.

    This is essentially forward-KL with the constant H(P_teacher) term
    dropped — same gradient, same convergence behavior, mode-covering.

    Returns (loss, metrics)."""
    B, T, V = student_logits.shape

    # Drop last position — no next-token label.
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]
    targets = target_ids[:, 1:]

    # Teacher's top-K
    t_top_v, t_top_idx = torch.topk(t, k=top_k, dim=-1)  # (B, T-1, K)
    # Teacher probabilities over top-K (renormalized)
    t_top_p = F.softmax(t_top_v.float(), dim=-1)         # (B, T-1, K)

    # Student log-probs over the FULL vocab (not renormalized to top-K).
    # This is the key — student is judged against teacher's top tokens
    # using its actual full-vocab probability, so a near-zero student
    # logit on a teacher-top token produces a huge loss.
    s_log_p_full = F.log_softmax(s, dim=-1)              # (B, T-1, V)
    s_log_p_at_topk = s_log_p_full.gather(-1, t_top_idx) # (B, T-1, K)

    # Cross-entropy at the K teacher-top tokens, weighted by teacher's prob
    ce_per_pos = -(t_top_p * s_log_p_at_topk).sum(-1)    # (B, T-1)
    distill_loss = ce_per_pos.mean()

    # SFT anchor: CE against ground-truth next token
    sft_loss = torch.zeros((), device=student_logits.device)
    if sft_anchor_weight > 0:
        sft_loss = F.cross_entropy(s.reshape(-1, V), targets.reshape(-1))

    total = (1 - sft_anchor_weight) * distill_loss + sft_anchor_weight * sft_loss

    # Diagnostics
    with torch.no_grad():
        # Student's renormalized prob at teacher's top tokens (for KL metrics)
        s_top_logits = s.gather(-1, t_top_idx)
        s_log_p_renorm = F.log_softmax(s_top_logits, dim=-1)
        t_log_p_renorm = F.log_softmax(t_top_v.float(), dim=-1)
        # Forward-KL (teacher → student) and reverse-KL for tracking
        fwd_kl = (t_top_p * (t_log_p_renorm - s_log_p_renorm)).sum(-1).mean()
        s_p_renorm = s_log_p_renorm.exp()
        rev_kl = (s_p_renorm * (s_log_p_renorm - t_log_p_renorm)).sum(-1).mean()

        # Top-1 agreement (student argmax over FULL vocab vs teacher's top-1)
        s_argmax = s.argmax(-1)
        t_top1 = t_top_idx[..., 0]
        top1_agree = (s_argmax == t_top1).float().mean()
        # Did student's top-1 land in teacher's top-K?
        s_in_topk = (s_argmax.unsqueeze(-1) == t_top_idx).any(-1).float().mean()
        # How much probability mass does student put on teacher's top-1?
        # (Higher is better; 1.0 = student also fully concentrates there)
        s_full_p = F.softmax(s, dim=-1)
        s_p_at_t_top1 = s_full_p.gather(-1, t_top1.unsqueeze(-1)).squeeze(-1)
        avg_p_at_t_top1 = s_p_at_t_top1.mean()

    return total, {
        "distill_ce": float(distill_loss.detach().cpu()),
        "rev_kl": float(rev_kl.detach().cpu()),
        "fwd_kl": float(fwd_kl.detach().cpu()),
        "sft_ce": float(sft_loss.detach().cpu()),
        "top1_agree": float(top1_agree.detach().cpu()),
        "s_in_topk": float(s_in_topk.detach().cpu()),
        "p_at_t_top1": float(avg_p_at_t_top1.detach().cpu()),
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-ckpt", required=True,
                    help="Path to TinyHetMoE best_qat.pt to resume from")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HF repo id for the teacher model")
    ap.add_argument("--domain", required=True, choices=list(STREAMERS.keys()))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--meaning-emb", default="data_v7/meaning_axes_full_132.npy")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--sft-anchor", type=float, default=0.1)
    ap.add_argument("--qat", action="store_true",
                    help="Stay in QAT (ternary). Default OFF: distill in bf16. "
                         "Re-QAT after distillation in a follow-up run.")
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--val-interval", type=int, default=200)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--vocab", type=int, default=151665)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    student_dev = "cuda:0"
    teacher_dev = "cuda:1"

    # Sanity: we need 2 GPUs
    if torch.cuda.device_count() < 2:
        raise SystemExit(f"Need 2 GPUs, found {torch.cuda.device_count()}")

    # ── Load student ─────────────────────────────────────────────────
    print(f"[distill] loading student from {args.student_ckpt}", flush=True)
    mcfg = TinyHetMoEConfig(
        vocab_size=args.vocab,
        meaning_dim=132, intuition_dim=132,
        input_dim=264, internal_dim=528, new_intuition=264,
        num_layers=4, num_heads=4,
        num_experts=4, top_k_experts=2,
        ffn_mult=2.0, max_seq_len=args.max_seq_len,
    )
    student = TinyHetMoE(mcfg).to(student_dev)
    student.load_meaning_embeddings(str(REPO / args.meaning_emb), freeze=True)
    ck = torch.load(args.student_ckpt, map_location=student_dev, weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in ck.get("model", ck).items()}
    missing, unexpected = student.load_state_dict(sd, strict=False)
    orig_val = ck.get("best_val") or ck.get("best_qat_val") or float("nan")
    print(f"[distill] student loaded: missing={len(missing)} unexpected={len(unexpected)} "
          f"orig_step={ck.get('step')} orig_val={orig_val:.4f} "
          f"orig_qat={ck.get('qat_currently_on')}", flush=True)
    set_qat_backward_mode("ste")
    set_quantize_mode(student, on=args.qat)
    print(f"[distill] training in {'QAT' if args.qat else 'bf16'} mode "
          f"(--qat={args.qat})", flush=True)
    student.train()
    n_params = sum(p.numel() for p in student.parameters())
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[distill] student params: total={n_params/1e6:.1f}M trainable={n_train/1e6:.1f}M", flush=True)

    # ── Load teacher ─────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[distill] loading teacher {args.teacher} on {teacher_dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map={"": teacher_dev},
    )
    teacher.eval()
    pad_id = tok.eos_token_id
    n_teacher = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"[distill] teacher loaded: {n_teacher:.1f}M params, vocab={tok.vocab_size}", flush=True)

    # Sanity: teacher and student must share vocab. Teacher's vocab is its
    # actual config size, may exceed student's if student vocab was capped.
    teacher_v = teacher.config.vocab_size
    if teacher_v < args.vocab:
        print(f"[distill] WARNING: teacher vocab ({teacher_v}) < student ({args.vocab})",
              flush=True)

    # ── Optimizer ────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )

    # ── Stream + train ───────────────────────────────────────────────
    eos_id = tok.eos_token_id
    stream = STREAMERS[args.domain]()
    buf = _token_buffer(stream, tok, eos_id)
    log_path = out_dir / "logs" / "train.jsonl"
    log_f = log_path.open("a")

    best_rev_kl = float("inf")
    t_start = time.time()
    print(f"[distill] starting domain={args.domain} steps={args.max_steps} "
          f"lr={args.lr} top_k={args.top_k} sft_anchor={args.sft_anchor} "
          f"seq_len={args.max_seq_len} batch={args.batch_size}",
          flush=True)
    for step in range(1, args.max_steps + 1):
        try:
            input_ids, attn_mask = make_batch(buf, args.batch_size, args.max_seq_len)
        except StopIteration:
            print(f"[distill] stream exhausted at step {step}; restarting", flush=True)
            stream = STREAMERS[args.domain]()
            buf = _token_buffer(stream, tok, eos_id)
            continue

        # Send to teacher GPU, run forward (no grad)
        t_input = input_ids.to(teacher_dev, non_blocking=True)
        t_attn = attn_mask.to(teacher_dev, non_blocking=True)
        with torch.no_grad():
            t_logits = teacher(t_input, attention_mask=t_attn).logits  # (B, T, V_t)

        # Adjust to student's vocab size if necessary
        if t_logits.shape[-1] > args.vocab:
            t_logits = t_logits[..., :args.vocab]
        elif t_logits.shape[-1] < args.vocab:
            pad_amt = args.vocab - t_logits.shape[-1]
            pad = torch.full(
                (t_logits.shape[0], t_logits.shape[1], pad_amt),
                float("-inf"), device=t_logits.device, dtype=t_logits.dtype,
            )
            t_logits = torch.cat([t_logits, pad], dim=-1)
        t_logits = t_logits.to(student_dev, non_blocking=True)

        # Forward student
        s_input = input_ids.to(student_dev, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            s_logits, _ = student(s_input)

        # Loss in fp32
        with torch.amp.autocast("cuda", enabled=False):
            s_logits_fp32 = s_logits.float()
            loss, metrics = topk_ce_distill(
                s_logits_fp32, t_logits, target_ids=s_input,
                top_k=args.top_k, sft_anchor_weight=args.sft_anchor,
            )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            tok_s = (step * args.batch_size * args.max_seq_len) / elapsed
            print(f"step {step:>5} | distill_ce {metrics['distill_ce']:.4f} | "
                  f"rev_kl {metrics['rev_kl']:.4f} fwd_kl {metrics['fwd_kl']:.4f} | "
                  f"top1_agree {metrics['top1_agree']*100:.1f}% | "
                  f"s_p_at_t_top1 {metrics['p_at_t_top1']*100:.1f}% | "
                  f"{tok_s:.0f} tok/s",
                  flush=True)
            log_f.write(json.dumps({"step": step, **metrics, "tok_s": tok_s}) + "\n")
            log_f.flush()

        # Track best by rev_kl (lower = closer to teacher)
        if step % args.val_interval == 0:
            if metrics["rev_kl"] < best_rev_kl:
                best_rev_kl = metrics["rev_kl"]
                torch.save({
                    "model": student.state_dict(),
                    "step": step,
                    "rev_kl": best_rev_kl,
                    "qat_currently_on": args.qat,
                    "config": vars(mcfg) if hasattr(mcfg, '__dict__') else None,
                }, out_dir / "checkpoints/best_distill_qat.pt")
                print(f"  saved best_distill_qat.pt (rev_kl={best_rev_kl:.4f}) ←★", flush=True)

        if step % args.save_interval == 0:
            torch.save({
                "model": student.state_dict(),
                "step": step,
                "qat_currently_on": True,
            }, out_dir / f"checkpoints/ckpt_distill_{step}.pt")

    # Final
    torch.save({
        "model": student.state_dict(),
        "step": args.max_steps,
        "rev_kl": best_rev_kl,
        "qat_currently_on": True,
    }, out_dir / f"checkpoints/ckpt_final_distill_{args.max_steps}.pt")
    print(f"[distill] done. best_rev_kl={best_rev_kl:.4f}", flush=True)
    log_f.close()


if __name__ == "__main__":
    main()
