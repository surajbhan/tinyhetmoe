#!/usr/bin/env python3
"""val_distill.py — measure student val CE on a held-out slice of the
distillation corpus, before vs after distillation. Compares against
the teacher's val CE as an upper bound.

Reports CE per token (lower=better) and PPL on the same val data the
distillation training streamed. This is the honest "did distillation
work" metric — Paris-style probes are out-of-distribution for
conversational corpora and don't reflect the actual training signal.

Run:
  python3 scripts/val_distill.py \\
      --student-ckpts before=runs/foo/best.pt after=runs/foo_distill/best.pt \\
      --teacher Qwen/Qwen2.5-0.5B \\
      --domain general \\
      --n-batches 50
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
from training.distill_v7 import STREAMERS, _token_buffer, make_batch  # noqa: E402


def load_student(path: str, vocab: int, qat: bool, device: str) -> TinyHetMoE:
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
def measure_ce(model, val_batches, device: str, is_teacher: bool = False) -> tuple[float, float]:
    """Returns (mean_ce, perplexity) over val_batches."""
    total_ce = 0.0
    total_n = 0
    for input_ids in val_batches:
        ids = input_ids.to(device)
        if is_teacher:
            logits = model(ids).logits
        else:
            logits, _ = model(ids)
        # Predict t+1 from t: shift
        s = logits[:, :-1, :]  # (B, T-1, V)
        targets = ids[:, 1:]   # (B, T-1)
        ce = F.cross_entropy(s.reshape(-1, s.shape[-1]).float(),
                             targets.reshape(-1),
                             reduction="sum")
        total_ce += float(ce.detach().cpu())
        total_n += targets.numel()
    mean_ce = total_ce / total_n
    return mean_ce, float(torch.tensor(mean_ce).exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-ckpts", nargs="+", required=True,
                    help="name=path pairs (e.g. before=ckpt1.pt after=ckpt2.pt)")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--domain", required=True, choices=list(STREAMERS.keys()))
    ap.add_argument("--meaning-emb", default="data_v7/meaning_axes_full_132.npy")
    ap.add_argument("--n-batches", type=int, default=50,
                    help="how many val batches to draw from the stream")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=151665)
    ap.add_argument("--seed", type=int, default=12345,
                    help="seed for skipping training-stream prefix to get held-out region")
    ap.add_argument("--skip", type=int, default=10000,
                    help="number of training-stream segments to skip before "
                         "starting val sampling (avoids overlap with training)")
    args = ap.parse_args()

    student_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    teacher_dev = "cuda:1" if torch.cuda.device_count() >= 2 else student_dev

    # ─ Build val batches (do this ONCE so all models see the same data) ─
    print(f"[val] streaming {args.domain} corpus, skipping {args.skip} segments…", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.teacher)
    eos_id = tok.eos_token_id
    stream = STREAMERS[args.domain]()
    # Skip ahead to avoid overlap with the training prefix
    for i, _ in enumerate(stream):
        if i >= args.skip:
            break
    # Keep going from this point as the val source
    buf = _token_buffer(stream, tok, eos_id)
    val_batches = []
    for _ in range(args.n_batches):
        try:
            ids, _ = make_batch(buf, args.batch_size, args.seq_len)
            val_batches.append(ids)
        except StopIteration:
            print(f"[val] stream exhausted at batch {len(val_batches)}", flush=True)
            break
    n_tok = sum(b.numel() for b in val_batches)
    print(f"[val] built {len(val_batches)} batches, {n_tok:,} tokens total", flush=True)

    # ─ Teacher CE (upper bound — what we're chasing) ─
    print(f"[val] loading teacher {args.teacher} on {teacher_dev}", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map={"": teacher_dev},
    )
    teacher.eval()
    t_ce, t_ppl = measure_ce(teacher, val_batches, teacher_dev, is_teacher=True)
    print(f"\n  TEACHER ({args.teacher}):  CE={t_ce:.4f}  PPL={t_ppl:.2f}", flush=True)
    del teacher
    torch.cuda.empty_cache()

    # ─ Student CE for each ckpt ─
    print()
    rows = []
    for spec in args.student_ckpts:
        name, path = spec.split("=", 1)
        print(f"[val] {name}: loading {path}", flush=True)
        # Try both qat and bf16 — best to load qat=False since the
        # forward path already gates on QuantizedLinear.quantize. We
        # decide qat mode based on the ckpt's "qat_currently_on" if set.
        ck_meta = torch.load(path, map_location="cpu", weights_only=False)
        qat = bool(ck_meta.get("qat_currently_on", False))
        m = load_student(path, args.vocab, qat, student_dev)
        s_ce, s_ppl = measure_ce(m, val_batches, student_dev)
        gap = s_ce - t_ce
        rows.append((name, path, qat, s_ce, s_ppl, gap))
        print(f"  {name} (qat={qat}): CE={s_ce:.4f}  PPL={s_ppl:.2f}  gap-to-teacher={gap:+.4f}",
              flush=True)
        del m
        torch.cuda.empty_cache()

    # ─ Summary table ─
    print(f"\n{'='*80}\n  SUMMARY (domain={args.domain}, n_tok={n_tok:,})  TEACHER CE={t_ce:.4f}  PPL={t_ppl:.2f}\n{'='*80}")
    print(f"  {'name':<40} {'qat':<5} {'CE':>8} {'PPL':>8} {'gap':>10}")
    for name, path, qat, s_ce, s_ppl, gap in rows:
        print(f"  {name:<40} {str(qat):<5} {s_ce:>8.4f} {s_ppl:>8.2f} {gap:>+10.4f}")


if __name__ == "__main__":
    main()
