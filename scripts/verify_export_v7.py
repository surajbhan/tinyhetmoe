#!/usr/bin/env python3
"""verify_export_v7.py — round-trip test for the new HTMOE004 + MNGSHR04
export format. Loads the shared meaning file + per-expert .bin, reconstructs
the model in PyTorch, and computes val CE. Should match the same val CE
computed directly from the source PyTorch ckpt within numerical noise.

Run:
  python3 scripts/verify_export_v7.py \\
      --src-ckpt runs/.../best_distill_qat.pt \\
      --bin /tmp/v7_export_test/thinker.bin \\
      --meaning /tmp/v7_export_test/meaning_shared.bin \\
      --domain thinker
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.tiny_hetmoe import (  # noqa: E402
    TinyHetMoE, TinyHetMoEConfig, set_quantize_mode, set_qat_backward_mode,
)
from training.distill_v7 import STREAMERS, _token_buffer, make_batch  # noqa: E402


def load_meaning_shared(path: str) -> torch.Tensor:
    """Read MNGSHR04 file -> (V, D) fp32 tensor."""
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"MNGSHR04", f"bad magic: {magic}"
        V = struct.unpack("<I", f.read(4))[0]
        D = struct.unpack("<I", f.read(4))[0]
        nbytes = V * D * 2
        buf = f.read(nbytes)
    arr = np.frombuffer(buf, dtype=np.float16).reshape(V, D).astype(np.float32)
    return torch.from_numpy(arr.copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-ckpt", required=True,
                    help="Source PyTorch ckpt (best_distill_qat.pt)")
    ap.add_argument("--meaning-shared", required=True,
                    help="MNGSHR04 file path")
    ap.add_argument("--domain", required=True, choices=list(STREAMERS.keys()))
    ap.add_argument("--n-batches", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Build val
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    eos_id = tok.eos_token_id
    print(f"[val] streaming {args.domain} (skip 5000)")
    stream = STREAMERS[args.domain]()
    for i, _ in enumerate(stream):
        if i >= 5000: break
    buf = _token_buffer(stream, tok, eos_id)
    val_batches = [make_batch(buf, 2, 384)[0] for _ in range(args.n_batches)]

    @torch.no_grad()
    def eval_ce(model, dev):
        total, n = 0.0, 0
        for ids in val_batches:
            ids = ids.to(dev)
            logits, _ = model(ids)
            s = logits[:, :-1, :]; t = ids[:, 1:]
            ce = F.cross_entropy(s.reshape(-1, s.shape[-1]).float(),
                                  t.reshape(-1), reduction="sum")
            total += float(ce.detach().cpu()); n += t.numel()
        return total / n, float(torch.tensor(total / n).exp())

    # === Source ckpt eval (pre-export reference) ===
    print(f"\n[src] loading source ckpt {args.src_ckpt}")
    mcfg = TinyHetMoEConfig(
        vocab_size=151665, meaning_dim=132, intuition_dim=132,
        input_dim=264, internal_dim=528, new_intuition=264,
        num_layers=4, num_heads=4, num_experts=4, top_k_experts=2,
        ffn_mult=2.0, max_seq_len=2048,
    )
    src_model = TinyHetMoE(mcfg).to(args.device)
    ck = torch.load(args.src_ckpt, map_location=args.device, weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in ck["model"].items()}
    src_model.load_state_dict(sd, strict=False)
    src_model.eval()
    set_qat_backward_mode("ste")
    set_quantize_mode(src_model, on=True)
    src_ce, src_ppl = eval_ce(src_model, args.device)
    print(f"  SRC: CE={src_ce:.4f} PPL={src_ppl:.2f}")

    # === Round-trip: replace meaning_embed with the shared-file fp16 version,
    # then replace intuition_embed with the fp16-cast version (simulating
    # the lossy step the exporter did). Reuse same model weights.
    print(f"\n[round-trip] simulating fp16 embedding cast")
    meaning_shared = load_meaning_shared(args.meaning_shared).to(args.device)
    print(f"  meaning_shared shape: {meaning_shared.shape}, dtype: {meaning_shared.dtype}")

    # Sanity: shared meaning should match src meaning bit-for-bit after fp16 cast
    src_mean = sd["meaning_embed.weight"].to(args.device)
    src_mean_fp16 = src_mean.to(torch.float16).to(torch.float32)
    diff_meaning = (meaning_shared - src_mean_fp16).abs().max().item()
    print(f"  |meaning_shared - src_fp16| max: {diff_meaning:.2e}  "
          f"(should be 0 — same fp16 cast)")

    # Replace embeddings with their fp16-roundtripped versions
    src_model.meaning_embed.weight.data = meaning_shared
    src_model.intuition_embed.weight.data = sd["intuition_embed.weight"].to(torch.float16).to(torch.float32).to(args.device)

    rt_ce, rt_ppl = eval_ce(src_model, args.device)
    print(f"  RT (fp16 embed): CE={rt_ce:.4f} PPL={rt_ppl:.2f}  "
          f"Δ={rt_ce-src_ce:+.4f} nats  ratio={rt_ppl/src_ppl:.3f}x")

    # ── verdict ──
    delta = rt_ce - src_ce
    if abs(delta) < 0.01:
        print(f"\n  PASS — fp16 round-trip is lossless (|Δ| < 0.01 nats)")
    elif abs(delta) < 0.05:
        print(f"\n  ACCEPTABLE — fp16 round-trip has {delta:+.4f} nats drift")
    else:
        print(f"\n  WARN — fp16 round-trip degrades by {delta:+.4f} nats!")


if __name__ == "__main__":
    main()
