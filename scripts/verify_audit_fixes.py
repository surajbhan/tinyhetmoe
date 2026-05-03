#!/usr/bin/env python3
"""verify_audit_fixes.py — train↔deploy correspondence smoke test.

Goal: confirm that after the v6.x audit fixes (RMSNorm eps=1e-6, fp32
chunked-CE with ternary-pre-quantized lm_head, GELU tanh approx, ckpt
QAT-state persistence), the loss reported by the trainer matches the
loss the deploy/inference path would compute, at the same numerical
precision the wasm engine uses (fp32 forward + ternary quant).

Test plan:
  1. Build a fresh TinyHetMoE at production v7 config.
  2. Load meaning embeddings, optionally a checkpoint.
  3. Run a small validation batch through TWO paths:
       (a) train path:  model(ids, targets) -> (None, loss)
                        chunked CE in fp32 with ternary lm_head pre-quant
       (b) deploy path: model(ids, None)   -> (logits, _)
                        manual fp32 CE on returned logits
  4. Diff (a) vs (b). Pass: |Δ| < 0.05 nats. Fail: anything larger
     points at a remaining train↔deploy drift bug.
  5. Also report aux_total (router load-balance loss) so we can confirm
     it's the only intentional offset.

Run:
    python3 scripts/verify_audit_fixes.py \\
        --val-bin data_v7/v7_val_general.bin \\
        --meaning data_v7/meaning_axes_full_132.npy \\
        --vocab 151665 \\
        [--ckpt runs/foo/checkpoints/best.pt] \\
        [--qat]
"""
from __future__ import annotations

import argparse
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


def load_val_batch(bin_path: str, vocab: int, seq_len: int, batch: int,
                   seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    sz = Path(bin_path).stat().st_size
    n = sz // 4  # uint32
    if n < seq_len + 1:
        raise RuntimeError(f"{bin_path} has {n} tokens, need {seq_len+1}")
    arr = np.memmap(bin_path, dtype=np.uint32, mode="r", shape=(n,))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - seq_len - 1, size=batch)
    ids = np.stack([np.asarray(arr[s:s+seq_len], dtype=np.int64) for s in starts])
    tgt = np.stack([np.asarray(arr[s+1:s+seq_len+1], dtype=np.int64) for s in starts])
    over = (ids >= vocab).sum() + (tgt >= vocab).sum()
    if over > 0:
        print(f"[warn] {over} token ids are >= vocab_size; clamping")
        ids = np.clip(ids, 0, vocab - 1)
        tgt = np.clip(tgt, 0, vocab - 1)
    return torch.from_numpy(ids), torch.from_numpy(tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-bin", required=True)
    ap.add_argument("--meaning", required=True)
    ap.add_argument("--vocab", type=int, default=151665)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--qat", action="store_true",
                    help="enable ternary forward (default off; emulates "
                         "bf16-phase trainer behavior)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol-nats", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # v7 production config — matches production TinyHetMoE settings
    mcfg = TinyHetMoEConfig(
        vocab_size=args.vocab,
        meaning_dim=132,
        intuition_dim=132,
        input_dim=264,
        internal_dim=528,
        new_intuition=264,
        num_layers=4,
        num_heads=4,
        num_experts=4,
        top_k_experts=2,
        ffn_mult=2.0,
        max_seq_len=args.seq_len,
    )
    model = TinyHetMoE(mcfg).to(device)
    model.load_meaning_embeddings(args.meaning, freeze=True)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt)
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
        qat_was_on = ckpt.get("qat_currently_on")
        print(f"[ckpt] qat_currently_on={qat_was_on}")

    model.eval()
    set_quantize_mode(model, args.qat)
    set_qat_backward_mode("ste")

    ids, tgt = load_val_batch(args.val_bin, args.vocab, args.seq_len,
                              args.batch, seed=args.seed)
    ids, tgt = ids.to(device), tgt.to(device)

    print(f"[smoke] vocab={args.vocab} seq={args.seq_len} batch={args.batch} "
          f"qat={args.qat} ckpt={'yes' if args.ckpt else 'fresh'}")

    # ----- Path A: trainer's chunked-CE (fp32 + ternary pre-quant if QAT)
    with torch.no_grad():
        _, loss_train = model(ids, targets=tgt)
    loss_train_v = float(loss_train.detach().cpu())

    # ----- Path B: deploy-style: forward returns logits, do CE externally
    #               in fp32 over the same ids. Aux loss is by definition
    #               not part of deploy CE.
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        logits, _ = model(ids, targets=None)
        logits_fp32 = logits.float()
        ce_deploy = F.cross_entropy(
            logits_fp32.reshape(-1, logits_fp32.size(-1)),
            tgt.reshape(-1),
            reduction="mean",
        )
    loss_deploy_v = float(ce_deploy.detach().cpu())

    # The trainer adds aux_total to its returned loss. To compare against
    # deploy CE we have to subtract it, but it's not directly returned —
    # we re-run the layer hooks (cheap) by reading the cached _aux from
    # the deploy-path forward we just did.
    aux = float(sum(layer._aux for layer in model.layers).detach().cpu())
    loss_train_minus_aux = loss_train_v - aux

    print()
    print(f"  train path (CE + aux):    {loss_train_v:.6f}")
    print(f"  aux load-balance:         {aux:.6f}")
    print(f"  train CE only (- aux):    {loss_train_minus_aux:.6f}")
    print(f"  deploy CE (logits route): {loss_deploy_v:.6f}")
    delta = abs(loss_train_minus_aux - loss_deploy_v)
    print(f"  |Δ|:                      {delta:.6f}")

    if delta < args.tol_nats:
        print(f"[smoke] PASS — train↔deploy match within {args.tol_nats} nats")
        sys.exit(0)
    else:
        print(f"[smoke] FAIL — train↔deploy diverge by {delta:.4f} nats "
              f"(tol={args.tol_nats})")
        sys.exit(2)


if __name__ == "__main__":
    main()
