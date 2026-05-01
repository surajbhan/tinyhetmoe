#!/usr/bin/env python3
"""stitch_demo_v66.py — manual-eval demo for hierarchical-MoE stitching.

Loads v6.6 padded best_qat.pt for pg19 + wiki experts (both at vocab=32772
and QAT-on so they reflect deployed reality). Then for a fixed prompt,
produces 4 continuations side-by-side:

  1. pg19_only (always route to pg19)
  2. wiki_only (always route to wiki)
  3. oracle (oracle would only matter for stitched eval — for free-form
     generation we instead show "switch every 50 tokens" as a simple
     deterministic alternation, demonstrating that the experts compose
     architecturally)
  4. learned (132→64→2 classifier picks per token from EMA meaning vector)

Plus the cross-domain val matrix on the v6.6 deployed-honest checkpoints.

Output: prints all 4 generations + the matrix. Manually inspect to verify:
  - pg19_only on a wiki prompt: should drift to literary register
  - wiki_only on a literary prompt: should drift to encyclopedic register
  - learned routing: should pick the right expert based on prompt style
  - alternation: should still be coherent within each chunk (proves
    architectural composability)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.tiny_hetmoe import (
    TinyHetMoE, TinyHetMoEConfig, set_quantize_mode,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_TOKENS = 100_000
SEQ_LEN = 512
EMA_ALPHA = 0.05
PROSE_MASK_TIDS = [32768, 32769, 32770, 32771]  # ChatML specials

PROMPTS = [
    "The old wooden house at the edge of the meadow had been empty for many years, until one autumn afternoon a stranger arrived",
    "The Battle of Hastings, fought on 14 October 1066, was a decisive Norman victory that",
    "She walked through the moonlit garden, where",
    "Mount Everest is the highest mountain above sea level. It is located in",
]


def load_v66_padded(ckpt_path: Path, cfg_path: Path) -> TinyHetMoE:
    cfg_dict = json.load(cfg_path.open())
    cfg_dict["vocab_size"] = 32772
    mc_fields = {
        "vocab_size", "meaning_dim", "intuition_dim", "input_dim",
        "internal_dim", "new_intuition", "num_layers", "num_heads",
        "num_experts", "top_k_experts", "ffn_mult", "max_seq_len",
        "load_balance_weight",
    }
    mc_kwargs = {k: v for k, v in cfg_dict.items() if k in mc_fields}
    mcfg = TinyHetMoEConfig(**mc_kwargs)
    model = TinyHetMoE(mcfg)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob.get("state_dict", blob))
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    set_quantize_mode(model, on=True)
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def eval_corpus(model: TinyHetMoE, ids: np.ndarray, label: str) -> float:
    n = (len(ids) // SEQ_LEN) * SEQ_LEN
    arr = ids[:n].reshape(-1, SEQ_LEN)
    if len(arr) > EVAL_TOKENS // SEQ_LEN:
        arr = arr[:EVAL_TOKENS // SEQ_LEN]
    inputs = torch.from_numpy(arr[:, :-1].astype(np.int64)).to(DEVICE)
    targets = torch.from_numpy(arr[:, 1:].astype(np.int64)).to(DEVICE)
    total_loss = 0.0
    total_count = 0
    BATCH = 8
    mask_t = torch.tensor(PROSE_MASK_TIDS, device=DEVICE, dtype=torch.long)
    for i in range(0, inputs.shape[0], BATCH):
        x = inputs[i:i + BATCH]
        y = targets[i:i + BATCH]
        logits, _ = model(x)
        logits[..., mask_t] = -1e9
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1), reduction="sum"
        )
        total_loss += ce.item()
        total_count += y.numel()
    avg = total_loss / total_count
    print(f"     {label}: ce {avg:.4f}  ppl {np.exp(avg):.2f}  "
          f"({total_count:,} tokens)")
    return avg


def train_classifier(axes: np.ndarray, vals: list[tuple[str, np.ndarray]],
                     window: int = 512, hidden: int = 64,
                     epochs: int = 30, batch: int = 256) -> dict:
    """Reusable: train a 132→hidden→K MLP on EMA-meaning of val windows."""
    feats, labels = [], []
    for label_idx, (name, val_ids) in enumerate(vals):
        n_w = len(val_ids) // window
        seqs = val_ids[:n_w * window].reshape(n_w, window)
        m = axes[seqs]  # (n_w, window, 132)
        decay = (1.0 - EMA_ALPHA)
        w = (EMA_ALPHA *
             (decay ** np.arange(window - 1, -1, -1, dtype=np.float32)))
        f = (m * w[None, :, None]).sum(axis=1).astype(np.float32)
        feats.append(f)
        labels.append(np.full(len(f), label_idx, dtype=np.int64))
        print(f"     {name}: {len(f)} windows")

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    n_val = int(len(X) * 0.2)
    X_tr, y_tr = X[perm[n_val:]], y[perm[n_val:]]
    X_va, y_va = X[perm[:n_val]], y[perm[:n_val]]
    mu, sd = X_tr.mean(axis=0, keepdims=True), X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd
    K = len(vals)
    print(f"     train {len(X_tr)}, val {len(X_va)}, classes={K}")

    Xt = torch.from_numpy(X_tr).to(DEVICE)
    yt = torch.from_numpy(y_tr).to(DEVICE)
    Xv = torch.from_numpy(X_va).to(DEVICE)
    yv = torch.from_numpy(y_va).to(DEVICE)
    clf = nn.Sequential(
        nn.Linear(132, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, K),
    ).to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=3e-4, weight_decay=1e-4)
    best_va, best_state = 0.0, None
    for ep in range(epochs):
        clf.train()
        order = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), batch):
            ix = order[i:i + batch]
            loss = F.cross_entropy(clf(Xt[ix]), yt[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        clf.eval()
        with torch.no_grad():
            va_acc = (clf(Xv).argmax(-1) == yv).float().mean().item()
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.detach().clone() for k, v in clf.state_dict().items()}
    clf.load_state_dict(best_state)
    print(f"     best val acc: {best_va*100:.2f}%")
    return {"clf": clf, "mu": mu, "sd": sd, "best_va": best_va, "K": K}


@torch.no_grad()
def generate(models: list[TinyHetMoE], prompt_tids: list[int],
             router_fn, n_tokens: int = 80, temp: float = 0.7,
             top_k: int = 40) -> tuple[list[int], list[int]]:
    """Generate n_tokens. router_fn(tokens_so_far, position) -> expert_idx.
    Returns (output_tids, route_per_token). For each generation step,
    we recompute logits on the FULL current sequence under the chosen
    expert (no KV cache reuse — slower but trivially correct for swapping).
    """
    out = list(prompt_tids)
    routes = [-1] * len(prompt_tids)  # routes for prompt tokens are unset
    mask_t = torch.tensor(PROSE_MASK_TIDS, device=DEVICE, dtype=torch.long)
    for _ in range(n_tokens):
        chosen = router_fn(out, len(out))
        m = models[chosen]
        ids = torch.tensor([out], dtype=torch.long, device=DEVICE)
        logits, _ = m(ids)
        next_logits = logits[0, -1].clone()
        next_logits[mask_t] = -1e9
        next_logits = next_logits / temp
        v, _ = torch.topk(next_logits, top_k)
        next_logits[next_logits < v[-1]] = -1e9
        nid = int(torch.multinomial(F.softmax(next_logits, dim=-1), 1).item())
        out.append(nid)
        routes.append(chosen)
    return out, routes


def main():
    print(f"[stitch] device {DEVICE}")
    # ── Load tokenizer + axes ────────────────────────────────────────
    qwen_tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True,
    )
    vocab = json.load((REPO / "tokenizer/unified_vocab.json").open())
    qwen_to_tiny = {int(k): v for k, v in vocab["qwen_to_tiny"].items()}
    tiny_to_qwen = vocab["tiny_to_qwen"]

    def encode(text):
        qids = qwen_tok(text, add_special_tokens=False)["input_ids"]
        return [qwen_to_tiny.get(int(q), 0) for q in qids]

    def decode(tids):
        qs = [tiny_to_qwen[t] for t in tids
              if t > 3 and tiny_to_qwen[t] is not None]
        return qwen_tok.decode(qs)

    axes = np.load(REPO / "data/unified_meaning_axes_132.npy").astype(np.float32)
    print(f"[stitch] axes shape: {axes.shape}")

    # ── Load v6.6 models ─────────────────────────────────────────────
    print(f"\n[stitch] loading v6.6 models")
    pg19 = load_v66_padded(
        REPO / "runs/tiny_hetmoe_v6_6_pg19_qat/checkpoints/best_qat_padded.pt",
        REPO / "training/configs/tiny_hetmoe_v6_6_pg19_qat.json",
    )
    print(f"  pg19 v6.6 loaded (vocab=32772, QAT on)")
    wiki = load_v66_padded(
        REPO / "runs/tiny_hetmoe_v6_6_wiki_qat/checkpoints/best_qat_padded.pt",
        REPO / "training/configs/tiny_hetmoe_v6_6_wiki_qat.json",
    )
    print(f"  wiki v6.6 loaded (vocab=32772, QAT on)")
    models = [pg19, wiki]
    NAMES = ["pg19", "wiki"]

    # ── Load val .bins ───────────────────────────────────────────────
    pg19_val = np.fromfile(REPO / "data/unified_val_pg19.bin", dtype=np.uint16)
    wiki_val = np.fromfile(REPO / "data/unified_val_wiki.bin", dtype=np.uint16)
    print(f"\n[stitch] pg19_val: {len(pg19_val):,}, wiki_val: {len(wiki_val):,}")

    # ── 1. Cross-domain matrix ────────────────────────────────────────
    print(f"\n[1] cross-domain val matrix on v6.6 (deployed-honest, lower=better)")
    print(f"  pg19_expert on:")
    pp = eval_corpus(pg19, pg19_val, "pg19_val (in-domain)")
    pw = eval_corpus(pg19, wiki_val, "wiki_val (cross)")
    print(f"  wiki_expert on:")
    wp = eval_corpus(wiki, pg19_val, "pg19_val (cross)")
    ww = eval_corpus(wiki, wiki_val, "wiki_val (in-domain)")

    print(f"\n  Matrix:")
    print(f"                   pg19_val   wiki_val")
    print(f"    pg19_expert:    {pp:.4f}     {pw:.4f}")
    print(f"    wiki_expert:    {wp:.4f}     {ww:.4f}")
    print(f"  in-domain advantage:")
    print(f"    pg19: {pp:.4f} vs cross {wp:.4f}  → Δ = {wp - pp:+.4f}")
    print(f"    wiki: {ww:.4f} vs cross {pw:.4f}  → Δ = {pw - ww:+.4f}")

    # ── 2. Train classifier on v6.6 unified val ──────────────────────
    print(f"\n[2] training 132→64→2 domain classifier on unified val")
    clf_blob = train_classifier(
        axes,
        [("pg19", pg19_val), ("wiki", wiki_val)],
    )

    # ── 3. Generation under different routing policies ────────────────
    print(f"\n[3] generating continuations under 4 routing policies")
    print(f"     ROUTING:  pg19_only | wiki_only | alternate-50 | learned")
    print(f"     Mark each generated token with [P] or [W] for routing trace")

    def route_pg19_only(seq, pos): return 0
    def route_wiki_only(seq, pos): return 1
    def route_alternate(seq, pos): return (pos // 50) % 2

    @torch.no_grad()
    def route_learned(seq, pos):
        # Compute EMA meaning over the sequence so far, classify
        if len(seq) == 0:
            return 0
        meaning = axes[np.array(seq)]
        decay = 1.0 - EMA_ALPHA
        weights = EMA_ALPHA * decay ** np.arange(len(seq) - 1, -1, -1, dtype=np.float32)
        feat = (meaning * weights[:, None]).sum(axis=0)  # (132,)
        feat = (feat - clf_blob["mu"][0]) / clf_blob["sd"][0]
        with torch.no_grad():
            logits = clf_blob["clf"](torch.from_numpy(feat).float().to(DEVICE))
            return int(logits.argmax().item())

    routers = [
        ("pg19_only",     route_pg19_only),
        ("wiki_only",     route_wiki_only),
        ("alternate-50",  route_alternate),
        ("learned",       route_learned),
    ]

    for prompt in PROMPTS:
        prompt_tids = encode(prompt)
        print(f"\n--- PROMPT: {prompt!r} ---")
        for label, rfn in routers:
            out_tids, routes = generate(models, prompt_tids, rfn, n_tokens=60, temp=0.7)
            generated_only = out_tids[len(prompt_tids):]
            generated_routes = routes[len(prompt_tids):]
            text = decode(out_tids)
            # Compose route trace string
            route_str = "".join(["P" if r == 0 else "W" for r in generated_routes])
            print(f"\n  [{label}] routes: {route_str}")
            print(f"    {text!r}")

    # ── 4. Summary ──────────────────────────────────────────────────
    print(f"\n[4] summary")
    print(f"  cross-domain Δ (in vs cross, larger = stronger specialization):")
    print(f"    pg19: {wp - pp:+.4f} nats  (PPL ratio {np.exp(wp - pp):.2f}x)")
    print(f"    wiki: {pw - ww:+.4f} nats  (PPL ratio {np.exp(pw - ww):.2f}x)")
    print(f"  classifier acc on held-out unified val: {clf_blob['best_va']*100:.2f}%")


if __name__ == "__main__":
    main()
