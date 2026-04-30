#!/usr/bin/env python3
"""eval_stitching.py — does specialization + routing actually compose?

Three questions answered:

  1. **Cross-domain matrix** (4 cells): both checkpoints × both vals.
     If `wiki_expert(wiki_val) < pg19_expert(wiki_val)` AND symmetrically,
     specialization is real.

  2. **Oracle routing on a stitched sequence**: alternate chunks of
     pg19_val and wiki_val, route each token to the ground-truth expert
     for its chunk. Compares against single-expert baselines on the
     same stitched sequence.

  3. **Learned routing**: same stitched sequence, but the classifier
     drives routing decisions per token. Tells us if the classifier
     trained today is good enough for production.

Outputs raw numbers; no plots, no decisions — those go in the journal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
sys.path.insert(0, str(REPO))

from model.tiny_hetmoe import TinyHetMoE, TinyHetMoEConfig  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_TOKENS = 200_000     # cap eval tokens for speed
SEQ_LEN = 512             # match training
CHUNK_LEN = 200           # tokens per single-domain chunk in stitched seq
EMA_ALPHA = 0.05


def load_model(ckpt_path: Path, cfg_path: Path) -> TinyHetMoE:
    cfg_dict = json.load(cfg_path.open())
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
    model.eval()
    model.to(DEVICE)
    return model


@torch.no_grad()
def eval_corpus(model: TinyHetMoE, ids: np.ndarray, label: str) -> float:
    """Return average cross-entropy (nats per token) on `ids`."""
    n = (len(ids) // SEQ_LEN) * SEQ_LEN
    ids = ids[:n].reshape(-1, SEQ_LEN)
    if len(ids) > EVAL_TOKENS // SEQ_LEN:
        ids = ids[: EVAL_TOKENS // SEQ_LEN]
    inputs = torch.from_numpy(ids[:, :-1].astype(np.int64)).to(DEVICE)
    targets = torch.from_numpy(ids[:, 1:].astype(np.int64)).to(DEVICE)

    total_loss = 0.0
    total_count = 0
    BATCH = 8
    for i in range(0, inputs.shape[0], BATCH):
        x = inputs[i:i + BATCH]
        y = targets[i:i + BATCH]
        logits, _ = model(x)  # logits: (B, T, V)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += ce.item()
        total_count += y.numel()
    avg = total_loss / total_count
    print(f"     {label}: ce {avg:.4f}  ppl {np.exp(avg):.2f}  "
          f"({total_count:,} tokens)")
    return avg


def build_stitched(pg19_ids: np.ndarray, wiki_ids: np.ndarray,
                   chunk_len: int, n_chunks: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (token_ids, domain_label_per_token) — alternating chunks."""
    rng = np.random.default_rng(7)
    out_ids = []
    out_lab = []  # 0 = pg19, 1 = wiki
    for c in range(n_chunks):
        if c % 2 == 0:
            src, lab = pg19_ids, 0
        else:
            src, lab = wiki_ids, 1
        start = rng.integers(0, len(src) - chunk_len)
        chunk = src[start : start + chunk_len]
        out_ids.append(chunk)
        out_lab.append(np.full(chunk_len, lab, dtype=np.int64))
    return np.concatenate(out_ids), np.concatenate(out_lab)


@torch.no_grad()
def eval_routed(models: dict, stitched_ids: np.ndarray,
                router_fn, label: str) -> float:
    """For each token i, use `router_fn(token_ids[:i+1], i)` to choose
    which model produces logits[i]. Score CE on stitched_ids[1:] vs
    those routed logits. router_fn returns 0 or 1 (model index).
    """
    n = (len(stitched_ids) // SEQ_LEN) * SEQ_LEN
    ids = stitched_ids[:n].reshape(-1, SEQ_LEN)
    if len(ids) > EVAL_TOKENS // SEQ_LEN:
        ids = ids[: EVAL_TOKENS // SEQ_LEN]

    inputs = torch.from_numpy(ids[:, :-1].astype(np.int64)).to(DEVICE)
    targets = torch.from_numpy(ids[:, 1:].astype(np.int64)).to(DEVICE)

    total_loss = 0.0
    total_count = 0
    routes_used = [0, 0]
    BATCH = 8
    model_pg19, model_wiki = models["pg19"], models["wiki"]
    for i in range(0, inputs.shape[0], BATCH):
        x = inputs[i:i + BATCH]                     # (B, T)
        y = targets[i:i + BATCH]                    # (B, T)
        logits_pg, _ = model_pg19(x)                # (B, T, V)
        logits_wk, _ = model_wiki(x)
        # Per-token route. Build a mask (B, T) of 0/1.
        route_mask = router_fn(x, i)                # numpy (B, T)
        rm = torch.from_numpy(route_mask.astype(np.bool_)).to(DEVICE)
        # logits: pick wiki where rm=True else pg19
        logits = torch.where(rm.unsqueeze(-1), logits_wk, logits_pg)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += ce.item()
        total_count += y.numel()
        routes_used[0] += (route_mask == 0).sum()
        routes_used[1] += (route_mask == 1).sum()
    avg = total_loss / total_count
    frac_wiki = routes_used[1] / sum(routes_used)
    print(f"     {label}: ce {avg:.4f}  ppl {np.exp(avg):.2f}  "
          f"(routed wiki {frac_wiki*100:.1f}% of tokens)")
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg19_ckpt", default="runs/tiny_hetmoe_v6_pg19/checkpoints/best.pt")
    ap.add_argument("--wiki_ckpt", default="runs/tiny_hetmoe_v6_wiki/checkpoints/best.pt")
    ap.add_argument("--pg19_cfg",  default="training/configs/tiny_hetmoe_v6_pg19.json")
    ap.add_argument("--wiki_cfg",  default="training/configs/tiny_hetmoe_v6_wiki.json")
    args = ap.parse_args()

    print(f"[stitch] device {DEVICE}")
    print(f"[stitch] loading models")
    pg19_model = load_model(REPO / args.pg19_ckpt, REPO / args.pg19_cfg)
    wiki_model = load_model(REPO / args.wiki_ckpt, REPO / args.wiki_cfg)

    print(f"[stitch] loading val .bins")
    pg19_val = np.fromfile(DATA_DIR / "unified_val_pg19.bin", dtype=np.uint16)
    wiki_val = np.fromfile(DATA_DIR / "unified_val_wiki.bin", dtype=np.uint16)
    print(f"     pg19_val: {len(pg19_val):,}, wiki_val: {len(wiki_val):,}")

    # ── 1. Cross-domain matrix ────────────────────────────────────────
    print(f"\n[1] cross-domain val matrix (lower is better)")
    print(f"  pg19_expert on:")
    pg19_on_pg19 = eval_corpus(pg19_model, pg19_val, "pg19_val (in-domain)")
    pg19_on_wiki = eval_corpus(pg19_model, wiki_val, "wiki_val (cross)")
    print(f"  wiki_expert on:")
    wiki_on_pg19 = eval_corpus(wiki_model, pg19_val, "pg19_val (cross)")
    wiki_on_wiki = eval_corpus(wiki_model, wiki_val, "wiki_val (in-domain)")

    print(f"\n  Matrix:")
    print(f"                   pg19_val   wiki_val")
    print(f"    pg19_expert:    {pg19_on_pg19:.4f}     {pg19_on_wiki:.4f}")
    print(f"    wiki_expert:    {wiki_on_pg19:.4f}     {wiki_on_wiki:.4f}")
    print(f"  in-domain advantage:")
    print(f"    pg19: {pg19_on_pg19:.4f} vs cross {wiki_on_pg19:.4f}  "
          f"→ Δ = {wiki_on_pg19 - pg19_on_pg19:+.4f}")
    print(f"    wiki: {wiki_on_wiki:.4f} vs cross {pg19_on_wiki:.4f}  "
          f"→ Δ = {pg19_on_wiki - wiki_on_wiki:+.4f}")

    # ── 2. Stitched sequence ──────────────────────────────────────────
    print(f"\n[2] building stitched sequence (alternating {CHUNK_LEN}-token chunks)")
    n_chunks = (EVAL_TOKENS // CHUNK_LEN)
    stitched, gt_label = build_stitched(pg19_val, wiki_val, CHUNK_LEN, n_chunks)
    print(f"     stitched: {len(stitched):,} tokens, {n_chunks} chunks")

    models = {"pg19": pg19_model, "wiki": wiki_model}

    # 2a. Single-expert baselines on stitched
    print(f"\n  single-expert baselines on stitched seq:")
    pg_only = lambda x, _i: np.zeros((x.shape[0], x.shape[1]), dtype=np.int64)
    wk_only = lambda x, _i: np.ones((x.shape[0], x.shape[1]), dtype=np.int64)
    base_pg = eval_routed(models, stitched, pg_only, "pg19_only")
    base_wk = eval_routed(models, stitched, wk_only, "wiki_only")

    # 2b. Oracle routing — use ground-truth label
    print(f"\n  oracle routing (ground-truth chunk labels):")
    # Reshape gt_label to match (B, T) batches
    gt_reshape = gt_label[: (len(gt_label) // SEQ_LEN) * SEQ_LEN].reshape(-1, SEQ_LEN)
    if len(gt_reshape) > EVAL_TOKENS // SEQ_LEN:
        gt_reshape = gt_reshape[: EVAL_TOKENS // SEQ_LEN]

    # Predict next token: target is at position+1, so route by gt_label[t+1]
    # But gt_label is per-input-token. We score on inputs[:,:-1] → targets[:,1:],
    # so route_for_logits[t] = gt_label_at_target_position = gt_reshape[:,1:][t]
    # The routing call sees x = inputs[:, :-1] which has shape (B, T-1), so
    # we want gt_for_targets = gt_reshape[:, 1:]
    BATCH = 8
    def oracle(_x, batch_start):
        end = min(batch_start + BATCH, len(gt_reshape))
        return gt_reshape[batch_start:end, 1:]
    oracle_loss = eval_routed(models, stitched, oracle, "oracle")

    # 2c. Learned routing — classifier drives decisions
    print(f"\n  learned routing (132→64→2 classifier from earlier):")
    clf_path = DATA_DIR / "domain_classifier_pg19_wiki.pt"
    clf_blob = torch.load(clf_path, map_location=DEVICE, weights_only=False)
    clf_w = clf_blob["state_dict"]
    feat_mu = torch.from_numpy(clf_blob["feat_mean"]).to(DEVICE)
    feat_sd = torch.from_numpy(clf_blob["feat_std"]).to(DEVICE)
    axes = np.load(DATA_DIR / "unified_meaning_axes_132.npy")
    axes_t = torch.from_numpy(axes).to(DEVICE)

    # Reconstruct the classifier net (132→64→2)
    import torch.nn as nn
    clf = nn.Sequential(
        nn.Linear(132, 64), nn.ReLU(), nn.Dropout(0.0), nn.Linear(64, 2)
    ).to(DEVICE)
    clf.load_state_dict({
        "0.weight": clf_w["net.0.weight"],
        "0.bias":   clf_w["net.0.bias"],
        "3.weight": clf_w["net.3.weight"],
        "3.bias":   clf_w["net.3.bias"],
    })
    clf.eval()

    # For each input token position, build its EMA-of-meaning feature
    # using only tokens up to and including itself (causal). Then predict
    # the route to use for the NEXT token.
    def learned(x, _batch_start):
        # x: (B, T) input tokens (T = SEQ_LEN-1 here)
        meaning = axes_t[x]  # (B, T, 132)
        # Causal EMA over T: e[t] = (1-α)e[t-1] + α m[t]
        B, T, M = meaning.shape
        e = torch.zeros(B, T, M, device=DEVICE)
        decay = 1.0 - EMA_ALPHA
        prev = torch.zeros(B, M, device=DEVICE)
        for t in range(T):
            prev = decay * prev + EMA_ALPHA * meaning[:, t, :]
            e[:, t, :] = prev
        e_norm = (e - feat_mu) / feat_sd
        logits = clf(e_norm.reshape(-1, M)).reshape(B, T, 2)
        return logits.argmax(-1).cpu().numpy()

    learned_loss = eval_routed(models, stitched, learned, "learned")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n[summary]")
    print(f"  baselines on stitched: pg19_only={base_pg:.4f}  wiki_only={base_wk:.4f}")
    print(f"  oracle routing:        {oracle_loss:.4f}  "
          f"(saves {min(base_pg,base_wk)-oracle_loss:+.4f} vs best baseline)")
    print(f"  learned routing:       {learned_loss:.4f}  "
          f"(gap to oracle: {learned_loss-oracle_loss:+.4f})")


if __name__ == "__main__":
    main()
