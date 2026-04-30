#!/usr/bin/env python3
"""train_domain_classifier.py — outer router for hierarchical HetMoE.

The classifier takes the EMA of meaning-axis vectors over a token window
and predicts which domain expert should serve the next token.

This is the binding question for the routing thesis: can the frozen
132-axis meaning embedding distinguish domains well enough that a
2-layer MLP gets clean held-out accuracy?

Inputs:
  data/unified_val_*.bin               (uint16 token streams)
  data/unified_meaning_axes_132.npy    (vocab → 132-axis lookup)

Outputs:
  data/domain_classifier.pt            (trained MLP weights)
  + console eval: held-out accuracy, confidence margins, acc-vs-position

Usage:
  python3 scripts/train_domain_classifier.py
  python3 scripts/train_domain_classifier.py --domains pg19,wiki,tool,code
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

WINDOW = 512        # tokens per classification example
EMA_ALPHA = 0.05    # exponential smoothing over ~20-token effective horizon
HIDDEN = 64
DROPOUT = 0.1
LR = 3e-4
WD = 1e-4
EPOCHS = 30
BATCH = 256
VAL_FRAC = 0.2
SEED = 42


def load_corpus_windows(name: str, axes: np.ndarray, window: int) -> np.ndarray:
    """Read uint16 .bin, slice into non-overlapping windows, return EMA-meaning."""
    path = DATA_DIR / f"unified_val_{name}.bin"
    raw = np.fromfile(path, dtype=np.uint16)
    n_windows = len(raw) // window
    raw = raw[: n_windows * window].reshape(n_windows, window)

    # Look up meaning vectors for every token in every window
    meaning = axes[raw]  # (n_windows, window, 132)

    # Per-window EMA: y_t = (1-α)·y_{t-1} + α·x_t.
    # Vectorized via cumulative weighting; final EMA value is what we use.
    decay = (1.0 - EMA_ALPHA)
    weights = (EMA_ALPHA *
               (decay ** np.arange(window - 1, -1, -1, dtype=np.float32)))
    feat = (meaning * weights[None, :, None]).sum(axis=1)
    feat += decay ** window * 0.0  # initial EMA is zero
    return feat.astype(np.float32)


def windows_at_positions(name: str, axes: np.ndarray,
                         positions: list[int]) -> dict[int, np.ndarray]:
    """For acc-vs-position: same windows but EMA truncated at each position."""
    path = DATA_DIR / f"unified_val_{name}.bin"
    raw = np.fromfile(path, dtype=np.uint16)
    max_pos = max(positions)
    n_windows = len(raw) // max_pos
    raw = raw[: n_windows * max_pos].reshape(n_windows, max_pos)
    meaning = axes[raw]  # (n_windows, max_pos, 132)

    out: dict[int, np.ndarray] = {}
    for p in positions:
        decay = (1.0 - EMA_ALPHA)
        w = (EMA_ALPHA *
             (decay ** np.arange(p - 1, -1, -1, dtype=np.float32)))
        feat = (meaning[:, :p, :] * w[None, :, None]).sum(axis=1)
        out[p] = feat.astype(np.float32)
    return out


class DomainMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="pg19,wiki",
                    help="comma-separated; needs a unified_val_<name>.bin per")
    ap.add_argument("--device", default="cpu",
                    help="cpu is fine for ~10K windows × 132 dim")
    args = ap.parse_args()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    print(f"[clf] domains: {domains}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[clf] loading meaning axes")
    axes = np.load(DATA_DIR / "unified_meaning_axes_132.npy")
    print(f"     axes shape: {axes.shape}")

    # Build dataset
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label, name in enumerate(domains):
        f = load_corpus_windows(name, axes, WINDOW)
        feats.append(f)
        labels.append(np.full(len(f), label, dtype=np.int64))
        print(f"     {name}: {len(f)} windows of {WINDOW} tokens")

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)

    # Train/val split
    perm = np.random.permutation(len(X))
    n_val = int(len(X) * VAL_FRAC)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    print(f"[clf] train {len(X_tr)}, val {len(X_va)}")

    # Normalize features (per-dim z-score on train)
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd

    dev = torch.device(args.device)
    Xt = torch.from_numpy(X_tr).to(dev)
    yt = torch.from_numpy(y_tr).to(dev)
    Xv = torch.from_numpy(X_va).to(dev)
    yv = torch.from_numpy(y_va).to(dev)

    model = DomainMLP(132, HIDDEN, len(domains), DROPOUT).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    print(f"[clf] training {EPOCHS} epochs, batch {BATCH}")
    n_tr = len(Xt)
    best_va = 0.0
    best_state = None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_tr, device=dev)
        tot_loss = 0.0
        for i in range(0, n_tr, BATCH):
            ix = perm[i:i + BATCH]
            logits = model(Xt[ix])
            loss = F.cross_entropy(logits, yt[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(ix)
        tr_loss = tot_loss / n_tr

        model.eval()
        with torch.no_grad():
            va_logits = model(Xv)
            va_acc = (va_logits.argmax(-1) == yv).float().mean().item()
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  ep {ep:>3} | tr_loss {tr_loss:.4f} | va_acc {va_acc*100:.2f}% "
                  f"(best {best_va*100:.2f}%)")

    model.load_state_dict(best_state)

    # Save
    out_path = DATA_DIR / f"domain_classifier_{'_'.join(domains)}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "domains": domains,
        "feat_mean": mu.astype(np.float32),
        "feat_std": sd.astype(np.float32),
        "window": WINDOW,
        "ema_alpha": EMA_ALPHA,
        "best_val_acc": best_va,
    }, out_path)
    print(f"[clf] saved {out_path}  best_va={best_va*100:.2f}%")

    # ── Detailed eval ─────────────────────────────────────────────────
    print(f"\n[eval] confidence-margin distribution on val set")
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(Xv), dim=-1)
        top2 = probs.topk(min(2, len(domains)), dim=-1).values
        margin = (top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)).cpu().numpy()
    qs = [10, 25, 50, 75, 90]
    pcts = np.percentile(margin, qs)
    print("     margin  %ile:", "  ".join(f"p{q}={p:.3f}" for q, p in zip(qs, pcts)))
    print(f"     mean margin {margin.mean():.3f}, "
          f"frac_high_conf(>0.9): {(margin > 0.9).mean()*100:.1f}%")

    # Per-domain acc
    print(f"\n[eval] per-domain held-out accuracy")
    with torch.no_grad():
        preds = model(Xv).argmax(-1).cpu().numpy()
    yv_np = yv.cpu().numpy()
    for label, name in enumerate(domains):
        m = (yv_np == label)
        if m.sum():
            acc = (preds[m] == label).mean()
            print(f"     {name}: {m.sum()} examples, acc {acc*100:.2f}%")

    # Acc vs position
    print(f"\n[eval] accuracy vs window length (how many tokens before decision)")
    positions = [16, 32, 64, 128, 256, 512]
    pos_feats: dict[int, np.ndarray] = {}
    pos_labels: list[int] = []
    for label, name in enumerate(domains):
        per_pos = windows_at_positions(name, axes, positions)
        for p, f in per_pos.items():
            pos_feats.setdefault(p, []).append(f)
        pos_labels.extend([label] * len(per_pos[positions[0]]))
    yp = torch.tensor(pos_labels, device=dev)
    for p in positions:
        Xp = np.concatenate(pos_feats[p], axis=0)
        Xp = (Xp - mu) / sd
        Xp_t = torch.from_numpy(Xp.astype(np.float32)).to(dev)
        with torch.no_grad():
            acc = (model(Xp_t).argmax(-1) == yp).float().mean().item()
        print(f"     pos={p:>4}: acc {acc*100:.2f}%")


if __name__ == "__main__":
    main()
