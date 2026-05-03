#!/usr/bin/env python3
"""train_domain_classifier_v7.py — outer router for v7 hierarchical HetMoE.

Six v7 domains: general, thinker, code_py, code_js, medical, legal.
Uses the same meaning_axes (151,665 × 132 frozen) all experts share.

Differences from v6 trainer:
  - Streams text from HF datasets directly (no on-disk val bins needed)
  - 6 classes instead of 4
  - Token dtype is implicit (we tokenize on the fly with Qwen tokenizer)

Output:
  data_v7/domain_classifier_v7.pt — sd + mu/sd + domains list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.distill_v7 import STREAMERS  # noqa: E402

WINDOW = 512
EMA_ALPHA = 0.05
HIDDEN = 64
DROPOUT = 0.1
LR = 3e-4
WD = 1e-4
EPOCHS = 30
BATCH = 256
SEED = 42

DOMAINS = ["general", "thinker", "code_py", "code_js", "medical", "legal"]


def stream_features(domain: str, axes: np.ndarray, n_windows: int,
                    tok) -> np.ndarray:
    """Pull text from the domain stream, tokenize it, sample windows of
    VARYING length (mimicking what the classifier sees at test time on
    short prompts), lookup meaning vectors, EMA over the window.
    Returns (n_windows, 132).

    The window-length distribution is chosen so the classifier sees both
    short (cold-EMA) and long (saturated-EMA) views of each domain. The
    v1 trainer only used WINDOW=512 which made the classifier bad on
    test prompts of <30 tokens.
    """
    # Window length distribution: heavy on short windows so cold-EMA
    # cases (the actual test-time scenario) are well-represented.
    WIN_LENS = [4, 8, 16, 32, 64, 128, 256, 512]
    WIN_WEIGHTS = np.array([0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.07, 0.03])
    WIN_WEIGHTS = WIN_WEIGHTS / WIN_WEIGHTS.sum()

    print(f"[clf] streaming {domain} for {n_windows} variable-length windows...")
    stream = STREAMERS[domain]()

    # Buffer roughly enough text for the requested windows.
    avg_window = int((np.array(WIN_LENS) * WIN_WEIGHTS).sum())
    needed = n_windows * avg_window * 2  # 2x slack for sampling
    buf = []
    for text in stream:
        buf.extend(tok.encode(text, add_special_tokens=False))
        if len(buf) >= needed:
            break

    rng = np.random.default_rng(SEED + hash(domain) % 100)
    feats = np.zeros((n_windows, 132), dtype=np.float32)
    n_buf = len(buf)
    for w in range(n_windows):
        win_len = int(rng.choice(WIN_LENS, p=WIN_WEIGHTS))
        if n_buf < win_len + 1:
            continue
        start = int(rng.integers(0, n_buf - win_len))
        ids = np.clip(np.array(buf[start:start + win_len]),
                      0, axes.shape[0] - 1)
        meaning = axes[ids]  # (win_len, 132)
        decay = (1.0 - EMA_ALPHA)
        weights = (EMA_ALPHA *
                   (decay ** np.arange(win_len - 1, -1, -1, dtype=np.float32)))
        feats[w] = (meaning * weights[:, None]).sum(axis=0).astype(np.float32)
    return feats


class DomainMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
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
    ap.add_argument("--out", default="data_v7/domain_classifier_v7.pt")
    ap.add_argument("--axes", default="data_v7/meaning_axes_full_132.npy")
    ap.add_argument("--n-windows-per-domain", type=int, default=2000,
                    help="how many 512-token windows to sample per domain")
    ap.add_argument("--device", default="cpu",
                    help="cpu is fine: ~12K windows × 132 features")
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[clf] loading meaning axes from {args.axes}")
    axes = np.load(REPO / args.axes).astype(np.float32)
    print(f"[clf]   axes shape: {axes.shape}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

    # Build dataset by streaming each domain
    feats, labels = [], []
    for label, name in enumerate(DOMAINS):
        f = stream_features(name, axes, args.n_windows_per_domain, tok)
        feats.append(f)
        labels.append(np.full(len(f), label, dtype=np.int64))
        print(f"[clf]   {name}: {len(f)} windows")

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)

    # Train/val split
    perm = np.random.permutation(len(X))
    n_val = int(len(X) * args.val_frac)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    print(f"[clf] train {len(X_tr)}, val {len(X_va)}")

    # z-score on train
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd

    dev = torch.device(args.device)
    Xt = torch.from_numpy(X_tr).to(dev); yt = torch.from_numpy(y_tr).to(dev)
    Xv = torch.from_numpy(X_va).to(dev); yv = torch.from_numpy(y_va).to(dev)

    model = DomainMLP(132, HIDDEN, len(DOMAINS), DROPOUT).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    print(f"[clf] training {EPOCHS} epochs")
    n_tr = len(Xt)
    best_va, best_state = 0.0, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_tr, device=dev)
        tot_loss = 0.0
        for i in range(0, n_tr, BATCH):
            ix = perm[i:i + BATCH]
            logits = model(Xt[ix])
            loss = F.cross_entropy(logits, yt[ix])
            opt.zero_grad(); loss.backward(); opt.step()
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

    # Confusion matrix
    print(f"\n[eval] confusion matrix (rows=true, cols=pred)")
    model.eval()
    with torch.no_grad():
        preds = model(Xv).argmax(-1).cpu().numpy()
    yv_np = yv.cpu().numpy()
    K = len(DOMAINS)
    cm = np.zeros((K, K), dtype=np.int32)
    for t, p in zip(yv_np, preds):
        cm[t, p] += 1
    # Normalize by row
    cm_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    header = 'true/pred'
    print(f"  {header:<10}" + ''.join(f"{d:>10}" for d in DOMAINS))
    for i, d in enumerate(DOMAINS):
        print(f"  {d:<10}" + ''.join(f"{cm_pct[i,j]:>9.1f}%" for j in range(K)))

    # Save
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "domains": DOMAINS,
        "feat_mean": mu.astype(np.float32),
        "feat_std": sd.astype(np.float32),
        "window": WINDOW,
        "ema_alpha": EMA_ALPHA,
        "best_val_acc": best_va,
    }, out_path)
    print(f"\n[clf] saved {out_path} — best_va={best_va*100:.2f}%")


if __name__ == "__main__":
    main()
