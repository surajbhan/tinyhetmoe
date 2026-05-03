#!/usr/bin/env python3
"""dpq_compress.py — Differentiable Product Quantization for embedding tables.

Compresses a (V, D) float embedding table into:
  - M codebooks of shape (K, D/M)              — small (~33 KB each at M=4 K=256)
  - One index array of shape (V, M) uint8     — ~600 KB at V=151665 M=4

Reconstruction: emb_dpq[v] = concat(codebook[m][indices[v,m]] for m in 0..M)

Storage: 4 bytes per row instead of D*4 (132×4=528). Compression ~132x at M=4 K=256.

Run:
  python3 scripts/dpq_compress.py \\
      --ckpt /home/suraj/v7_runs/tiny_hetmoe_v7_general_distill_qat/checkpoints/best_distill_qat.pt \\
      --target intuition_embed \\
      --out data_v7/dpq_general_intuition.pt \\
      --M 4 --K 256

Or for the shared meaning_embed (only need once):
  python3 scripts/dpq_compress.py \\
      --ckpt data_v7/meaning_axes_full_132.npy \\
      --target raw_npy \\
      --out data_v7/dpq_meaning_shared.pt \\
      --M 4 --K 256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_target(ckpt_path: str, target: str) -> torch.Tensor:
    """Load the target embedding tensor.
    target: 'intuition_embed', 'meaning_embed', or 'raw_npy' for a numpy file."""
    p = Path(ckpt_path)
    if target == "raw_npy":
        arr = np.load(p)
        return torch.from_numpy(arr).float()
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    key_map = {
        "intuition_embed": "intuition_embed.weight",
        "meaning_embed": "meaning_embed.weight",
    }
    key = key_map.get(target, target)
    if key not in sd:
        raise KeyError(f"target {target} (key {key}) not in ckpt; "
                       f"available: {[k for k in sd.keys() if 'embed' in k]}")
    return sd[key].float()


class DPQ(nn.Module):
    """Differentiable Product Quantization layer.

    Each row of the (V, D) embedding is split into M chunks of D/M dims.
    For each chunk, we softly assign to one of K prototypes via attention.
    Training: soft assignment (gradient flows). Inference: hard argmax."""

    def __init__(self, V: int, D: int, M: int, K: int, init_emb: torch.Tensor):
        super().__init__()
        assert D % M == 0, f"D={D} not divisible by M={M}"
        self.V, self.D, self.M, self.K = V, D, M, K
        self.chunk_dim = D // M

        # Codebooks: M copies of (K, chunk_dim)
        self.codebooks = nn.Parameter(torch.randn(M, K, self.chunk_dim) * 0.1)

        # Per-row, per-chunk attention logits over K prototypes
        # Shape: (V, M, K). Initialized so each row picks K chunks at random.
        self.assign_logits = nn.Parameter(torch.randn(V, M, K) * 0.1)

        # Initialize codebooks via k-means-like seeding from init_emb
        with torch.no_grad():
            init_chunks = init_emb.view(V, M, self.chunk_dim)  # (V, M, cd)
            for m in range(M):
                # Sample K random rows for codebook[m]
                idx = torch.randperm(V)[:K]
                self.codebooks.data[m] = init_chunks[idx, m, :]
                # Initial assignment: nearest codebook entry per row
                d2 = ((init_chunks[:, m, :].unsqueeze(1) - self.codebooks.data[m].unsqueeze(0)) ** 2).sum(-1)  # (V, K)
                # Set logits to be peaked at nearest entry
                self.assign_logits.data[:, m, :] = -d2 * 10  # peak at min-distance

    def forward(self, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        """Reconstruct the full (V, D) embedding."""
        # Soft assignment: (V, M, K)
        logits = self.assign_logits / temperature
        if hard:
            idx = logits.argmax(dim=-1)  # (V, M)
            # One-hot for clean reconstruction
            attn = F.one_hot(idx, num_classes=self.K).float()  # (V, M, K)
        else:
            attn = F.softmax(logits, dim=-1)  # (V, M, K)
        # Reconstruct: einsum(attn, codebooks) -> (V, M, chunk_dim)
        # attn (V, M, K), codebooks (M, K, chunk_dim) -> (V, M, chunk_dim)
        chunks = torch.einsum("vmk,mkd->vmd", attn, self.codebooks)
        return chunks.reshape(self.V, self.D)

    def get_indices(self) -> torch.Tensor:
        """Hard indices for storage. Returns (V, M) uint8."""
        with torch.no_grad():
            return self.assign_logits.argmax(dim=-1).to(torch.uint8)

    def get_codebooks(self) -> torch.Tensor:
        """Returns codebooks (M, K, chunk_dim) fp16 for storage."""
        with torch.no_grad():
            return self.codebooks.detach().half()


def kmeans_chunk(data: torch.Tensor, K: int, n_iter: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """K-means on (N, d) data with K clusters. Returns (centroids (K,d), labels (N,))."""
    N, d = data.shape
    # Init: K-means++ style — random first, then pick farthest
    centroids = torch.empty(K, d, device=data.device, dtype=data.dtype)
    centroids[0] = data[torch.randint(N, (1,))].squeeze(0)
    for k in range(1, K):
        # Distance from each point to nearest existing centroid
        d2 = ((data.unsqueeze(1) - centroids[:k].unsqueeze(0)) ** 2).sum(-1)  # (N, k)
        nearest = d2.min(dim=-1).values  # (N,)
        # Sample next centroid weighted by squared distance
        if k < K:
            probs = nearest / nearest.sum().clamp(min=1e-8)
            idx = torch.multinomial(probs, 1).item()
            centroids[k] = data[idx]

    # Lloyd iterations
    for it in range(n_iter):
        # Assign: each point → nearest centroid
        d2 = ((data.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(-1)  # (N, K)
        labels = d2.argmin(dim=-1)  # (N,)
        # Update: each centroid = mean of its assigned points
        for k in range(K):
            mask = labels == k
            if mask.any():
                centroids[k] = data[mask].mean(0)
            # else: leave as-is (degenerate cluster — could re-seed but rare)

    # Final labels
    d2 = ((data.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(-1)
    labels = d2.argmin(dim=-1)
    return centroids, labels


def train_dpq(target_emb: torch.Tensor, M: int, K: int,
              epochs: int = 20, batch_size: int = 4096, lr: float = 1e-3,
              device: str = "cuda") -> DPQ:
    """Two-phase: (1) per-chunk k-means for codebook + initial assignment,
    (2) gradient descent fine-tune on assign_logits + codebooks jointly."""
    V, D = target_emb.shape
    M_K_d = (M, K, D // M)
    print(f"[dpq] training: V={V} D={D} M={M} K={K} chunk_dim={D//M}, "
          f"compression {(V*D*4)/(V*M + M*K*(D//M)*2):.1f}x")
    target = target_emb.to(device)

    # Phase 1: per-chunk k-means
    print(f"[dpq] phase 1: per-chunk k-means")
    target_chunks = target.view(V, M, D // M)
    codebooks = torch.zeros(M, K, D // M, device=device)
    indices = torch.zeros(V, M, dtype=torch.long, device=device)
    for m in range(M):
        chunk_data = target_chunks[:, m, :]  # (V, chunk_dim)
        centroids, labels = kmeans_chunk(chunk_data, K, n_iter=15)
        codebooks[m] = centroids
        indices[:, m] = labels
        # Diagnostic
        recon_chunk = centroids[labels]  # (V, chunk_dim)
        chunk_mse = ((recon_chunk - chunk_data) ** 2).mean().item()
        print(f"  chunk {m}: MSE {chunk_mse:.6f}")

    # Build DPQ module from k-means result
    dpq = DPQ(V, D, M, K, init_emb=target_emb)
    dpq = dpq.to(device)
    with torch.no_grad():
        dpq.codebooks.data = codebooks.clone()
        # Set assign_logits peaked at the k-means label
        dpq.assign_logits.data.zero_()
        dpq.assign_logits.data.scatter_(2, indices.unsqueeze(-1), 10.0)

    # Sanity check after phase 1
    with torch.no_grad():
        recon = dpq(temperature=1.0, hard=True)
        mse_p1 = (recon - target).pow(2).mean().item()
        cosine_p1 = F.cosine_similarity(recon, target, dim=-1).mean().item()
        print(f"[dpq] after phase 1: hard_mse {mse_p1:.6f} cosine {cosine_p1:.4f}")

    # Phase 2: gradient fine-tune
    print(f"[dpq] phase 2: gradient fine-tune ({epochs} epochs)")
    opt = torch.optim.Adam(dpq.parameters(), lr=lr)
    n_rows = V
    for ep in range(epochs):
        temp = max(0.05, 0.5 - 0.45 * ep / epochs)
        perm = torch.randperm(n_rows, device=device)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, n_rows, batch_size):
            ix = perm[i:i+batch_size]
            attn = F.softmax(dpq.assign_logits[ix] / temp, dim=-1)
            chunks = torch.einsum("bmk,mkd->bmd", attn, dpq.codebooks)
            recon = chunks.reshape(len(ix), D)
            loss = (recon - target[ix]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.detach())
            n_batches += 1

        with torch.no_grad():
            recon_hard = dpq(temperature=1.0, hard=True)
            mse_hard = (recon_hard - target).pow(2).mean().item()
            cosine = F.cosine_similarity(recon_hard, target, dim=-1).mean().item()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep {ep:>3} | T={temp:.3f} | tr_mse {ep_loss/n_batches:.6f} | "
                  f"hard_mse {mse_hard:.6f} | cosine {cosine:.4f}")

    return dpq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="path to best_distill_qat.pt OR raw .npy file")
    ap.add_argument("--target", default="intuition_embed",
                    choices=["intuition_embed", "meaning_embed", "raw_npy"])
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[dpq] loading target from {args.ckpt} ({args.target})")
    target = load_target(args.ckpt, args.target)
    print(f"[dpq] target shape: {target.shape}, dtype: {target.dtype}")

    dpq = train_dpq(target, M=args.M, K=args.K,
                     epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr, device=args.device)

    # Save: codebooks (fp16) + indices (uint8)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "codebooks": dpq.get_codebooks().cpu(),       # (M, K, chunk_dim) fp16
        "indices":   dpq.get_indices().cpu(),         # (V, M) uint8
        "V": dpq.V, "D": dpq.D, "M": args.M, "K": args.K,
        "source": str(args.ckpt),
        "target": args.target,
    }
    torch.save(payload, out_path)
    sz = out_path.stat().st_size / 1e6
    raw_sz = dpq.V * dpq.D * 4 / 1e6
    print(f"\n[dpq] saved {out_path} — {sz:.2f} MB (vs {raw_sz:.1f} MB raw, "
          f"{raw_sz/sz:.1f}x compression)")


if __name__ == "__main__":
    main()
