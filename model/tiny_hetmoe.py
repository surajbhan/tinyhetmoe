"""tiny_hetmoe.py — the model.

The recipe is identical to the production 418M Highway-B run:
  - Meaning + Intuition split embedding (132 + 132 here)
  - Highway expansion (264 → 528 → 264)
  - 4 HetMoE expert types (Standard, SwiGLU, DeepNarrow, Bottleneck), top-2 routing
  - NoPE — no positional encoding, just a causal mask
  - QAT-from-step-0 — every Linear is a QuantizedLinear, ternary forward
    with vote-backward gradient
  - QK-Norm — RMSNorm per-head on q and k for FP↔ternary stability
  - B recipe — meaning embedding is trainable from a contextual init
    (no .detach in forward)

What's different from production:
  - Smaller (vocab 5967, hidden 264, 4 layers, head_dim 66)
  - Trains on uint16 token ids (vocab fits)
  - No DDP — single GPU is plenty for this size
  - No grad checkpointing — model is small enough that activations fit
  - Single-corpus (TinyStories), no variable-length needed (most stories
    fit in 256 tokens, we'll train at 512)

The educational story panels expose internal state, so the forward pass
has a `return_trace=True` mode that returns per-layer attention,
expert routing weights, and hidden states alongside the logits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Config
# ============================================================================

@dataclass
class TinyHetMoEConfig:
    vocab_size: int = 5967
    meaning_dim: int = 132
    intuition_dim: int = 132
    input_dim: int = 264          # meaning + intuition
    internal_dim: int = 528       # highway 2× expansion
    new_intuition: int = 264      # internal - meaning - intuition = 528-132-132
    num_layers: int = 4
    num_heads: int = 4            # head_dim = 528/4 = 132
    num_experts: int = 4
    top_k_experts: int = 2
    ffn_mult: float = 2.0
    max_seq_len: int = 512
    load_balance_weight: float = 0.01


# ============================================================================
# Ternary QAT
# ============================================================================

class TernaryQuantizeFn(torch.autograd.Function):
    """Forward: w_q = round(w/alpha).clamp(-1,1) * alpha, alpha=mean(|w|).
    Backward: vote-backward — grad = grad_out * sign(w), with 0.1× rescue
    at w=0. Learned to converge in production at 130M-1.5B; we apply the
    same recipe at 26M."""

    @staticmethod
    def forward(ctx, w):
        alpha = w.detach().abs().mean()
        if alpha == 0:
            ctx.save_for_backward(w)
            return torch.zeros_like(w)
        w_q = torch.round(w / alpha).clamp(-1.0, 1.0) * alpha
        ctx.save_for_backward(w)
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        (w,) = ctx.saved_tensors
        w_sign = torch.sign(w)
        zero_mask = (w_sign == 0)
        grad_w = grad_output * w_sign
        grad_w = torch.where(zero_mask, grad_output * 0.1, grad_w)
        return grad_w


def ternary_quantize(w):
    return TernaryQuantizeFn.apply(w)


class QuantizedLinear(nn.Module):
    """Drop-in nn.Linear that supports FP↔ternary mode switch via
    `quantize` flag. The FP weight is preserved across the transition
    so the optimizer momentum remains meaningful."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.quantize = False
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w = ternary_quantize(self.weight) if self.quantize else self.weight
        return F.linear(x, w, self.bias)


def set_quantize_mode(model: nn.Module, on: bool) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            m.quantize = on
            n += 1
    return n


# ============================================================================
# Attention with QK-Norm + trace export
# ============================================================================

class Attention(nn.Module):
    """Multi-head attention with QK-Norm. Causal mask only (NoPE)."""

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.w_q = QuantizedLinear(dim, dim, bias=False)
        self.w_k = QuantizedLinear(dim, dim, bias=False)
        self.w_v = QuantizedLinear(dim, dim, bias=False)
        self.w_o = QuantizedLinear(dim, dim, bias=False)
        # QK-Norm — RMSNorm per-head, applied after projection
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        q = self.w_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if return_attn:
            # explicit attention for trace export (not for training)
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1,
            )
            attn = attn + mask[None, None, :, :]
            attn = attn.softmax(dim=-1)
            out = attn @ v
            return self.w_o(out.transpose(1, 2).reshape(B, T, C)), attn
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.0,
        )
        return self.w_o(out.transpose(1, 2).reshape(B, T, C))


# ============================================================================
# HetMoE — 4 heterogeneous expert FFNs with top-2 routing
# ============================================================================

class StandardFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.up = QuantizedLinear(dim, hidden, bias=False)
        self.down = QuantizedLinear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = QuantizedLinear(dim, hidden, bias=False)
        self.w2 = QuantizedLinear(dim, hidden, bias=False)
        self.down = QuantizedLinear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.w1(x)) * self.w2(x))


class DeepNarrowFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        mid = dim * 2
        self.l1 = QuantizedLinear(dim, mid, bias=False)
        self.l2 = QuantizedLinear(mid, hidden, bias=False)
        self.l3 = QuantizedLinear(hidden, mid, bias=False)
        self.l4 = QuantizedLinear(mid, dim, bias=False)

    def forward(self, x):
        return self.l4(F.gelu(self.l3(F.gelu(self.l2(F.gelu(self.l1(x)))))))


class BottleneckFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        neck = dim
        self.down_proj = QuantizedLinear(dim, neck, bias=False)
        self.up_proj = QuantizedLinear(neck, hidden, bias=False)
        self.out_proj = QuantizedLinear(hidden, dim, bias=False)

    def forward(self, x):
        return self.out_proj(F.gelu(self.up_proj(F.gelu(self.down_proj(x)))))


EXPERT_TYPES = [StandardFFN, SwiGLUFFN, DeepNarrowFFN, BottleneckFFN]
EXPERT_NAMES = ["Standard", "SwiGLU", "DeepNarrow", "Bottleneck"]


class HetMoE(nn.Module):
    """4 heterogeneous expert FFNs, top-K routing. Returns output and
    aux loss. With `return_trace=True`, also returns per-token routing
    weights for the UI."""

    def __init__(self, dim, num_experts, top_k, ffn_mult, lb_weight):
        super().__init__()
        self.hidden = int(dim * ffn_mult)
        self.dim = dim
        self.experts = nn.ModuleList([
            EXPERT_TYPES[i % len(EXPERT_TYPES)](dim, self.hidden)
            for i in range(num_experts)
        ])
        self.gate = QuantizedLinear(dim, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts
        self.lb_weight = lb_weight

    def forward(self, x, return_trace=False):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        N = x_flat.shape[0]
        logits = torch.nan_to_num(self.gate(x_flat), 0.0)
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_ids = torch.topk(probs, self.top_k, dim=-1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # Load-balance aux loss
        avg_probs = probs.mean(0)
        freq = torch.zeros(self.num_experts, device=x.device)
        for k in range(self.top_k):
            freq.scatter_add_(
                0,
                topk_ids[:, k].clamp(0, self.num_experts - 1),
                torch.ones(N, device=x.device),
            )
        freq = freq / (N * self.top_k)
        aux = self.lb_weight * self.num_experts * (avg_probs * freq).sum()

        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (topk_ids[:, k] == e)
                if mask.any():
                    out = self.experts[e](x_flat[mask])
                    output[mask] += topk_vals[mask, k:k + 1] * out
        out = output.view(B, T, C)
        if return_trace:
            return out, aux, probs.view(B, T, self.num_experts)
        return out, aux


# ============================================================================
# Highway block
# ============================================================================

class HighwayBlock(nn.Module):
    """Transformer block at internal dim. Meaning dims (0:meaning_dim) are
    protected: attention output and FFN output are zeroed on those dims
    so they pass through unchanged. The compress projection at the end
    of the model brings them back."""

    def __init__(self, cfg: TinyHetMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.n1 = nn.RMSNorm(cfg.internal_dim)
        self.attn = Attention(cfg.internal_dim, cfg.num_heads)
        self.n2 = nn.RMSNorm(cfg.internal_dim)
        self.moe = HetMoE(
            cfg.internal_dim, cfg.num_experts, cfg.top_k_experts,
            cfg.ffn_mult, cfg.load_balance_weight,
        )
        self._aux = 0.0
        # Trace caches (set during forward when return_trace=True)
        self._attn_cache = None
        self._route_cache = None

    def forward(self, x, return_trace=False):
        # Attention with meaning protection
        if return_trace:
            attn_out, attn_weights = self.attn(self.n1(x), return_attn=True)
            self._attn_cache = attn_weights
        else:
            attn_out = self.attn(self.n1(x))
        m = self.cfg.meaning_dim
        attn_out = torch.cat(
            [torch.zeros_like(attn_out[:, :, :m]), attn_out[:, :, m:]],
            dim=-1,
        )
        x = x + attn_out

        # MoE with meaning protection
        if return_trace:
            moe_out, self._aux, routing = self.moe(self.n2(x), return_trace=True)
            self._route_cache = routing
        else:
            moe_out, self._aux = self.moe(self.n2(x))
        moe_out = torch.cat(
            [torch.zeros_like(moe_out[:, :, :m]), moe_out[:, :, m:]],
            dim=-1,
        )
        x = x + moe_out
        return x


# ============================================================================
# Full model
# ============================================================================

class TinyHetMoE(nn.Module):
    """13-26M parameter Highway-B + HetMoE + QAT model for TinyStories.

    Forward modes:
      - `(ids, targets=None)`: standard forward. Returns (logits, loss).
      - `(ids, return_trace=True)`: trace mode. Returns
        (logits, trace_dict) where trace_dict has per-layer attention,
        expert routing, and hidden states for the UI to consume.
    """

    def __init__(self, cfg: TinyHetMoEConfig):
        super().__init__()
        self.cfg = cfg

        # Embeddings — both trainable (B recipe). meaning is initialized
        # from the 132-axis contextual file via `load_meaning_embeddings`.
        self.meaning_embed = nn.Embedding(cfg.vocab_size, cfg.meaning_dim)
        self.intuition_embed = nn.Embedding(cfg.vocab_size, cfg.intuition_dim)

        # Highway expansion (intuition → new_intuition); meaning copies through
        self.expand = QuantizedLinear(cfg.intuition_dim, cfg.new_intuition, bias=False)

        self.layers = nn.ModuleList([HighwayBlock(cfg) for _ in range(cfg.num_layers)])

        # Compress (intuition halves merge back to intuition_dim)
        intuition_internal = cfg.intuition_dim + cfg.new_intuition
        self.compress = QuantizedLinear(intuition_internal, cfg.intuition_dim, bias=False)

        self.norm = nn.RMSNorm(cfg.input_dim)
        self.lm_head = QuantizedLinear(cfg.input_dim, cfg.vocab_size, bias=False)

    def load_meaning_embeddings(self, path: str, freeze: bool = False):
        """Load (vocab_size, meaning_dim) contextual axis embeddings from
        a .npy file. If `freeze=True`, sets requires_grad=False (A recipe).
        For B recipe (default), keeps trainable."""
        emb = np.load(path)
        assert emb.shape == (self.cfg.vocab_size, self.cfg.meaning_dim), (
            f"meaning embedding shape {emb.shape} does not match config "
            f"({self.cfg.vocab_size}, {self.cfg.meaning_dim})"
        )
        with torch.no_grad():
            self.meaning_embed.weight.copy_(torch.from_numpy(emb).float())
        self.meaning_embed.weight.requires_grad = not freeze

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids, targets=None, return_trace=False):
        cfg = self.cfg

        meaning = self.meaning_embed(ids)            # (B, T, meaning_dim) — B: no detach
        intuition = self.intuition_embed(ids)         # (B, T, intuition_dim)
        new_int = self.expand(intuition)              # (B, T, new_intuition)
        x = torch.cat([meaning, intuition, new_int], dim=-1)  # (B, T, internal_dim)

        aux_total = 0.0
        for layer in self.layers:
            x = layer(x, return_trace=return_trace)
            aux_total = aux_total + layer._aux

        # Compress back to input_dim. Meaning halves preserved.
        meaning_out = x[:, :, :cfg.meaning_dim]
        intuition_all = x[:, :, cfg.meaning_dim:]
        intuition_out = self.compress(intuition_all)
        x_out = torch.cat([meaning_out, intuition_out], dim=-1)

        hidden = self.norm(x_out)

        if return_trace:
            logits = self.lm_head(hidden)
            trace = {
                "meaning": meaning,                                  # (B, T, M)
                "intuition_input": intuition,                         # (B, T, I)
                "attn_per_layer": [l._attn_cache for l in self.layers],   # list of (B, H, T, T)
                "route_per_layer": [l._route_cache for l in self.layers], # list of (B, T, E)
                "hidden_out": hidden,                                # (B, T, input_dim)
                "logits": logits,
            }
            return logits, trace

        if targets is None:
            return self.lm_head(hidden), None

        # Chunked CE for memory efficiency
        h_flat = hidden.view(-1, hidden.size(-1))
        t_flat = targets.view(-1)
        CHUNK = 1024
        total = 0.0
        count = 0
        for i in range(0, h_flat.shape[0], CHUNK):
            h = h_flat[i:i + CHUNK]
            t = t_flat[i:i + CHUNK]
            logits = h @ self.lm_head.weight.t()  # FP — no QAT on this matmul
            total = total + F.cross_entropy(logits, t, reduction="sum")
            count += t.shape[0]
        return None, total / count + aux_total


def count_params(model: TinyHetMoE) -> dict:
    """Return a breakdown of parameters by named submodule."""
    breakdown = {}
    for name, p in model.named_parameters():
        head = name.split(".")[0]
        breakdown[head] = breakdown.get(head, 0) + p.numel()
    breakdown["TOTAL"] = sum(breakdown.values())
    return breakdown
