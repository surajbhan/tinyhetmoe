#!/usr/bin/env python3
"""export_model.py — pack TinyHetMoE checkpoint to the binary format
the Rust/WASM inference engine consumes.

Format: HTMOE002 — extends the production HTMOE001 format with:
  - M+I split embedding (meaning + intuition tables)
  - Highway expand + compress projections (ternary)
  - NoPE (no rope_cos / rope_sin)
  - QK-Norm (RMSNorm weights for q and k per layer)
  - Untied lm_head (separate ternary weight from embeddings)

Wire format (little-endian):
    magic            "HTMOE002" (8 bytes)
    vocab_size       u32
    meaning_dim      u32
    intuition_dim    u32
    input_dim        u32       # meaning + intuition (i.e. hidden out)
    internal_dim     u32       # highway internal (input goes through this)
    new_intuition    u32       # internal - meaning - intuition
    num_layers       u32
    num_heads        u32
    head_dim         u32       # internal_dim / num_heads
    max_seq_len      u32
    num_experts      u32
    top_k_experts    u32
    ffn_mult_x100    u32       # int(ffn_mult * 100); decoder divides by 100

    expert_arch_types [num_layers * num_experts] u8
        0=Standard, 1=SwiGLU, 2=DeepNarrow, 3=Bottleneck

    meaning_embed     fp32 tensor (vocab_size * meaning_dim)
    intuition_embed   fp32 tensor (vocab_size * intuition_dim)
    expand            ternary tensor (new_intuition * intuition_dim)
    for each layer:
        attn_norm     fp32 (internal_dim,)
        q_norm        fp32 (head_dim,)         # QK-Norm
        k_norm        fp32 (head_dim,)         # QK-Norm
        ffn_norm      fp32 (internal_dim,)
        q_proj        ternary (internal_dim * internal_dim)
        k_proj        ternary
        v_proj        ternary
        o_proj        ternary
        for each expert (according to expert_arch_types[layer][e]):
            arch=0 (Standard): up, down                       # 2 weights
            arch=1 (SwiGLU):   w1, w2, down                   # 3 weights
            arch=2 (DeepNarrow): l1, l2, l3, l4               # 4 weights
            arch=3 (Bottleneck): down_proj, up_proj, out_proj # 3 weights
        gate          fp32 (num_experts * internal_dim)
    compress          ternary ((meaning_dim + intuition_dim_internal) * intuition_dim)
                      where intuition_dim_internal = intuition_dim + new_intuition
    final_norm        fp32 (input_dim,)
    lm_head           ternary (vocab_size * input_dim)

Each ternary tensor is encoded as:
    scale            fp32 (alpha = mean(|w|) at training time)
    ndim             u32
    shape            ndim x u32
    data             rows * cols int8 values in {-1, 0, +1}

The Rust side packs ternary into 2-bits-per-weight at load time.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.tiny_hetmoe import TinyHetMoE, TinyHetMoEConfig  # noqa: E402


ARCH_MAP = {
    "StandardFFN":    0,
    "SwiGLUFFN":      1,
    "DeepNarrowFFN":  2,
    "BottleneckFFN":  3,
}
# How many ternary weights each expert architecture has, in fixed order
EXPERT_WEIGHT_NAMES = {
    0: ["up", "down"],
    1: ["w1", "w2", "down"],
    2: ["l1", "l2", "l3", "l4"],
    3: ["down_proj", "up_proj", "out_proj"],
}


def quantize_ternary(weight: torch.Tensor) -> tuple[float, np.ndarray]:
    """Apply training-time ternary recipe: alpha = mean(|w|),
    w_q = round(w/alpha).clamp(-1, 1). Returns (alpha, int8 array)."""
    w = weight.detach().float().cpu()
    alpha = float(w.abs().mean().clamp(min=1e-5).item())
    w_norm = w / alpha
    w_t = w_norm.round().clamp(-1, 1).to(torch.int8).contiguous().numpy()
    return alpha, w_t


def write_fp32(f, tensor: torch.Tensor) -> int:
    """Write tensor as: u32 count, count*f32. Returns bytes written."""
    data = tensor.detach().float().cpu().contiguous().numpy()
    f.write(struct.pack("<I", data.size))
    f.write(data.tobytes())
    return 4 + data.nbytes


def write_ternary(f, weight: torch.Tensor) -> tuple[int, int, int, int, int]:
    """Write tensor as: f32 scale, u32 ndim, ndim u32 shape, total int8 values.
    Returns (bytes_written, total, zeros, pos, neg).

    HTMOE002 (legacy): one int8 per weight.
    """
    alpha, w_t = quantize_ternary(weight)
    f.write(struct.pack("<f", alpha))
    f.write(struct.pack("<I", w_t.ndim))
    for dim in w_t.shape:
        f.write(struct.pack("<I", dim))
    f.write(w_t.tobytes())
    total = int(w_t.size)
    zeros = int((w_t == 0).sum())
    pos = int((w_t == 1).sum())
    neg = int((w_t == -1).sum())
    bytes_written = 4 + 4 + 4 * w_t.ndim + w_t.nbytes
    return bytes_written, total, zeros, pos, neg


def write_ternary_packed(f, weight: torch.Tensor) -> tuple[int, int, int, int, int]:
    """HTMOE003 packed-ternary writer: 4 weights per byte (2 bits each).

    Encoding (matches Rust tensor.rs):
      00 → 0
      01 → +1
      11 → -1
      10 → reserved (unused)

    Layout: f32 scale, u32 ndim, ndim u32 shape, ceil(total/4) packed bytes.
    """
    alpha, w_t = quantize_ternary(weight)
    flat = w_t.reshape(-1)
    n = flat.size
    n_bytes = (n + 3) // 4
    packed = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n):
        v = int(flat[i])
        bits = 0b00 if v == 0 else (0b01 if v == 1 else 0b11)
        packed[i // 4] |= bits << ((i % 4) * 2)

    f.write(struct.pack("<f", alpha))
    f.write(struct.pack("<I", w_t.ndim))
    for dim in w_t.shape:
        f.write(struct.pack("<I", dim))
    f.write(packed.tobytes())

    total = int(n)
    zeros = int((w_t == 0).sum())
    pos = int((w_t == 1).sum())
    neg = int((w_t == -1).sum())
    bytes_written = 4 + 4 + 4 * w_t.ndim + n_bytes
    return bytes_written, total, zeros, pos, neg


def export_model(ckpt_path: str, out_path: str, packed: bool = True):
    print(f"[export] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg_kwargs = {k: cfg_dict[k] for k in [
        "vocab_size", "meaning_dim", "intuition_dim", "input_dim",
        "internal_dim", "new_intuition", "num_layers", "num_heads",
        "num_experts", "top_k_experts", "ffn_mult", "max_seq_len",
        "load_balance_weight",
    ] if k in cfg_dict}
    cfg = TinyHetMoEConfig(**cfg_kwargs)
    head_dim = cfg.internal_dim // cfg.num_heads
    print(f"[export] cfg: vocab={cfg.vocab_size} hidden={cfg.input_dim} "
          f"internal={cfg.internal_dim} L={cfg.num_layers} H={cfg.num_heads} "
          f"head_dim={head_dim} E={cfg.num_experts}/{cfg.top_k_experts}")

    model = TinyHetMoE(cfg)
    sd = ckpt["model"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[export] WARN missing: {missing[:5]}")
    if unexpected:
        print(f"[export] WARN unexpected: {unexpected[:5]}")
    model.eval()

    expert_archs: list[list[int]] = []
    for layer in model.layers:
        archs_for_layer = []
        for expert in layer.moe.experts:
            archs_for_layer.append(ARCH_MAP[type(expert).__name__])
        expert_archs.append(archs_for_layer)

    total_params = 0
    total_zeros = 0
    bytes_packed = 0
    bytes_fp = 0

    magic = b"HTMOE003" if packed else b"HTMOE002"
    write_ternary_fn = write_ternary_packed if packed else write_ternary
    print(f"[export] writing {out_path} ({'packed 2-bit' if packed else 'int8'} ternary)")
    with open(out_path, "wb") as f:
        f.write(magic)

        # Config header
        f.write(struct.pack("<I", cfg.vocab_size))
        f.write(struct.pack("<I", cfg.meaning_dim))
        f.write(struct.pack("<I", cfg.intuition_dim))
        f.write(struct.pack("<I", cfg.input_dim))
        f.write(struct.pack("<I", cfg.internal_dim))
        f.write(struct.pack("<I", cfg.new_intuition))
        f.write(struct.pack("<I", cfg.num_layers))
        f.write(struct.pack("<I", cfg.num_heads))
        f.write(struct.pack("<I", head_dim))
        f.write(struct.pack("<I", cfg.max_seq_len))
        f.write(struct.pack("<I", cfg.num_experts))
        f.write(struct.pack("<I", cfg.top_k_experts))
        f.write(struct.pack("<I", int(cfg.ffn_mult * 100)))

        # Expert architecture types per layer
        for layer_archs in expert_archs:
            for arch in layer_archs:
                f.write(struct.pack("<B", arch))

        # M+I embeddings (FP32, not ternary — small + needs precision)
        print("[export] meaning_embed", model.meaning_embed.weight.shape)
        bytes_fp += write_fp32(f, model.meaning_embed.weight)
        print("[export] intuition_embed", model.intuition_embed.weight.shape)
        bytes_fp += write_fp32(f, model.intuition_embed.weight)

        # Expand projection (ternary)
        print("[export] expand", model.expand.weight.shape)
        bw, t, z, p, n = write_ternary_fn(f, model.expand.weight)
        bytes_packed += bw
        total_params += t
        total_zeros += z

        # Layers
        for l_idx, layer in enumerate(model.layers):
            print(f"[export] layer {l_idx}")
            # Norms (FP32)
            bytes_fp += write_fp32(f, layer.n1.weight)             # attn_norm
            bytes_fp += write_fp32(f, layer.attn.q_norm.weight)    # QK-Norm q
            bytes_fp += write_fp32(f, layer.attn.k_norm.weight)    # QK-Norm k
            bytes_fp += write_fp32(f, layer.n2.weight)             # ffn_norm

            # Attention projections (ternary)
            for name in ["w_q", "w_k", "w_v", "w_o"]:
                proj = getattr(layer.attn, name)
                bw, t, z, p, n = write_ternary_fn(f, proj.weight)
                bytes_packed += bw
                total_params += t
                total_zeros += z

            # Experts (ternary, by arch type)
            for e_idx, expert in enumerate(layer.moe.experts):
                arch = expert_archs[l_idx][e_idx]
                weight_names = EXPERT_WEIGHT_NAMES[arch]
                for wn in weight_names:
                    proj = getattr(expert, wn)
                    bw, t, z, p, n = write_ternary_fn(f, proj.weight)
                    bytes_packed += bw
                    total_params += t
                    total_zeros += z

            # Router gate — Python's gate is QuantizedLinear, so during QAT
            # forward it becomes ternary. Pre-ternarize the FP weight at
            # export time so Rust's FP dot product gives the same answer.
            with torch.no_grad():
                gw = layer.moe.gate.weight.detach().float()
                alpha = float(gw.abs().mean().clamp(min=1e-5).item())
                gw_q = (gw / alpha).round().clamp(-1, 1) * alpha
                bytes_fp += write_fp32(f, gw_q)

        # Compress (ternary)
        print("[export] compress", model.compress.weight.shape)
        bw, t, z, p, n = write_ternary_fn(f, model.compress.weight)
        bytes_packed += bw
        total_params += t
        total_zeros += z

        # Final norm (FP32)
        bytes_fp += write_fp32(f, model.norm.weight)

        # lm_head (ternary — biggest single tensor, big win)
        print("[export] lm_head", model.lm_head.weight.shape)
        bw, t, z, p, n = write_ternary_fn(f, model.lm_head.weight)
        bytes_packed += bw
        total_params += t
        total_zeros += z

    file_size = Path(out_path).stat().st_size
    print()
    print(f"=== {out_path} ===")
    print(f"  file size:        {file_size/1e6:.2f} MB")
    print(f"  fp32 segment:     {bytes_fp/1e6:.2f} MB")
    print(f"  ternary segment:  {bytes_packed/1e6:.2f} MB (int8, "
          f"would be {bytes_packed/4/1e6:.2f} MB if 2-bit-packed)")
    print(f"  ternary params:   {total_params:,}")
    print(f"  sparsity (zeros): {total_zeros/total_params*100:.1f}%")

    # Sidecar metadata for the UI
    meta_path = Path(out_path).with_suffix(".meta.json")
    meta = {
        "format": "HTMOE003" if packed else "HTMOE002",
        "training_step": int(ckpt.get("step", 0)),
        "best_val": float(ckpt.get("best_val", float("nan"))),
        "config": {
            "vocab_size": cfg.vocab_size,
            "meaning_dim": cfg.meaning_dim,
            "intuition_dim": cfg.intuition_dim,
            "input_dim": cfg.input_dim,
            "internal_dim": cfg.internal_dim,
            "new_intuition": cfg.new_intuition,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "head_dim": head_dim,
            "max_seq_len": cfg.max_seq_len,
            "num_experts": cfg.num_experts,
            "top_k_experts": cfg.top_k_experts,
            "ffn_mult": cfg.ffn_mult,
        },
        "expert_archs": expert_archs,
        "expert_arch_names": ["Standard", "SwiGLU", "DeepNarrow", "Bottleneck"],
        "ternary_params": total_params,
        "sparsity_zeros_frac": total_zeros / max(1, total_params),
    }
    json.dump(meta, meta_path.open("w"), indent=2)
    print(f"  metadata:         {meta_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="model.bin")
    p.add_argument("--legacy", action="store_true",
                   help="Write HTMOE002 (1 byte/weight) instead of HTMOE003 packed")
    args = p.parse_args()
    export_model(args.ckpt, args.out, packed=not args.legacy)
