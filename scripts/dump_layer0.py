#!/usr/bin/env python3
"""Dump Python's layer-0 intermediates at position 10 for the canonical prompt.
Numbers can be cross-checked against the Rust debug binary.
"""
import torch, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.tiny_hetmoe import TinyHetMoE, TinyHetMoEConfig, set_quantize_mode

ckpt = torch.load("/tmp/best_snap.pt", map_location="cpu", weights_only=False)
cfg_dict = ckpt["config"]
cfg = TinyHetMoEConfig(**{k: cfg_dict[k] for k in [
    "vocab_size", "meaning_dim", "intuition_dim", "input_dim", "internal_dim",
    "new_intuition", "num_layers", "num_heads", "num_experts", "top_k_experts",
    "ffn_mult", "max_seq_len", "load_balance_weight",
]})
model = TinyHetMoE(cfg)
sd = ckpt["model"]
sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.eval()
set_quantize_mode(model, on=True)

prompt = [1, 59, 63, 10, 40, 7, 42, 11, 10, 43, 41]
ids = torch.tensor([prompt], dtype=torch.long)

# Hook layer 0 to capture intermediates
captured = {}

# Manually walk through the forward pass at the same level the Rust engine does
with torch.no_grad():
    # Embedding + Highway expand
    meaning = model.meaning_embed(ids)              # (1, 11, 132)
    intuition = model.intuition_embed(ids)          # (1, 11, 132)
    new_int = model.expand(intuition)                # (1, 11, 264)
    x = torch.cat([meaning, intuition, new_int], dim=-1)  # (1, 11, 528)

    print("=" * 60)
    print("AFTER M+I + EXPAND (input to layer 0)")
    print("=" * 60)
    print(f"x[0, 10, 0..5]:           {x[0, 10, :5].tolist()}")
    print(f"x[0, 10, 132..137]:       {x[0, 10, 132:137].tolist()}")
    print(f"x[0, 10, 264..269]:       {x[0, 10, 264:269].tolist()}")
    print(f"x[0, 10] norm:            {x[0, 10].norm().item():.4f}")

    # Layer 0 attention
    layer0 = model.layers[0]

    # n1
    n1_out = layer0.n1(x)
    print()
    print("=" * 60)
    print("AFTER LAYER 0 n1 (attn_norm RMSNorm)")
    print("=" * 60)
    print(f"n1[0, 10, 0..5]:          {n1_out[0, 10, :5].tolist()}")
    print(f"n1[0, 10, 132..137]:      {n1_out[0, 10, 132:137].tolist()}")
    print(f"n1[0, 10] norm:           {n1_out[0, 10].norm().item():.4f}")

    # Attention forward — replicate manually
    # q_proj is QuantizedLinear with set_quantize_mode(on=True) so it ternarizes weights
    B, T, C = n1_out.shape
    q = layer0.attn.w_q(n1_out).view(B, T, layer0.attn.num_heads, layer0.attn.head_dim).transpose(1, 2)
    k = layer0.attn.w_k(n1_out).view(B, T, layer0.attn.num_heads, layer0.attn.head_dim).transpose(1, 2)
    v = layer0.attn.w_v(n1_out).view(B, T, layer0.attn.num_heads, layer0.attn.head_dim).transpose(1, 2)
    print()
    print("=" * 60)
    print("AFTER Q/K/V PROJECTIONS (pre-QK-Norm)")
    print("=" * 60)
    print(f"q[0, head=0, t=10, 0..5]: {q[0, 0, 10, :5].tolist()}")
    print(f"k[0, head=0, t=10, 0..5]: {k[0, 0, 10, :5].tolist()}")
    print(f"v[0, head=0, t=10, 0..5]: {v[0, 0, 10, :5].tolist()}")
    print(f"q[0, head=0, t=10] norm:  {q[0, 0, 10].norm().item():.4f}")

    # QK-Norm
    q = layer0.attn.q_norm(q)
    k = layer0.attn.k_norm(k)
    print()
    print("=" * 60)
    print("AFTER QK-NORM")
    print("=" * 60)
    print(f"q[0, head=0, t=10, 0..5]: {q[0, 0, 10, :5].tolist()}")
    print(f"k[0, head=0, t=10, 0..5]: {k[0, 0, 10, :5].tolist()}")
    print(f"q[0, head=0, t=10] norm:  {q[0, 0, 10].norm().item():.4f}")

    # SDPA
    import torch.nn.functional as F
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    out_concat = out.transpose(1, 2).reshape(B, T, C)
    print()
    print("=" * 60)
    print("AFTER ATTENTION (pre-o_proj)")
    print("=" * 60)
    print(f"out[0, 10, 0..5]:         {out_concat[0, 10, :5].tolist()}")
    print(f"out[0, 10] norm:          {out_concat[0, 10].norm().item():.4f}")

    # o_proj
    o_out = layer0.attn.w_o(out_concat)
    print()
    print("=" * 60)
    print("AFTER o_proj")
    print("=" * 60)
    print(f"o_out[0, 10, 0..5]:       {o_out[0, 10, :5].tolist()}")
    print(f"o_out[0, 10, 132..137]:   {o_out[0, 10, 132:137].tolist()}")

    # Meaning protection
    o_out_protected = torch.cat([torch.zeros_like(o_out[..., :cfg.meaning_dim]), o_out[..., cfg.meaning_dim:]], dim=-1)

    # Residual
    x_after_attn = x + o_out_protected
    print()
    print("=" * 60)
    print("AFTER RESIDUAL (x + protected o_out)")
    print("=" * 60)
    print(f"x_after[0, 10, 0..5]:     {x_after_attn[0, 10, :5].tolist()}")
    print(f"x_after[0, 10, 132..137]: {x_after_attn[0, 10, 132:137].tolist()}")

    # n2 (ffn_norm)
    n2_out = layer0.n2(x_after_attn)

    # Router
    moe = layer0.moe
    logits_router = F.linear(n2_out.view(-1, C), moe.gate.weight)
    probs = F.softmax(logits_router, dim=-1)
    print()
    print("=" * 60)
    print("ROUTING at position 10")
    print("=" * 60)
    print(f"router probs (last pos): {probs[10].tolist()}")

    # ── Run a full layer 0 forward and dump output ─────────────────
    # Use the actual layer's forward, since that's what Rust mirrors
    x_l0_out = layer0(x)
    print()
    print("=" * 60)
    print("AFTER FULL LAYER 0 (output to layer 1)")
    print("=" * 60)
    print(f"x_l0[0, 10, 0..5]:      {x_l0_out[0, 10, :5].tolist()}")
    print(f"x_l0[0, 10, 130..137]:  {x_l0_out[0, 10, 130:137].tolist()}")
    print(f"x_l0[0, 10, 264..269]:  {x_l0_out[0, 10, 264:269].tolist()}")
    print(f"x_l0[0, 10] norm:       {x_l0_out[0, 10].norm().item():.4f}")

    # Run layer 1 attention norm + projections, see the routing at L1
    layer1 = model.layers[1]
    n1_l1 = layer1.n1(x_l0_out)
    print()
    print(f"L1 n1[0, 10, 0..5]:     {n1_l1[0, 10, :5].tolist()}")
    print(f"L1 n1[0, 10] norm:      {n1_l1[0, 10].norm().item():.4f}")
