#!/usr/bin/env python3
"""Dump expert 2 (DeepNarrow) and expert 3 (Bottleneck) Python outputs
on layer 0's n2(x_after_attn) input at position 10. So we can compare
to Rust expert outputs.
"""
import torch, sys
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

with torch.no_grad():
    # Replicate the forward up to layer-0 MoE input
    meaning = model.meaning_embed(ids)
    intuition = model.intuition_embed(ids)
    new_int = model.expand(intuition)
    x = torch.cat([meaning, intuition, new_int], dim=-1)

    layer0 = model.layers[0]
    n1_out = layer0.n1(x)
    B, T, C = n1_out.shape
    q = layer0.attn.w_q(n1_out).view(B, T, layer0.attn.num_heads, layer0.attn.head_dim).transpose(1, 2)
    k = layer0.attn.w_k(n1_out).view(B, T, layer0.attn.num_heads, layer0.attn.head_dim).transpose(1, 2)
    v = layer0.attn.w_v(n1_out).view(B, T, layer0.attn.num_heads, layer0.attn.head_dim).transpose(1, 2)
    q = layer0.attn.q_norm(q)
    k = layer0.attn.k_norm(k)
    import torch.nn.functional as F
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    out_concat = out.transpose(1, 2).reshape(B, T, C)
    o_out = layer0.attn.w_o(out_concat)
    o_out = torch.cat([torch.zeros_like(o_out[..., :cfg.meaning_dim]), o_out[..., cfg.meaning_dim:]], dim=-1)
    x_after_attn = x + o_out
    n2_out = layer0.n2(x_after_attn)

    # Get position-10 row
    moe_in = n2_out[0, 10]
    print(f"moe_input[0..10]:    {moe_in[:10].tolist()}")
    print(f"moe_input norm:      {moe_in.norm().item():.4f}")
    print()

    # Run each expert individually on this single input
    for i, expert in enumerate(layer0.moe.experts):
        out = expert(moe_in.unsqueeze(0)).squeeze(0)
        print(f"Expert {i} ({type(expert).__name__}):")
        print(f"  output[0..5]:      {out[:5].tolist()}")
        print(f"  output[132..137]:  {out[132:137].tolist()}")
        print(f"  norm:              {out.norm().item():.4f}")
