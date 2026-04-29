#!/usr/bin/env python3
"""export_trace.py — load a TinyHetMoE checkpoint, run a prompt, and dump
the full per-token internal state as a JSON file the UI can consume.

Output shape (one JSON file per export):
  {
    "model_config": {hidden, layers, heads, experts, ...},
    "tokens": [
      {"id": int, "tiny_id": int, "str": "Once",
       "is_special": false, "is_prompt": true},
      ...
    ],
    "meaning_axis_names": ["I", "YOU", ...],          # length 132
    "expert_names": ["Standard", "SwiGLU", ...],       # length 4
    "per_token": [
      {
        "meaning": [132 floats],                       # post-embedding
        "intuition_input": [132 floats],
        "attn_per_layer": [                            # length=num_layers
          {                                            # one per layer
            "by_head": [                               # length=num_heads
              [T floats],                              # row T-1 of attn (last token's attention to all positions)
              ...
            ]
          },
          ...
        ],
        "route_per_layer": [                           # length=num_layers
          [4 floats]                                   # softmax probs over experts
        ],
        "hidden_out": [264 floats],
        "next_token_topk": [
          {"tiny_id": int, "str": "happy", "prob": 0.45},
          ...
        ]
      },
      ...
    ]
  }

Notes:
  - For the attention panel we emit the FULL attention matrix per layer
    per head (T×T). This is `attn_full_per_layer` if --full-attn is set.
    Default is "last token's attention row" only (lighter).
  - All numbers are float32 → JSON. File can be ~MB for short prompts.
    Use compact format (no indentation) to keep size sane.

Usage:
    python scripts/export_trace.py \\
        --ckpt runs/tiny_hetmoe/checkpoints/best.pt \\
        --prompt "Once upon a time" \\
        --gen 20 \\
        --out trace.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.tiny_hetmoe import (  # noqa: E402
    TinyHetMoE, TinyHetMoEConfig, set_quantize_mode, EXPERT_NAMES,
)


def load_vocab():
    vocab = json.load((REPO / "tokenizer" / "vocab.json").open())
    tiny_to_gpt2 = vocab["tiny_to_gpt2"]
    gpt2_to_tiny = {int(k): v for k, v in vocab["gpt2_to_tiny"].items()}
    specials = vocab["specials"]
    return vocab, tiny_to_gpt2, gpt2_to_tiny, specials


def encode_prompt(prompt: str, gpt2_to_tiny: dict, specials: dict,
                   gpt2_tok) -> list[int]:
    """Encode a UTF-8 prompt into tiny token ids. Unmappable GPT-2 ids
    become <unk>. Always prepends <bos>."""
    bos = specials["<bos>"]
    unk = specials["<unk>"]
    gpt2_ids = gpt2_tok(prompt, add_special_tokens=False)["input_ids"]
    tiny_ids = [bos]
    for gid in gpt2_ids:
        tiny_ids.append(gpt2_to_tiny.get(int(gid), unk))
    return tiny_ids


def decode_tiny_id(tiny_id: int, tiny_to_gpt2: list, specials: dict,
                    gpt2_tok) -> tuple[str, bool]:
    """Return (display_string, is_special)."""
    if tiny_id < len(specials):
        rev = {v: k for k, v in specials.items()}
        return rev.get(tiny_id, f"<{tiny_id}>"), True
    gpt2_id = tiny_to_gpt2[tiny_id]
    if gpt2_id is None:
        return f"<unk-{tiny_id}>", True
    return gpt2_tok.decode([gpt2_id]), False


def to_list(t: torch.Tensor) -> list:
    """Convert tensor to nested python lists (for JSON)."""
    return t.detach().float().cpu().tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    p.add_argument("--prompt", default="Once upon a time", help="text prompt")
    p.add_argument("--gen", type=int, default=20,
                   help="number of tokens to generate after the prompt")
    p.add_argument("--top_k", type=int, default=8,
                   help="top-K next tokens to record at each position")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--out", default="trace.json")
    p.add_argument("--full-attn", action="store_true",
                   help="export full T×T attention per layer per head "
                        "(default: only last-row T floats)")
    p.add_argument("--no-qat", action="store_true",
                   help="run in FP mode (turn QAT off). Default is QAT on, "
                        "matching how the model was trained.")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = args.device
    print(f"[trace] loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", {})
    cfg_kwargs = {k: model_cfg[k] for k in [
        "vocab_size", "meaning_dim", "intuition_dim", "input_dim",
        "internal_dim", "new_intuition", "num_layers", "num_heads",
        "num_experts", "top_k_experts", "ffn_mult", "max_seq_len",
        "load_balance_weight",
    ] if k in model_cfg}
    cfg = TinyHetMoEConfig(**cfg_kwargs)
    print(f"[trace] cfg: vocab={cfg.vocab_size} hidden={cfg.input_dim} "
          f"L={cfg.num_layers} H={cfg.num_heads}")

    model = TinyHetMoE(cfg)
    sd = ckpt["model"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[trace] missing keys: {missing[:3]}")
        print(f"[trace] unexpected keys: {unexpected[:3]}")
    model.eval().to(device)
    if not args.no_qat:
        n_qat = set_quantize_mode(model, on=True)
        print(f"[trace] QAT on for {n_qat} modules")

    print(f"[trace] loading vocab + tokenizer")
    vocab_dict, tiny_to_gpt2, gpt2_to_tiny, specials = load_vocab()
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")

    axis_names = json.load((REPO / "data" / "meaning_axis_names.json").open())

    # Encode prompt
    tiny_ids = encode_prompt(args.prompt, gpt2_to_tiny, specials, gpt2_tok)
    print(f"[trace] prompt encoded to {len(tiny_ids)} tokens: "
          f"{[decode_tiny_id(t, tiny_to_gpt2, specials, gpt2_tok)[0] for t in tiny_ids]}")

    # Per-token records: we generate one at a time so each forward pass
    # exposes that step's full state.
    tokens_meta = []
    per_token = []

    for i, tid in enumerate(tiny_ids):
        s, is_special = decode_tiny_id(tid, tiny_to_gpt2, specials, gpt2_tok)
        tokens_meta.append({
            "tiny_id": int(tid),
            "str": s,
            "is_special": is_special,
            "is_prompt": True,
            "step_i": i,
        })

    # We do generation by repeatedly running the model with the full
    # prefix and taking the last position. Inefficient but simple, and
    # for trace export we want every position's state once.
    rng = np.random.default_rng(123)

    for step in range(args.gen + 1):
        # When step==0 we just want the trace for the existing prompt.
        # For step >= 1 we sample a new token, append, and re-run.
        ids = torch.tensor([tiny_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, trace = model(ids, return_trace=True)

        T = ids.shape[1]

        # Extract per-position state for the most-recently-added token
        # (the last token in the prefix). On step==0 we record state for
        # ALL prompt positions; on step>0 we record only the new last.
        positions_to_record = (range(T) if step == 0 else [T - 1])

        for pos in positions_to_record:
            if pos >= len(per_token):
                # Prepare entry
                rec = {}
            else:
                rec = per_token[pos]

            rec["meaning"] = to_list(trace["meaning"][0, pos])
            rec["intuition_input"] = to_list(trace["intuition_input"][0, pos])
            rec["hidden_out"] = to_list(trace["hidden_out"][0, pos])

            attn_per_layer = []
            for layer_attn in trace["attn_per_layer"]:
                # layer_attn: (B=1, H, T, T)
                if args.full_attn:
                    by_head = [to_list(layer_attn[0, h]) for h in range(layer_attn.shape[1])]
                else:
                    by_head = [to_list(layer_attn[0, h, pos, :pos + 1])
                                for h in range(layer_attn.shape[1])]
                attn_per_layer.append({"by_head": by_head})
            rec["attn_per_layer"] = attn_per_layer

            route_per_layer = []
            for layer_route in trace["route_per_layer"]:
                # layer_route: (B=1, T, E)
                route_per_layer.append(to_list(layer_route[0, pos]))
            rec["route_per_layer"] = route_per_layer

            # Top-K next-token predictions at this position
            pos_logits = logits[0, pos]
            probs = F.softmax(pos_logits / args.temperature, dim=-1)
            topk = torch.topk(probs, args.top_k)
            topk_list = []
            for k in range(args.top_k):
                tk_id = int(topk.indices[k].item())
                tk_p = float(topk.values[k].item())
                tk_s, tk_is_sp = decode_tiny_id(
                    tk_id, tiny_to_gpt2, specials, gpt2_tok,
                )
                topk_list.append({
                    "tiny_id": tk_id,
                    "str": tk_s,
                    "prob": tk_p,
                    "is_special": tk_is_sp,
                })
            rec["next_token_topk"] = topk_list

            if pos >= len(per_token):
                per_token.append(rec)
            else:
                per_token[pos] = rec

        # Sample the next token from the last-position distribution
        if step < args.gen:
            last_logits = logits[0, -1]
            probs = F.softmax(last_logits / args.temperature, dim=-1)
            # Top-K nucleus-ish sampling: pick from top_k by prob mass
            top_p, top_i = torch.topk(probs, args.top_k)
            top_p = top_p / top_p.sum()
            choice = rng.choice(args.top_k, p=top_p.cpu().numpy())
            next_id = int(top_i[choice].item())
            tiny_ids.append(next_id)
            s, is_special = decode_tiny_id(next_id, tiny_to_gpt2, specials, gpt2_tok)
            tokens_meta.append({
                "tiny_id": next_id,
                "str": s,
                "is_special": is_special,
                "is_prompt": False,
                "step_i": len(tiny_ids) - 1,
            })

    # Assemble output
    out = {
        "model_config": {
            "vocab_size": cfg.vocab_size,
            "meaning_dim": cfg.meaning_dim,
            "intuition_dim": cfg.intuition_dim,
            "input_dim": cfg.input_dim,
            "internal_dim": cfg.internal_dim,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "num_experts": cfg.num_experts,
            "top_k_experts": cfg.top_k_experts,
            "ffn_mult": cfg.ffn_mult,
        },
        "training_step": int(ckpt.get("step", 0)),
        "best_val": float(ckpt.get("best_val", float("nan"))),
        "qat_enabled": not args.no_qat,
        "tokens": tokens_meta,
        "meaning_axis_names": axis_names,
        "expert_names": EXPERT_NAMES,
        "prompt": args.prompt,
        "generation": {
            "n_generated": args.gen,
            "top_k": args.top_k,
            "temperature": args.temperature,
        },
        "per_token": per_token,
    }

    out_path = Path(args.out)
    with out_path.open("w") as f:
        json.dump(out, f)  # compact (no indent) to keep size small
    sz_mb = out_path.stat().st_size / 1e6
    print(f"\n[trace] wrote {out_path} ({sz_mb:.2f} MB)")
    print(f"[trace] tokens: {len(tokens_meta)} ({args.gen} generated)")
    print(f"[trace] generated text:")
    print(f"  {' / '.join(t['str'] for t in tokens_meta)}")


if __name__ == "__main__":
    main()
