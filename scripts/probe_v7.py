#!/usr/bin/env python3
"""probe_v7.py — quick generation probe for a v7 best_qat.pt.

Loads a v7 checkpoint, samples completions for a few domain-relevant
prompts. Eyeball-only — no scoring. Use to gauge whether the model
has learned anything coherent before spending wall-clock on the next
batch.

Run:
    python3 scripts/probe_v7.py \\
        --ckpt runs/tiny_hetmoe_v7_code_py/checkpoints/best_qat.pt \\
        --vocab 151665
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.tiny_hetmoe import (  # noqa: E402
    TinyHetMoE, TinyHetMoEConfig, set_quantize_mode, set_qat_backward_mode,
)


PROMPTS = {
    "code_py": [
        "def fibonacci(n):\n    ",
        "import numpy as np\n\ndef softmax(x):\n    ",
        "# Compute the mean of a list\ndef mean(xs):\n    ",
    ],
    "code_js": [
        "function fibonacci(n) {\n    ",
        "const sum = (xs) => ",
        "// Async fetch with retry\nasync function fetchWithRetry(url) {\n    ",
    ],
    "general": [
        "The capital of France is",
        "<|im_start|>user\nExplain photosynthesis in one sentence.<|im_end|>\n<|im_start|>assistant\n",
        "Once upon a time,",
    ],
    "thinker": [
        "<|im_start|>user\nIf a train leaves at 3pm going 60 mph, where is it at 5pm?<|im_end|>\n<|im_start|>assistant\n",
        "Let me think step by step. We have 12 apples and",
    ],
    "medical": [
        "<|im_start|>user\nWhat are the symptoms of pneumonia?<|im_end|>\n<|im_start|>assistant\n",
        "The patient presented with",
    ],
    "legal": [
        "The court held that",
        "<|im_start|>user\nWhat is the difference between civil and criminal law?<|im_end|>\n<|im_start|>assistant\n",
    ],
}


def load_model(ckpt_path: str, vocab: int, device: str) -> TinyHetMoE:
    mcfg = TinyHetMoEConfig(
        vocab_size=vocab,
        meaning_dim=132,
        intuition_dim=132,
        input_dim=264,
        internal_dim=528,
        new_intuition=264,
        num_layers=4,
        num_heads=4,
        num_experts=4,
        top_k_experts=2,
        ffn_mult=2.0,
        max_seq_len=2048,
    )
    model = TinyHetMoE(mcfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck.get("model", ck)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[probe] step={ck.get('step')} val={ck.get('best_val'):.4f} "
          f"qat={ck.get('qat_currently_on')} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    set_qat_backward_mode("ste")
    set_quantize_mode(model, on=bool(ck.get("qat_currently_on")))
    return model


@torch.no_grad()
def generate(model, ids: torch.Tensor, n_tokens: int = 60,
             temperature: float = 0.8, top_k: int = 40,
             eos_id: int | None = None) -> list[int]:
    out = ids[0].tolist()
    for _ in range(n_tokens):
        logits, _ = model(ids)
        nl = logits[0, -1] / temperature
        if top_k > 0:
            v, _ = torch.topk(nl, top_k)
            nl[nl < v[-1]] = -1e9
        probs = F.softmax(nl, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        if eos_id is not None and nxt == eos_id:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", type=int, default=151665)
    ap.add_argument("--n-tokens", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--domain", default="",
                    help="filter prompts to one domain (default: infer from "
                         "ckpt path)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Infer domain from checkpoint path (e.g. tiny_hetmoe_v7_code_py)
    if args.domain:
        domain = args.domain
    else:
        domain = "general"
        for k in PROMPTS:
            if k in args.ckpt:
                domain = k; break
    print(f"[probe] ckpt={args.ckpt}  domain={domain}")

    print(f"[probe] loading Qwen tokenizer")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
    eos_id = tok.eos_token_id

    model = load_model(args.ckpt, args.vocab, device)

    print()
    for prompt in PROMPTS.get(domain, PROMPTS["general"]):
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out_ids = generate(model, ids, n_tokens=args.n_tokens,
                            temperature=args.temp, top_k=args.top_k,
                            eos_id=eos_id)
        text = tok.decode(out_ids)
        print("─" * 72)
        print(f"PROMPT: {prompt!r}")
        print(f"OUTPUT:")
        print(text)
        print()


if __name__ == "__main__":
    main()
