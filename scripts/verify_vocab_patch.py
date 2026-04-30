#!/usr/bin/env python3
"""verify_vocab_patch.py — sanity-check that the vocab patch left existing
checkpoints intact.

The earlier swap-mode patch briefly rewired tids 32764-32767 on disk to
ChatML qids. Currently-running training jobs loaded the meaning axes
into GPU RAM at startup, so they shouldn't have been affected — but
we want empirical confirmation.

This script:
  1. Loads pg19's best.pt with the ORIGINAL vocab=32768 config
  2. Runs a tokenizer round-trip on a probe sentence containing all
     four displaced words ('erial', ' twitch', ' tomato', '_med')
  3. Verifies tokens decode cleanly back to source text
  4. Generates a 50-token continuation and prints it

If output is coherent and includes/follows-from those words sensibly,
the embedding rows for tids 32764-32767 are intact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.tiny_hetmoe import TinyHetMoE, TinyHetMoEConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_pg19_with_orig_vocab():
    """Load pg19 best.pt using ORIGINAL vocab=32768 config (matches the
    shape the model was trained at). Bypasses the patched config file."""
    cfg_path = REPO / "training/configs/tiny_hetmoe_v6_5_pg19.json"
    cfg_dict = json.load(cfg_path.open())
    cfg_dict["vocab_size"] = 32768  # force original size

    mc_fields = {
        "vocab_size", "meaning_dim", "intuition_dim", "input_dim",
        "internal_dim", "new_intuition", "num_layers", "num_heads",
        "num_experts", "top_k_experts", "ffn_mult", "max_seq_len",
        "load_balance_weight",
    }
    mc_kwargs = {k: v for k, v in cfg_dict.items() if k in mc_fields}
    mcfg = TinyHetMoEConfig(**mc_kwargs)
    model = TinyHetMoE(mcfg)

    ckpt_path = REPO / "runs/tiny_hetmoe_v6_5_pg19/checkpoints/best.pt"
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob.get("state_dict", blob))
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(DEVICE)
    print(f"[verify] loaded pg19 best.pt at vocab=32768")
    return model


def tokenize_with_unified(text: str, qwen_tok, qwen_to_tiny: dict[int, int]):
    qids = qwen_tok(text, add_special_tokens=False)["input_ids"]
    tids = []
    for qid in qids:
        tids.append(qwen_to_tiny.get(int(qid), 0))  # 0 = <unk>
    return qids, tids


def decode_unified(tids: list[int], tiny_to_qwen: list, qwen_tok):
    qids_back = [tiny_to_qwen[tid] for tid in tids if tid > 3 and tiny_to_qwen[tid] is not None]
    return qwen_tok.decode(qids_back)


@torch.no_grad()
def generate(model, prompt_tids: list[int], n_tokens: int = 50,
             temperature: float = 0.8, top_k: int = 40):
    """Greedy + top-k sampling generation."""
    ids = torch.tensor([prompt_tids], dtype=torch.long, device=DEVICE)
    out = list(prompt_tids)
    for _ in range(n_tokens):
        logits, _ = model(ids)
        next_logits = logits[0, -1] / temperature
        if top_k > 0:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[-1]] = -1e9
        probs = F.softmax(next_logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        out.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        if next_id == 2:  # <eos>
            break
    return out


def main():
    print(f"[verify] loading Qwen tokenizer + unified vocab")
    qwen_tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True,
    )
    vocab = json.load((REPO / "tokenizer/unified_vocab.json").open())
    qwen_to_tiny = {int(k): val for k, val in vocab["qwen_to_tiny"].items()}
    tiny_to_qwen = vocab["tiny_to_qwen"]
    print(f"[verify] vocab_size: {vocab['vocab_size']}")

    # The 4 displaced original tokens (from when swap-mode briefly applied):
    test_qids = [(2848, "erial"), (59413, " twitch"),
                 (41020, " tomato"), (44085, "_med")]

    print(f"\n[verify] tid lookups for displaced original tokens:")
    for qid, name in test_qids:
        tid = qwen_to_tiny.get(qid)
        decoded = qwen_tok.decode([qid])
        print(f"  qid {qid:>6} ({decoded!r}): tid={tid}")

    print(f"\n[verify] tid lookups for ChatML specials (should be at 32768+):")
    for qid, name in [(151644, "<|im_start|>"), (151645, "<|im_end|>"),
                       (151657, "<tool_call>"), (151658, "</tool_call>")]:
        tid = qwen_to_tiny.get(qid)
        print(f"  qid {qid:>6} ({name}): tid={tid}")

    # ── Tokenizer round-trip ───────────────────────────────────────────
    print(f"\n[verify] round-trip test: text → tids → decode")
    probes = [
        "The serial number on the tomato can was rusted and unreadable.",
        "Her medical condition caused an involuntary twitch.",
        "Pictorial evidence of celestial events fascinated the imperial astronomers.",
    ]
    for text in probes:
        qids, tids = tokenize_with_unified(text, qwen_tok, qwen_to_tiny)
        decoded = decode_unified(tids, tiny_to_qwen, qwen_tok)
        unk_count = sum(1 for t in tids if t == 0)
        match = decoded.strip() == text.strip()
        print(f"  in:  {text!r}")
        print(f"  out: {decoded!r}")
        print(f"  unk_count: {unk_count}, exact_match: {match}")

    # ── Generation test on pg19 ─────────────────────────────────────────
    print(f"\n[verify] loading pg19 best.pt for generation test")
    model = load_pg19_with_orig_vocab()
    prompt = "She found a serial number etched into the tomato can."
    qids, tids = tokenize_with_unified(prompt, qwen_tok, qwen_to_tiny)
    print(f"\n[verify] prompt: {prompt!r}")
    print(f"  prompt tids ({len(tids)}): {tids}")
    print(f"  contains tid 32764 (erial): {32764 in tids}")
    print(f"  contains tid 32766 (tomato): {32766 in tids}")

    # Use only valid tids (filter those >= 32768 since model is vocab=32768)
    safe_tids = [t if t < 32768 else 0 for t in tids]
    out_tids = generate(model, safe_tids, n_tokens=80, temperature=0.7)
    out_text = decode_unified(out_tids, tiny_to_qwen, qwen_tok)
    print(f"\n[verify] generated continuation:")
    print(f"  {out_text!r}")
    print(f"\n[verify] eyeball check: does this read as coherent prose?")


if __name__ == "__main__":
    main()
