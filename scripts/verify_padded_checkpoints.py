#!/usr/bin/env python3
"""verify_padded_checkpoints.py — sanity-check that the vocab-padded
checkpoints work the same as their originals.

For each (pg19, wiki) × (best, best_qat) padded checkpoint:
  1. Load with vocab_size=32772 config
  2. Evaluate val loss on the corpus's val.bin
  3. Compare against the headline number from training
  4. Generate a 50-token continuation from a probe prompt
  5. Sanity-print the output

If val loss matches training and generation is coherent, padding worked.
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

from model.tiny_hetmoe import TinyHetMoE, TinyHetMoEConfig, set_quantize_mode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_TOKENS = 50_000
SEQ_LEN = 512
# Mask logits for these tids on pg19/wiki/code (prose) experts. They were
# zero-padded at vocab grow time and have no learned semantics; the mask
# prevents spurious emission of ChatML tokens that shouldn't appear in
# raw-prose continuation.
PROSE_MASK_TIDS = list(range(32768, 32772))


def load_padded(ckpt_path: Path, cfg_template: Path, new_vocab: int = 32772,
                qat: bool = False):
    cfg_dict = json.load(cfg_template.open())
    cfg_dict["vocab_size"] = new_vocab

    mc_fields = {
        "vocab_size", "meaning_dim", "intuition_dim", "input_dim",
        "internal_dim", "new_intuition", "num_layers", "num_heads",
        "num_experts", "top_k_experts", "ffn_mult", "max_seq_len",
        "load_balance_weight",
    }
    mc_kwargs = {k: v for k, v in cfg_dict.items() if k in mc_fields}
    mcfg = TinyHetMoEConfig(**mc_kwargs)
    model = TinyHetMoE(mcfg)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob.get("state_dict", blob))
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    if qat:
        # QuantizedLinear.quantize is a Python attr, not a parameter, so
        # load_state_dict doesn't restore it. QAT checkpoints' weights
        # only make sense when forward-quantized.
        n = set_quantize_mode(model, on=True)
        print(f"  enabled QAT forward on {n} modules")
    model.eval()
    model.to(DEVICE)
    return model


@torch.no_grad()
def eval_val(model, val_bin: Path, label: str) -> float:
    """Match training-time eval exactly:
      - sample sequences of (input, target) where target is shifted-by-1
      - call model(input, target) which uses the FP lm_head shortcut
      - loss is mean CE + small aux load-balance term (same as training)

    Uses the targets-CE path (line 465 in tiny_hetmoe.py) which bypasses
    QuantizedLinear.forward for lm_head — same as training. The deployed
    Rust engine should match this by keeping lm_head as fp16 (not ternary).
    """
    raw = np.fromfile(val_bin, dtype=np.uint16)
    # Build sequences of (T+1) so we can split into input[T] and target[T]
    n_full = (len(raw) // (SEQ_LEN + 1)) * (SEQ_LEN + 1)
    seqs = raw[:n_full].reshape(-1, SEQ_LEN + 1)
    max_seqs = EVAL_TOKENS // SEQ_LEN
    if len(seqs) > max_seqs:
        seqs = seqs[:max_seqs]

    total_loss = 0.0
    total_count = 0
    BATCH = 8
    for i in range(0, seqs.shape[0], BATCH):
        chunk = torch.from_numpy(seqs[i:i + BATCH].astype(np.int64)).to(DEVICE)
        ids = chunk[:, :-1].contiguous()       # (B, T)
        tgt = chunk[:, 1:].contiguous()        # (B, T) — shifted by 1
        _, loss = model(ids, tgt)
        n_tok = tgt.numel()
        total_loss += loss.item() * n_tok
        total_count += n_tok
    avg = total_loss / total_count
    print(f"  [{label}] val loss: {avg:.4f}  ppl {np.exp(avg):.2f}  "
          f"({total_count:,} tokens)")
    return avg


@torch.no_grad()
def generate(model, prompt_tids: list[int], n_tokens: int = 60,
             temperature: float = 0.8, top_k: int = 40) -> list[int]:
    ids = torch.tensor([prompt_tids], dtype=torch.long, device=DEVICE)
    out = list(prompt_tids)
    mask_tids_t = torch.tensor(PROSE_MASK_TIDS, device=DEVICE, dtype=torch.long)
    for _ in range(n_tokens):
        logits, _ = model(ids)
        next_logits = logits[0, -1].clone()
        next_logits[mask_tids_t] = -1e9
        next_logits = next_logits / temperature
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
    print(f"[verify] device {DEVICE}")
    print(f"[verify] loading Qwen tokenizer + unified vocab")
    qwen_tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True,
    )
    vocab = json.load((REPO / "tokenizer/unified_vocab.json").open())
    qwen_to_tiny = {int(k): v for k, v in vocab["qwen_to_tiny"].items()}
    tiny_to_qwen = vocab["tiny_to_qwen"]
    print(f"[verify] vocab_size: {vocab['vocab_size']}")

    def encode(text: str) -> list[int]:
        qids = qwen_tok(text, add_special_tokens=False)["input_ids"]
        return [qwen_to_tiny.get(int(q), 0) for q in qids]

    def decode(tids: list[int]) -> str:
        qids_back = [tiny_to_qwen[t] for t in tids
                     if t > 3 and tiny_to_qwen[t] is not None]
        return qwen_tok.decode(qids_back)

    cases = [
        ("pg19_best",     "runs/tiny_hetmoe_v6_5_pg19/checkpoints/best_padded.pt",
                          "training/configs/tiny_hetmoe_v6_5_pg19.json",
                          "data/unified_val_pg19.bin", 3.9331, False,
                          "She walked through the moonlit garden, where"),
        ("pg19_best_qat", "runs/tiny_hetmoe_v6_5_pg19/checkpoints/best_qat_padded.pt",
                          "training/configs/tiny_hetmoe_v6_5_pg19.json",
                          "data/unified_val_pg19.bin", 4.1848, True,
                          "She walked through the moonlit garden, where"),
        ("wiki_best",     "runs/tiny_hetmoe_v6_5_wiki/checkpoints/best_padded.pt",
                          "training/configs/tiny_hetmoe_v6_5_wiki.json",
                          "data/unified_val_wiki.bin", 3.4214, False,
                          "The Battle of Waterloo was fought in 1815 between"),
        ("wiki_best_qat", "runs/tiny_hetmoe_v6_5_wiki/checkpoints/best_qat_padded.pt",
                          "training/configs/tiny_hetmoe_v6_5_wiki.json",
                          "data/unified_val_wiki.bin", 3.7471, True,
                          "The Battle of Waterloo was fought in 1815 between"),
    ]

    print(f"\n[verify] running 4 cases (val loss + generation)\n")
    for label, ckpt, cfg, val_bin, expected_val, qat, prompt in cases:
        print(f"=== {label} ===")
        model = load_padded(REPO / ckpt, REPO / cfg, new_vocab=32772, qat=qat)
        val_loss = eval_val(model, REPO / val_bin, label)
        delta = val_loss - expected_val
        verdict = "OK" if abs(delta) < 0.05 else "DIFF"
        print(f"  expected ~{expected_val:.4f}, got {val_loss:.4f}  "
              f"(Δ={delta:+.4f}) [{verdict}]")
        prompt_tids = encode(prompt)
        out_tids = generate(model, prompt_tids, n_tokens=50, temperature=0.7)
        out_text = decode(out_tids)
        print(f"  prompt: {prompt!r}")
        print(f"  out:    {out_text!r}")
        print()
        # Free GPU memory between cases
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
