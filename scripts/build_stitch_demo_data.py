#!/usr/bin/env python3
"""build_stitch_demo_data.py — produce JSON helpers for the browser demo.

Outputs (into <out_dir>):
  - decode.json:    tid -> string mapping (one entry per non-special tid)
  - prompts.json:   list of pre-tokenized example prompts

The browser demo can't easily run Qwen BPE encoding in JS. So we ship a
fixed set of prompts pre-encoded to tids, and rely on `decode.json` to
turn generated tids back into text for display.

This is a manual-eval tool, not the production chatbot — full encode/decode
in browser comes later.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer


PROMPTS = [
    {
        "label": "Literary (PG-19 style)",
        "text": "She walked through the moonlit garden, where",
        "domain_hint": "pg19",
    },
    {
        "label": "Literary 2",
        "text": "The old wooden house at the edge of the meadow had been empty for many years",
        "domain_hint": "pg19",
    },
    {
        "label": "Encyclopedic (Wiki style)",
        "text": "The Battle of Hastings, fought on 14 October 1066, was a decisive Norman victory",
        "domain_hint": "wiki",
    },
    {
        "label": "Encyclopedic 2",
        "text": "Mount Everest is the highest mountain above sea level. It is located in",
        "domain_hint": "wiki",
    },
    {
        "label": "Tool call",
        "text": "<|im_start|>system\nYou are a helpful assistant with tools.<|im_end|>\n<|im_start|>user\nWhat's the weather in Tokyo?<|im_end|>\n<|im_start|>assistant\n",
        "domain_hint": "tool",
    },
    {
        "label": "Code (Python)",
        "text": "def fibonacci(n):\n    if n < 2:\n        return n\n    return ",
        "domain_hint": "code",
    },
    {
        "label": "Mixed (literary -> wiki)",
        "text": "She walked along the dusty road. The road, paved",
        "domain_hint": "mixed",
    },
]


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/stitch_v66")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True,
    )
    vocab = json.load((REPO / "tokenizer/unified_vocab.json").open())
    qwen_to_tiny = {int(k): v for k, v in vocab["qwen_to_tiny"].items()}
    tiny_to_qwen = vocab["tiny_to_qwen"]
    n_tids = vocab["vocab_size"]

    # Encode prompts
    prompts_out = []
    for p in PROMPTS:
        qids = tok(p["text"], add_special_tokens=False)["input_ids"]
        tids = [qwen_to_tiny.get(int(q), 0) for q in qids]
        n_unk = sum(1 for t in tids if t == 0)
        prompts_out.append({
            "label": p["label"],
            "text": p["text"],
            "tids": tids,
            "n_unk": n_unk,
            "domain_hint": p["domain_hint"],
        })
        print(f"  {p['label']:35s} : {len(tids)} tids ({n_unk} unk)")

    (out_dir / "prompts.json").write_text(json.dumps(prompts_out, indent=2))
    print(f"[demo] wrote {out_dir / 'prompts.json'}")

    # Build decode table: tid -> string. For specials use a placeholder.
    decode = []
    for tid in range(n_tids):
        qid = tiny_to_qwen[tid] if tid < len(tiny_to_qwen) else None
        if qid is None:
            # Special token (0..3 = unk/bos/eos/pad)
            decode.append(["<unk>", "<bos>", "<eos>", "<pad>"][tid] if tid < 4 else "<?>")
        else:
            decode.append(tok.decode([qid]))

    decode_path = out_dir / "decode.json"
    decode_path.write_text(json.dumps(decode))
    size_mb = decode_path.stat().st_size / 1e6
    print(f"[demo] wrote {decode_path} ({size_mb:.2f} MB, {n_tids} entries)")


if __name__ == "__main__":
    main()
