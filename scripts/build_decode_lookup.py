#!/usr/bin/env python3
"""build_decode_lookup.py — produce a JSON map from tiny token id to
its decoded string (using GPT-2 byte-level decode). Shipped to the UI
so it can render token strings without needing a JS BPE library.
"""
import json
from pathlib import Path

from transformers import GPT2TokenizerFast

REPO = Path(__file__).resolve().parent.parent
vocab = json.load((REPO / "tokenizer" / "vocab.json").open())
specials = vocab["specials"]
specials_rev = {v: k for k, v in specials.items()}
tiny_to_gpt2 = vocab["tiny_to_gpt2"]
tok = GPT2TokenizerFast.from_pretrained("gpt2")

decode = {}
for tid, gid in enumerate(tiny_to_gpt2):
    if gid is None:
        decode[str(tid)] = specials_rev.get(tid, f"<{tid}>")
        continue
    decode[str(tid)] = tok.decode([gid])

out = REPO / "ui" / "decode_lookup.json"
json.dump(decode, out.open("w"))
print(f"wrote {out} ({len(decode)} entries, {out.stat().st_size/1e3:.1f} KB)")
