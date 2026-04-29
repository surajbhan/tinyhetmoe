#!/usr/bin/env python3
"""build_encode_lookup.py — emit JSON encoder data for the UI.

Outputs `ui/encode_lookup.json` containing:
  - `byte_encoder`: GPT-2's byte→unicode mapping (for byte-level BPE)
  - `gpt2_to_tiny`: pruned vocab remap (gpt2 id → tiny id)
  - `specials`: special token ids
  - `bpe_ranks`: GPT-2 merge rules as { "tokenA tokenB": rank }
  - `gpt2_token_to_id`: GPT-2 vocab string → id

JS will use these to:
  1. Encode raw text → bytes → byte-encoded chars
  2. Apply BPE merges greedily (lowest rank wins)
  3. Look up resulting tokens in gpt2_token_to_id
  4. Remap to tiny ids via gpt2_to_tiny (or <unk> if not in pruned vocab)
"""
import json
from pathlib import Path

from transformers import GPT2TokenizerFast

REPO = Path(__file__).resolve().parent.parent
vocab = json.load((REPO / "tokenizer" / "vocab.json").open())
gpt2_to_tiny = vocab["gpt2_to_tiny"]
specials = vocab["specials"]

tok = GPT2TokenizerFast.from_pretrained("gpt2")

# GPT-2 byte→unicode map. The slow tokenizer exposes byte_encoder.
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer, bytes_to_unicode
byte_encoder = bytes_to_unicode()

# BPE ranks. The fast tokenizer's underlying BPE model has them.
# Easiest: instantiate the slow tokenizer to get .bpe_ranks (a dict).
slow = GPT2Tokenizer.from_pretrained("gpt2")
bpe_ranks = {f"{a} {b}": r for (a, b), r in slow.bpe_ranks.items()}

# GPT-2 vocab string → id
gpt2_token_to_id = slow.encoder

out = {
    "byte_encoder": {str(b): c for b, c in byte_encoder.items()},
    "gpt2_to_tiny": gpt2_to_tiny,
    "gpt2_token_to_id": gpt2_token_to_id,
    "bpe_ranks": bpe_ranks,
    "specials": specials,
}
out_path = REPO / "ui" / "encode_lookup.json"
json.dump(out, out_path.open("w"))
print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")
print(f"  byte_encoder:    {len(out['byte_encoder'])} entries")
print(f"  bpe_ranks:       {len(out['bpe_ranks'])} merges")
print(f"  gpt2_token_to_id: {len(out['gpt2_token_to_id'])} entries")
