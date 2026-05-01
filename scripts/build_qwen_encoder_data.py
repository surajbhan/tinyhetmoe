#!/usr/bin/env python3
"""build_qwen_encoder_data.py — produce encoder + decoder data for the
browser BPE encoder.

Outputs (into <out_dir>):
  - encode.json: byte_encoder, bpe_ranks, qwen_token_to_id, qwen_to_tiny,
                 specials. Used by the in-browser BPE encoder.
  - decode.json: tid -> string (already produced by build_stitch_demo_data;
                 we DON'T duplicate here unless --include-decode is passed)

Design:
  - We need the FULL Qwen merges + vocab for correct BPE; pruning to just
    our trimmed vocab would change tokenization. Full vocab JSON is ~5 MB.
  - Special tokens: <unk>, <bos>, <eos>, <pad> at tids 0..3.
  - The BPE merge rule format: a single string "left right" → rank.
  - The byte_encoder: maps byte values (0..255) to printable Unicode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def bytes_to_unicode():
    """Same byte-encoder as GPT-2/Qwen. Maps 0..255 → printable Unicode chars."""
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/stitch_v66")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find Qwen tokenizer files
    qwen_dir = Path(
        "/home/suraj/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B"
        "/snapshots/8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
    )
    vocab_path = qwen_dir / "vocab.json"
    merges_path = qwen_dir / "merges.txt"
    print(f"[enc] reading {vocab_path} + {merges_path}")
    qwen_vocab: dict[str, int] = json.load(vocab_path.open())
    print(f"[enc] qwen vocab: {len(qwen_vocab)} entries")

    # merges.txt: each non-comment line is "left right"
    raw = merges_path.read_text().splitlines()
    merges = [line for line in raw if line and not line.startswith("#")]
    bpe_ranks = {pair: i for i, pair in enumerate(merges)}
    print(f"[enc] merges: {len(merges)} pairs")

    # Our unified vocab: qwen_id -> tiny_id
    unified = json.load((REPO / "tokenizer/unified_vocab.json").open())
    qwen_to_tiny: dict[str, int] = unified["qwen_to_tiny"]  # str(qwen_id) -> tiny_id
    print(f"[enc] unified vocab: {len(qwen_to_tiny)} qwen→tiny mappings, "
          f"vocab_size={unified['vocab_size']}")

    # Byte encoder
    be = bytes_to_unicode()
    byte_encoder = {str(b): c for b, c in be.items()}

    # Build a direct qwen_token (string) -> tiny_id lookup. Skip the
    # intermediate qwen_id step by composing qwen_vocab and qwen_to_tiny.
    # If a qwen token's id isn't in qwen_to_tiny, omit it — the encoder
    # will emit <unk> when its merge result lands on a missing token,
    # which matches the prepared corpora's tokenization behavior.
    qwen_token_to_tiny: dict[str, int] = {}
    for token, qid in qwen_vocab.items():
        tid = qwen_to_tiny.get(str(qid))
        if tid is not None:
            qwen_token_to_tiny[token] = tid
    print(f"[enc] qwen_token_to_tiny: {len(qwen_token_to_tiny)} kept "
          f"(of {len(qwen_vocab)} qwen tokens)")

    # Specials
    specials = unified["specials"]

    # Read added_tokens from tokenizer.json so the JS encoder can match
    # them atomically (BEFORE BPE) the same way Qwen does.
    tj_path = qwen_dir / "tokenizer.json"
    tj = json.load(tj_path.open())
    added_tokens_list = []
    for at in tj.get("added_tokens", []):
        # Only include if the qwen_id is in our trimmed vocab (i.e., we
        # have a tiny_id for it). Otherwise the JS encoder shouldn't
        # match it specially.
        qid = at["id"]
        tid = qwen_to_tiny.get(str(qid))
        if tid is not None:
            added_tokens_list.append({"text": at["content"], "tid": tid})
    print(f"[enc] added_tokens (in trimmed vocab): {len(added_tokens_list)}")
    for at in added_tokens_list:
        print(f"  {at['text']!r} -> tid {at['tid']}")

    encode_data = {
        "byte_encoder": byte_encoder,
        "bpe_ranks": bpe_ranks,
        "qwen_token_to_tiny": qwen_token_to_tiny,
        "added_tokens": added_tokens_list,  # split on these BEFORE BPE
        "specials": specials,
        "vocab_size": unified["vocab_size"],
    }

    out_path = out_dir / "encode.json"
    out_path.write_text(json.dumps(encode_data))
    size_mb = out_path.stat().st_size / 1e6
    print(f"[enc] wrote {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
