#!/usr/bin/env python3
"""build_qwen_encoder_v7.py — produce encoder + decoder data for the v7
browser BPE tokenizer.

V7 uses the FULL Qwen 2.5 vocab (151,665 tokens) as-is — no trimming, no
unified mapping. tiny_id == qwen_id throughout.

Outputs (into <out_dir>):
  - encode.json: byte_encoder, bpe_ranks, qwen_token_to_id, added_tokens,
                 vocab_size. Used by the in-browser BPE encoder.
  - decode.json: tid -> string. Used to detokenize the model's outputs.

Run:
  python3 scripts/build_qwen_encoder_v7.py docs/stitch_v7
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def bytes_to_unicode():
    """GPT-2/Qwen byte→unicode map."""
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
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/stitch_v7")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find Qwen tokenizer files (Coder-0.5B has the same vocab as 0.5B base)
    candidates = [
        Path("/home/suraj/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B"
             "/snapshots/8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"),
    ]
    # also accept any Qwen2.5 base / coder snapshot dir
    hf_root = Path("/home/suraj/.cache/huggingface/hub")
    for d in hf_root.glob("models--Qwen--Qwen2.5*/snapshots/*"):
        if (d / "vocab.json").exists() and (d / "merges.txt").exists():
            candidates.append(d)
    qwen_dir = next((d for d in candidates if (d/"vocab.json").exists()), None)
    if qwen_dir is None:
        raise SystemExit("[enc] no Qwen tokenizer dir found in HF cache")

    vocab_path = qwen_dir / "vocab.json"
    merges_path = qwen_dir / "merges.txt"
    tj_path = qwen_dir / "tokenizer.json"

    print(f"[enc] reading {vocab_path}")
    qwen_vocab: dict[str, int] = json.load(vocab_path.open())
    print(f"[enc] qwen vocab: {len(qwen_vocab)} entries")

    raw = merges_path.read_text().splitlines()
    merges = [line for line in raw if line and not line.startswith("#")]
    bpe_ranks = {pair: i for i, pair in enumerate(merges)}
    print(f"[enc] merges: {len(merges)} pairs")

    # In v7, qwen_token_to_tiny is the identity (qwen_token -> qwen_id).
    qwen_token_to_tiny: dict[str, int] = dict(qwen_vocab)
    print(f"[enc] qwen_token_to_tiny: {len(qwen_token_to_tiny)} entries (identity map)")

    # Read added_tokens from tokenizer.json so the JS encoder can match them
    # atomically before running BPE (e.g., <|im_start|>, <|endoftext|>).
    tj = json.load(tj_path.open())
    added_tokens_list = []
    for at in tj.get("added_tokens", []):
        added_tokens_list.append({"text": at["content"], "tid": at["id"]})
    print(f"[enc] added_tokens: {len(added_tokens_list)}")
    for at in added_tokens_list[:8]:
        print(f"  {at['text']!r} -> tid {at['tid']}")

    # Byte encoder
    be = bytes_to_unicode()
    byte_encoder = {str(b): c for b, c in be.items()}

    # Special tokens (just the basics — for v7 ChatML these are added_tokens)
    # We expose the Qwen names directly as specials for compatibility.
    specials = {
        "endoftext": 151643,
        "im_start": 151644,
        "im_end": 151645,
    }
    # Tally vocab size: take max of qwen_vocab values + 1, plus added tokens
    max_tid = max(qwen_vocab.values())
    if added_tokens_list:
        max_tid = max(max_tid, max(at["tid"] for at in added_tokens_list))
    vocab_size = max_tid + 1
    print(f"[enc] vocab_size = {vocab_size}")

    encode_data = {
        "byte_encoder": byte_encoder,
        "bpe_ranks": bpe_ranks,
        "qwen_token_to_tiny": qwen_token_to_tiny,
        "added_tokens": added_tokens_list,
        "specials": specials,
        "vocab_size": vocab_size,
    }
    encode_path = out_dir / "encode.json"
    encode_path.write_text(json.dumps(encode_data))
    print(f"[enc] wrote {encode_path} ({encode_path.stat().st_size / 1e6:.2f} MB)")

    # Decode table: tid -> raw string (with byte-decoder un-applied)
    # Build inverse byte-decoder
    byte_decoder = {c: b for b, c in be.items()}

    def decode_token(tok_str: str) -> str:
        # Each char in tok_str maps via byte_decoder back to a byte
        try:
            byte_arr = bytes(byte_decoder[c] for c in tok_str)
            return byte_arr.decode("utf-8", errors="replace")
        except KeyError:
            return tok_str  # fallback (e.g., for already-utf8 special tokens)

    decode_table = [""] * vocab_size
    for tok, tid in qwen_vocab.items():
        decode_table[tid] = decode_token(tok)
    for at in added_tokens_list:
        decode_table[at["tid"]] = at["text"]

    decode_path = out_dir / "decode.json"
    decode_path.write_text(json.dumps(decode_table))
    print(f"[enc] wrote {decode_path} ({decode_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
