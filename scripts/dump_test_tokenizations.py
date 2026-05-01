#!/usr/bin/env python3
"""dump_test_tokenizations.py — produce a corpus of (text, tids) pairs
from the real Python Qwen tokenizer + unified vocab map.

Used to verify the JS encoder matches.

Usage:
  python3 scripts/dump_test_tokenizations.py [out_path]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer


# Diverse test cases covering the corner cases:
#   - Pure ASCII prose
#   - Numbers (Qwen splits 1-3 digit chunks)
#   - Punctuation density
#   - Whitespace patterns (single/double spaces, leading/trailing)
#   - Newlines
#   - Code with symbols
#   - Tool ChatML markers
#   - Unicode (not ASCII)
TEST_CASES = [
    "She walked through the moonlit garden, where",
    "The Battle of Hastings, fought on 14 October 1066, was a decisive Norman victory",
    "Mount Everest is the highest mountain above sea level. It is located in",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat's the weather?<|im_end|>",
    "Numbers: 1, 12, 123, 1234, 12345, 123456789",
    "Multiple   spaces\t\tand\ttabs   here.",
    "Trailing whitespace   ",
    "   Leading whitespace",
    "Punctuation!!! ??? ... ;;; (((",
    "She said, \"Hello, world!\" — the words echoed.",
    "Unicode: café, naïve, résumé, München, 北京",
    "Empty parens: () [] {} <> empty",
    "URL: https://example.com/path?query=1&val=2",
    "Code symbols: a += b; c <<= 1; d.method().chain();",
    "Mixed:She walked through the<|im_start|>tool_call<|im_end|>and continued",
]


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/expected_tokenizations.jsonl")

    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True,
    )
    vocab = json.load((REPO / "tokenizer/unified_vocab.json").open())
    qwen_to_tiny = {int(k): v for k, v in vocab["qwen_to_tiny"].items()}
    unk = vocab["specials"]["<unk>"]

    n_unk_total = 0
    with out_path.open("w") as f:
        for text in TEST_CASES:
            qids = tok(text, add_special_tokens=False)["input_ids"]
            tids = [qwen_to_tiny.get(int(q), unk) for q in qids]
            n_unk_total += sum(1 for t in tids if t == unk)
            f.write(json.dumps({"text": text, "tids": tids}) + "\n")

    print(f"[dump] wrote {len(TEST_CASES)} cases to {out_path}")
    print(f"[dump] {n_unk_total} unk tokens across all cases")


if __name__ == "__main__":
    main()
