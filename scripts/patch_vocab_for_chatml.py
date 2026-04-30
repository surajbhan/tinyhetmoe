#!/usr/bin/env python3
"""patch_vocab_for_chatml.py — append ChatML specials, undoing any prior swap.

Goal end-state:
  tids 0..32767     unchanged from original prep
  tid  32768        → qid 151644 (<|im_start|>)
  tid  32769        → qid 151645 (<|im_end|>)
  tid  32770        → qid 151657 (<tool_call>)
  tid  32771        → qid 151658 (</tool_call>)
  vocab_size        = 32772

If a prior "swap-mode" patch is detected (tids 32764-32767 currently hold
the ChatML qids), undo it first by restoring the original qids that the
swap displaced.

Re-runs are idempotent — if vocab is already in the goal state, nothing
happens.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOK_DIR = REPO / "tokenizer"

CHATML_SPECIALS = [
    (151644, "<|im_start|>"),
    (151645, "<|im_end|>"),
    (151657, "<tool_call>"),
    (151658, "</tool_call>"),
]

# Original qids displaced by the earlier swap-mode patch (logged when
# the swap happened). These map back to tids 32764-32767 in that order.
DISPLACED_ORIGINALS = [
    (32764, 2848),   # 'erial'
    (32765, 59413),  # ' twitch'
    (32766, 41020),  # ' tomato'
    (32767, 44085),  # '_med'
]


def main():
    p = TOK_DIR / "unified_vocab.json"
    print(f"[patch] loading {p}")
    v = json.load(p.open())
    qwen_to_tiny: dict[str, int] = v["qwen_to_tiny"]
    tiny_to_qwen: list = v["tiny_to_qwen"]
    initial_size = v["vocab_size"]
    print(f"[patch] initial vocab_size: {initial_size}")

    # ── Step 1: detect & undo prior swap ────────────────────────────────
    chatml_qids = {qid for qid, _ in CHATML_SPECIALS}
    swap_tids = sorted(
        tid for tid in range(initial_size)
        if tid < len(tiny_to_qwen) and tiny_to_qwen[tid] in chatml_qids
        and tid < initial_size  # not in append range
    )
    if swap_tids and any(tid < 32768 for tid in swap_tids):
        print(f"[patch] detected swap-mode artifact at tids {swap_tids}; undoing")
        # Remove ChatML qids from qwen_to_tiny
        for qid in chatml_qids:
            qwen_to_tiny.pop(str(qid), None)
        # Restore displaced originals at their original tids
        for tid, orig_qid in DISPLACED_ORIGINALS:
            tiny_to_qwen[tid] = orig_qid
            qwen_to_tiny[str(orig_qid)] = tid
            print(f"  restored tid {tid} → qid {orig_qid}")

    # ── Step 2: check end-state & append if needed ──────────────────────
    # If all 4 ChatML qids already at end (append already done), skip.
    if v["vocab_size"] == initial_size:
        # Vocab still at original 32768 size; need to append
        target_size = initial_size + len(CHATML_SPECIALS)
    else:
        target_size = v["vocab_size"]

    already_appended = all(
        str(qid) in qwen_to_tiny and qwen_to_tiny[str(qid)] >= initial_size
        for qid, _ in CHATML_SPECIALS
    )
    if already_appended:
        print(f"[patch] all ChatML specials already appended; nothing to do")
    else:
        print(f"[patch] appending {len(CHATML_SPECIALS)} ChatML specials at tids "
              f"{initial_size}..{initial_size + len(CHATML_SPECIALS) - 1}")
        for offset, (qid, name) in enumerate(CHATML_SPECIALS):
            tid = initial_size + offset
            # Ensure the appended position exists in tiny_to_qwen
            while len(tiny_to_qwen) <= tid:
                tiny_to_qwen.append(None)
            tiny_to_qwen[tid] = qid
            qwen_to_tiny[str(qid)] = tid
            print(f"  tid {tid:>5} → qid {qid:>6}  ({name})")
        v["vocab_size"] = initial_size + len(CHATML_SPECIALS)

    v["chatml_patch"] = {
        "applied": True,
        "mode": "append",
        "specials": [{"tid": initial_size + i, "qid": qid, "name": name}
                     for i, (qid, name) in enumerate(CHATML_SPECIALS)],
        "displaced_restored": True,
    }
    json.dump(v, p.open("w"), indent=2)
    print(f"[patch] wrote {p} (vocab_size now {v['vocab_size']})")


if __name__ == "__main__":
    main()
