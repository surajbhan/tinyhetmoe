#!/usr/bin/env python3
"""filter_tool_for_classifier.py — extract only the structural-signal
portions of the tool corpus to use as classifier training data.

Problem: the tool corpus is ChatML conversations where the bulk of tokens
are natural-language user messages and assistant replies. When the
classifier sees these via EMA-of-meaning, they look like generic prose,
not "tool". So the 4-way classifier mis-routes literary prompts to tool.

Fix: extract only the spans inside <tool_call>...</tool_call> blocks
(JSON tool calls). These are the structurally distinctive parts that
identify "this conversation is in tool mode" — JSON syntax, function
names, structured arguments. The expert is still trained on full ChatML;
only the classifier sees the filtered version.

Output:
  data/unified_val_tool_clf.bin    — concatenation of all <tool_call>
                                      blocks from unified_val_tool.bin
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

TID_IM_START = 32768
TID_IM_END = 32769
TID_TOOL_CALL_OPEN = 32770
TID_TOOL_CALL_CLOSE = 32771


def main():
    # Use TRAIN bin as source — much more data than val, gives the
    # classifier more <tool_call> blocks to learn from.
    src = DATA_DIR / "unified_train_tool.bin"
    out = DATA_DIR / "unified_val_tool_clf.bin"
    print(f"[filter] reading {src}")
    raw = np.fromfile(src, dtype=np.uint16)
    print(f"[filter] {len(raw):,} tokens total")

    # Strategy: keep any ChatML turn (im_start...im_end) that contains a
    # <tool_call>...</tool_call> block. This includes:
    #   - the system message defining tools (if it's near a tool turn)
    #   - assistant turns that emit tool calls (with their structure)
    #   - tool-response turns
    # This way the classifier sees not just the JSON inside <tool_call>
    # but also the surrounding ChatML markers, which are the strongest
    # "this is tool mode" signal at the classifier window scale.
    #
    # Also includes the ChatML markers themselves — those are the ChatML
    # specials whose meaning vectors point strongly to "tool" once the
    # classifier learns them.
    out_chunks = []
    n_blocks = 0
    n_tokens_kept = 0

    # Walk the stream, splitting into im_start..im_end turns. Keep turns
    # that contain a tool_call.
    i = 0
    while i < len(raw):
        if raw[i] == TID_IM_START:
            turn_start = i
            j = i + 1
            has_tool_call = False
            while j < len(raw) and raw[j] != TID_IM_END:
                if raw[j] == TID_TOOL_CALL_OPEN:
                    has_tool_call = True
                j += 1
            turn_end = min(j + 1, len(raw))  # include im_end
            if has_tool_call:
                turn = raw[turn_start:turn_end]
                out_chunks.append(turn)
                n_tokens_kept += len(turn)
                n_blocks += 1
            i = turn_end
        else:
            i += 1

    if not out_chunks:
        print("[filter] WARNING: no <tool_call> blocks found")
        return

    out_arr = np.concatenate(out_chunks)
    out_arr.tofile(out)
    print(f"[filter] wrote {out} ({len(out_arr):,} tokens, "
          f"{n_blocks} blocks, "
          f"avg {len(out_arr) / n_blocks:.1f} tokens/block)")
    print(f"[filter] kept {n_tokens_kept / len(raw) * 100:.2f}% of source")


if __name__ == "__main__":
    main()
