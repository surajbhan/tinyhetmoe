#!/usr/bin/env python3
"""build_tool_corpus.py — unify glaive + xLAM + Hermes into one ChatML format.

Target format = Hermes-style (which matches Qwen2.5 function-calling schema):

  <|im_start|>system
  You are a function calling AI model. You are provided with function
  signatures within <tools></tools> XML tags. You may call one or more
  functions to assist with the user query. Don't make assumptions about
  what values to plug into functions.
  <tools>
  [{"type": "function", "function": {...JSON-Schema...}}]
  </tools>
  <|im_end|>
  <|im_start|>user
  {query}<|im_end|>
  <|im_start|>assistant
  <tool_call>
  {"name": "...", "arguments": {...}}
  </tool_call><|im_end|>
  <|im_start|>tool
  <tool_response>
  {result}
  </tool_response><|im_end|>
  <|im_start|>assistant
  {final_reply}<|im_end|>

Mixed formats teach the model "anything goes" which is the wrong lesson
for tool-use. We unify upfront so every example tokenizes the same way.

Output:
  data/tool_corpus_chatml.txt    — one conversation per <|endoftext|>-
                                    separated block, ready for tokenization
  data/tool_corpus_stats.json    — per-source counts, total bytes

Then:
  scripts/prepare_data_unified.py picks this up via a new code path,
  re-tokenizes into unified_train_tool.bin / unified_val_tool.bin.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

GLAIVE_JSON = Path(
    "/home/suraj/.cache/huggingface/hub/datasets--glaiveai--glaive-function-calling-v2"
    "/snapshots/e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac/glaive-function-calling-v2.json"
)

# ChatML markers
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DOC_SEP = "<|endoftext|>"

SYSTEM_PREAMBLE = (
    "You are a function calling AI model. You are provided with function "
    "signatures within <tools></tools> XML tags. You may call one or more "
    "functions to assist with the user query. Don't make assumptions about "
    "what values to plug into functions."
)


def emit_chatml_turn(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def emit_system_with_tools(tools: list) -> str:
    """Emit system turn with tools array embedded in <tools></tools>."""
    body = f"{SYSTEM_PREAMBLE}\n<tools>\n{json.dumps(tools)}\n</tools>"
    return emit_chatml_turn("system", body)


def emit_tool_call(name: str, arguments: dict) -> str:
    body = f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"
    return emit_chatml_turn("assistant", body)


def emit_tool_response(content) -> str:
    body_str = content if isinstance(content, str) else json.dumps(content)
    body = f"<tool_response>\n{body_str}\n</tool_response>"
    return emit_chatml_turn("tool", body)


# ─── normalisers per source ────────────────────────────────────────────


def normalize_xlam_tool(tool: dict) -> dict:
    """xLAM tool format → JSON Schema format.
    xLAM: {"name": ..., "description": ..., "parameters": {param: {type, desc, default}}}
    Target: {"type":"function","function":{"name":...,"description":...,
              "parameters":{"type":"object","properties":{...},"required":[...]}}}
    """
    name = tool.get("name", "")
    desc = tool.get("description", "")
    params = tool.get("parameters") or {}
    properties = {}
    required = []
    if not isinstance(params, dict):
        params = {}
    for pname, pinfo in params.items():
        if not isinstance(pinfo, dict):
            continue
        properties[pname] = {
            "type": pinfo.get("type", "string"),
            "description": pinfo.get("description", ""),
        }
        if "default" not in pinfo:
            required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def normalize_glaive_tool(tool: dict) -> dict:
    """Glaive tools are already JSON Schema-shaped. Just wrap in
    {type: function, function: {...}} envelope.
    """
    name = tool.get("name", "")
    desc = tool.get("description", "")
    params = tool.get("parameters") or {"type": "object", "properties": {}}
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": params,
        },
    }


def normalize_glaive_tools_from_system(system_text: str) -> list:
    """Extract tool defs from glaive's system prose-JSON. Returns canonical list.

    Glaive system text looks like:
        SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -
        {
            "name": "get_exchange_rate",
            "description": "Get the exchange rate between two currencies",
            "parameters": { ... }
        }
        ...
    Multiple JSON blocks back-to-back. Some entries have NO tools.
    """
    # Strip "SYSTEM:" prefix if present
    text = re.sub(r'^\s*SYSTEM\s*:\s*', '', system_text, flags=re.IGNORECASE)
    # Find all top-level {...} JSON blocks (greedy brace-matching).
    tools = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                blob = text[start:i + 1]
                try:
                    obj = json.loads(blob)
                    if "name" in obj:
                        tools.append(normalize_glaive_tool(obj))
                except json.JSONDecodeError:
                    pass
                start = None
    return tools


def _extract_glaive_call(content: str):
    """Find <functioncall> {...} in an assistant turn. Returns (name, args)
    or None if no functioncall. Glaive's args is a JSON STRING (not object)
    containing more JSON, so we need to find the outer braces with a
    brace-depth scan instead of regex."""
    idx = content.find('<functioncall>')
    if idx < 0:
        return None
    # Find first '{' after the marker
    j = content.find('{', idx)
    if j < 0:
        return None
    depth = 0
    end = -1
    in_str = False
    esc = False
    for k in range(j, len(content)):
        ch = content[k]
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = k + 1
                break
    if end < 0:
        return None
    blob = content[j:end]
    # Glaive uses Python-ish mixed quoting: arguments is a STRING value
    # wrapped in single quotes containing JSON, e.g.:
    #   {"name": "...", "arguments": '{"country": "US"}'}
    # That's not valid JSON. Detect this pattern and rewrite to:
    #   {"name": "...", "arguments": {"country": "US"}}
    # before parsing.
    fixed = re.sub(
        r'"arguments"\s*:\s*\'(\{.*?\})\'',
        r'"arguments": \1',
        blob,
        flags=re.DOTALL,
    )
    try:
        obj = json.loads(fixed)
    except json.JSONDecodeError:
        # Try the original (in case some entries are already correct)
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            return None
    name = obj.get("name", "")
    args_raw = obj.get("arguments", {})
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {"_raw": args_raw}
    elif isinstance(args_raw, dict):
        args = args_raw
    else:
        args = {}
    return (name, args)


def parse_glaive_chat(chat_text: str) -> list[tuple[str, str]]:
    """Glaive chat text → list of (role, content) tuples.

    Glaive uses markers: USER: ..., A: ..., FUNCTION RESPONSE: ...
    (Note: assistant marker is 'A:' not 'ASSISTANT:'.)
    Inside A turns: <functioncall> {...} <|endoftext|> for tool calls,
    where arguments is a STRING containing JSON, not a JSON object.
    """
    # Split on the role markers. Glaive inconsistently uses 'A:' or
    # 'ASSISTANT:' for assistant turns. Match both, plus USER and FUNCTION
    # RESPONSE. The marker must be at start-of-line (after \n) and end with ':'.
    parts = re.split(
        r'(?:\n|^)\s*(USER|ASSISTANT|A|FUNCTION RESPONSE)\s*:\s*',
        '\n' + chat_text,
    )
    turns = []
    for i in range(1, len(parts) - 1, 2):
        role_raw = parts[i].strip()
        content = parts[i + 1].strip()
        if not content:
            continue
        content = re.sub(r'\s*<\|endoftext\|>\s*$', '', content).strip()
        if role_raw == 'USER':
            turns.append(('user', content))
        elif role_raw in ('A', 'ASSISTANT'):
            call = _extract_glaive_call(content)
            if call:
                name, args = call
                turns.append(('assistant_tool_call', (name, args)))
            else:
                turns.append(('assistant', content))
        elif role_raw == 'FUNCTION RESPONSE':
            turns.append(('tool', content))
    return turns


# ─── per-source iterators ──────────────────────────────────────────────


def iter_glaive_chatml(byte_cap: int) -> Iterator[str]:
    """Yield one canonical-ChatML conversation per call."""
    raw = json.loads(GLAIVE_JSON.read_text())
    print(f"[glaive] {len(raw)} conversations available")
    total = 0
    n_emitted = 0
    n_skipped = 0
    n_errors = 0
    for ex in raw:
        if not isinstance(ex, dict):
            continue
        sys_text = ex.get("system", "") or ""
        chat_text = ex.get("chat", "") or ""
        if not chat_text.strip():
            continue
        try:
            tools = normalize_glaive_tools_from_system(sys_text)
            turns = parse_glaive_chat(chat_text)
        except Exception:
            n_errors += 1
            continue
        if not turns:
            n_skipped += 1
            continue
        out = emit_system_with_tools(tools) if tools else emit_chatml_turn(
            "system", "You are a helpful assistant.")
        for role, content in turns:
            if role == 'user':
                out += emit_chatml_turn("user", content)
            elif role == 'assistant':
                out += emit_chatml_turn("assistant", content)
            elif role == 'assistant_tool_call':
                name, args = content
                out += emit_tool_call(name, args)
            elif role == 'tool':
                out += emit_tool_response(content)
        out += DOC_SEP + "\n"
        total += len(out.encode("utf-8"))
        n_emitted += 1
        yield out
        if total >= byte_cap:
            break
    print(f"[glaive] emitted {n_emitted}, skipped {n_skipped}, "
          f"errors {n_errors}, {total/1e6:.1f} MB")


def iter_xlam_chatml(byte_cap: int) -> Iterator[str]:
    """xLAM-60k → ChatML. Single-turn: query → tool_call(s)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    print(f"[xlam] {len(ds)} examples")
    total = 0
    n = 0
    for ex in ds:
        query = ex.get("query", "")
        tools_raw = ex.get("tools", "[]")
        answers_raw = ex.get("answers", "[]")
        # tools/answers are stringified JSON
        try:
            tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
            answers = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
        except json.JSONDecodeError:
            continue
        if not query or not isinstance(answers, list):
            continue
        canonical_tools = [normalize_xlam_tool(t) for t in tools]
        out = emit_system_with_tools(canonical_tools)
        out += emit_chatml_turn("user", query)
        # xLAM may have multiple parallel tool calls → emit each in one assistant turn
        if answers:
            calls_body = "\n".join(
                f"<tool_call>\n{json.dumps({'name': a.get('name',''), 'arguments': a.get('arguments',{})})}\n</tool_call>"
                for a in answers
            )
            out += emit_chatml_turn("assistant", calls_body)
        out += DOC_SEP + "\n"
        total += len(out.encode("utf-8"))
        n += 1
        yield out
        if total >= byte_cap:
            break
    print(f"[xlam] emitted {n}, {total/1e6:.1f} MB")


def iter_hermes_chatml(byte_cap: int) -> Iterator[str]:
    """Hermes-FC-v1 — already nearly in target format. Just remap roles."""
    from datasets import load_dataset
    # Multiple configs; use the main 'func_calling' subset
    try:
        ds = load_dataset(
            "NousResearch/hermes-function-calling-v1",
            "func_calling", split="train",
        )
    except Exception:
        ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
    print(f"[hermes] {len(ds)} examples")
    total = 0
    n = 0
    for ex in ds:
        convs = ex.get("conversations", [])
        if not convs:
            continue
        out = ""
        for turn in convs:
            role_raw = (turn.get("from", "") or "").lower()
            content = turn.get("value", "")
            if not content:
                continue
            role_map = {
                "system": "system",
                "human": "user",
                "user": "user",
                "gpt": "assistant",
                "assistant": "assistant",
                "tool": "tool",
                "function": "tool",
            }
            role = role_map.get(role_raw, role_raw)
            out += emit_chatml_turn(role, content)
        out += DOC_SEP + "\n"
        total += len(out.encode("utf-8"))
        n += 1
        yield out
        if total >= byte_cap:
            break
    print(f"[hermes] emitted {n}, {total/1e6:.1f} MB")


# ─── main ──────────────────────────────────────────────────────────────


def main():
    out_path = DATA_DIR / "tool_corpus_chatml.txt"
    stats_path = DATA_DIR / "tool_corpus_stats.json"

    PER_SOURCE_BYTES = 350 * 1000 * 1000  # 350 MB per source max

    sources = [
        ("glaive", iter_glaive_chatml),
        ("xlam",   iter_xlam_chatml),
        ("hermes", iter_hermes_chatml),
    ]

    stats = {"sources": {}, "format": "chatml-hermes-style"}
    total_bytes = 0
    with out_path.open("w") as f:
        for name, iterator in sources:
            print(f"\n[build] running source: {name}")
            n = 0
            bytes_this = 0
            for chunk in iterator(PER_SOURCE_BYTES):
                f.write(chunk)
                n += 1
                bytes_this += len(chunk.encode("utf-8"))
            stats["sources"][name] = {
                "conversations": n,
                "bytes": bytes_this,
            }
            total_bytes += bytes_this
    stats["total_bytes"] = total_bytes
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\n[build] wrote {out_path} ({total_bytes/1e6:.1f} MB)")
    print(f"[build] stats:\n{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
