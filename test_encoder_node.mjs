// Node test: verify the JS BPE encoder matches Python's Qwen tokenizer
// for our unified vocab. Run after build_qwen_encoder_data.py.
//
// Usage:
//   node test_encoder_node.mjs <bundle_dir> <expected_jsonl>
//
// expected_jsonl is produced by `scripts/dump_test_tokenizations.py`
// and contains lines of: {"text": "...", "tids": [...]}

import fs from "node:fs";

// Inline-port of stitch.js's encoder (no DOM/wasm dependency).
let byteEncoder = null;
let bpeRanks = null;
let qwenTokenToTiny = null;
let specials = null;
let addedTokens = null;          // [{text, tid}, ...]
let addedTokensSplitRe = null;   // RegExp matching any added_token literal
const bpeCache = new Map();

// Qwen2.5 pretokenizer regex (lifted verbatim from tokenizer.json).
// Differences from GPT-2: \p{N} (single digit, not 1-3 chunks),
// explicit newline handling. Note Qwen's regex uses (?i:'s|...) for
// case-insensitive contraction match on those alternatives only —
// JS doesn't have inline flag groups, so we expand the alternatives
// with explicit case alternatives.
// Use character classes [sS], [tT], etc. to handle Qwen's (?i:) flag.
const QWEN_PAT =
  /'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function loadEncoder(encodeData) {
  byteEncoder = {};
  for (const [b, c] of Object.entries(encodeData.byte_encoder)) {
    byteEncoder[parseInt(b, 10)] = c;
  }
  bpeRanks = new Map();
  for (const [pair, rank] of Object.entries(encodeData.bpe_ranks)) {
    bpeRanks.set(pair, rank);
  }
  qwenTokenToTiny = encodeData.qwen_token_to_tiny;
  specials = encodeData.specials;
  addedTokens = encodeData.added_tokens || [];
  if (addedTokens.length > 0) {
    // Sort by length desc so longer matches win (e.g. <|im_start|> over <|im)
    const sorted = [...addedTokens].sort((a, b) => b.text.length - a.text.length);
    const pattern = sorted.map(t => escapeRegex(t.text)).join("|");
    addedTokensSplitRe = new RegExp(pattern, "g");
  }
}

function bpeEncode(text) {
  if (bpeCache.has(text)) return bpeCache.get(text);
  let symbols = Array.from(text);
  while (true) {
    let bestRank = Infinity;
    let bestPair = null;
    for (let i = 0; i < symbols.length - 1; i++) {
      const pair = symbols[i] + " " + symbols[i + 1];
      const r = bpeRanks.get(pair);
      if (r !== undefined && r < bestRank) {
        bestRank = r;
        bestPair = [symbols[i], symbols[i + 1]];
      }
    }
    if (bestPair === null) break;
    const ns = [];
    let i = 0;
    while (i < symbols.length) {
      if (
        i < symbols.length - 1 &&
        symbols[i] === bestPair[0] &&
        symbols[i + 1] === bestPair[1]
      ) {
        ns.push(bestPair[0] + bestPair[1]);
        i += 2;
      } else {
        ns.push(symbols[i]);
        i++;
      }
    }
    symbols = ns;
  }
  bpeCache.set(text, symbols);
  return symbols;
}

function byteEncode(text) {
  // utf8 each char → map each byte through byte_encoder → join
  const utf8 = new TextEncoder().encode(text);
  let out = "";
  for (const b of utf8) out += byteEncoder[b];
  return out;
}

function encodeFragment(text) {
  // text is an arbitrary substring with NO added_tokens inside.
  // Qwen's pipeline: pretokenizer (on RAW text) → byte_encoder per
  // pre-token → BPE per pre-token → vocab lookup.
  if (!text) return [];
  const out = [];
  for (const match of text.matchAll(QWEN_PAT)) {
    const rawPiece = match[0];
    const bePiece = byteEncode(rawPiece);
    const symbols = bpeEncode(bePiece);
    for (const s of symbols) {
      const tid = qwenTokenToTiny[s];
      out.push(tid !== undefined ? tid : specials["<unk>"]);
    }
  }
  return out;
}

function encodeText(text) {
  // Split on added_tokens (e.g., <|im_start|>) FIRST. These are emitted
  // as their atom tids; the in-between fragments get the BPE pipeline.
  if (!addedTokensSplitRe) return encodeFragment(text);
  const out = [];
  let lastEnd = 0;
  // matchAll resets state because regex is /g
  for (const m of text.matchAll(addedTokensSplitRe)) {
    const start = m.index;
    const end = m.index + m[0].length;
    if (start > lastEnd) {
      out.push(...encodeFragment(text.slice(lastEnd, start)));
    }
    // Look up the tid for this added_token
    const at = addedTokens.find(t => t.text === m[0]);
    if (at) out.push(at.tid);
    else out.push(specials["<unk>"]);
    lastEnd = end;
  }
  if (lastEnd < text.length) {
    out.push(...encodeFragment(text.slice(lastEnd)));
  }
  return out;
}

// ─── Main ────────────────────────────────────────────────────────────

const bundleDir = process.argv[2] || "docs/stitch_v66";
const expectedPath = process.argv[3] || "/tmp/expected_tokenizations.jsonl";

console.log(`[test] loading encoder from ${bundleDir}/encode.json`);
const encodeData = JSON.parse(fs.readFileSync(`${bundleDir}/encode.json`, "utf8"));
loadEncoder(encodeData);
console.log(`[test] ${Object.keys(qwenTokenToTiny).length} tokens, ${bpeRanks.size} merge rules`);

console.log(`[test] reading expected from ${expectedPath}`);
const lines = fs.readFileSync(expectedPath, "utf8").split("\n").filter(l => l.trim());
console.log(`[test] ${lines.length} test cases`);

let pass = 0, fail = 0;
for (const line of lines) {
  const obj = JSON.parse(line);
  const got = encodeText(obj.text);
  const expected = obj.tids;
  const match = got.length === expected.length && got.every((v, i) => v === expected[i]);
  if (match) {
    pass++;
  } else {
    fail++;
    console.log(`\n--- FAIL ---`);
    console.log(`text: ${JSON.stringify(obj.text)}`);
    console.log(`expected (${expected.length}): ${expected.join(" ")}`);
    console.log(`got      (${got.length}): ${got.join(" ")}`);
    // Show first differing position
    const minLen = Math.min(got.length, expected.length);
    for (let i = 0; i < minLen; i++) {
      if (got[i] !== expected[i]) {
        console.log(`  first diff at pos ${i}: expected ${expected[i]}, got ${got[i]}`);
        break;
      }
    }
  }
}
console.log(`\n[test] ${pass} pass, ${fail} fail (of ${lines.length})`);
process.exit(fail > 0 ? 1 : 0);
