// Verify the JS BPE encoder agrees with Python's GPT-2 tokenizer + remap.

import fs from "node:fs";

const data = JSON.parse(fs.readFileSync("./encode_lookup.json", "utf-8"));
const { byte_encoder, gpt2_to_tiny, gpt2_token_to_id, bpe_ranks, specials } = data;

const byteEnc = {};
for (const [b, c] of Object.entries(byte_encoder)) byteEnc[parseInt(b, 10)] = c;
const ranks = new Map();
for (const [pair, rank] of Object.entries(bpe_ranks)) ranks.set(pair, rank);

const GPT2_PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function bpeEncode(text) {
  let symbols = Array.from(text);
  while (true) {
    let bestRank = Infinity, bestPair = null;
    for (let i = 0; i < symbols.length - 1; i++) {
      const pair = symbols[i] + " " + symbols[i + 1];
      const r = ranks.get(pair);
      if (r !== undefined && r < bestRank) {
        bestRank = r;
        bestPair = [symbols[i], symbols[i + 1]];
      }
    }
    if (!bestPair) break;
    const out = [];
    let i = 0;
    while (i < symbols.length) {
      if (i < symbols.length - 1 && symbols[i] === bestPair[0] && symbols[i + 1] === bestPair[1]) {
        out.push(bestPair[0] + bestPair[1]);
        i += 2;
      } else {
        out.push(symbols[i]);
        i++;
      }
    }
    symbols = out;
  }
  return symbols;
}

function encodeText(text) {
  const out = [];
  const utf8 = new TextEncoder().encode(text);
  let byteEncoded = "";
  for (const b of utf8) byteEncoded += byteEnc[b];
  for (const match of byteEncoded.matchAll(GPT2_PAT)) {
    const piece = match[0];
    const symbols = bpeEncode(piece);
    for (const s of symbols) {
      const gid = gpt2_token_to_id[s];
      if (gid === undefined) {
        out.push(specials["<unk>"]);
      } else {
        const tid = gpt2_to_tiny[String(gid)];
        if (tid === undefined) {
          out.push(specials["<unk>"]);
        } else {
          out.push(tid);
        }
      }
    }
  }
  return out;
}

const tests = [
  ["Once upon a time", [59, 63, 10, 40]],
  ["Once upon a time, there was a happy little", [59, 63, 10, 40, 7, 42, 11, 10, 43, 41]],
];

let pass = 0;
for (const [text, expected] of tests) {
  const got = encodeText(text);
  const ok = got.length === expected.length && got.every((v, i) => v === expected[i]);
  console.log(`${ok ? "✓" : "✗"} "${text}"`);
  console.log(`  expected: ${expected.join(",")}`);
  console.log(`  got:      ${got.join(",")}`);
  if (ok) pass++;
}
console.log(`\n${pass}/${tests.length} passed`);
