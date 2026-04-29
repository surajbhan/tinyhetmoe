// Headless test: load the WASM model in Node, run a prompt, print top-K.
// This validates that the WASM build works end-to-end without a browser.

import fs from "node:fs";
import { WasmModel } from "./wasm-pkg-node/tiny_hetmoe_wasm.js";

const bytes = fs.readFileSync("./tiny.bin");
console.log(`[test] loading ${bytes.length} bytes...`);
const m = new WasmModel(bytes);
console.log(`[test] model: ${m.vocab_size} vocab, ${m.num_layers} layers x ${m.num_heads} heads, ${m.num_experts} experts`);

const promptIds = [1, 59, 63, 10, 40, 7, 42, 11, 10, 43, 41];  // "<bos> Once upon a time, there was a happy little"
console.log(`[test] running prompt ${JSON.stringify(promptIds)}`);

let lastStep = null;
for (const tid of promptIds) {
  lastStep = m.step(tid);
}

console.log(`[test] last token results:`);
console.log(`  hidden[0..5]   = ${Array.from(lastStep.hidden.slice(0, 5))}`);
console.log(`  meaning[0..5]  = ${Array.from(lastStep.meaning.slice(0, 5))}`);

// Top-5 logits
const logits = Array.from(lastStep.logits);
const ranked = logits.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v).slice(0, 5);
console.log(`  top-5: ${ranked.map(r => `id=${r.i} logit=${r.v.toFixed(3)}`).join(", ")}`);

// Routing
const numLayers = m.num_layers;
const numExperts = m.num_experts;
const r = Array.from(lastStep.routing_flat);
console.log(`  routing layer 0 = ${r.slice(0, numExperts).map(v => v.toFixed(3))}`);
console.log(`  routing layer ${numLayers - 1} = ${r.slice((numLayers - 1) * numExperts, numLayers * numExperts).map(v => v.toFixed(3))}`);

// Attention shape sanity
const attnLengths = Array.from(lastStep.attn_lengths);
console.log(`  attn_lengths (${attnLengths.length} entries): first=${attnLengths.slice(0, 4)}, last=${attnLengths.slice(-4)}`);
