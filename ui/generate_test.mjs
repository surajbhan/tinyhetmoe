// Headless quality test: generate 50 tokens with different strategies.

import fs from "node:fs";
import { WasmModel } from "./wasm-pkg-node/tiny_hetmoe_wasm.js";

const bytes = fs.readFileSync("./tiny.bin");
const decode = JSON.parse(fs.readFileSync("./decode_lookup.json", "utf-8"));
const m = new WasmModel(bytes);

const PROMPTS = [
  // <bos> Once upon a time, there was a happy little
  [1, 59, 63, 10, 40, 7, 42, 11, 10, 43, 41],
  // <bos> Once upon a time
  [1, 59, 63, 10, 40],
];

function softmax(logits, T) {
  const max = Math.max(...logits);
  const e = logits.map(v => Math.exp((v - max) / T));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

function generate(promptIds, n, strategy) {
  m.reset();
  let last;
  for (const tid of promptIds) last = m.step(tid);
  const out = [...promptIds];
  for (let k = 0; k < n; k++) {
    const logits = Array.from(last.logits);
    let pick;
    if (strategy === "greedy") {
      let bestI = 0, bestV = -Infinity;
      for (let i = 0; i < logits.length; i++) {
        if (logits[i] > bestV) { bestV = logits[i]; bestI = i; }
      }
      pick = bestI;
    } else if (strategy === "topk-T0.5") {
      const idxs = Array.from(logits.keys());
      idxs.sort((a, b) => logits[b] - logits[a]);
      const top = idxs.slice(0, 8);
      const probs = softmax(top.map(i => logits[i]), 0.5);
      const r = Math.random();
      let acc = 0;
      pick = top[0];
      for (let i = 0; i < top.length; i++) {
        acc += probs[i];
        if (r < acc) { pick = top[i]; break; }
      }
    } else if (strategy === "topk-T0.8") {
      const idxs = Array.from(logits.keys());
      idxs.sort((a, b) => logits[b] - logits[a]);
      const top = idxs.slice(0, 8);
      const probs = softmax(top.map(i => logits[i]), 0.8);
      const r = Math.random();
      let acc = 0;
      pick = top[0];
      for (let i = 0; i < top.length; i++) {
        acc += probs[i];
        if (r < acc) { pick = top[i]; break; }
      }
    }
    out.push(pick);
    last = m.step(pick);
  }
  const text = out.map(t => decode[String(t)] || `<${t}>`).join("");
  return { ids: out, text };
}

for (const prompt of PROMPTS) {
  console.log("\n=== Prompt:", prompt.map(t => decode[String(t)]).join(""), "===");
  for (const strategy of ["greedy", "topk-T0.5", "topk-T0.8"]) {
    const r = generate(prompt, 50, strategy);
    console.log(`\n[${strategy}]`);
    console.log(`  ${r.text}`);
  }
}
