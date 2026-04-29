// Benchmark WASM inference: tokens/sec and per-token latency.

import fs from "node:fs";
import { WasmModel } from "./wasm-pkg-node/tiny_hetmoe_wasm.js";

const bytes = fs.readFileSync("./tiny.bin");
console.log(`Loading model (${(bytes.length / 1e6).toFixed(1)} MB)...`);
const t0 = performance.now();
const m = new WasmModel(bytes);
const tLoad = performance.now() - t0;
console.log(`  load+pack: ${tLoad.toFixed(0)} ms`);

const promptIds = [1, 59, 63, 10, 40, 7, 42, 11, 10, 43, 41];

// Warmup
m.step(promptIds[0]);
m.reset();

// Run prompt + 50 generated tokens; measure the generation phase
const t1 = performance.now();
for (const tid of promptIds) m.step(tid);
const tPrompt = performance.now() - t1;
console.log(`  prompt fill (${promptIds.length} tokens): ${tPrompt.toFixed(0)} ms total, ${(tPrompt / promptIds.length).toFixed(1)} ms/token`);

const t2 = performance.now();
const N = 50;
for (let i = 0; i < N; i++) {
  // Just feed token 41 (some valid id) repeatedly — doesn't matter for timing
  m.step(41);
}
const tGen = performance.now() - t2;
console.log(`  generation (${N} tokens):  ${tGen.toFixed(0)} ms total, ${(tGen / N).toFixed(1)} ms/token`);
console.log(`  → ${(N / (tGen / 1000)).toFixed(1)} tokens/sec`);
