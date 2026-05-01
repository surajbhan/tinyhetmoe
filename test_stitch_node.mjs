// Node test for the wasm stitched engine.
// Loads the stitch bundle (stitch.json + N expert .bin files), constructs
// the engine, runs a literary + encyclopedic prompt and prints route + text.
//
// Usage:
//   node test_stitch_node.mjs <bundle_dir>

import fs from "node:fs";
import path from "node:path";
import { WasmStitchedEngine } from "./wasm/pkg-node/tiny_hetmoe_wasm.js";

const bundleDir = process.argv[2] || "docs/stitch_v66";
console.log(`[test] loading bundle from ${bundleDir}`);

// Read stitch.json
const stitchJson = fs.readFileSync(path.join(bundleDir, "stitch.json"), "utf8");
const stitch = JSON.parse(stitchJson);
console.log(`[test] format: ${stitch.format_version}, ema_alpha: ${stitch.ema_alpha}`);
console.log(`[test] experts: ${stitch.experts.map(e => e.name).join(", ")}`);
console.log(`[test] classifier: hidden=${stitch.classifier.hidden}, k=${stitch.classifier.k}, domains=${stitch.classifier.domains}`);

// Read each expert .bin and concatenate into flat_bytes + offsets
const buffers = stitch.experts.map(e =>
  fs.readFileSync(path.join(bundleDir, e.bin))
);
const totalSize = buffers.reduce((acc, b) => acc + b.length, 0);
const flat = new Uint8Array(totalSize);
const offsets = [0];
let cursor = 0;
for (const b of buffers) {
  flat.set(b, cursor);
  cursor += b.length;
  offsets.push(cursor);
}
console.log(`[test] total expert bytes: ${(totalSize / 1e6).toFixed(1)} MB`);
console.log(`[test] offsets: ${offsets}`);

// Construct engine
const t0 = performance.now();
const engine = new WasmStitchedEngine(flat, new Uint32Array(offsets), stitchJson);
const tLoad = performance.now() - t0;
console.log(`[test] engine constructed in ${tLoad.toFixed(0)} ms`);
console.log(`[test] num_experts=${engine.num_experts}, vocab_size=${engine.vocab_size}, expert_names=${engine.expert_names}`);

// Helper: argmax of logits
function argmax(arr) {
  let best = 0, bestV = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > bestV) { bestV = arr[i]; best = i; }
  }
  return best;
}

// Run a prompt + greedy generation
function runPrompt(label, promptTids, nGen) {
  console.log(`\n=== ${label} ===`);
  console.log(`  prompt tids: ${promptTids.join(" ")}`);
  engine.reset();
  let lastNext = 0;
  for (const tid of promptTids) {
    const step = engine.step(tid);
    const probs = step.classifier_probs;
    const probsStr = Array.from(probs).map(p => p.toFixed(3)).join(",");
    const expertName = engine.expert_names.split(",")[step.chosen_expert];
    console.log(`  PROMPT pos=${engine.position-1} tok=${tid} -> ${expertName} conf=${step.confidence.toFixed(3)} probs=[${probsStr}]`);
    if (step.warmup_happened) {
      console.log(`    (warmup: replayed ${step.warmup_tokens} tokens)`);
    }
    lastNext = argmax(step.logits);
  }
  console.log(`  -- gen ${nGen} tokens (greedy) --`);
  const generated = [];
  for (let i = 0; i < nGen; i++) {
    const step = engine.step(lastNext);
    const expertName = engine.expert_names.split(",")[step.chosen_expert];
    const probs = step.classifier_probs;
    const probsStr = Array.from(probs).map(p => p.toFixed(3)).join(",");
    console.log(`  GEN    pos=${engine.position-1} tok=${lastNext} -> ${expertName} conf=${step.confidence.toFixed(3)} probs=[${probsStr}]`);
    generated.push(lastNext);
    lastNext = argmax(step.logits);
  }
  return generated;
}

// Literary prompt: "She walked through the moonlit garden, where"
runPrompt("LITERARY", [1389, 3013, 279, 5, 3658, 17590, 3428, 4, 234], 20);

// Encyclopedic prompt: "The Battle of Hastings, fought on 14 October 1066"
runPrompt("ENCYCLOPEDIC", [127, 3999, 8, 21004, 4, 5081, 44, 7, 13, 35, 1385, 7, 13, 12, 41, 41], 20);

console.log("\n[test] done.");
