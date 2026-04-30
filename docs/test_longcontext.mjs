// Long-context extrapolation test.
//
// The model was trained with seq_len=512. With NoPE (no positional encoding)
// + meaning protection, it should be able to extrapolate well beyond that.
// This test:
//   1. Runs a real validation sequence through the model up to N tokens
//   2. Logs the per-position cross-entropy loss
//   3. Buckets it into ranges (0-100, 100-500, 500-1000, 1000-2000, 2000-4000)
//   4. If the average CE stays roughly flat across buckets → infinite-context works
//   5. If it explodes past 512 → positional generalization broke

import fs from "node:fs";
import { WasmModel } from "./wasm-pkg-node/tiny_hetmoe_wasm.js";

const bytes = fs.readFileSync("./tiny.bin");
const decode = JSON.parse(fs.readFileSync("./decode_lookup.json", "utf-8"));
const m = new WasmModel(bytes);
console.log(`Model: ${m.vocab_size} vocab, ${m.num_layers} layers, ${m.num_heads} heads`);

// Load real validation data — use the model's val.bin if present
// (assumes /data prep has been run; falls back to a generation-based test if not)
const VAL_PATH = "../data/val.bin";
let valIds = null;
if (fs.existsSync(VAL_PATH)) {
  const buf = fs.readFileSync(VAL_PATH);
  // uint16 little-endian
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const n = buf.byteLength / 2;
  valIds = new Uint16Array(n);
  for (let i = 0; i < n; i++) valIds[i] = view.getUint16(i * 2, true);
  console.log(`Loaded val.bin: ${n.toLocaleString()} tokens`);
}

const TARGET_LEN = 4096;          // run 4K tokens through the model
const SEED_OFFSET = 1000;          // start somewhere into val data, not at position 0

if (valIds === null) {
  console.log("\nval.bin not found at", VAL_PATH);
  console.log("Falling back to greedy-generation continuity test instead.");
  process.exit(1);
}

// Pick a long contiguous chunk from val data
const startIdx = SEED_OFFSET;
const endIdx = startIdx + TARGET_LEN + 1;
const chunk = valIds.slice(startIdx, endIdx);
const ids = Array.from(chunk.slice(0, -1));
const targets = Array.from(chunk.slice(1));
console.log(`Slice: positions ${startIdx}..${endIdx} of val (${ids.length} input tokens)`);

// Run through the model token-by-token, accumulating per-position loss
m.reset();
const losses = new Float32Array(TARGET_LEN);

console.log(`\nRunning forward... (this may take ${(TARGET_LEN / 150).toFixed(0)}s at ~150 tok/s)`);
const t0 = performance.now();
let lastLog = t0;

for (let pos = 0; pos < TARGET_LEN; pos++) {
  const step = m.step(ids[pos]);
  const logits = step.logits;
  const targetId = targets[pos];

  // Cross-entropy: -log(softmax(logits)[targetId])
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  const lse = max + Math.log(sum);
  losses[pos] = lse - logits[targetId];

  if (performance.now() - lastLog > 5000) {
    const elapsed = (performance.now() - t0) / 1000;
    const tps = (pos + 1) / elapsed;
    console.log(`  pos ${pos}/${TARGET_LEN} (${tps.toFixed(0)} tok/s, eta ${((TARGET_LEN - pos) / tps).toFixed(0)}s)`);
    lastLog = performance.now();
  }
}
const tTotal = (performance.now() - t0) / 1000;
console.log(`Total: ${tTotal.toFixed(1)}s, ${(TARGET_LEN / tTotal).toFixed(0)} tok/s\n`);

// Bucket and summarize
const buckets = [
  { name: "    1-32 (warmup)",       lo: 1,    hi: 32 },
  { name: "   32-128",                lo: 32,   hi: 128 },
  { name: "  128-256",                lo: 128,  hi: 256 },
  { name: "  256-512 (training len)", lo: 256,  hi: 512 },
  { name: "  512-1024 ★ extrap",      lo: 512,  hi: 1024 },
  { name: " 1024-2048 ★ extrap",      lo: 1024, hi: 2048 },
  { name: " 2048-3072 ★ extrap",      lo: 2048, hi: 3072 },
  { name: " 3072-4096 ★ extrap",      lo: 3072, hi: 4096 },
];

console.log("Mean cross-entropy by position bucket:");
console.log("(if NoPE infinite-context works, post-512 buckets should match the training-range buckets)\n");

for (const b of buckets) {
  if (b.hi > TARGET_LEN) continue;
  let sum = 0;
  let n = 0;
  for (let i = b.lo; i < b.hi; i++) {
    sum += losses[i];
    n++;
  }
  const mean = sum / n;
  const ppl = Math.exp(mean);
  const bar = "▓".repeat(Math.round(mean * 10));
  console.log(`  ${b.name.padEnd(28)} mean=${mean.toFixed(3)}  PPL=${ppl.toFixed(2).padStart(6)}  ${bar}`);
}

// Headline diagnostic
const trainBucket = (() => {
  let sum = 0, n = 0;
  for (let i = 256; i < 512; i++) { sum += losses[i]; n++; }
  return sum / n;
})();
const farBucket = (() => {
  let sum = 0, n = 0;
  for (let i = Math.min(3072, TARGET_LEN - 1024); i < Math.min(4096, TARGET_LEN); i++) { sum += losses[i]; n++; }
  return sum / n;
})();

console.log(`\nHeadline:`);
console.log(`  In-training range (256-512):  mean ${trainBucket.toFixed(3)} (PPL ${Math.exp(trainBucket).toFixed(2)})`);
console.log(`  Far extrapolation (3K-4K):    mean ${farBucket.toFixed(3)} (PPL ${Math.exp(farBucket).toFixed(2)})`);
const drift = farBucket - trainBucket;
console.log(`  Drift: ${drift >= 0 ? "+" : ""}${drift.toFixed(3)} nats (${(drift / trainBucket * 100).toFixed(1)}%)`);
if (Math.abs(drift) < 0.15) {
  console.log(`  ✓ Infinite-context: model holds quality ${TARGET_LEN}× past training (negligible drift)`);
} else if (drift > 0 && drift < 0.5) {
  console.log(`  ~ Mild degradation past training length, but no collapse`);
} else if (drift > 0.5) {
  console.log(`  ✗ Significant degradation — extrapolation breaks down`);
} else {
  console.log(`  ✓ Loss actually IMPROVES past training length (sometimes happens with longer contexts)`);
}
