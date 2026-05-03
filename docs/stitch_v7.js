// stitch_v7.js — browser driver for the v7 stitched-MoE engine.
//
// Bundle layout (in stitch_v7/):
//   stitch.json            classifier weights + per-expert metadata
//   meaning_shared.bin     fp16 frozen meaning_embed (40 MB, downloaded once)
//   <domain>.bin           HTMOE004 expert (intuition fp16 + ternary, ~57 MB)
//   encode.json            byte_encoder + bpe_ranks + qwen_token_to_tiny + added_tokens
//   decode.json            tid -> raw string (for the output renderer)
//   tiny_hetmoe_wasm.js    + tiny_hetmoe_wasm_bg.wasm
//
// Boot phase:
//   - fetch stitch.json + tokenizer files
//   - fetch meaning_shared.bin + all 6 expert .bin (parallel)
//   - assemble flat byte buffer, call WasmStitchedEngine.new_v7(...)
//
// Generation:
//   - encode prompt via byte-level BPE
//   - feed each token to engine.step() — classifier picks expert, returns logits
//   - sample next token, render under expert color
//
// On-demand loading is a v8 plan; v7 ships with all 6 experts loaded
// (~393 MB total). The architecture supports lazy-load via add_expert(),
// but the wasm-bindgen API doesn't expose it yet.

import init, { WasmStitchedEngine } from "./stitch_v7/tiny_hetmoe_wasm.js";

const BUNDLE_DIR = "stitch_v7";
const EXPERT_NAMES = ["general", "thinker", "code_py", "code_js", "medical", "legal"];
const N_EXPERTS = EXPERT_NAMES.length;

const $ = (id) => document.getElementById(id);
const setStatus = (msg, isError = false) => {
  const el = $("status");
  el.textContent = msg;
  el.classList.toggle("error", isError);
};

let engine = null;
let decodeTable = null;
let encodeData = null;
let stitch = null;
let bpeCache = new Map();

// ─── Boot ────────────────────────────────────────────────────────────

async function boot() {
  setStatus("Loading wasm runtime…");
  await init({ module_or_path: `./${BUNDLE_DIR}/tiny_hetmoe_wasm_bg.wasm` });

  setStatus("Fetching stitch bundle metadata…");
  stitch = await fetch(`${BUNDLE_DIR}/stitch.json`).then((r) => r.json());
  if (stitch.format_version !== "STITCHV7_001") {
    throw new Error(`unexpected stitch format ${stitch.format_version}`);
  }

  // Fetch shared meaning + all 6 experts in parallel. We still flatten
  // into one buffer for the wasm constructor.
  setStatus("Fetching shared meaning + 6 expert .bin files in parallel…");
  const t0 = performance.now();
  const meaningP = fetch(`${BUNDLE_DIR}/${stitch.shared_meaning.url}`)
    .then((r) => r.arrayBuffer())
    .then((b) => new Uint8Array(b));
  const expertPs = stitch.experts.map((e) =>
    fetch(`${BUNDLE_DIR}/${e.url}`)
      .then((r) => r.arrayBuffer())
      .then((b) => new Uint8Array(b))
  );
  // Tokenizer in parallel too.
  const encodeP = fetch(`${BUNDLE_DIR}/encode.json`).then((r) => r.json());
  const decodeP = fetch(`${BUNDLE_DIR}/decode.json`).then((r) => r.json());

  const [meaning, ...experts] = await Promise.all([meaningP, ...expertPs]);
  encodeData = await encodeP;
  decodeTable = await decodeP;
  buildEncoderTables();

  const tFetch = performance.now() - t0;
  const totalMB = (meaning.length + experts.reduce((s, b) => s + b.length, 0)) / 1e6;
  setStatus(`Fetched ${totalMB.toFixed(1)} MB in ${(tFetch / 1000).toFixed(1)}s — assembling engine…`);

  // Build flat buffer + offsets for the wasm constructor.
  // Layout: [meaning][expert0][expert1]...[expertN-1]
  // bin_offsets length = N+2: [0, end_of_meaning, end_of_expert0, ..., end_of_expertN-1]
  const total = meaning.length + experts.reduce((s, b) => s + b.length, 0);
  const flat = new Uint8Array(total);
  const offsets = new Uint32Array(N_EXPERTS + 2);
  let cursor = 0;
  flat.set(meaning, cursor);
  offsets[0] = 0;
  cursor += meaning.length;
  offsets[1] = cursor;
  experts.forEach((b, i) => {
    flat.set(b, cursor);
    cursor += b.length;
    offsets[2 + i] = cursor;
  });

  const tEng0 = performance.now();
  engine = WasmStitchedEngine.new_v7(flat, offsets, JSON.stringify(stitch));
  const tEng = performance.now() - tEng0;

  $("run-btn").disabled = false;
  setStatus(
    `Ready. ${engine.num_experts} experts (${engine.expert_names}), ` +
    `vocab=${engine.vocab_size}. Engine built in ${tEng.toFixed(0)} ms. ` +
    `Bundle ${totalMB.toFixed(0)} MB; classifier acc=${(stitch.classifier.best_val_acc*100).toFixed(1)}% on training val.`
  );
}

// ─── BPE encoder (Qwen2.5) — same logic as v6 stitch.js ────────────────

let byteEncoder = null;
let bpeRanks = null;
let qwenTokenToTiny = null;
let addedTokens = null;
let addedTokensSplitRe = null;

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function buildEncoderTables() {
  byteEncoder = {};
  for (const [b, c] of Object.entries(encodeData.byte_encoder)) {
    byteEncoder[parseInt(b, 10)] = c;
  }
  bpeRanks = new Map();
  for (const [pair, rank] of Object.entries(encodeData.bpe_ranks)) {
    bpeRanks.set(pair, rank);
  }
  qwenTokenToTiny = encodeData.qwen_token_to_tiny;
  addedTokens = encodeData.added_tokens || [];
  if (addedTokens.length > 0) {
    const sorted = [...addedTokens].sort((a, b) => b.text.length - a.text.length);
    const pattern = sorted.map((t) => escapeRegex(t.text)).join("|");
    addedTokensSplitRe = new RegExp(pattern, "g");
  }
}

// Qwen2.5 pretokenizer regex
const QWEN_PAT =
  /'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

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
  const utf8 = new TextEncoder().encode(text);
  let out = "";
  for (const b of utf8) out += byteEncoder[b];
  return out;
}

function encodeFragment(text) {
  if (!text) return [];
  const out = [];
  for (const match of text.matchAll(QWEN_PAT)) {
    const rawPiece = match[0];
    const bePiece = byteEncode(rawPiece);
    const symbols = bpeEncode(bePiece);
    for (const s of symbols) {
      const tid = qwenTokenToTiny[s];
      if (tid !== undefined) out.push(tid);
      // else: silently skip — should never happen with full Qwen vocab
    }
  }
  return out;
}

function encodeText(text) {
  if (!addedTokensSplitRe) return encodeFragment(text);
  const out = [];
  let lastEnd = 0;
  for (const m of text.matchAll(addedTokensSplitRe)) {
    const start = m.index;
    const end = m.index + m[0].length;
    if (start > lastEnd) out.push(...encodeFragment(text.slice(lastEnd, start)));
    const at = addedTokens.find((t) => t.text === m[0]);
    if (at) out.push(at.tid);
    lastEnd = end;
  }
  if (lastEnd < text.length) out.push(...encodeFragment(text.slice(lastEnd)));
  return out;
}

function decodeOne(tid) {
  if (tid >= 0 && tid < decodeTable.length) return decodeTable[tid] || "";
  return "";
}

// ─── Sampling ────────────────────────────────────────────────────────

function makeRng(seed) {
  let s = BigInt(Math.max(1, Math.floor(seed)));
  return () => {
    s ^= s << 13n;
    s ^= s >> 7n;
    s ^= s << 17n;
    s &= 0xffffffffffffffffn;
    return Number(s & 0xffffffn) / 0x1000000;
  };
}

function argmax(arr) {
  let best = 0, v = arr[0];
  for (let i = 1; i < arr.length; i++) if (arr[i] > v) { v = arr[i]; best = i; }
  return best;
}

function sampleTopK(logits, temperature, topK, rng) {
  if (temperature <= 0) return argmax(logits);
  const idx = Array.from(logits.keys());
  idx.sort((a, b) => logits[b] - logits[a]);
  const thresh = logits[idx[Math.min(topK, idx.length - 1)]];
  const scaled = new Float32Array(logits.length);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i] >= thresh ? logits[i] / temperature : -Infinity;
    scaled[i] = v;
    if (v > max) max = v;
  }
  let sum = 0;
  for (let i = 0; i < scaled.length; i++) {
    scaled[i] = Math.exp(scaled[i] - max);
    sum += scaled[i];
  }
  const u = rng() * sum;
  let acc = 0;
  for (let i = 0; i < scaled.length; i++) {
    acc += scaled[i];
    if (acc >= u) return i;
  }
  return scaled.length - 1;
}

// ─── Token rendering ─────────────────────────────────────────────────

function tokenSpan(tid, expertIdx, isPrompt) {
  const span = document.createElement("span");
  span.className = "tok " + (isPrompt ? "prompt" : `expert-${expertIdx}`);
  const text = decodeOne(tid);
  if (text.includes("\n")) {
    span.innerHTML = text.replace(/\n/g, "<br>");
  } else {
    span.textContent = text;
  }
  span.title = `tid=${tid}${expertIdx >= 0 ? `, expert=${EXPERT_NAMES[expertIdx]}` : ""}`;
  return span;
}

function renderStats(routes) {
  const counts = new Array(N_EXPERTS).fill(0);
  routes.forEach((r) => { if (r >= 0) counts[r]++; });
  const total = routes.filter((r) => r >= 0).length;
  let longest = 0, cur = 0, prev = -1;
  for (const r of routes) {
    if (r === prev && r >= 0) cur++;
    else { if (cur > longest) longest = cur; cur = 1; prev = r; }
  }
  if (cur > longest) longest = cur;
  const $stats = $("stats");
  $stats.innerHTML = "";
  const t = document.createElement("span");
  t.textContent = `Total: ${total} tokens`;
  $stats.appendChild(t);
  for (let i = 0; i < N_EXPERTS; i++) {
    if (counts[i] === 0) continue;
    const s = document.createElement("span");
    s.style.color = ["#4a8fbb", "#c59c4f", "#6f9c4d", "#8b5fb4", "#c2603e", "#777777"][i];
    s.textContent = `${EXPERT_NAMES[i]}: ${counts[i]} (${total ? ((counts[i] / total) * 100).toFixed(0) : 0}%)`;
    $stats.appendChild(s);
  }
  const sr = document.createElement("span");
  sr.textContent = `longest run: ${longest}`;
  $stats.appendChild(sr);
}

function renderTraceRow(pos, tid, step, isPrompt) {
  const row = document.createElement("div");
  row.className = "row";
  if (step.warmup_happened) row.classList.add("warmup");
  const phase = isPrompt ? "P" : "G";
  const exp = step.chosen_expert;
  const probs = Array.from(step.classifier_probs).map((p) => p.toFixed(2)).join(",");
  row.innerHTML =
    `<span class="pos">${phase}${pos}</span>` +
    `<span class="expert-${exp}">${EXPERT_NAMES[exp]}</span>` +
    `<span class="conf">conf ${step.confidence.toFixed(3)}</span>` +
    `<span class="probs">[${probs}]${step.warmup_happened ? `  warmup +${step.warmup_tokens}` : ""}</span>`;
  $("trace").appendChild(row);
  $("trace").scrollTop = $("trace").scrollHeight;
}

// ─── Generation ──────────────────────────────────────────────────────

function getRouteMode() {
  const v = document.querySelector('input[name="route-mode"]:checked').value;
  return v === "classifier" ? -1 : parseInt(v, 10);
}

function getLatency() {
  return parseInt(document.querySelector('input[name="latency"]:checked').value, 10);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function generate() {
  if (!engine) return;
  const promptText = $("prompt-input").value;
  if (!promptText.trim()) {
    setStatus("Empty prompt.", true);
    return;
  }
  const promptTids = encodeText(promptText);
  if (promptTids.length === 0) {
    setStatus("Prompt encoded to zero tokens.", true);
    return;
  }
  const nGen = parseInt($("n-gen").value, 10);
  const temp = parseFloat($("temp").value);
  const topK = parseInt($("top-k").value, 10);
  const seed = parseInt($("seed").value, 10);
  const routeMode = getRouteMode();
  const latencyMs = getLatency();

  $("output").innerHTML = "";
  $("trace").innerHTML = "";
  $("stats").innerHTML = "";
  $("run-btn").disabled = true;
  setStatus(`Encoded prompt → ${promptTids.length} tokens. Generating…`);

  engine.reset();
  engine.force_expert(routeMode);

  const rng = makeRng(seed);
  const routes = [];
  let lastNext = 0;

  // Phase 1: feed prompt
  for (const tid of promptTids) {
    const step = engine.step(tid);
    routes.push(step.chosen_expert);
    $("output").appendChild(tokenSpan(tid, step.chosen_expert, true));
    renderTraceRow(engine.position - 1, tid, step, true);
    if (latencyMs > 0) await sleep(latencyMs);
    lastNext = argmax(step.logits);
  }

  // Phase 2: generate
  for (let i = 0; i < nGen; i++) {
    const step = engine.step(lastNext);
    routes.push(step.chosen_expert);
    $("output").appendChild(tokenSpan(lastNext, step.chosen_expert, false));
    renderTraceRow(engine.position - 1, lastNext, step, false);
    renderStats(routes);
    await sleep(Math.max(latencyMs, 1));
    lastNext = sampleTopK(step.logits, temp, topK, rng);

    // Stop early on EOS / endoftext
    if (lastNext === encodeData.specials?.endoftext) break;
  }

  setStatus(`Done. ${routes.length} tokens generated.`);
  $("run-btn").disabled = false;
}

// ─── Wire up ─────────────────────────────────────────────────────────

$("run-btn").addEventListener("click", generate);
$("reset-btn").addEventListener("click", () => {
  if (engine) engine.reset();
  $("output").innerHTML = "";
  $("trace").innerHTML = "";
  $("stats").innerHTML = "";
  setStatus("Reset.");
});

document.querySelectorAll('input[name="route-mode"]').forEach((el) => {
  el.addEventListener("change", () => {
    if (engine) engine.force_expert(getRouteMode());
  });
});

document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $("prompt-input").value = btn.dataset.text;
  });
});

boot().catch((e) => {
  console.error(e);
  setStatus(`Error: ${e.message || e}`, true);
});
