// stitch.js — browser driver for the wasm stitched-MoE engine.
//
// Loads the stitch bundle (4 expert .bin files + stitch.json + decode.json
// + encode.json), constructs the WasmStitchedEngine, and runs token-by-
// token generation with per-token routing visible in the UI.

import init, { WasmStitchedEngine } from "./wasm-pkg/tiny_hetmoe_wasm.js";

const BUNDLE_DIR = "stitch_v66";

const $ = (id) => document.getElementById(id);
const setStatus = (msg, isError = false) => {
  const el = $("status");
  el.textContent = msg;
  el.classList.toggle("error", isError);
};

const EXPERT_NAMES = ["pg19", "wiki", "tool", "code"];

let engine = null;
let decodeTable = null;
let encodeData = null;
let stitch = null;
let bpeCache = new Map();

// ─── Boot ────────────────────────────────────────────────────────────

async function boot() {
  setStatus("Loading wasm runtime…");
  await init({ module_or_path: "./wasm-pkg/tiny_hetmoe_wasm_bg.wasm" });

  setStatus("Fetching stitch bundle metadata…");
  const stitchRes = await fetch(`${BUNDLE_DIR}/stitch.json`);
  stitch = await stitchRes.json();
  setStatus(`Bundle: ${stitch.format_version}, ${stitch.experts.length} experts: ${stitch.experts.map(e => e.name).join(", ")}`);

  setStatus("Loading expert .bin.gz files (gzip-decompressed in-stream)…");
  const t0 = performance.now();
  const bufs = await Promise.all(
    stitch.experts.map(async (e) => {
      // Try gzipped first; fall back to raw if missing. The browser's
      // built-in DecompressionStream("gzip") inflates as bytes arrive.
      let resp = await fetch(`${BUNDLE_DIR}/${e.bin}.gz`);
      if (resp.ok) {
        const stream = resp.body.pipeThrough(new DecompressionStream("gzip"));
        const ab = await new Response(stream).arrayBuffer();
        return new Uint8Array(ab);
      }
      resp = await fetch(`${BUNDLE_DIR}/${e.bin}`);
      const ab = await resp.arrayBuffer();
      return new Uint8Array(ab);
    })
  );
  const tFetch = performance.now() - t0;
  const totalMB = bufs.reduce((s, b) => s + b.length, 0) / 1e6;
  setStatus(`Inflated ${totalMB.toFixed(1)} MB in ${(tFetch / 1000).toFixed(1)}s, building engine…`);

  // Flatten + offsets
  const totalSize = bufs.reduce((s, b) => s + b.length, 0);
  const flat = new Uint8Array(totalSize);
  const offsets = new Uint32Array(bufs.length + 1);
  let cursor = 0;
  bufs.forEach((b, i) => {
    flat.set(b, cursor);
    offsets[i] = cursor;
    cursor += b.length;
  });
  offsets[bufs.length] = cursor;

  const tEngineStart = performance.now();
  engine = new WasmStitchedEngine(flat, offsets, JSON.stringify(stitch));
  const tEngine = performance.now() - tEngineStart;

  setStatus("Loading tokenizer (decode + encode tables)…");
  const [decodeRes, encodeRes] = await Promise.all([
    fetch(`${BUNDLE_DIR}/decode.json`).then((r) => r.json()),
    fetch(`${BUNDLE_DIR}/encode.json`).then((r) => r.json()),
  ]);
  decodeTable = decodeRes;
  encodeData = encodeRes;
  buildEncoderTables();

  $("run-btn").disabled = false;

  setStatus(
    `Ready. ${engine.num_experts} experts (${engine.expert_names}), ` +
    `vocab=${engine.vocab_size}. Engine built in ${tEngine.toFixed(0)} ms. ` +
    `Tokenizer: ${Object.keys(encodeData.qwen_token_to_tiny).length} BPE tokens, ` +
    `${bpeRanks.size} merge rules.`
  );
}

// ─── BPE encoder (Qwen2.5) ──────────────────────────────────────────
//
// Pipeline: pretokenizer (raw text) → byte-encode each pretoken
// → BPE merges per pretoken → vocab lookup → tiny ids.
// Special tokens (<|im_start|>, etc) are split out atomically BEFORE
// any of this so they round-trip as their own tids.

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

// Qwen2.5 pretokenizer regex (verbatim from tokenizer.json's Split rule).
// JS doesn't have inline (?i:) so we use char classes for the
// case-insensitive contraction alternatives.
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
  // Pretokenizer on RAW text → byte-encode each pretoken → BPE.
  if (!text) return [];
  const out = [];
  for (const match of text.matchAll(QWEN_PAT)) {
    const rawPiece = match[0];
    const bePiece = byteEncode(rawPiece);
    const symbols = bpeEncode(bePiece);
    for (const s of symbols) {
      const tid = qwenTokenToTiny[s];
      out.push(tid !== undefined ? tid : encodeData.specials["<unk>"]);
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
    out.push(at ? at.tid : encodeData.specials["<unk>"]);
    lastEnd = end;
  }
  if (lastEnd < text.length) out.push(...encodeFragment(text.slice(lastEnd)));
  return out;
}

function decodeTokens(tids) {
  return tids
    .map((t) => (t > 3 && t < decodeTable.length ? decodeTable[t] : ""))
    .join("");
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
  let best = 0,
    v = arr[0];
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
  let text = decodeTable[tid] || "";
  if (text.includes("\n")) {
    span.innerHTML = text.replace(/\n/g, "<br>");
  } else {
    span.textContent = text;
  }
  span.title = `tid=${tid}${expertIdx >= 0 ? `, expert=${EXPERT_NAMES[expertIdx]}` : ""}`;
  return span;
}

function renderStats(routes) {
  const counts = [0, 0, 0, 0];
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
  for (let i = 0; i < 4; i++) {
    const s = document.createElement("span");
    s.className = `stat-${EXPERT_NAMES[i]}`;
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
  }

  setStatus(`Done. ${routes.length} tokens.`);
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
