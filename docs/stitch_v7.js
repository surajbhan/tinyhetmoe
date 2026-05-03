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

// Lazy fetch helper — gzip first, then raw fallback
async function fetchBin(url) {
  let resp = await fetch(`${url}.gz`);
  if (resp.ok) {
    const stream = resp.body.pipeThrough(new DecompressionStream("gzip"));
    const ab = await new Response(stream).arrayBuffer();
    return new Uint8Array(ab);
  }
  resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} -> ${resp.status}`);
  return new Uint8Array(await resp.arrayBuffer());
}

// Track in-flight + completed loads, keyed by expert idx.
const expertFetches = new Map();  // idx -> Promise<void>


async function ensureExpertLoaded(idx) {
  if (engine.is_loaded(idx)) return;
  if (expertFetches.has(idx)) return expertFetches.get(idx);

  const e = stitch.experts[idx];
  const node = document.getElementById(`ng-node-${idx}`);
  node?.classList.add("loading");
  const p = (async () => {
    const t0 = performance.now();
    const bytes = await fetchBin(`${BUNDLE_DIR}/${e.url}`);
    engine.add_expert_lazy(idx, bytes);
    const dt = performance.now() - t0;
    node?.classList.remove("loading");
    syncNodeLoadedState();
    setStatus(`Loaded expert ${e.name} (${(bytes.length/1e6).toFixed(0)} MB in ${(dt/1000).toFixed(1)}s) ` +
              `· ${engine.loaded_count}/${N_EXPERTS} experts cached`);
  })();
  expertFetches.set(idx, p);
  return p;
}

async function boot() {
  setStatus("Loading wasm runtime…");
  await init({ module_or_path: `./${BUNDLE_DIR}/tiny_hetmoe_wasm_bg.wasm` });

  setStatus("Fetching stitch bundle metadata + tokenizer + shared meaning…");
  // Light-weight fetches first: manifest + tokenizer + shared meaning
  const t0 = performance.now();
  const [stitchJson, encJson, decJson, meaningBytes] = await Promise.all([
    fetch(`${BUNDLE_DIR}/stitch.json`).then((r) => r.json()),
    fetch(`${BUNDLE_DIR}/encode.json`).then((r) => r.json()),
    fetch(`${BUNDLE_DIR}/decode.json`).then((r) => r.json()),
    fetchBin(`${BUNDLE_DIR}/meaning_shared.bin`),
  ]);
  stitch = stitchJson;
  encodeData = encJson;
  decodeTable = decJson;
  if (stitch.format_version !== "STITCHV7_001") {
    throw new Error(`unexpected stitch format ${stitch.format_version}`);
  }
  buildEncoderTables();

  // Build the engine with classifier + meaning, but ZERO experts loaded.
  setStatus(`Assembling engine (lazy: 0/${N_EXPERTS} experts loaded)…`);
  const tEng0 = performance.now();
  engine = WasmStitchedEngine.new_v7_lazy(meaningBytes, JSON.stringify(stitch));
  const tEng = performance.now() - tEng0;

  // Stickier routing: classifier needs a 50% margin to switch experts
  // mid-generation. Stops the code_py↔code_js / general↔medical flutter.
  engine.set_switch_threshold(0.5);

  // Build the distributed-node graph (SVG nodes + edges)
  buildNodeGraph();

  // Fetch the FIRST expert (general) before enabling the run button.
  // This is enough to start generating; other experts fetch on demand.
  setStatus("Fetching first expert (general) for warm start…");
  await ensureExpertLoaded(0);

  const tFetch = performance.now() - t0;
  const firstLoadMB = (meaningBytes.length / 1e6) +
    (stitch.experts[0].size_bytes / 1e6);
  $("run-btn").disabled = false;
  setStatus(
    `Ready. First-load ~${firstLoadMB.toFixed(0)} MB in ${(tFetch/1000).toFixed(1)}s. ` +
    `Other 5 experts (~57 MB each) fetch on demand. ` +
    `Vocab=${engine.vocab_size}; classifier acc=${(stitch.classifier.best_val_acc*100).toFixed(1)}%.`
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

// ─── Distributed-node graph viz ──────────────────────────────────────
//
// Six expert nodes arranged on a hexagon around a central "router".
// Per-token: router pulses → arrow flashes from router to chosen expert
// → expert node lights up while the token is generated. The flash
// duration scales with the latency-sim selector so WAN feels like an
// actual network round-trip.

const NODE_POSITIONS = (() => {
  // Hexagonal layout. Center at (360, 160), radius 110, six nodes
  // starting at angle -90° (top) going clockwise.
  const cx = 360, cy = 160, r = 110;
  const positions = [];
  for (let i = 0; i < N_EXPERTS; i++) {
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / N_EXPERTS;
    positions.push({
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
    });
  }
  return positions;
})();

const NODE_COLORS = ["#4a8fbb", "#c59c4f", "#6f9c4d", "#8b5fb4", "#c2603e", "#777777"];

let ngTokenCounts = new Array(N_EXPERTS).fill(0);

function syncNodeLoadedState() {
  // Reflect engine.is_loaded(idx) -> CSS .loaded class on each node.
  if (!engine) return;
  for (let i = 0; i < N_EXPERTS; i++) {
    const node = document.getElementById(`ng-node-${i}`);
    if (!node) continue;
    if (engine.is_loaded(i)) node.classList.add("loaded");
    else node.classList.remove("loaded");
  }
}

function buildNodeGraph() {
  const ROUTER_X = 360, ROUTER_Y = 160;
  const NODE_R = 28;

  const edgesG = document.getElementById("ng-edges");
  const dotsG = document.getElementById("ng-token-dots");
  const nodesG = document.getElementById("ng-nodes");
  if (!edgesG || !nodesG) return;

  edgesG.innerHTML = "";
  dotsG.innerHTML = "";
  nodesG.innerHTML = "";

  for (let i = 0; i < N_EXPERTS; i++) {
    const { x, y } = NODE_POSITIONS[i];

    // Edge router → node
    const edge = document.createElementNS("http://www.w3.org/2000/svg", "line");
    edge.setAttribute("x1", ROUTER_X);
    edge.setAttribute("y1", ROUTER_Y);
    edge.setAttribute("x2", x);
    edge.setAttribute("y2", y);
    edge.setAttribute("class", "ng-edge");
    edge.setAttribute("data-idx", String(i));
    edge.setAttribute("id", `ng-edge-${i}`);
    edgesG.appendChild(edge);

    // Travelling dot (hidden until flashed)
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("class", "ng-token-dot");
    dot.setAttribute("id", `ng-dot-${i}`);
    dot.setAttribute("cx", ROUTER_X);
    dot.setAttribute("cy", ROUTER_Y);
    dotsG.appendChild(dot);

    // Node group
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "ng-node");
    g.setAttribute("data-idx", String(i));
    g.setAttribute("id", `ng-node-${i}`);

    // Pulse halo (under bg)
    const pulse = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    pulse.setAttribute("class", "pulse");
    pulse.setAttribute("cx", x);
    pulse.setAttribute("cy", y);
    pulse.setAttribute("r", "26");
    g.appendChild(pulse);

    // Background circle
    const bg = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    bg.setAttribute("class", "bg");
    bg.setAttribute("cx", x);
    bg.setAttribute("cy", y);
    bg.setAttribute("r", String(NODE_R));
    g.appendChild(bg);

    // Domain label
    const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
    lbl.setAttribute("class", "label");
    lbl.setAttribute("x", x);
    lbl.setAttribute("y", y - 6);
    lbl.textContent = EXPERT_NAMES[i];
    g.appendChild(lbl);

    // Sub-tag: "node-N" (suggests separate machines)
    const sub = document.createElementNS("http://www.w3.org/2000/svg", "text");
    sub.setAttribute("class", "subtag");
    sub.setAttribute("x", x);
    sub.setAttribute("y", y + 6);
    sub.textContent = `node-${i + 1}`;
    g.appendChild(sub);

    // Token count
    const cnt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    cnt.setAttribute("class", "count");
    cnt.setAttribute("x", x);
    cnt.setAttribute("y", y + 17);
    cnt.setAttribute("id", `ng-count-${i}`);
    cnt.textContent = "0 tok";
    g.appendChild(cnt);

    nodesG.appendChild(g);
  }
  // Build the router's classifier-prob bars (6 vertical mini-bars inside
  // the central router circle). Each bar grows in height with that
  // expert's probability for the current step.
  const barsG = document.getElementById("ng-router-bars");
  if (barsG) {
    barsG.innerHTML = "";
    const BAR_W = 4;
    const BAR_GAP = 2;
    const TOTAL_W = N_EXPERTS * BAR_W + (N_EXPERTS - 1) * BAR_GAP;
    const X0 = -TOTAL_W / 2;
    const MAX_H = 22;  // max bar height (px)
    for (let i = 0; i < N_EXPERTS; i++) {
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("class", "rb");
      r.setAttribute("data-idx", String(i));
      r.setAttribute("x", String(X0 + i * (BAR_W + BAR_GAP)));
      r.setAttribute("y", "0");
      r.setAttribute("width", String(BAR_W));
      r.setAttribute("height", "1");
      r.setAttribute("fill", NODE_COLORS[i]);
      r.setAttribute("opacity", "0.85");
      r.setAttribute("rx", "1");
      barsG.appendChild(r);
    }
    // Tag for max height
    barsG.setAttribute("data-max-h", String(MAX_H));
  }
  syncNodeLoadedState();
}

function updateRouterBars(probs) {
  const barsG = document.getElementById("ng-router-bars");
  if (!barsG) return;
  const MAX_H = parseInt(barsG.getAttribute("data-max-h") || "22", 10);
  for (let i = 0; i < N_EXPERTS; i++) {
    const r = barsG.querySelector(`rect.rb[data-idx="${i}"]`);
    if (!r) continue;
    const p = Math.max(0, Math.min(1, probs[i] || 0));
    const h = Math.max(1, p * MAX_H);
    r.setAttribute("height", String(h));
    r.setAttribute("y", String(-h));  // grow upward from baseline
  }
}

function resetNodeGraph() {
  ngTokenCounts = new Array(N_EXPERTS).fill(0);
  for (let i = 0; i < N_EXPERTS; i++) {
    const node = document.getElementById(`ng-node-${i}`);
    if (node) node.classList.remove("active", "pulsing");
    const bg = node?.querySelector("circle.bg");
    if (bg) bg.classList.remove("active");
    const edge = document.getElementById(`ng-edge-${i}`);
    if (edge) edge.classList.remove("active");
    const cnt = document.getElementById(`ng-count-${i}`);
    if (cnt) cnt.textContent = "0 tok";
  }
  // Reset router bars to baseline (1px height each)
  updateRouterBars(new Array(N_EXPERTS).fill(0));
}

// Trigger the routing animation for one token.
//   expertIdx: which node to light up
//   flashMs:   how long the arrow + node pulse last (scaled by latency)
//   probs:     classifier probabilities for the central router bars
function flashRouting(expertIdx, flashMs = 200, probs = null) {
  // Update the router's live classifier-prob bars
  if (probs) updateRouterBars(probs);

  // Increment count for this expert
  ngTokenCounts[expertIdx]++;
  const cnt = document.getElementById(`ng-count-${expertIdx}`);
  if (cnt) cnt.textContent = `${ngTokenCounts[expertIdx]} tok`;

  // Clear all "active" state, then set this one
  for (let i = 0; i < N_EXPERTS; i++) {
    const node = document.getElementById(`ng-node-${i}`);
    const bg = node?.querySelector("circle.bg");
    const edge = document.getElementById(`ng-edge-${i}`);
    if (i === expertIdx) {
      bg?.classList.add("active");
      edge?.classList.add("active");
      node?.classList.add("active");
    } else {
      bg?.classList.remove("active");
      edge?.classList.remove("active");
      node?.classList.remove("active");
    }
  }

  // Re-trigger the pulse animation
  const node = document.getElementById(`ng-node-${expertIdx}`);
  if (node) {
    node.classList.remove("pulsing");
    // Force reflow so the animation restarts
    void node.offsetWidth;
    node.classList.add("pulsing");
  }

  // Travelling dot: animate position from router → node manually so we
  // can scale duration with latency. CSS keyframes only fade opacity.
  const dot = document.getElementById(`ng-dot-${expertIdx}`);
  if (dot) {
    const { x, y } = NODE_POSITIONS[expertIdx];
    dot.setAttribute("cx", "360");
    dot.setAttribute("cy", "160");
    dot.classList.remove("flying");
    void dot.getBoundingClientRect();  // reflow
    dot.classList.add("flying");
    // Animate via Web Animations API (so we can pick a duration)
    dot.animate(
      [
        { cx: "360", cy: "160", opacity: 1 },
        { cx: String(x), cy: String(y), opacity: 1 },
      ],
      { duration: Math.max(80, flashMs * 0.7), fill: "forwards" }
    );
  }
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
  resetNodeGraph();
  setStatus(`Encoded prompt → ${promptTids.length} tokens. Generating…`);

  engine.reset();
  engine.force_expert(routeMode);

  const rng = makeRng(seed);
  const routes = [];
  let lastNext = 0;

  // Latency sim drives both pacing and node-flash duration. Off=fast,
  // LAN ~5ms = brief flash, WAN ~50ms = each routing visibly travels.
  const flashMs = Math.max(120, latencyMs * 4);

  // Step helper that handles lazy-load of pending experts.
  // Peek the classifier BEFORE committing the token: if the classifier
  // wants an unloaded expert, AWAIT the fetch first, THEN do the real
  // step. This makes "first generation on a new domain" pause briefly
  // while that expert downloads — like inserting a cartridge before
  // pressing play.
  const stepWithLazyLoad = async (tid) => {
    const intended = engine.peek_expert(tid);
    if (!engine.is_loaded(intended)) {
      const name = stitch.experts[intended].name;
      setStatus(`Classifier picked ${name} — fetching expert (~57 MB) before generating…`);
      await ensureExpertLoaded(intended);
      setStatus(`${name} loaded. Resuming generation.`);
    }
    return engine.step(tid);
  };

  // Phase 1: feed prompt
  for (const tid of promptTids) {
    const step = await stepWithLazyLoad(tid);
    routes.push(step.chosen_expert);
    $("output").appendChild(tokenSpan(tid, step.chosen_expert, true));
    renderTraceRow(engine.position - 1, tid, step, true);
    flashRouting(step.chosen_expert, flashMs, Array.from(step.classifier_probs));
    if (latencyMs > 0) await sleep(latencyMs);
    lastNext = argmax(step.logits);
  }

  // Phase 2: generate
  // The classifier uses a sliding window of the last 64 meaning vectors,
  // so prompt context stays load-bearing for many generation tokens
  // without any explicit prompt-lock. Genuine drift still flips the
  // route; surface noise doesn't.
  for (let i = 0; i < nGen; i++) {
    const step = await stepWithLazyLoad(lastNext);
    routes.push(step.chosen_expert);
    $("output").appendChild(tokenSpan(lastNext, step.chosen_expert, false));
    renderTraceRow(engine.position - 1, lastNext, step, false);
    flashRouting(step.chosen_expert, flashMs, Array.from(step.classifier_probs));
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
  resetNodeGraph();
  setStatus("Reset.");
});

document.querySelectorAll('input[name="route-mode"]').forEach((el) => {
  el.addEventListener("change", () => {
    if (engine) engine.force_expert(getRouteMode());
  });
});

// Routing blend slider — wires the classifier's flat-vs-attention ratio.
const flatBlendEl = $("flat-blend");
const flatBlendValEl = $("flat-blend-val");
if (flatBlendEl) {
  flatBlendEl.addEventListener("input", () => {
    const v = parseInt(flatBlendEl.value, 10) / 100;
    if (engine) engine.set_flat_blend(v);
    if (flatBlendValEl) flatBlendValEl.textContent = `flat=${v.toFixed(2)}`;
  });
}

document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $("prompt-input").value = btn.dataset.text;
  });
});

boot().catch((e) => {
  console.error(e);
  setStatus(`Error: ${e.message || e}`, true);
});
