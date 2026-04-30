"use strict";

import init, { WasmModel } from "./wasm-pkg/tiny_hetmoe_wasm.js";

// ── State ────────────────────────────────────────────────────────────
let model = null;
let modelMeta = null;
let decode = null;          // tiny_id → string
let encodeData = null;      // GPT-2 BPE encoder data
let axisNames = null;
let axisDescriptions = null;  // [{name, short, pos, neg, category}, ...]
const SPECIALS_REV = ["<unk>", "<bos>", "<eos>", "<pad>"];
const EXPERT_NAMES = ["Standard", "SwiGLU", "DeepNarrow", "Bottleneck"];

// Each entry: { tiny_id, str, is_special, is_prompt, trace }
let history = [];
let inspectPos = 0;            // which token is currently selected for the inspect panels

// Rolling-window TPS measurement. Holds the last N step durations (ms).
const TPS_WINDOW = 12;
let stepTimings = [];

function recordStepTime(ms) {
  stepTimings.push(ms);
  if (stepTimings.length > TPS_WINDOW) stepTimings.shift();
  updateTpsChip();
}

function updateTpsChip() {
  const el = document.getElementById("tps-val");
  if (!el) return;
  if (stepTimings.length === 0) {
    el.textContent = "—";
    return;
  }
  const avgMs = stepTimings.reduce((a, b) => a + b, 0) / stepTimings.length;
  const tps = 1000 / avgMs;
  el.textContent = tps.toFixed(0);
}

// ── Boot ─────────────────────────────────────────────────────────────
async function boot() {
  setMeta("Loading WASM module…");
  await init();

  setMeta("Loading vocab + axis names + encoder…");
  const [decodeRes, axisRes, metaRes, encodeRes, axisDescRes] = await Promise.all([
    fetch("decode_lookup.json").then(r => r.json()),
    fetch("meaning_axis_names.json").then(r => r.json()),
    fetch("tiny.meta.json").then(r => r.json()),
    fetch("encode_lookup.json").then(r => r.json()),
    fetch("axis_descriptions.json").then(r => r.json()),
  ]);
  decode = decodeRes;
  axisNames = axisRes;
  modelMeta = metaRes;
  encodeData = encodeRes;
  axisDescriptions = axisDescRes;
  buildEncoderTables();

  // First do a HEAD request to get the actual size before streaming.
  let knownSize = 0;
  try {
    const head = await fetch("tiny.bin", { method: "HEAD" });
    knownSize = parseInt(head.headers.get("Content-Length") || "0", 10);
  } catch (e) { /* fall back to 0; we'll show "X MB" without a denominator */ }
  const sizeStr = knownSize > 0 ? `${(knownSize / 1e6).toFixed(1)} MB` : "model weights";
  setMeta(`Loading ${sizeStr}…`);

  const binRes = await fetch("tiny.bin");
  const total = parseInt(binRes.headers.get("Content-Length") || String(knownSize), 10);
  const reader = binRes.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) {
      setMeta(`Downloading: ${(received / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB…`);
    } else {
      setMeta(`Downloading: ${(received / 1e6).toFixed(1)} MB…`);
    }
  }
  const blob = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { blob.set(c, off); off += c.length; }
  // Remember actual model size for later display
  modelMeta._download_bytes = received;

  setMeta("Building model in WASM…");
  const t0 = performance.now();
  model = new WasmModel(blob);
  const tLoad = performance.now() - t0;
  const actualMB = (received / 1e6).toFixed(1);
  setMeta(
    `${actualMB} MB model · ${model.vocab_size} vocab, ${model.num_layers}L × ` +
    `${model.num_heads}H, ${model.num_experts} experts. ` +
    `Trained at step ${modelMeta.training_step}, val ${modelMeta.best_val.toFixed(4)} ` +
    `(PPL ${Math.exp(modelMeta.best_val).toFixed(1)}). Load ${tLoad.toFixed(0)}ms.`
  );

  document.getElementById("ckpt-step").textContent = modelMeta.training_step;
  document.getElementById("ckpt-val").textContent = modelMeta.best_val.toFixed(4);
  document.getElementById("ckpt-ppl").textContent = Math.exp(modelMeta.best_val).toFixed(1);
  document.getElementById("qat-status").textContent = "(active)";

  setupControls();
  await runPrompt();
}

function setMeta(s) {
  const el = document.getElementById("meta");
  if (el) el.textContent = s;
  console.log("[boot]", s);
}

function setupControls() {
  document.getElementById("run-btn").addEventListener("click", runPrompt);
  document.getElementById("reset-btn").addEventListener("click", runPrompt);
  document.getElementById("auto-btn").addEventListener("click", () => autoComplete());
  document.getElementById("step-btn").addEventListener("click", () => autoComplete(1));
  const promptInput = document.getElementById("prompt-input");
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runPrompt();
  });

  // Wire slider→label sync
  const sliders = [
    { id: "cfg-temp",  fmt: v => parseFloat(v).toFixed(2) },
    { id: "cfg-topk",  fmt: v => v },
    { id: "cfg-rep",   fmt: v => parseFloat(v).toFixed(2) },
    { id: "cfg-win",   fmt: v => v },
    { id: "cfg-autoN", fmt: v => v },
  ];
  for (const s of sliders) {
    const slider = document.getElementById(s.id);
    const label = document.getElementById(s.id + "-val");
    const sync = () => { label.textContent = s.fmt(slider.value); };
    slider.addEventListener("input", sync);
    sync();
  }

  // When mode is "greedy", grey out temp/top-K (they don't apply)
  const modeSel = document.getElementById("strategy");
  const updateGrey = () => {
    const isGreedy = modeSel.value === "greedy";
    document.getElementById("cfg-temp").disabled = isGreedy;
    document.getElementById("cfg-topk").disabled = isGreedy;
  };
  modeSel.addEventListener("change", updateGrey);
  updateGrey();

  // Sync auto-N label into the button text so it reflects the slider
  const autoBtn = document.getElementById("auto-btn");
  const autoNSlider = document.getElementById("cfg-autoN");
  const syncAutoBtn = () => { autoBtn.textContent = `Auto+${autoNSlider.value}`; };
  autoNSlider.addEventListener("input", syncAutoBtn);
  syncAutoBtn();
}

// ── GPT-2 BPE encoder (port to JS) ───────────────────────────────────
//
// Steps to encode raw UTF-8 text into tiny token ids:
// 1. UTF-8 encode → bytes
// 2. Map each byte through `byte_encoder` to a printable Unicode char
// 3. Tokenize the result with regex (GPT-2 pattern); for each match:
//    a. apply BPE merges greedily until no merge applies
//    b. look up each resulting symbol in gpt2_token_to_id
// 4. Map gpt2_id → tiny_id via gpt2_to_tiny (or <unk>)

let byteEncoder = null;     // {byte_int: unicode_char_string}
let bpeRanks = null;        // Map<"a b", rank>
let gpt2TokenToId = null;
let gpt2ToTiny = null;

function buildEncoderTables() {
  byteEncoder = {};
  for (const [b, c] of Object.entries(encodeData.byte_encoder)) {
    byteEncoder[parseInt(b, 10)] = c;
  }
  bpeRanks = new Map();
  for (const [pair, rank] of Object.entries(encodeData.bpe_ranks)) {
    bpeRanks.set(pair, rank);
  }
  gpt2TokenToId = encodeData.gpt2_token_to_id;
  gpt2ToTiny = encodeData.gpt2_to_tiny;
}

// GPT-2 pretokenizer regex
// (matches the official tokenizer's pre_tokenize step)
const GPT2_PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function bpeEncode(text) {
  // Convert each character (which is already in byte-encoded space) into
  // its symbol list, then apply merges greedily.
  let symbols = Array.from(text);  // unicode codepoints (but our text is byte-encoded so ASCII-ish)
  while (true) {
    let bestPair = null, bestRank = Infinity, bestIdx = -1;
    for (let i = 0; i < symbols.length - 1; i++) {
      const pair = symbols[i] + " " + symbols[i + 1];
      const r = bpeRanks.get(pair);
      if (r !== undefined && r < bestRank) {
        bestRank = r;
        bestPair = [symbols[i], symbols[i + 1]];
        bestIdx = i;
      }
    }
    if (bestPair === null) break;
    // Merge all non-overlapping occurrences of this pair (left-to-right)
    const newSymbols = [];
    let i = 0;
    while (i < symbols.length) {
      if (i < symbols.length - 1 && symbols[i] === bestPair[0] && symbols[i + 1] === bestPair[1]) {
        newSymbols.push(bestPair[0] + bestPair[1]);
        i += 2;
      } else {
        newSymbols.push(symbols[i]);
        i++;
      }
    }
    symbols = newSymbols;
  }
  return symbols;
}

function encodeText(text) {
  // Returns array of tiny ids (no <bos> prefix)
  const out = [];
  const utf8 = new TextEncoder().encode(text);
  // Byte-encode the whole utf8
  let byteEncoded = "";
  for (const b of utf8) byteEncoded += byteEncoder[b];
  // Apply pretokenizer regex on the byte-encoded text
  for (const match of byteEncoded.matchAll(GPT2_PAT)) {
    const piece = match[0];
    const symbols = bpeEncode(piece);
    for (const s of symbols) {
      const gid = gpt2TokenToId[s];
      if (gid === undefined) {
        out.push(encodeData.specials["<unk>"]);
      } else {
        const tid = gpt2ToTiny[String(gid)];
        if (tid === undefined) {
          out.push(encodeData.specials["<unk>"]);
        } else {
          out.push(tid);
        }
      }
    }
  }
  return out;
}

function decodeToken(tinyId) {
  if (tinyId < 4) return SPECIALS_REV[tinyId] || `<${tinyId}>`;
  return decode[String(tinyId)] ?? `[${tinyId}]`;
}

// ── Run prompt (resets state, processes prompt, shows top-K) ─────────
async function runPrompt() {
  if (!model) return;
  const text = document.getElementById("prompt-input").value;
  const ids = [encodeData.specials["<bos>"], ...encodeText(text)];
  console.log(`[runPrompt] ${ids.length} tokens:`, ids.map(decodeToken).join("|"));

  model.reset();
  history = [];
  stepTimings = [];  // reset TPS rolling window when resetting story
  for (let i = 0; i < ids.length; i++) {
    const tid = ids[i];
    const t0 = performance.now();
    const step = model.step(tid);
    recordStepTime(performance.now() - t0);
    history.push({
      tiny_id: tid,
      str: decodeToken(tid),
      is_special: tid < 4,
      is_prompt: true,
      trace: stepToTrace(step),
    });
  }

  trimLogitsHistory(1);
  inspectPos = history.length - 1;
  renderAll();
}

function appendToken(tid) {
  const t0 = performance.now();
  const step = model.step(tid);
  recordStepTime(performance.now() - t0);
  history.push({
    tiny_id: tid,
    str: decodeToken(tid),
    is_special: tid < 4,
    is_prompt: false,
    trace: stepToTrace(step),
  });
  trimLogitsHistory(1);
  inspectPos = history.length - 1;
  renderAll();
}

// Read sampler config from the UI controls.
function readSamplerConfig() {
  return {
    strategy: document.getElementById("strategy").value,
    temperature: parseFloat(document.getElementById("cfg-temp").value),
    topK: parseInt(document.getElementById("cfg-topk").value, 10),
    repetitionPenalty: parseFloat(document.getElementById("cfg-rep").value),
    repetitionWindow: parseInt(document.getElementById("cfg-win").value, 10),
    autoN: parseInt(document.getElementById("cfg-autoN").value, 10),
  };
}

// Auto-complete with repetition penalty + temperature + top-K sampling.
// Uses the full logits vector so any recently-seen token in the vocab can
// be penalized, not just those in the visible top-K.
function autoComplete(n) {
  if (!model || history.length === 0) return;
  const cfg = readSamplerConfig();
  const stepCount = (n === undefined) ? cfg.autoN : n;

  for (let k = 0; k < stepCount; k++) {
    const last = history[history.length - 1];
    const fullLogits = last.trace.logits;
    if (!fullLogits) {
      console.warn("[autoComplete] missing full logits");
      appendToken(last.trace.topK[0].tiny_id);
      continue;
    }

    // Recent tokens window (for repetition penalty)
    const recent = new Set();
    if (cfg.repetitionPenalty > 1.0001 && cfg.repetitionWindow > 0) {
      const start = Math.max(0, history.length - cfg.repetitionWindow);
      for (let i = start; i < history.length; i++) {
        const tok = history[i];
        if (!tok.is_special) recent.add(tok.tiny_id);
      }
    }

    // Apply repetition penalty (HF-style: divide if positive, multiply if negative)
    const adj = new Float32Array(fullLogits.length);
    const p = cfg.repetitionPenalty;
    for (let i = 0; i < fullLogits.length; i++) {
      const v = fullLogits[i];
      if (recent.has(i)) {
        adj[i] = v > 0 ? v / p : v * p;
      } else {
        adj[i] = v;
      }
    }

    let pick;
    if (cfg.strategy === "greedy") {
      let bestI = 0, bestV = -Infinity;
      for (let i = 0; i < adj.length; i++) {
        if (adj[i] > bestV) { bestV = adj[i]; bestI = i; }
      }
      pick = bestI;
    } else {
      // Top-K sampling on adjusted logits with temperature
      const TOP = Math.min(cfg.topK, adj.length);
      const idxs = Array.from({ length: adj.length }, (_, i) => i);
      idxs.sort((a, b) => adj[b] - adj[a]);
      const sorted = idxs.slice(0, TOP);
      const T = Math.max(0.05, cfg.temperature);
      const max = adj[sorted[0]];
      const expArr = sorted.map(i => Math.exp((adj[i] - max) / T));
      const sum = expArr.reduce((a, b) => a + b, 0);
      const r = Math.random() * sum;
      let acc = 0;
      pick = sorted[0];
      for (let i = 0; i < sorted.length; i++) {
        acc += expArr[i];
        if (r < acc) { pick = sorted[i]; break; }
      }
    }
    appendToken(pick);
  }
}

// ── Convert WasmStep to the structure renderers expect ───────────────
function stepToTrace(step) {
  const numLayers = model.num_layers;
  const numHeads = model.num_heads;
  const numExperts = model.num_experts;

  const attnFlat = step.attn_flat;
  const attnLengths = step.attn_lengths;
  const attn_per_layer = [];
  let off = 0;
  for (let l = 0; l < numLayers; l++) {
    const by_head = [];
    for (let h = 0; h < numHeads; h++) {
      const len = attnLengths[l * numHeads + h];
      by_head.push(Array.from(attnFlat.slice(off, off + len)));
      off += len;
    }
    attn_per_layer.push({ by_head });
  }

  const route_per_layer = [];
  const routingFlat = step.routing_flat;
  for (let l = 0; l < numLayers; l++) {
    route_per_layer.push(Array.from(routingFlat.slice(l * numExperts, (l + 1) * numExperts)));
  }

  // Top-8 next-token by raw softmax (no temperature here — temperature
  // is applied at sampling time in autoComplete; the displayed probs
  // are the model's actual posterior).
  const logits = step.logits;
  const max = logits.reduce((a, b) => Math.max(a, b), -Infinity);
  let sumExp = 0;
  const expArr = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const v = Math.exp(logits[i] - max);
    expArr[i] = v;
    sumExp += v;
  }
  const ranked = [];
  for (let i = 0; i < logits.length; i++) {
    ranked.push({ tiny_id: i, prob: expArr[i] / sumExp });
  }
  ranked.sort((a, b) => b.prob - a.prob);
  const topK = ranked.slice(0, 8).map(r => ({
    tiny_id: r.tiny_id,
    prob: r.prob,
    str: decodeToken(r.tiny_id),
    is_special: r.tiny_id < 4,
  }));

  return {
    meaning: Array.from(step.meaning),
    intuition: Array.from(step.intuition),
    hidden: Array.from(step.hidden),
    attn_per_layer,
    route_per_layer,
    topK,
    // Full logits (needed by autoComplete for repetition penalty over the
    // whole vocab). Float32Array view of the WASM buffer.
    logits: Array.from(logits),
  };
}

// Free older history entries' full logits to keep memory bounded.
// Called after each appendToken(). Keeps only the most recent N entries
// with logits (older ones drop the field; topK is still kept).
function trimLogitsHistory(keepN = 1) {
  if (history.length <= keepN) return;
  for (let i = 0; i < history.length - keepN; i++) {
    if (history[i].trace.logits) delete history[i].trace.logits;
  }
}

// ── Renderers ────────────────────────────────────────────────────────
function renderAll() {
  renderStory();
  renderTopK();
  renderInspect();
}

function renderStory() {
  const el = document.getElementById("story");
  el.innerHTML = "";
  history.forEach((tok, i) => {
    const span = document.createElement("span");
    span.className = "tok " + (tok.is_special ? "special" : (tok.is_prompt ? "prompt" : "gen"));
    if (i === inspectPos) span.classList.add("selected");
    span.textContent = tok.str;
    span.dataset.pos = i;
    span.addEventListener("click", () => {
      inspectPos = i;
      renderStory();
      renderInspect();
    });
    el.appendChild(span);
  });
}

function renderTopK() {
  const el = document.getElementById("topk");
  el.innerHTML = "";
  // The top-K we show is for the END of the story (the next token to pick),
  // which is the trace of the LAST history entry.
  if (history.length === 0) return;
  const last = history[history.length - 1];
  const topK = last.trace.topK;
  topK.forEach((entry, i) => {
    const row = document.createElement("button");
    row.className = "topk-row clickable";
    row.innerHTML = `
      <span class="topk-rank">${i + 1}.</span>
      <span class="topk-tok">${escapeHtml(entry.str)}</span>
      <span class="topk-bar"><span class="topk-bar-fill" style="width:${(entry.prob * 100).toFixed(1)}%"></span></span>
      <span class="topk-prob">${(entry.prob * 100).toFixed(1)}%</span>
    `;
    row.addEventListener("click", () => appendToken(entry.tiny_id));
    el.appendChild(row);
  });
}

function renderInspect() {
  const tok = history[inspectPos];
  document.getElementById("topk-pos").textContent =
    `position ${inspectPos} (token "${tok.str.replace(/^\s/, "·")}")`;
  renderExplain();
  renderMeaning();
  renderAttn();
  renderRoute();
}

// ── "What the model sees" — synthesize plain-language reading ─────
//
// Inputs:
//   - meaning vector (132 floats, per-axis activation)
//   - attention per layer per head (rows of variable length)
//   - routing per layer (probs over 4 experts)
//   - top-K next tokens
//   - the token itself (string + position)
// Output: a short HTML paragraph mixing English with subtle markup
// for the activated concepts.
function renderExplain() {
  const el = document.getElementById("explain");
  el.innerHTML = "";
  const tok = history[inspectPos];
  const trace = tok.trace;

  // ── Extract top axes (positive and negative separately) ──
  const m = trace.meaning;
  const ranked = m.map((v, i) => ({ v, i })).filter(r => Math.abs(r.v) > 0.5);
  ranked.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const top = ranked.slice(0, 6);
  const topPos = top.filter(r => r.v > 0).slice(0, 3);
  const topNeg = top.filter(r => r.v < 0).slice(0, 2);

  // ── Detect strongest attention target (last layer, max across heads) ──
  let attnTarget = null;
  if (inspectPos > 0) {
    const lastL = trace.attn_per_layer[trace.attn_per_layer.length - 1];
    let bestScore = 0;
    let bestPos = -1;
    for (const headRow of lastL.by_head) {
      for (let j = 0; j < headRow.length - 1; j++) {  // skip self (last position)
        if (headRow[j] > bestScore) {
          bestScore = headRow[j];
          bestPos = j;
        }
      }
    }
    if (bestPos >= 0 && bestScore > 0.15 && bestPos < history.length) {
      attnTarget = { pos: bestPos, str: history[bestPos].str, score: bestScore };
    }
  }

  // ── Determine which experts ran (top-2 in last layer) ──
  const lastRouting = trace.route_per_layer[trace.route_per_layer.length - 1];
  const expRanked = lastRouting.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
  const topExperts = expRanked.slice(0, 2);

  // ── Top-K next predictions ──
  const topKNext = trace.topK.slice(0, 5);

  // ── Build paragraphs ──
  const paragraphs = [];

  // Paragraph 1: what this token is about
  const tokDisplay = tok.is_special
    ? `<i>${escapeHtml(tok.str)}</i>`
    : `"${escapeHtml(tok.str.replace(/^ /, " "))}"`;
  if (topPos.length === 0 && topNeg.length === 0) {
    paragraphs.push(
      `The token ${tokDisplay} has very little semantic content — ` +
      `mostly a separator or function word.`
    );
  } else {
    // Positive axes are the "is about" half. Use only the strongest 2-3
    // and join with "and" for natural prose.
    const posPills = topPos.map(r =>
      `<span class="axis-pill pos">${escapeHtml(axisDescriptions[r.i].short)}</span>`
    );
    let posPhrase = "";
    if (posPills.length === 1) posPhrase = posPills[0];
    else if (posPills.length === 2) posPhrase = `${posPills[0]} and ${posPills[1]}`;
    else if (posPills.length >= 3) posPhrase = `${posPills.slice(0, -1).join(", ")}, and ${posPills[posPills.length - 1]}`;

    // Negative axes are "is NOT about" — single best one only,
    // phrased without double-negation (axis "neg" field is correctly
    // singular, but "not [neg-form]" reads awkwardly. Use the axis's
    // *positive* short form prefixed by "not" instead).
    let negPhrase = "";
    if (topNeg.length > 0) {
      const r = topNeg[0];  // strongest negative
      negPhrase = `<span class="axis-pill neg">not ${escapeHtml(axisDescriptions[r.i].short)}</span>`;
    }

    let p1 = `The model reads ${tokDisplay} as `;
    if (posPhrase && negPhrase) {
      p1 += `${posPhrase} — and ${negPhrase}.`;
    } else if (posPhrase) {
      p1 += `${posPhrase}.`;
    } else {
      p1 += `${negPhrase}.`;
    }
    paragraphs.push(p1);
  }

  // Paragraph 2: attention narrative
  if (attnTarget && inspectPos > 0) {
    paragraphs.push(
      `Looking back at the prefix, attention focuses on ` +
      `<span class="attn-mention">"${escapeHtml(attnTarget.str.replace(/^ /, " "))}"</span> ` +
      `(${(attnTarget.score * 100).toFixed(0)}% in the last layer).`
    );
  }

  // Paragraph 3: expert routing in plain language
  const expertSpecialty = {
    "Standard": "the simple feed-forward expert",
    "SwiGLU": "the gated multiplicative expert",
    "DeepNarrow": "the deeper four-layer expert",
    "Bottleneck": "the compress-then-expand expert",
  };
  const e0 = EXPERT_NAMES[topExperts[0].i];
  const e1 = EXPERT_NAMES[topExperts[1].i];
  paragraphs.push(
    `It routes through ` +
    `<span class="expert-mention">${escapeHtml(e0)}</span> ` +
    `(${(topExperts[0].p * 100).toFixed(0)}%) and ` +
    `<span class="expert-mention">${escapeHtml(e1)}</span> ` +
    `(${(topExperts[1].p * 100).toFixed(0)}%) in the last layer.`
  );

  // Paragraph 4: prediction
  if (topKNext.length > 0 && !tok.is_special) {
    const top1 = topKNext[0];
    // Skip top-1 for the "alternatives" list to avoid repetition
    const alts = topKNext.slice(1, 4).map(t =>
      `"${escapeHtml(t.str.replace(/^ /, " "))}"`
    ).join(", ");
    let p = `The model expects the next token to most likely be ` +
      `<span class="next-mention">"${escapeHtml(top1.str.replace(/^ /, " "))}"</span> ` +
      `(${(top1.prob * 100).toFixed(0)}%)`;
    if (alts) p += `, with ${alts} as runners-up`;
    p += `.`;
    paragraphs.push(p);
  }

  // Render
  for (const p of paragraphs) {
    const div = document.createElement("p");
    div.className = "explain-para";
    div.innerHTML = p;
    el.appendChild(div);
  }
}

function renderMeaning() {
  const el = document.getElementById("meaning");
  el.innerHTML = "";
  const m = history[inspectPos].trace.meaning;
  const ranked = m.map((v, i) => ({ v, i, abs: Math.abs(v) }))
                   .sort((a, b) => b.abs - a.abs)
                   .slice(0, 24);
  const maxAbs = Math.max(...ranked.map((r) => r.abs));
  ranked.forEach((r) => {
    const row = document.createElement("div");
    row.className = "axis-row";
    const widthPct = Math.min(50, (r.abs / maxAbs) * 50);
    const sideClass = r.v >= 0 ? "pos" : "neg";
    const valStr = (r.v >= 0 ? "+" : "") + r.v.toFixed(2);
    row.innerHTML = `
      <span class="axis-name">${escapeHtml(axisNames[r.i])}</span>
      <span class="axis-bar-container">
        <span class="axis-bar-mid"></span>
        <span class="axis-bar ${sideClass}" style="width:${widthPct}%"></span>
      </span>
      <span class="axis-val">${valStr}</span>
    `;
    el.appendChild(row);
  });
}

function renderAttn() {
  const el = document.getElementById("attn");
  el.innerHTML = "";
  const layers = history[inspectPos].trace.attn_per_layer;
  const nLayers = layers.length;
  const nHeads = layers[0].by_head.length;

  const corner = document.createElement("div");
  corner.className = "attn-corner";
  corner.textContent = "L\\H";
  el.appendChild(corner);
  for (let h = 0; h < nHeads; h++) {
    const lbl = document.createElement("div");
    lbl.className = "attn-head-label";
    lbl.textContent = `H${h}`;
    el.appendChild(lbl);
  }
  for (let l = 0; l < nLayers; l++) {
    const lblRow = document.createElement("div");
    lblRow.className = "attn-layer-label";
    lblRow.textContent = `L${l}`;
    el.appendChild(lblRow);
    for (let h = 0; h < nHeads; h++) {
      const cell = document.createElement("div");
      cell.className = "attn-cell";
      cell.title = `layer ${l}, head ${h}`;
      const row = layers[l].by_head[h];
      const max = Math.max(...row, 0.001);
      row.forEach((a, j) => {
        const sq = document.createElement("span");
        sq.className = "attn-square";
        const intensity = Math.min(1.0, a / max);
        const r = Math.round(255 - 220 * intensity);
        const g = Math.round(255 - 180 * intensity);
        const b = Math.round(255 - 80 * intensity);
        sq.style.background = `rgb(${r},${g},${b})`;
        sq.title = `attn → "${history[j]?.str || j}" = ${(a * 100).toFixed(1)}%`;
        cell.appendChild(sq);
      });
      el.appendChild(cell);
    }
  }
}

function renderRoute() {
  const el = document.getElementById("route");
  el.innerHTML = "";
  const routes = history[inspectPos].trace.route_per_layer;
  const topK = 2;
  routes.forEach((probs, l) => {
    const ranked = probs.map((p, i) => ({ p, i }))
                         .sort((a, b) => b.p - a.p);
    const topSet = new Set(ranked.slice(0, topK).map((r) => r.i));
    const row = document.createElement("div");
    row.className = "route-row";
    const layer = document.createElement("div");
    layer.className = "route-layer";
    layer.textContent = `L${l}`;
    row.appendChild(layer);
    const bars = document.createElement("div");
    bars.className = "route-bars";
    probs.forEach((p, i) => {
      const exp = document.createElement("div");
      exp.className = "route-expert";
      const isRouted = topSet.has(i);
      exp.innerHTML = `
        <span class="route-expert-name">${escapeHtml(EXPERT_NAMES[i])}${isRouted ? " ✓" : ""}</span>
        <div class="route-expert-bar">
          <div class="route-expert-bar-fill ${isRouted ? "routed" : ""}" style="width:${(p * 100).toFixed(1)}%"></div>
          <span class="route-expert-prob">${(p * 100).toFixed(1)}%</span>
        </div>
      `;
      bars.appendChild(exp);
    });
    row.appendChild(bars);
    el.appendChild(row);
  });
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

window.addEventListener("DOMContentLoaded", boot);
