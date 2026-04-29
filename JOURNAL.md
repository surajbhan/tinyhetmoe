# TinyHetMoE — Engineering Journal

Daily-ish log of what was tried, what worked, what blocked. Updated at
each gate (not just end-of-day). New entries on top.

---

## 2026-04-29 22:00 — Switched to plateau-triggered QAT mid-run

User redirect: instead of fixed `qat_start_step=48000` (guess), trigger
QAT on the **first val plateau** during the bf16 phase. The model
tells us when it's done learning structure in FP, no need to guess.

### Why it's better

- Model dictates timing. If bf16 converges fast (which v2 was doing
  — val 1.86 at step 4500), we'd otherwise waste compute in bf16
  doing nothing. With plateau trigger, we move to QAT exactly when
  bf16 stops improving.
- One mechanism, two purposes. Existing plateau detector now does
  LR-halving in QAT phase + first-time QAT enable in bf16 phase.
- If training runs slow, we wait longer in bf16 — never undertrain
  the FP foundation.

### Resumed not restarted

v2 had built up to **val 1.86** at step 4500 in ~12 min. Stopped,
resumed from `best.pt` with new config — no work lost. Resume code
restores model + optimizer state + step counter cleanly.

### New config

[`training/configs/tiny_hetmoe_v2_plateau.json`](training/configs/tiny_hetmoe_v2_plateau.json):
- `qat_from_zero: false`
- `qat_start_step: 0` (no fixed-step trigger)
- `qat_on_plateau: true` (NEW — first plateau enables QAT instead of halving LR)
- `gpu_resident_corpus: true`
- `resume_from: runs/tiny_hetmoe_v2/checkpoints/best.pt`

### Implementation note

Detector logic: when `qat_on_plateau=True`, the FIRST plateau detected
in the bf16 phase sets `qat_trigger_pending=True` (rank 0). At the top
of the next iteration, all ranks broadcast-synchronize this flag and
flip `set_quantize_mode(model, on=True)`. After QAT enables, the
detector reverts to normal LR-halving for the QAT phase.

`val_history` is reset on the QAT switch so the first 8 post-QAT vals
don't immediately trigger LR halving (val will jump up briefly when
ternary kicks in).

### Status

Live at 96K tok/s, step ~4560, training cleanly. Plateau detector will
fire when val flatlines (likely 6K-15K steps from now based on current
slope).

---

## 2026-04-29 23:30 — TinyHetMoE v2 launched: bf16-then-QAT + GPU-resident corpus

Stopped v1 cleanly at step 29040 (best val 3.0512 at step 23000, was
already plateauing — LR had halved to ×0.500). Launched v2 with two
recipe changes:

### Recipe changes

1. **bf16-then-QAT** — train pure FP for 80% of steps (0..48000), then
   enable ternary forward for the final 20% (48000..60000). User's
   call: instead of QAT-from-zero forcing the model to learn under
   permanent ternary constraint with vote-backward gradient
   approximation, let it learn structure cleanly in FP first, then
   "compress" into ternary. End artifact is still 100% ternary.

2. **GPU-resident corpus** — load entire 952 MB train.bin + 9.6 MB
   val.bin to GPU at startup (cast to int64). Skip per-batch
   memmap+CPU+transfer overhead. Each rank holds its own copy
   (cheaper than sharing for our scale).

### Smoke test (30 steps, QAT switch at 20)

- bf16 phase: loss 8.86 → 6.96 in 20 steps (FP descent — much faster
  than QAT-from-zero v1 which took ~700 steps for same descent)
- QAT switch fired cleanly: `★★★ QAT mode ENABLED at step 20`
- val 7.17 → 7.33 across the switch (small bump, expected)
- Throughput: **94K tok/s vs v1's 84K = +12%** from GPU-resident corpus
- GPU memory: 6.8 GB / 16 GB per rank (3.7 GB for corpus + 3.1 GB
  model+activations). Plenty of headroom.

### Production run launched

Config: [`training/configs/tiny_hetmoe_v2.json`](training/configs/tiny_hetmoe_v2.json):
- `qat_from_zero: false`, `qat_start_step: 48000`
- `gpu_resident_corpus: true`
- `max_steps: 60000`, all other hyperparams identical to v1

Live throughput: **~96K tok/s** at step 60. Projected ETA: ~5.7 hours
total wall time (4.5h bf16 + 1.1h QAT-finetune + overhead).

### What we expect to learn

- **Phase 1 floor (val at step 48000):** if FP-equivalent of our recipe
  hits val ~1.5-2.0, we know the architecture is sound and v1's PPL ~21
  was a QAT-from-zero artifact. If FP also plateaus at 2.5+, the
  architecture itself has overhead.
- **Phase 2 recovery (val at step 60000):** how much capacity does QAT
  cost when bootstrapped from a good FP starting point? If we land
  within 0.1-0.3 nats of phase 1, the recipe works cleanly. If we lose
  >0.5 nats, ternary is genuinely lossy on this size class.
- Result feeds production decision: should next-gen Highway-B do
  bf16-then-QAT instead of QAT-from-zero?

---

## 2026-04-29 22:30 — WASM engine: VALIDATED + FAST

Late-night marathon paid off. The Rust/WASM engine is **end-to-end
working with bit-correct math against Python.**

### Final speed

- **205 tps native** (release, x86_64-linux)
- **152 tps WASM** in Node v20 (no SIMD intrinsics)
- **WASM size:** 78 KB (after wasm-bindgen) + 42 MB raw int8 weights
  (will be 9 MB after 2-bit packing — TODO for production)

### Final validation

5/5 top-K overlap with Python, top-1 exact match, **logit deltas all
< 0.005**. The remaining tiny noise is fp32 rounding from doing the
gather-adds in different order than torch's matmul.

```
Python top-5: [(64, 5.218), (95, 4.846), (1269, 4.291), (44, 3.789), (151, 3.186)]
Rust   top-5: [(64, 5.218), (95, 4.845), (1269, 4.292), (44, 3.791), (151, 3.186)]
```

### The bug that ate 4 hours

The validation drift was the **MoE gate**. In our Python model class,
`self.gate = QuantizedLinear(...)` — so when QAT mode is on, the gate
weights are also ternarized at forward time. My export script was
writing the FP gate weights to tiny.bin, but Rust was using those
FP weights as-is in the gate dot product. So Python ran ternary-gate
forward, Rust ran FP-gate forward — **same gate, different forward**.
Routing diverged at every layer, compounding through 4 layers.

Fix: pre-ternarize the gate weights **at export time** using the same
ternarize formula Python uses. Now Rust's FP dot product against
the pre-ternarized gate matches Python's ternary forward exactly.
One-line change in [`scripts/export_model.py`](scripts/export_model.py).

### Architecture / engine notes

- Replaced PackedTernaryWeight + LUT matvec with **flat gather-add**
  (v3.1 layout from production). `output[r] = scale * (sum(input[plus_idx]) - sum(input[minus_idx]))`.
- 8-way unrolled accumulators with `unsafe get_unchecked` (sound:
  indices validated at construction).
- No INT8 activation quantization — pure FP32 input, ternary weight.
- Speed: 0.5 tps → 127 tps (LUT loop reorder) → 205 tps (gather-add).
- WASM SIMD128 (`-C target-feature=+simd128`) didn't help: gather
  pattern doesn't auto-vectorize without manual intrinsics. Future
  optimization (~1.5-2× possible).

### What's running

- **HTTP server on :8765** serving `ui/`. Smoke page at
  `http://localhost:8765/smoke.html` (minimal — just generation),
  full UI at `http://localhost:8765/index.html` (panels: meaning
  axes, attention heatmaps, expert routing, top-K).
- Training continues in background at step 26500, val 3.05 best.

### Open / deferred

- Hot-token cache: would help but small win on 6K vocab. Skip for v1.
- 2-bit packing of weights file for download size reduction (42 MB →
  9 MB). Tractable but requires updating both Python export and Rust load.
- Hand-written WASM SIMD for gather inner loop (~1.5-2× possible).
- Final model swap when training finishes (recipe-locked, just
  re-run export_model.py + replace tiny.bin).

---

## 2026-04-29 21:30 — WASM engine: works but slow, then fast, then needs gather-add port

Long evening session. Everything compiles, training keeps running in
the background (paused-Highway-B GPUs reused for TinyHetMoE — at step
24000+, val 3.05 best @ step 23000).

### What's built

- [`scripts/export_model.py`](scripts/export_model.py) — packs a TinyHetMoE
  checkpoint into HTMOE002 binary format (Highway/M+I/QK-Norm aware).
  ~42 MB raw int8, ~9 MB after 2-bit packing in Rust at load time.
- [`wasm/src/`](wasm/src/) — Rust crate, native + WASM targets, full
  forward pass with QK-Norm, meaning protection, Highway expand/
  compress, NoPE.
- [`scripts/validate_rust.py`](scripts/validate_rust.py) — runs Python
  and the native Rust binary on the same prompt, compares top-K.
- [`wasm/src/bin/bench.rs`](wasm/src/bin/bench.rs) — native benchmark.
- [`ui/wasm-pkg/`](ui/wasm-pkg/) — wasm-bindgen bundle for the browser.
- [`ui/wasm-pkg-node/`](ui/wasm-pkg-node/) — node-bindings for headless
  testing.
- [`ui/bench_node.mjs`](ui/bench_node.mjs) — runs the WASM build under Node.

### Speed journey

1. **First compile, naïve LUT layout (LUT rebuilt per-row)**: 0.5 tps
   native, 0.4 tps WASM. Unusable. The hot loop was rebuilding a
   256-entry LUT for every row × column-group pair.
2. **Pre-build LUTs once per col-group**: 127 tps native, 107 tps WASM.
   200× speedup. Validation passed at step ~12500.
3. **At step ~17500+, validation drifts**: top-1 mismatches Python by
   ~0.7 logits, only 2/5 top-K overlap. Root cause: INT8 activation
   quantization + RMSNorm eps differences compound across 4 layers.
   Worse as routing sharpens with training.

### User-shared reference: v3.1 gather-add engine

User pulled `main.rs` from a Mac-mini engine that hits **~1000 tps on
~30M models, ~200+ tps on larger ones**. Saved to
[`wasm/REFERENCE_v3_1.md`](wasm/REFERENCE_v3_1.md). Key insights:
- Gather-add with **flat contiguous index arrays** (not Vec<Vec<u16>>)
  — prefetcher-friendly, no pointer chasing
- 8-way unrolled accumulators, `unsafe get_unchecked` after validation
- **No INT8 activation quantization** — pure FP32 × ternary, more
  accurate. Probably fixes our validation drift.
- Ternary lm_head built from embedding at load time
- Hot-token cache (~90% hit rate skips full vocab scan)
- Per-user concurrent batch mode

### Next session

Port the gather-add engine. Steps:
1. Replace `PackedTernaryWeight` + LUT matvec with `GatherWeight` +
   gather-add. Should fix correctness AND get to 200+ tps.
2. Re-validate against Python — expect 5/5 top-K with Δ < 0.01 logits.
3. Add hot-token cache (small win on our 6K vocab but cleaner code).
4. Wire UI to live infer (replace static JSON load with WasmModel
   .step()). Already designed in [`ui/app.js`](ui/app.js) — just needs
   re-build of WASM + retest.
5. Build a UI toggle for engine choice ("LUT" vs "gather-add") to
   show the speed difference live.

### What works right now (despite drift)

- Native bench: 127 tps prompt fill + generation
- WASM bench (Node): 107 tps. ~0% WASM overhead.
- Top-K outputs are *story-relevant* even with drift (rust → "boy",
  "girl", "rabbit"; python → "boy", "girl", "bunny") — model is
  learning correctly, the engine is just amplifying slightly different
  ones from the same neighborhood.

---

## 2026-04-29 18:10 — model + trainer + DDP smoke tests pass

**Model** ([model/tiny_hetmoe.py](model/tiny_hetmoe.py)):
- Vendored Highway-B + HetMoE + QAT + QK-Norm into a self-contained
  module (no production dependencies; the folder is GitHub-ready).
- Forward modes: standard (returns logits, loss) and `return_trace=True`
  (returns logits + per-layer attention, expert routing, hiddens for the
  UI to consume).
- 37.8M params total, 71 QuantizedLinear modules, all FP/QAT/trace
  paths verified by [scripts/check_model.py](scripts/check_model.py).

Param size landed at 37.8M, above the 12-30M I'd estimated. The MoE
budget I'd computed by hand was off (forgot SwiGLU and DeepNarrow have
multiple internal projections). **Decision: keep 37.8M.** Still small
enough that:
- WASM ternary packed ≈ 9.5 MB pre-zstd, ~5-7 MB after = phone-loadable
- Eldan & Li's TinyStories paper used 30M as baseline, so we're +27%,
  reasonable for "coherent generation" target

**Trainer** ([training/train_tiny_hetmoe.py](training/train_tiny_hetmoe.py)):
- Forked from production `train_highway_b.py`, stripped multi-corpus
  + variable-length + grad-checkpointing (none needed at this scale).
- Initially built single-GPU; user pointed out DDP was the right call
  since both GPUs are free. Added DDP — 1.78× speedup confirmed in
  smoke test (46K → 81K tok/s).
- HetMoE top-2 routing → `find_unused_parameters=True` (Day 6 lesson).
  At smoke-test scale DDP warns "no unused parameters" since untrained
  routing is roughly uniform; will become true (some experts unused on
  some ranks) as routing sharpens. Keep the flag.
- Plateau detect, lr_scale persistence, ckpt rotation, chunked CE all
  carried over.
- Single-GPU path also works (no torchrun → falls back gracefully).

**Smoke tests** (30 steps, both single-GPU and DDP):
- Loss descends 8.88 → 8.30 (random ≈ 8.69, untrained ceiling)
- val descends 8.52 → 8.31
- 3.0 GB memory per GPU (huge headroom; could 4× batch if needed)
- No NaN, gradients stable, plateau wiring tested

**Next:** launch the 60K-step DDP run. ETA ~7-8h. ~1.97B tokens of
training (4 epochs of TinyStories train). Watch: does QAT-from-zero
descend cleanly at this small scale? Production saw it slower than FP
but converging — at 38M we're at the smaller end of "QAT works."

---

## 2026-04-29 17:55 — data + meaning artifacts ready

Two scripts ran clean:

**`scripts/prepare_data.py`** — pruned GPT-2 tokenizer, TinyStories
download + tokenize + remap. ~22 min CPU.
- Loaded GPT-2 (50,257 vocab) via HF `transformers`
- Counted token frequencies on full TinyStories train (2.1M stories,
  ~476M GPT-2 tokens)
- Kept top **5,963 GPT-2 ids** that hit 99.5% coverage; +4 specials
  (`<unk>`, `<bos>`, `<eos>`, `<pad>`) → tiny vocab size **5,967**
- Wrote `data/train.bin` (476M uint16 tokens, 952 MB) and
  `data/val.bin` (4.8M, 9.6 MB). Fits comfortably in uint16.
- `<unk>` rate 0.5% — exactly the coverage target, sanity check passes
- Vocab map in `tokenizer/vocab.json` (167 KB)

**`scripts/make_meaning_axes.py`** — 132 production axes remapped to
the tiny vocab. ~30 sec CPU.
- Reused `/data/sematic-embedding/contextual_embeddings/contextual_axis_embeddings.npy`
  (the 151665 × 132 production matrix, keyed on Qwen vocab)
- For each tiny token: GPT-2 decode → string → Qwen re-encode → look up
  the axis vector. Multi-subtoken Qwen encodings averaged.
- Coverage: 97.5% direct Qwen match, 2.4% averaged, 0% fallback. Very
  clean. Specials (4) zero-vector.
- Output: `data/meaning_axes_132.npy` (5967 × 132, 3.1 MB) and
  `data/meaning_axis_names.json`
- Sanity checks confirm semantic landing: "good"→GOOD (+3.47),
  "bad"→BAD (+3.97), "you"→YOU (+8.71), "dog"/"cat"→ANIMATE (+3.80,
  +3.91). Some rare words noisier (e.g. "happy" doesn't top out on
  VALENCE_POS) but acceptable.

### Decision: switch from 32 to 132 axes (reverted at user request)

Earlier I'd planned 32 hand-picked Wierzbicka primes for the small
model on the assumption that fewer axes = simpler visualization.
Walked back: the production model uses 132, the recipe is the recipe,
and we already have the 132-axis matrix sitting on disk. Use it. The
UI can still highlight a curated subset of axes per token — the model
itself sees all 132.

### Param budget redo (132 meaning, 132 intuition, Highway 2×)

```
Vocab=5967, hidden=264 (132+132), internal=528, L=4, experts=4
embedding (M+I): 5967 × 264 = 1.6M
Highway expand:  256 × 528 ≈ 0.2M
attention/L:     4 × 528² = 1.1M  ×4 = 4.5M
HetMoE/L:        4 experts × (528 × 1056 × 2) = 4.5M  ×4 = 17.8M
gate/L:          528 × 4 = 2K  ×4 = neg
compress:        528 × 132 = 0.07M
lm_head:         5967 × 264 = 1.6M
                                    -------
                                     ~26M total
```

Settles at **~26M params**, well within the 12-30M target. Backbone is
22M, embeddings/lm_head 3.2M, expand+compress 0.3M.

**Next:** write `model/tiny_hetmoe.py` — the model class. Will fork
from production `train_mi_highway.py`'s Highway Block and adapt for
small scale + uint16 tokens.

---

## 2026-04-29 16:50 — phase c skipped, going straight to phase b

User redirect: pause the parent Highway-B run (paused cleanly at step
22760, best.pt at step 22000, val 6.0857) and start TinyHetMoE phase b
now rather than waiting for Highway-B to converge.

This is fine because:
- Highway-B's recipe is the recipe TinyHetMoE will use. There are no
  new architectural questions to settle.
- The single-GPU availability is exactly what TinyHetMoE needs.
- The blog/UI is the main deliverable here, and the model is the
  precursor to that — get it built.

Phase a (design) ✓ done. Phase b (build) starts now. Phase c (wait)
collapsed into "resume Highway-B from step 22000 best.pt later."

**Next:** step 1 — train the 5K-vocab tokenizer on TinyStories.

---

## 2026-04-29 — folder scaffolded, design doc moved

Created the standalone `tinyhetmoe/` folder, separate from the parent
`experiments_may/` work. The intent: put this on GitHub and host the UI
on GitHub Pages, so it has to live in its own self-contained tree.

Layout: `model/`, `tokenizer/`, `training/`, `wasm/`, `ui/`, `docs/`,
`blog/`, `scripts/`. The original design doc moved to `docs/design.md`.

Started the blog with post 0 (why this exists). The journal will track
the build day-by-day; the blog will narrate the interesting bits as
posts after the fact.

**Phase status:** a (design) ✓ done. c (wait for Highway-B) — in
progress, parent run is at step ~19200 / 100000, val_mixed 6.13. b
(build) — gated on c.

**Next:** when Highway-B converges, copy `qat_utils.py` and the trainer
skeleton into `model/` and `training/`, then start phase b step 1
(tokenizer).

---

## 2026-04-29 — design doc drafted

Wrote a 12-section design doc covering model spec (~13M params, hidden
256, 4 layers, 4 experts), tokenizer (5K vocab: 3.5K word-level + 1.2K
BPE + 256 byte fallback), data (TinyStories), 32 hand-picked
Wierzbicka-prime meaning axes, training plan, WASM export approach
(Rust + wasm-bindgen), UI sketch, and risks.

Decisions made:
- Hidden 256 not 192 — pushes model from ~8M to ~13M, gives more room
  for coherent generation
- Word-level tokenizer not pure BPE — interpretability matters more than
  compression for a teaching demo
- 32 axes not 132 — the production count is too many for visualization
- Rust + wasm-bindgen for inference, not ONNX — reuses existing ternary
  inference codebase from production

Open for review: vocab size (5K vs 2.5K vs 10K), param target (13M vs
20-25M), WASM approach, axis selection, scope of phase b.
