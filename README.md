# TinyHetMoE

A **38M-parameter ternary language model** trained on TinyStories, running
entirely in the browser via WebAssembly. Built around a 132-axis named
meaning embedding so every prediction can be **explained, not just
measured**. Designed for **infinite context** through meaning-protected
attention.

> *Click any token to inspect the model's internal state at that
> position. Click a next-token candidate to extend the story. Pull
> sliders to play with temperature, top-K, repetition penalty.*

## What's interesting about it

- **Every weight is one of three values: −α, 0, +α.** No multipliers in
  the inner loop — matrix multiplication becomes pure gather-add.
- **Half the embedding has named meaning.** 132 hand-curated axes
  (HAPPY, WANT, PAST_TENSE, ANIMATE, …) initialized from contextual
  Qwen2.5-Coder hidden states. The model can be queried — for any
  token, what does it think this token means?
- **Heterogeneous Mixture of Experts.** 4 different FFN architectures
  (Standard / SwiGLU / DeepNarrow / Bottleneck) per layer with top-2
  routing. The router learns which expert handles which token type.
- **Highway expansion** with **meaning protection**. Hidden state
  expands 2× internally for compute; meaning dimensions pass through
  unchanged so the semantic anchors don't drift.
- **No positional encoding.** Just a causal attention mask. The model
  learns position from the attention pattern itself.
- **Bf16-then-QAT training recipe.** First ~80% of training runs in
  full precision; final ~20% switches to ternary forward + vote-backward
  gradient. Final artifact is 100% ternary, but the FP foundation gives
  a much stronger starting point.

## Try it

```bash
git clone https://github.com/surajbhan/tinyhetmoe
cd tinyhetmoe
# Download the model weights (separate from the repo)
curl -L -o ui/tiny.bin <release URL — coming after v2 finishes>
# Serve the UI
cd ui && python3 -m http.server 8765
# Open http://localhost:8765
```

The model file (`tiny.bin`, ~42 MB) is published as a release artifact,
not committed in the repo. Once downloaded, the entire experience runs
client-side — no backend, no API call, no telemetry.

## How it's organized

```
tinyhetmoe/
├── docs/design.md          ← full design doc (architecture rationale)
├── JOURNAL.md              ← engineering log, every decision and gate
├── blog/                   ← long-form narrative posts
│
├── model/tiny_hetmoe.py    ← PyTorch model definition
├── training/               ← trainer + configs
│   ├── train_tiny_hetmoe.py
│   └── configs/
├── scripts/                ← data prep + analysis
│   ├── prepare_data.py     (TinyStories → pruned-GPT-2 vocab → train.bin)
│   ├── make_meaning_axes.py (132-axis Qwen contextual extraction)
│   ├── export_model.py     (PyTorch checkpoint → HTMOE002 binary)
│   ├── export_trace.py     (per-token state dump for UI)
│   └── validate_rust.py    (Python ↔ Rust correctness check)
│
├── wasm/                   ← Rust gather-add inference engine
│   ├── src/lib.rs          (model loader, HTMOE002 format)
│   ├── src/forward.rs      (forward pass — Highway, M+I, QK-Norm, NoPE)
│   ├── src/tensor.rs       (gather-add ternary matvec)
│   ├── src/wasm_api.rs     (wasm-bindgen JS interface)
│   └── src/bin/            (native test harnesses)
│
├── ui/                     ← static dashboard, GitHub Pages target
│   ├── index.html, style.css, app.js
│   ├── wasm-pkg/           (compiled WASM + JS bindings)
│   ├── encode_lookup.json  (GPT-2 BPE → tiny vocab remap, JS-side)
│   ├── decode_lookup.json
│   ├── meaning_axis_names.json
│   └── axis_descriptions.json (plain-language axis labels)
│
└── data/                   ← derived artifacts (training data NOT included)
    ├── meaning_axes_132.npy  (5967 × 132 float32 — meaning embedding init)
    └── meaning_axis_names.json
```

## Building from scratch

If you want to retrain the model:

```bash
# 1. Prepare data (downloads TinyStories, builds pruned vocab + tokenizes)
python3 scripts/prepare_data.py
# 2. Build meaning-axis embeddings from production Qwen output
#    (requires the production extraction file — not included)
python3 scripts/make_meaning_axes.py
# 3. Train (DDP, two GPUs)
torchrun --nproc_per_node=2 training/train_tiny_hetmoe.py \
  --config training/configs/tiny_hetmoe_v2_plateau.json
# 4. Export to ternary binary
python3 scripts/export_model.py \
  --ckpt runs/tiny_hetmoe_v2/checkpoints/best.pt \
  --out ui/tiny.bin
```

## Building the inference engine

```bash
cd wasm
# Native (for testing/benchmarking)
cargo build --release
./target/release/tiny_hetmoe_native ../runs/tiny_hetmoe_v2/checkpoints/best.pt
# WASM (for the browser)
cargo build --release --target wasm32-unknown-unknown --features wasm
wasm-bindgen --target web --out-dir ../ui/wasm-pkg \
  target/wasm32-unknown-unknown/release/tiny_hetmoe_wasm.wasm
```

Native runs at ~205 tok/s on a single x86-64 core. WASM in browser
hits ~150 tok/s with no manual SIMD intrinsics.

## License

MIT. See [LICENSE](LICENSE).

## Status

This is **v1** — model trained with bf16-then-QAT recipe, val ~1.6 / PPL
~5 on TinyStories (matching published 30M FP baselines). Generation
quality is "coherent for ~30 tokens, then patterns repeat" — same level
as Eldan & Li (2023) reported for similar-scale TinyStories models.

Ongoing work and the engineering narrative live in
[JOURNAL.md](JOURNAL.md).
