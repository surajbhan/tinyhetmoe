# wasm/

Rust + wasm-bindgen inference engine. Out of scope for phase b — this
is its own project once the trained model exists.

Plan:
- Reuse the Rust gather-add codebase from the parent project's ternary
  inference work
- Strip the speed optimizations (we're not chasing performance for a
  teaching demo)
- Add trace export: each token, return per-layer attention, meaning
  embedding, expert routing weights, hidden states, top-K logits
- Compile to `wasm32-unknown-unknown`, bind via `wasm-bindgen`
- Target: ~1.5 MB compressed download, ~50–200 ms/token on phone

Empty for now.
