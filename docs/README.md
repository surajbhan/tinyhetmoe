# ui/

Static single-page UI. GitHub Pages target.

Will house:
- `index.html` — main page
- assets and JS to load the WASM module, render the panels
- panels: top-K next-token picker, 32 meaning-axis bars, 4×4 attention
  heatmaps, per-layer expert routing, ternary weight grid

See [docs/design.md §8](../docs/design.md) for the layout sketch.

Out of scope for phase b. Built after the model + WASM engine exist.
