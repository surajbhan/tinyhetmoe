# Reference: gather-add engine derivation

This crate's gather-add forward pass is adapted from a production
ternary inference engine (the Mac mini engine that
hit ~1000 tps on small models, ~200+ tps on larger ones). Key
techniques to port into our `tiny_hetmoe_wasm` crate:

## 1. Flat gather-add layout (replaces LUT)

Replace `PackedTernaryWeight` with `GatherWeight`:

```rust
struct GatherWeight {
    plus_data: Vec<u16>,      // all plus indices, flat
    plus_offsets: Vec<u32>,   // row i's plus start..end
    minus_data: Vec<u16>,
    minus_offsets: Vec<u32>,
    scale: f32,
    rows: usize,
    cols: usize,
}
```

Hot loop is `gather_matvec`: 8-way unrolled f32 sum over `plus_data`,
8-way subtract over `minus_data`, scale once. **Zero multiplies, no
INT8 quantization.**

This is MORE ACCURATE than my current INT8-LUT path (which is what's
causing the 0.7-logit Python/Rust drift in validation). The gather-add
version uses pure fp32 inputs, so it should match Python to within
fp32 rounding (~1e-6 relative).

## 2. Ternary lm_head (post-quantize from embedding)

The production engine doesn't store the lm_head separately — at load
time it ternarizes the embedding (per-row scale, threshold-based) and
uses gather-add for logits. **Big speedup: ~50K row matvec becomes
gather-add over sparse rows.**

For TinyHetMoE we already store lm_head ternary in HTMOE002, so we
can just convert it at load time the same way (no need to quantize
the embedding).

## 3. Hot token cache

Track up to 256 recently-generated tokens. For each forward, first
compute logits for hot tokens only. If best beats second by >8.0
(very confident), skip the full vocab scan. Hit rate ~90% on
agentic workloads, ~50%+ on natural text.

For TinyStories vocab is only ~6K, so the win is smaller (~12×
speedup on the lm_head, but lm_head is only one matmul) — still
worth a few % overall.

## 4. Parallel batch mode (multi-user)

One model, N user states, shared weights. Round-robin token-by-token.
Excellent throughput scaling. **Probably overkill for our demo** but
the technique is in the file if needed.

## 5. Per-user RNG (xorshift), top-K sampling

Already standard, but cleaner than my current ad-hoc impl.

## What needs adapting for TinyHetMoE

The reference has:
- Single-stream hidden (no Highway split)
- RoPE (not NoPE)
- No QK-Norm
- No M+I split embedding
- No meaning protection mask

So when porting: keep the gather-add core, drop RoPE/cos/sin, add
QK-Norm RMSNorm slabs, add M+I embedding lookup + Highway expand at
the start, add meaning protection on attn/MoE output, add Highway
compress at the end before lm_head.

## Estimated impact on our build

Current: 127 tps (LUT, INT8 quant, single-thread, no hot-cache).
After port: probably **200-400 tps** for our 38M model on this
machine. The Mac-mini's 1000 tps was on ~30M models, so 38M will be
maybe 60-70% of that.

## Open: validation drift

Once gather-add is in (no INT8 quant), the Python↔Rust validation
should jump from 2/5 top-K overlap to 5/5 with logit Δ < 0.01. If
not, the bug is somewhere besides matvec — most likely QK-Norm eps
or meaning-protection mask placement.
