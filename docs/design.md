# TinyHetMoE — Interactive Educational Model Design Doc

**Status:** Design draft, awaiting review.
**Date:** 2026-04-29
**Goal:** Train a 12-30M parameter model with the full Highway-B + HetMoE +
QAT + M+I recipe, on TinyStories, that runs in the browser via WASM and
serves as an interactive teaching tool for the architecture.

---

## 1. Why this is worth building

The 5-day plan + Highway-B work has produced a stack of architectural
ideas (NoPE, contextual semantic axes, HetMoE expert routing, Highway
expansion, ternary QAT, vote-backward gradient) that are individually
non-obvious and collectively unique. None of these are visible to a
student reading the code or papers — they're emergent properties of
forward passes you can't see.

**A live, interactive demo where someone can:**
- Hover over a token and see its 32 named meaning axes light up
- Click on attention to see which of L0-L3 has the sharpest pattern
- See the 4 HetMoE experts with their per-token routing weights
- Pick from top-K next tokens and watch the story branch
- Toggle between ternary (deployed) and FP (training) forward to see
  that the architecture works at both

…makes those ideas tangible. It's also a portfolio piece — if Kaman
ships a ternary 150M agent and someone asks "what's the architecture,"
the link to the demo is more convincing than an arXiv preprint.

**Audience:** ML curious folks (not researchers). Goal is "I get it"
not "I can replicate."

---

## 2. Model spec — 14-18M params

### Architecture (matches production recipe at small scale)

| Param | Value | Why |
|---|---|---|
| Vocab size | ~5,000 | Custom tokenizer (see §3). Trades off "I/O dwarfing model" vs "tokens are interpretable." |
| Hidden / input_dim | 192 (= 32 meaning + 160 intuition) | Visualizable as 8×24 heatmap. Big enough for 32 meaning axes + room for learned dims. |
| Internal_dim (Highway) | 384 | 2× expansion. Visualizable as 16×24. |
| Layers | 4 | Stackable view; few enough to show all attention maps simultaneously. |
| Heads | 4 (head_dim 48) | Each head's attention pattern fits in one tile. |
| Experts (HetMoE) | 4: Standard, SwiGLU, DeepNarrow, Bottleneck | Same 4 types as production. Top-2 routing. |
| FFN mult | 2.0 | Same as production Highway. |
| Position encoding | NoPE (causal mask only) | Same as production. |
| Meaning treatment | Trainable from contextual init | B recipe. Mirrors production. |
| QAT | From step 0, ternary forward + vote-backward | Same as production Highway-B. |
| QK-Norm | RMSNorm per-head | Required for FP↔ternary stability. |

### Param budget

```
embeddings:      5000 × 32 (meaning)  =   0.16 M
                 5000 × 160 (intuition) =  0.80 M
expand:          160 × 224 (intuition→new_int)  = 0.04 M  [192→384 less the 32 meaning copy]
attention/layer: 4 × 384² = 0.59 M  ×4 layers = 2.4 M
qk_norm:         8 × 48 = 384 params/layer × 4 layers ≈ negligible
moe/layer:       4 experts × ~200K each = 0.8 M  ×4 = 3.2 M
gate:            384 × 4 × 4 = 0.006 M
compress:        544 × 160 = 0.09 M
norm + lm_head:  192 × 5000 = 0.96 M
                 + 192 RMSNorm
                                    -------
                                     ~7.5 M total backbone
                            + 1.0 M I/O = ~8.5 M total

Hmm that's smaller than 12-30 target. Could increase:
  - hidden_dim 192 → 256 → ~12 M
  - hidden_dim 256 + layers 4 → 6 → ~16 M
  - Pick middle: hidden_dim 256, layers 4 → ~13 M

Final pick: hidden=256 (32 meaning + 224 intuition), internal=512,
layers=4, heads=4 (head_dim=64), experts=4. ≈ 13M params.
```

### Why this size, not bigger or smaller

- Smaller than 8M: tokens become incoherent garbage. Generation looks
  random, defeats the demo.
- Larger than 30M: WASM download size becomes prohibitive (>10MB),
  inference becomes slow on phones.
- 13M ternary packed = ~2.5 MB. zstd compressed: ~1.5 MB. **Loads
  in ~1 second on 4G.**

---

## 3. Tokenizer

The tokenizer choice is the biggest design decision because it affects:
- How interpretable individual tokens are
- How well meaning axes anchor
- How long sequences are

### Decision: word-level top-K + BPE fallback + byte fallback

Three-tier vocab structure:
1. **Top 3,500 words** from TinyStories — most words are full English
   words, completely readable
2. **~1,200 BPE merges** trained on the residual (rarer words/subwords)
3. **~256 bytes** for any character not covered above
4. Special: `<bos>`, `<eos>`, `<pad>`, `<unk>`

**Total: ~5,000 tokens.**

### Why not character-level

- 6× longer sequences for same content
- Meaning axes can't anchor (no "happy" anchor word, only h-a-p-p-y)
- Less interesting visualizations (each token is a letter)

### Why not pure BPE

- Subword tokens like `##ed`, `##ing` are confusing for non-ML audiences
- "I want to see the 'happy' token" should literally show 'happy'

### Custom tokenizer training

Use HuggingFace `tokenizers` library:
```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(
    vocab_size=5000,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
)
tok.train(["TinyStories-train.txt"], trainer)
```

Estimated: ~5 minutes on CPU, output ~200KB tokenizer file.

---

## 4. Data: TinyStories

- TinyStories-V2: ~2.1M synthetic stories of 100-1000 words each
- Total tokens after our tokenizer: ~500M-700M
- Single-corpus (no diverse mix needed — TinyStories is intentionally
  narrow)
- Hold out 1% as val (~5M tokens)

### Why TinyStories is a good fit

- **Coherent at small model scale** — Eldan & Li (2023) showed 30M
  models on TinyStories generate readable, narratively-consistent
  3-paragraph stories. 13M will be slightly worse but still
  demonstrably "alive."
- **Limited domain** = the model has time to actually learn it. Wiki
  at 30M would produce gibberish; TinyStories produces stories.
- **Clean format** = no code, no markup, no foreign characters.
  Tokenizer/visualization stay simple.

### Where to get it

```bash
# HuggingFace dataset
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="train")
# ~2.1M examples. Concatenate, tokenize, save as .bin
```

Verify it's not already on disk:
- Check `/data/training_kanam/clean_data/` — likely not (handoff §3
  mentions TinyStories was used for the 130M PoC, but we may need to
  re-download)

---

## 5. Meaning axes — 32 hand-picked

The 132 axes from production are too many for a teaching demo. Pick 32
that are:
- **Maximally interpretable** to a non-ML reader
- **Anchored on words that exist in the TinyStories vocab**

### Proposed 32-axis set

Drawn from Wierzbicka primes + affective:

```
Pronouns (4):     I, YOU, SOMEONE, PEOPLE
Existence (3):    THIS, THERE_IS, NOT
Quantity (4):     ONE, TWO, MANY, ALL
Mental (4):       THINK, KNOW, WANT, FEEL
Sensory (3):      SEE, HEAR, TOUCH
Action (3):       DO, MOVE, SAY
Time (3):         NOW, BEFORE, AFTER
Place (2):        HERE, THERE
Evaluation (4):   GOOD, BAD, BIG, SMALL
Affect (2):       HAPPY, SAD
```

**32 axes total.** Each anchored by 5-10 TinyStories words (e.g.
HAPPY axis anchors: "happy", "joyful", "cheerful", "smiled", "laughed").

### Generating the .npy

Adapt the contextual extraction approach (handoff §3 April 8):
- Run TinyStories through Qwen2.5-Coder (yes, Qwen — for the contextual
  vectors only; the small model itself uses our custom tokenizer)
- For each token in our 5K vocab, get average Qwen layer-12 hidden state
- Project 896-dim → 32 axes via QR-orthogonalized anchor directions

This gives us a `(5000, 32)` meaning embedding matrix that's
semantically meaningful AND visualizable. ~30 minutes on Atlas.

---

## 6. Training plan

### Phase 1 — tokenizer + data prep (~1 hour CPU)
- Train custom tokenizer on TinyStories
- Tokenize full corpus into a uint16 .bin (vocab fits in uint16)
- Generate 32-axis contextual meaning embeddings
- Build train/val split

### Phase 2 — model code (~1 day)
- Fork `train_highway_b.py` → `train_tiny_hetmoe.py`
- Adapt for small vocab (uint16 tokens, smaller embeddings)
- Adapt meaning loader for 32-axis .npy
- Same Highway + B + QAT + QK-Norm + variable-length stack
- Single GPU (this size doesn't need DDP)
- TinyStories-only (no multi-corpus needed)

### Phase 3 — training (~6-12 hours single GPU)
- 13M params, ~700M tokens, 1-2 epochs
- Variable-length: {128, 256, 512} (TinyStories sequences are short)
- Plateau-detect LR halving as in Highway-B
- Same QAT-from-zero recipe

### Phase 4 — generation quality eval
- Sample 50 stories from the model, prompted with story openings
- Human-readable check: do they make narrative sense?
- Reference: 30M models on TinyStories produce coherent 3-paragraph
  stories (Eldan & Li 2023)
- Our 13M target: coherent 1-2 paragraphs with occasional weirdness

---

## 7. WASM export & inference

### Approach: bf16-or-fp32 forward in WASM, ternary weights for download

The production runtime would use Rust gather-add for true ternary speed.
For the teaching demo, **simpler is better**:

1. **Pack ternary weights for download:** {-1, 0, +1} packed 4-per-byte
   = 1.6 bits/element nominal. Plus alpha scalar per Linear. Total
   ~1.5 MB after zstd.

2. **WASM inference:** unpack to fp32 on load (one-time, ~50ms),
   then run normal fp32 forward. WASM-SIMD handles 13M-param fp32
   inference at ~50-200ms/token on phone.

3. **No specialized ternary inference kernels in WASM** — we're not
   optimizing for speed. The "WOW it's ternary" reveal is in the
   weights view, not the runtime.

### What WASM needs to expose to JS

For the UI to be interactive, the inference engine must expose:
- Per-token, per-layer attention matrix (4 heads × T × T fp32)
- Per-token meaning embedding (32 fp32 floats)
- Per-token expert routing (4 floats per layer per token)
- Per-token, per-layer hidden state (256 fp32 — visualizable as 16×16)
- Top-K next-token logits + tokenizer decode

This is a lot of state — maybe 50KB per generated token. UI shows it
all live.

### Build path

- Rust + `wasm-bindgen` for the inference engine
- Or Python → ONNX → ONNX Runtime Web (faster to build, less control)
- Or `tinygrad` JS port (overkill)

**Recommendation: Rust + wasm-bindgen.** Reuses the existing Rust
gather-add codebase from handoff §3 that already does ternary
inference at 130M-1.5B scale. We strip out the speed optimizations,
keep the math, add the trace export.

Rough effort: 1-2 weeks for a competent Rust+WASM dev.

---

## 8. UI design (sketch)

Layout idea (browser, single-page app):

```
+----------------------------------------------------------+
|  TinyHetMoE — interact with a 13M ternary language model |
+----------------------------------------------------------+
|                                                          |
| Story so far: "Once upon a time, there was a happy"      |
|                                                          |
| Next token (top-5):                                      |
|   1. "boy"  (45%)  ← currently selected                  |
|   2. "girl" (28%)                                        |
|   3. "cat"  (12%)                                        |
|   4. "dog"  (8%)                                         |
|   5. "old"  (4%)                                         |
| [Pick another → branch the story]                        |
|                                                          |
+----------------------------------------------------------+
| 32 MEANING AXES (hover token to see)                     |
| For "happy": HAPPY 0.9 ████████░  GOOD 0.6 █████░        |
|              FEEL 0.4 ████░       SAD -0.2 ░             |
|              ... (28 more)                                |
+----------------------------------------------------------+
| 4 LAYERS × 4 HEADS attention (small heatmaps)            |
|  L0 H0 H1 H2 H3                                          |
|  L1 H0 H1 H2 H3   ← click any to enlarge                |
|  L2 H0 H1 H2 H3                                          |
|  L3 H0 H1 H2 H3                                          |
+----------------------------------------------------------+
| HetMoE expert routing (per token, per layer)             |
|  L0:  Std 0.4  SwiGLU 0.5  DN 0.05  BN 0.05              |
|  L1:  Std 0.1  SwiGLU 0.7  DN 0.15  BN 0.05              |
|  L2:  Std 0.2  SwiGLU 0.3  DN 0.45  BN 0.05              |
|  L3:  Std 0.5  SwiGLU 0.05 DN 0.05  BN 0.4               |
+----------------------------------------------------------+
| TERNARY WEIGHTS view (one Linear at a time)              |
|  L1 attention W_K (256 × 256):                           |
|  Red = -1, Gray = 0, Green = +1                          |
|  alpha = 0.072                                            |
|                                                          |
|  [animated, click to inspect a row]                      |
+----------------------------------------------------------+
```

Each panel is interactive:
- Click a token in the story → all panels update for that token
- Click a meaning axis → highlights tokens with high activation on it
- Click a layer/head → enlarges that attention map
- Click an expert → shows that expert's specialization (per-token type
  preference: nouns? action verbs? function words?)

### Educational arc

The demo can guide the user:
1. **"This is a ternary language model"** — show the weight view, point
   out red/gray/green
2. **"It uses 32 named meaning axes"** — show the axis bars, point out
   GOOD/HAPPY for happy words
3. **"It has 4 specialist FFNs"** — show the routing, point out which
   expert handles function words vs content
4. **"You can pick any next token"** — let them branch the story
5. **"The ternary weights are tiny"** — download size + inference speed

---

## 9. What we need from production work

This project doesn't depend on Highway-B converging — TinyHetMoE is
self-contained. But it will benefit from:
- The QAT code (`qat_utils.py`) — directly reusable
- The Highway model class — adaptable for tiny scale
- The contextual axis extraction script — adaptable for small vocab
- The trainer scaffolding (DDP off, plateau-detect, lr_scale resume) —
  reusable verbatim

**Disk impact:** model is small (5MB max ckpt × few). Disk no concern.

**GPU impact:** trains comfortably on a single 4060 Ti. We can wait
for Highway-B to converge or steal one GPU briefly.

---

## 10. Risks and mitigations

**Risk 1: 13M is too small to generate coherent text**
- Mitigation: 30M Eldan&Li reference. We're at 13M ternary which is
  ~3x effective compression of FP equivalent. Could expand to 20-25M
  if quality is bad.

**Risk 2: WASM development is harder than estimated**
- Mitigation: ONNX Runtime Web is a fallback path that gets 80% of the
  way without custom Rust work
- Could ship initial demo with Python+streamlit backend for UI to
  validate the design before WASM port

**Risk 3: The 32 chosen meaning axes don't anchor well on TinyStories vocab**
- Mitigation: anchor word selection is interactive — try a few axis
  sets, pick the cleanest separation score. Drop axes that don't
  anchor (like UNBOUNDED that we dropped in production).

**Risk 4: QAT-from-zero on 13M might not converge well**
- The 418M Highway-B run shows QAT-from-zero descends but slower than
  FP. At 13M scale, QAT might be too lossy
- Mitigation: train both FP-only and QAT-from-zero variants. Ship
  whichever has better generation quality. The "ternary" story still
  works either way (we can post-train quantize at the end).

**Risk 5: Eldan & Li used a specific recipe for TinyStories. Ours
might not work as well**
- Their secret was very high-quality synthetic data. We have the same
  data, so this risk is low.

---

## 11. Phasing

Following user's `a → c → b` request:

**a: Design doc (this).** Awaiting review.

**c: Wait for Highway-B production run to converge.** ETA 2-3 more
days at current rate, plus eval suite (1-2 days). During this time
the design can be refined based on review feedback.

**b: Build, in this order:**
1. Tokenizer training script + run on TinyStories (~1h)
2. 32-axis meaning .npy generator (~30min)
3. Training data preparation (.bin file) (~30min)
4. `train_tiny_hetmoe.py` adapted from train_highway_b.py (~half-day)
5. Smoke-test + full training run (~12h)
6. Generation quality check + adjust if needed (~1 day)
7. Trace export from Python forward pass (~half-day)
8. **Stop here** — that's the model. The WASM/UI work is a separate
   project that can be quoted/scoped independently.

---

## 12. Open questions for review

1. **Vocab size:** 5K target reasonable, or want smaller (2.5K = even
   simpler vocab) or larger (10K = more word coverage)?
2. **Param target:** 13M, or aim for 20-25M for more coherent
   generation?
3. **WASM approach:** Rust + wasm-bindgen (control), ONNX Runtime Web
   (faster), or something else?
4. **Meaning axes:** 32 my pick. Want different selection? Or keep as
   chosen?
5. **Scope of phase b:** stop at "model + Python trace export" and
   scope WASM/UI as separate project? Or include WASM build in
   phase b?
