# TinyHetMoE — Engineering Journal

Daily-ish log of what was tried, what worked, what blocked. Updated at
each gate (not just end-of-day). New entries on top.

---

## 2026-04-30 23:00 — Found and fixed a long-standing measurement bug (v5b numbers were wrong)

While debugging vocab padding for the v6.5 → v6.6 rebuild, hit a model
mismatch that exposed a real bug we've been carrying since v1: the
training-loop's loss path was computing CE through an FP shortcut for
`lm_head`, even when the rest of the network was running QAT
(ternary). The deployed inference path (and the exported tiny.bin in
the browser) **always** ran lm_head ternary. So the published "best
val" numbers on every QAT-trained model were measured in a forward
pass that doesn't match what gets deployed.

### The bug

`tiny_hetmoe.py:465` did:

```python
logits = h @ self.lm_head.weight.t()  # FP — no QAT on this matmul
```

…instead of going through `QuantizedLinear.forward(self.lm_head, h)`
which would apply `ternary_quantize` when `quantize=True`. The comment
said it was deliberate (probably for memory in the chunked-CE loop),
but it broke the train↔deploy correspondence.

### Effect on v5b — the published model

Re-evaluated `runs/tiny_hetmoe_v5b/checkpoints/best_qat.pt` properly
(`scripts/verify_v5b_deployed.py`):

| metric                       | published    | measured   | gap |
|------------------------------|-------------:|-----------:|----:|
| bf16 best.pt (val)           | 1.6056       | **1.5485** | sample noise |
| bf16 best.pt (PPL)           | 4.98         | **4.70**   | -- |
| **QAT best_qat.pt (val, deployed)**  | 1.5977       | **2.1873** | **+0.59 nats** |
| **QAT best_qat.pt (PPL, deployed)**  | **4.94**     | **8.91**   | **1.80×** |

The bf16 number was honest (no QAT, no bug). The QAT number — which
is what `docs/tiny.bin` actually delivers in the browser — is real
PPL ~9, not 4.94. The blog post and journal entries that say "PPL
4.94 deployed" are wrong.

### Affects every TinyHetMoE QAT model we've trained

v1–v5b all had this bug. v5b is the only one currently deployed
(`docs/tiny.bin`), so it's the only one with a public correction
needed. Future runs (v6.6+) use the fixed model code and will report
numbers that match deployment.

### The fix

Pre-quantize lm_head once outside the chunked-CE loop when QAT is on:

```python
if self.lm_head.quantize:
    w_lm = ternary_quantize(self.lm_head.weight)
else:
    w_lm = self.lm_head.weight
# ... loop computes logits = h @ w_lm.t()
```

This goes through the custom `TernaryQuantize` autograd function, so
STE backward is preserved. Memory cost is one extra (vocab,
input_dim) matrix during the forward — fits comfortably.

### Side effects to manage

  - `load_meaning_embeddings` made tolerant of larger axes files (take
    first vocab_size rows). Lets one axes file serve both old vocab=32768
    runs (v6.5) and new vocab=32772 runs (v6.6+).
  - QuantizedLinear.quantize is a Python attr, not a parameter →
    `load_state_dict` doesn't restore it. Eval/inference code needs
    explicit `set_quantize_mode(model, on=True)` for QAT checkpoints.
    Same applies to the Rust engine's per-expert QAT flag.

### Next: v6.6 fires the fix

Resuming pg19 + wiki from v6.5 `best.pt` with `--skip-optimizer`,
`qat_start_step=1`, `max_steps=30000` (3K QAT steps with the fixed
lm_head). First post-flip val is much higher than v6.5's was —
expected, since now lm_head also has to recover, not just the rest
of the network. Whether 3K is enough TBD.

---

## 2026-04-30 22:20 — v6.5 batch 1 done (pg19 + wiki, 1 GB, bf16 → QAT)

Both runs completed 30 K steps cleanly. QAT enabled at step 27 000
via the new 90 % backstop trigger (added to trainer earlier today).
3 K steps of QAT recovery after the hard-flip.

### Final numbers

| corpus | bf16 best (val) | bf16 PPL | QAT best (val) | QAT PPL | Δ (nats) |
|--------|----------------:|---------:|---------------:|--------:|---------:|
| pg19   |          3.9331 |     51.0 |         4.1848 |    65.7 |   +0.252 |
| wiki   |          3.4214 |     30.6 |         3.7471 |    42.4 |   +0.326 |

For comparison, v5b (TinyStories 38M) shipped at +0.4 nats QAT-vs-bf16.
Both v6.5 experts are below that on the harder corpora — comfortably
publishable as ternary deployments.

### Improvement vs v6 (200 MB / 10 K steps / bf16-only)

  • PG-19: 4.33 → 3.93 (bf16) = **−0.40 nats** from more data and
    longer training. QAT 4.18 still beats v6's bf16 by 0.15 nats.
  • Wiki:  3.78 → 3.42 (bf16) = **−0.36 nats**. QAT 3.75 also beats
    v6 bf16 by 0.03.

### QAT recovery curve

User flagged the 1.5B precedent — fast recovery in ~4 QAT steps then
beats fp. We saw a similar shape at smaller scale:

  • pg19: post-flip 4.28 → 4.29 → 4.24 → 4.23 → 4.18 → 4.21
    (steady descent from step 27 500 to 29 500, then mild oscillation)
  • wiki: post-flip 3.97 → 3.75 → 3.88 → 3.77 → 3.79 → 3.77
    (best @ 28 000, plateau-oscillation thereafter)

Different shapes — pg19 had more room to recover (deeper bf16 phase
was less converged), wiki settled fast. No soft-transition needed at
this scale; the longer bf16 pre-training (27 K steps on 1 GB) gave
enough weight stability that hard-flip is cheap.

### Why this matters for the swarm thesis

User's framing: 4 specialized 50 M experts, distributed over network
via WebSocket. Two distinct scenarios with very different latency
characteristics:

  • **Intra-LAN** (household swarm): per-token RTT ~1 ms. Network is
    invisible; compute on the active expert dominates.
  • **WAN swarm** (global distributed inference): per-token RTT
    ~50-200 ms (transcontinental ping baseline). Still cheaper per
    token than a current cloud LLM API call (30-80 ms first byte).

The interesting case is WAN — that's the true "dream" scenario where
any node anywhere serves a token. Even at 100 ms WAN RTT, the
architecture works because routing is sticky (98 % classifier acc =
neighbouring tokens overwhelmingly route to the same expert, so the
RTT cost fires once per turn, not once per token).

The UI for the 2-expert tab will surface this as a toggle: off /
LAN-sim (5 ms) / WAN-sim (50-100 ms), so the user can feel what each
distribution scenario actually looks like. Cold KV warmup on routing
switch is ~1-2 sec for a 500-token history regardless of network — a
fixed compute cost, not a transferred-bytes cost.

### Vocab patch interlude

Mid-run, realized the unified ChatML tool corpus needed `<|im_start|>`,
`<|im_end|>`, `<tool_call>`, `</tool_call>` as native single tokens.
Patched vocab from 32768 → 32772 by appending the 4 specials at the
end. Original tokens 0..32767 untouched (after recovering from a
swap-mode false start; verified by tokenizer round-trip + pg19 best.pt
generation test on a probe containing all 4 of the rare displaced
tokens — clean output, no embedding corruption).

In-flight pg19/wiki training was unaffected (axes loaded into GPU RAM
at startup; on-disk vocab patch happened mid-training but didn't
propagate). Their best.pt files have the original 32768 embedding
shape and need to be padded to 32772 (add 4 zero rows) before they
can be used with the new architecture for stitching.

### Tool corpus unification

Glaive + Salesforce/xLAM-60k + NousResearch/hermes-fc-v1 → unified
ChatML format (Hermes-style: `<tool_call>` / `<tool_response>` blocks,
JSON-Schema tool definitions, native ChatML special tokens).

  • glaive:  112 960 conversations, 263 MB
  • xlam:     60 000 conversations, 118 MB
  • hermes:    1 893 conversations,  17 MB
  • Total: **174 753 conversations, 386 MB raw**

Re-tokenized → 91 M training tokens, 0.93 M val. **Unk rate dropped
from 5.39 % (with mid-vocab swap) to 2.77 % (with append + ChatML
specials in vocab)** — every structural ChatML token now resolves to
its canonical single-token id, not `<unk>`.

### Padded-checkpoint verification — surfaced two real architecture facts

Padded all 4 (pg19/wiki × best/best_qat) from vocab=32768 → 32772 via
`scripts/pad_checkpoint_vocab.py`. New rows:
  - `meaning_embed`: filled from production Qwen contextual file
    (correct semantics for the 4 ChatML qids)
  - `intuition_embed`, `lm_head`: zero-padded
  - 4 ChatML tids must be **logit-masked at inference** for prose
    experts (pg19/wiki/code) since they should never emit ChatML.

Verified via `scripts/verify_padded_checkpoints.py` → val numbers
match training within 0.13 nats, generation is coherent prose for all
4 cases (bf16 + QAT both work).

**Two real architecture facts surfaced during this debugging:**

1. **`lm_head` must stay FP, not ternary**, in the deployed Rust
   engine. The model's `forward(ids, targets)` path uses an FP
   shortcut for the loss matmul (`tiny_hetmoe.py:465`) — the QAT
   training loop never saw the lm_head ternarized, so the learned
   weights are calibrated for FP. The current `export_model.py`
   ternarizes lm_head, which would compound error. Fix: keep
   lm_head as fp16 (~16 MB, modest bundle impact, big quality win).

2. **`QuantizedLinear.quantize` is a Python attr, not a parameter** —
   `load_state_dict` doesn't restore it. QAT checkpoints need
   explicit `set_quantize_mode(model, on=True)` after loading.
   Otherwise the model uses FP forward over the QAT-trained weights
   → garbage. Same fix needed in Rust engine: per-expert flag for
   "this expert's weights expect ternary forward."

3. **Per-expert `masked_tids` in sidecar metadata** — each .bin's
   `.meta.json` should include the list of vocab tids whose logits
   are masked at inference. For pg19/wiki/code: `[32768, 32769,
   32770, 32771]`. For tool: `[]` (it uses all of them). The wasm
   engine reads the sidecar, applies the mask in-engine post-logits.

Filed for the WASM dual-model work later in the session.

### Next gates

  1. Update v6.5 tool + code configs to vocab=32772.
  2. Commit + push (gitignore catches `.bin`/`.npy`/`.bak`/
     `tool_corpus_*`/`domain_classifier_*.pt`).
  3. Launch batch 2: tool + code on GPU 0/1.
  4. While batch 2 runs: build dual-model wasm engine + classifier-in-
     Rust (with per-expert masked_tids + QAT flag from sidecar).
  5. Manual browser testing of 2-expert stitched generation, with
     latency simulation toggles (off / LAN-5ms / WAN-50ms).
  6. After batch 2 lands: extend wasm + UI to 4-expert tab, publish.

---

## 2026-04-30 18:30 — v6 done; **stitching test passed, hierarchical MoE thesis confirmed**

Both bf16 v6 runs finished at step 10 000:

| corpus | best val | best PPL |
|--------|---------:|---------:|
| PG-19  |   4.3327 |     76.1 |
| Wiki   |   3.7815 |     43.9 |

(No QAT yet — these are the bf16 baseline. QAT comes in the next
phase with 1 GB corpora.)

### Cross-domain val matrix

The decisive question for the swarm thesis: do specialized experts
actually beat each other on their own domains? Ran both checkpoints
on both vals (200 K tokens each):

|              | pg19_val   | wiki_val   |
|--------------|-----------:|-----------:|
| pg19_expert  |   **4.38** |    7.99    |
| wiki_expert  |     8.69   |  **3.84**  |

In-domain advantage: **+4.30 nats** for PG-19, **+4.15 nats** for
Wiki. That's PPL ratio of roughly 75:1 in-domain vs cross-domain —
the wrong expert is barely literate on out-of-domain text. Not
"slight specialization." Specialization is the dominant signal.

### Stitched sequence (alternating 200-token chunks, 200 K tokens total)

Single-expert baselines on the stitched seq:
  • pg19_only: **CE 6.43** (PPL 618) — half the tokens are out-of-domain
  • wiki_only: **CE 6.47** (PPL 647) — symmetric

Oracle routing (ground-truth chunk labels, 50/50 routing):
  • **CE 4.40** (PPL 82) — almost matches in-domain numbers!
  • Saves **+2.03 nats** vs the best single-expert baseline.

Learned routing (132→64→2 classifier from earlier today, EMA over
meaning vectors per token):
  • **CE 4.79** (PPL 120) — beats single-expert by 1.64 nats
  • Gap to oracle: **+0.39 nats** — the cost of routing imperfection
    (mostly the 16-token lock-in window after each chunk transition,
    matching the 91 % acc-at-16-tokens from the classifier eval)

### What this proves (and what's next)

  • **Specialization is real** at this scale: per-domain experts crush
    cross-domain queries. The swarm thesis has empirical support.
  • **Experts compose**: oracle routing on a mixed sequence almost
    matches in-domain performance. Stitching works architecturally.
  • **Learned routing is viable** but the imperfection has a real cost
    (~0.4 nats). Reducible by (a) longer EMA windows, (b) sticky
    routing (only switch when confidence high), (c) bigger classifier.
    This is post-MVP optimization.

### Updated plan: phase 2

User direction: skip QAT for now (these were bf16-only), then do the
full production train next:

  1. Bump prep script to **1 GB per corpus** (5× current). Current
     200 MB / ~5 epochs caps PG-19 around step 5-6 K (saw the slope
     flatten). 1 GB / ~25 epochs gives plenty of room.
  2. Add **QAT recipe** to v6 configs: bf16-then-QAT with soft
     transition + plateau trigger + QAT-best tracking (all v5b
     validated tools).
  3. Train all **4 experts**: pg19, wiki, tool, code. Two GPUs → two
     batches of two, or one at a time.
  4. Re-run this stitching eval on ternary checkpoints (the
     production target). Expect oracle to take a small QAT hit but
     remain below single-expert baselines by a wide margin.

### Files written this gate

  • `scripts/eval_stitching.py` — cross-domain matrix + stitched
    routing eval. Reusable for the 4-expert version (just expand the
    `models` dict).
  • `data/domain_classifier_pg19_wiki.pt` — already saved earlier
    (10 KB).

---

## 2026-04-30 18:10 — domain classifier trained (while v6 still running)

The hierarchical-MoE thesis has two binding questions: can the experts
specialize, and can we route between them? With training still in
progress on GPU 0/1, decided to answer the routing question right
now — it's CPU-only and depends only on the meaning embedding (which
is frozen) and the val .bin files (which already exist).

### Setup

`scripts/train_domain_classifier.py`. Window of 512 tokens, EMA over
meaning vectors with α=0.05 (~20-token effective horizon), 132 → 64
→ 2 MLP, 30 epochs, AdamW. Trained on val .bin files only (held-out
from training data; classifier never sees what the experts saw).

  • pg19: 599 windows
  • wiki: 929 windows
  • train 1223, val 305 (80/20 split)

### Results

| metric                     | value     |
|----------------------------|----------:|
| held-out accuracy          | **98.69%** |
| pg19 per-class acc         |    99.15% |
| wiki per-class acc         |    98.40% |
| accuracy @ 16 tokens       |    91.30% |
| accuracy @ 32 tokens       |    96.14% |
| accuracy @ 64 tokens       |    98.04% |
| accuracy @ 256 tokens      |    98.30% |
| mean confidence margin     |     0.847 |
| frac high-conf (margin>0.9)|     57.0% |
| p50 margin                 |     0.935 |

### What this answers

The most uncertain part of the routing thesis was: do the frozen
132-axis Qwen-contextual meaning vectors actually encode *domain*
signal between two corpora that are both descriptive prose? Pg19 is
literary, wiki is encyclopedic — different in style and register,
not in surface syntax. If the axes mostly encode HAPPY / ANIMATE /
PAST_TENSE etc. without register, the classifier should top out
~70%.

It hits 98.69% — the axes have far more domain signal than I'd
calibrated. Possible explanations: (a) the production Qwen-coder
model was trained on data that gave its contextual embedding strong
register-awareness; (b) lexical density differs enough that even
register-naïve features separate the corpora; (c) both.

### What this implies

  • **Routing converges fast in real generation**. 91% accuracy at
    just 16 tokens means the outer router locks in within a sentence.
    For the chatbot, 50+ token prompts give effectively perfect
    routing.
  • **Routing is sticky**. 98%+ accuracy on 512-token windows means
    neighboring tokens overwhelmingly route to the same expert.
    "Switch every other token" failure mode that would kill LAN
    distribution doesn't happen at this corpus separation.
  • **The binary case is the hard one**. Pg19/wiki are closest
    semantically of the four corpora I prepped. Adding tool (JSON)
    and code (Python) — syntactically distinguishable in 2-3 tokens —
    should hit ≥99% trivially.

### Next gate

Wait for training (now ~step 7000+, both runs alive). Then:
  1. Cross-domain val matrix: pg19_expert × wiki_val, wiki_expert ×
     pg19_val. If `wiki_expert(wiki_val) > pg19_expert(wiki_val)` and
     symmetrically, **specialization is real** (the gap, not the
     absolute, decides the swarm thesis).
  2. Oracle-routing stitch: ground-truth-route a mixed text, see if
     the alternation produces coherent output. Isolates "do experts
     compose" from "can we route".
  3. Learned-routing stitch: same text but with the classifier from
     today driving routing decisions.

Saved: `data/domain_classifier_pg19_wiki.pt` (~10 KB).

---

## 2026-04-30 17:35 — v6 mid-run gate (step 5000 / 10 000)

Both runs healthy at the halfway mark. No plateau, LR still at
cosine-decayed 2.5e-4, throughput steady ~40 K tok/s on each 4060 Ti.

### Val trajectory (every 250 steps, only first val + key milestones)

| step | PG-19 val | PG-19 PPL | Wiki val | Wiki PPL |
|-----:|----------:|----------:|---------:|---------:|
|  250 |    6.6729 |       790 |   6.4113 |      609 |
|  500 |    5.7577 |       316 |   5.5014 |      245 |
| 1000 |    5.3745 |       216 |   5.0770 |      160 |
| 2000 |    4.9216 |       137 |   4.5085 |       91 |
| 3000 |    4.7613 |       117 |   4.3046 |       74 |
| 4000 |    4.6997 |       110 |   4.2183 |       68 |
| 5000 |    4.5591 |        96 |   4.1296 |       62 |

Wiki already crossed into the band I'd called "good at 10K" (4.2-4.7
val). PG-19 at the upper edge of its band (4.5-5.0). Expectation
revised upward — both should land below the bands by step 10 K.

### Slope analysis

PG-19 last 3 val deltas: -0.06, -0.09, -0.14, -0.03, -0.14
Wiki  last 3 val deltas: -0.09, -0.07, -0.09, -0.01, -0.09

Wiki slope flattening more than PG-19's — Wiki may be approaching its
bf16 floor for this 200 MB / ~5-epoch budget. PG-19 still has room
(literary text has more entropy per token, longer to learn).

### Why this is encouraging beyond the numbers

  • Unified vocab is **not** hurting either corpus. Was a real risk:
    the 32K Qwen vocab includes ~30% pure code tokens that wiki/pg19
    never see, which could waste embedding capacity. It doesn't.
  • Ratio Wiki/PG-19 (62 / 96 = 0.65) is consistent with what bigger
    models show on the same corpora — confirms PG-19 is genuinely
    harder, not that we're doing anything wrong on it.
  • No divergence, no NaN, no spikes. M+I + HetMoE handles harder
    text without the recipe changes that v5b needed for TinyStories.

### Next gate

Wait for completion (~25 min). Then:
  1. Eval both on each other's val (cross-domain). If pg19-expert is
     meaningfully worse on wiki than wiki-expert is on wiki — and
     symmetrically — specialization is real. That's the signal we
     need for the hierarchical-MoE thesis.
  2. Train 132→64→2 classifier on EMA of meaning vectors over the two
     domains. Measure routing stickiness on a mixed-text held-out.
  3. If both pass: kick off tool + code expert training. If they
     don't, debug specialization (might be that the domains aren't
     different enough at this scale).

---

## 2026-04-30 17:00 — v6 unified vocab dataprep done; PG-19 + Wiki training launched

Building the foundation for hierarchical HetMoE. Dropped TinyStories
(its vocab is a strict subset of PG-19, would give the per-token outer
router redundant domains) and standardised on 4 mutually-distinct
corpora, all sharing one trimmed Qwen2.5 vocab so domain experts can
be added later without re-tokenizing or re-training.

### Corpora (200 MB raw text each, streamed)

| corpus | source                     | tokens (train) | unk%  |
|--------|----------------------------|---------------:|------:|
| pg19   | emozilla/pg19 (parquet)    |     49,260,164 | 2.21  |
| wiki   | Salesforce/wikitext-103    |     46,018,387 | 3.24  |
| tool   | glaiveai func-calling-v2   |     43,701,052 | 1.55  |
| code   | bigcode/starcoderdata-py   |     49,455,287 | 5.60  |

Total ~187 M training tokens, ~2 M validation. `deepmind/pg19`
script-format breaks newer `datasets` (gzip-as-utf-8 decode error);
`emozilla/pg19` parquet mirror works.

### Vocab tuning

First run at MAX_VOCAB=24576 hit the cap (99.50% coverage of union)
and left code at 8.08% unk. Inspecting the dropped tokens showed real
Python subwords (`protobuf`, `_listener`, `(handler`, `Assignment`,
`_pending`) — not garbage. Bumped to **MAX_VOCAB=32768**: same 99.50%
union coverage but per-corpus unk dropped to 5.6% (code), 3.2% (wiki),
2.2% (pg19), 1.5% (tool). uint16-safe and only +14 M params on
embedding.

### Meaning axes

`scripts/make_unified_meaning_axes.py` against the production Qwen
contextual file: 100% direct lookup (32 764 of 32 764 non-special
tokens have non-zero rows in the production file), zero fallback
needed. Mean meaning-vector norm 8.80 — healthy.

### Smoke + parallel launch

50-step smoke on PG-19 GPU 0: loss 10.06 → 7.88, 43 K tok/s, 4.9 GB
mem on a 16 GB 4060 Ti. **Total params 51.97 M** — was 38 M at
vocab=16 K; the +14 M is purely the bigger embedding matrix.

Launched the real runs in parallel:

  • PG-19 → GPU 0, seed 1337, 10 000 steps (config `tiny_hetmoe_v6_pg19.json`)
  • Wiki  → GPU 1, seed 2025, 10 000 steps (config `tiny_hetmoe_v6_wiki.json`)

Same architecture as v5b (264/528, L=4, H=4, E=4 top-2). Pure bf16,
QAT off — this is a sanity-check for the unified vocab + harder
corpora. ETA ~50 min wall each. Outputs `runs/tiny_hetmoe_v6_pg19/`
and `runs/tiny_hetmoe_v6_wiki/`.

### Why this matters

This is the data foundation for the chatbot product:
1. PG-19 + Wiki experts trained now (this gate)
2. If their loss curves look reasonable and a 132→64→4 classifier on
   EMA-meaning vectors can route between them, we extend to
   {tool, code} → 4 domain experts
3. Outer per-token router selects which whole-HetMoE expert generates
   the next token; inner HetMoE routing handles architecture-level
   experts as before (M+I + Highway + 4 het experts top-2)

Doing it now with all 4 corpora baked into the vocab so we never have
to retokenize/retrain when adding the 3rd and 4th experts.

---

## 2026-04-30 16:00 — v5b extrapolation evaluation

Ran proper long-context evaluation on v5b's bf16 best (1.6056) and
QAT best (1.5977 / deployed). Three random val offsets averaged for
each, sequence length 4096 per offset.

### bf16 model — clean extrapolation through training range

| Range | PPL | Note |
|---|---|---|
| 256-512 | **4.09** | in training range |
| 512-1024 | **4.16** | in training range |
| 1024-2048 | **4.15** | in training range |
| 2048-3072 | 10.94 | 1.5× past training cap |
| 3072-4096 | 77.65 | 2× past training cap |

**Variable-length training works as designed.** PPL stays effectively
flat across 256 → 2048 (the longest training length). That's an
8× ratio with no degradation. Past the 2048 cap the model degrades —
expected, since no var_lengths sample was longer than 2048.

### QAT model — same shape, higher absolute PPL

| Range | PPL |
|---|---|
| 256-512 | 7.45 |
| 512-1024 | 7.78 |
| 1024-2048 | 8.23 |
| 2048-3072 | 34.57 |
| 3072-4096 | 159.58 |

The QAT model is uniformly worse than bf16 (~1.8× PPL), but the
**extrapolation pattern is identical**: flat in-range, sharp drop
past training cap. Ternary quantization didn't break the var-length
benefit.

### Comparison vs v4a (fixed-length 512)

v4a's earlier extrapolation test:
- in-range PPL 6
- 512-1024 PPL 38
- 1024-2048 PPL 237

v5b at 1024-2048 holds **PPL 8.23** (QAT) and **PPL 4.15** (bf16) —
about **30× better** than v4a in the same range. Var-length training
is the difference.

### Honest caveat

The 2048-cap collapse isn't a recipe failure but a data-cap failure
discussed in blog post 2: TinyStories docs are short, so 4K-token
val windows are sequences of unrelated short stories with `<eos>`
boundaries. Past the most recent boundary there's no useful signal
in the prefix, so attention to long prefix is mostly noise. A real
long-context test (PG-19 / WikiText) would tell a different story.

For now: **v5b deployed model holds clean PPL up to 2× the longest
training sequence**. That's the meaningful claim.

---

## 2026-04-30 15:10 — v5b finished. Hierarchical-MoE plan committed.

### v5b final result

**QAT best: val 1.5977 / PPL 4.94 at step 27500.** Pushed to docs/tiny.bin (live at https://surajbhan.github.io/tinyhetmoe/).

Trajectory across all the runs we've done:

| Run | Recipe | bf16 best | QAT best | Notes |
|---|---|---|---|---|
| v1 | QAT-from-zero, vote-backward | n/a | val 3.05 / PPL 21 | wrong gradient + low min_lr |
| v2 | bf16-then-QAT, low min_lr | 1.54 / 4.67 | 2.55 / 12.7 | LR starved QAT phase |
| v3 | bf16-then-QAT, **min_lr=2e-4** | 1.5155 / 4.55 | 2.40 / 11.0 | better but still vote-backward |
| v4a | + **STE backward** | 1.5155 / 4.55 (resumed) | **1.5734 / 4.82** | first ternary near FP floor |
| v5 | + **var-length training** | 1.6107 / 5.01 | (didn't finish QAT phase) | plateau detector bug masked stall |
| v5b | + **restart with --skip-optimizer** + plateau-bug fix | 1.6056 / 4.98 | **1.5977 / 4.94** | live deployment |

The ternary model deployed today is at PPL 4.94, **0.04 nats above the bf16 floor**. Effectively ternary matches FP at this scale.

### Diagnoses we had to make to get here

1. **Vote-backward is wrong at <200M scale.** Production 130M PoC uses STE; vote-backward only at 2B+. STE was the fix for v4.
2. **min_lr starved QAT phase.** Production uses min_lr=2e-4 (1.5× decay). We had 3e-5 (10× decay). Fixed in v3.
3. **NoPE alone doesn't give long-context.** Need variable-length training. v5+ uses {256/512/1024/2048} mix.
4. **bf16-then-QAT plateau detector bug:** rolling-window baseline let oscillating vals never trigger plateau. Fixed to compare against phase-appropriate all-time best.
5. **Trainer didn't save QAT-mode-best separately.** best.pt always landed on bf16 phase (lower val). Added best_qat.pt — saves whenever vloss < best_qat_val while QAT is on.
6. **Restart with --skip-optimizer un-sticks plateaus.** Validated empirically v5 → v5b: same config except fresh Adam state, broke through v5's plateau within 2K steps.

### Memory artifacts (under /home/suraj/.claude/projects/-data-sematic-embedding/memory/)

All saved as feedback/project memories so future-Claude doesn't re-discover:

- `feedback_data_dominates_architecture.md`
- `feedback_features_are_inductive_biases.md` (M+I/Highway/HetMoE are priors not overhead)
- `feedback_gpu_resident_corpus.md`
- `feedback_qat_after_lr_halve.md`, `feedback_qat_soft_transition.md`, `feedback_qat_trigger_logic.md`
- `feedback_ste_vs_vote_backward.md` (production 130M uses STE)
- `feedback_track_qat_and_fp_best_separately.md`
- `feedback_var_length_for_extrapolation.md`
- `feedback_restart_helps_convergence.md`
- `feedback_attention_dead_zones_at_4layer.md` (mid-layer attention is normally dead at shallow depth)
- `project_offline_browser_chatbot_product.md` (the productization vision)
- `project_hierarchical_domain_moe.md` (the next-gen architecture)

### What's deployed

- **Live demo:** https://surajbhan.github.io/tinyhetmoe/ — interactive WASM dashboard, click-to-extend story
- **Blog:** https://surajbhan.github.io/tinyhetmoe/blog.html — 2 posts narrating the build
- **Journal:** https://surajbhan.github.io/tinyhetmoe/journal.html — auto-rendered from this file
- Repo: https://github.com/surajbhan/tinyhetmoe
- Model: docs/tiny.bin, 15 MB packed HTMOE003, val 1.5977 / PPL 4.94

### Next-gen plan committed

User's vision: **HetMoE-of-HetMoEs**. Per-token outer router picks a domain expert (each whole HetMoE trained on different data); each domain expert internally does the existing HetMoE top-2 routing across architectures. Generation can fluidly switch domains: `[Tool] <call>...</call> [Tool] Result. [Stories] Wow, that's small. [Wiki] Mathematically, ...`.

System-level routing PER TOKEN. Inner architecture-level routing PER TOKEN PER LAYER (existing HetMoE). Two routing levels, both per-token.

**Sanity-test plan (next session):**

1. Stop v5b ✓ (done, snap at /tmp/v5b_best_qat.pt and /tmp/v5b_bf16_best.pt)
2. Build unified Qwen-tokenizer vocab covering TinyStories + WikiText (script ready: scripts/prepare_data_unified.py)
3. Train two models in parallel — one GPU each, no DDP:
   - GPU 0: Stories model, ~10K steps, unified vocab + v5b recipe (var-length, STE, min_lr=2e-4, qat_on_plateau, restart-on-plateau)
   - GPU 1: Wiki model, same recipe + same vocab + same architecture
4. Build small classifier (132 → 64 → 2 MLP, EMA over meaning vectors)
5. Sanity test: classifier should pick Stories on Stories prompts, Wiki on Wiki prompts, alternate token-by-token on mixed prompts
6. If passes: resume both to convergence, add Tool model (xLAM), publish

**Key architectural decisions:**
- Shared trunk, swap MoE per-token? OR independent models, swap per-token? (User's clarification: per-token-of-whole-HetMoE — option A in earlier discussion)
- KV cache implication: each domain model maintains its own KV; classifier switches active model. Feasible at our small scale.
- Vocab: Qwen2.5 tokenizer pruned to ~16K covering both corpora. Aligns with our existing meaning extraction pipeline.

**State of repo at end of this session:**
- v5b model deployed (PPL 4.94)
- Trainer has var_lengths, qat_backward_mode, qat_on_plateau, best_qat tracking, fixed plateau detector
- Long-context eval script exists (`scripts/eval_extrapolation.py`)
- Unified-vocab dataprep script ready (`scripts/prepare_data_unified.py`) — NOT YET RUN
- Disk: 198 GB used / 220 GB. ~11 GB free after cleaning v5/v5b intermediates.

### Tomorrow's workflow if context resets

1. Read CHECKPOINTS.md for snapshot inventory
2. Read this journal entry
3. Check memory/ for the project_hierarchical_domain_moe.md file — that's the live plan
4. v5b's snap is at `/tmp/v5b_best_qat.pt` (ternary, val 1.5977) and `/tmp/v5b_bf16_best.pt` (bf16, val 1.6056)
5. Run `scripts/prepare_data_unified.py` to build unified vocab + tokenize TinyStories + WikiText
6. Then train Stories + Wiki on separate GPUs (configs need to be written)
7. Build classifier, test routing

---

## 2026-04-30 11:25 — v4a stopped, v5 launched with var-length training

### v4a final result

QAT-best val **1.5734 / PPL 4.82** at step 38500. Stopped early because:
- Long-context test confirmed v4a collapses past seq_len=512 (training cap).
  PPL 6.1 in-range → PPL 38 at 1024 → PPL 237 at 2048 → PPL 405 at 3072.
  No amount of further training at fixed length 512 will fix that.
- v5 with variable-length training is the actual fix.

`docs/tiny.bin` updated to v4a's final best_qat (PPL 4.82, was 4.94).

### v5 plan: var-length from scratch

Trainer extended with `var_lengths` config (ported from production
Highway-B). Per-step sample one of:
- 256 × 32 (weight 0.25)
- 512 × 16 (weight 0.40) — primary
- 1024 × 8 (weight 0.25)
- 2048 × 4 (weight 0.10)

Plus full v4a recipe:
- STE backward (production 130M default; we validated this beats
  vote-backward at our scale)
- min_lr=2e-4 (production-aligned)
- plateau-trigger QAT
- max_seq_len bumped to 2048 in config
- 60K steps (~6h wall, ~2× v4a's wall time due to longer sequences)

### Why fresh from scratch

Resuming v4a's bf16 best (which was trained at 512) would propagate the
fixed-length attention statistics it learned. Cleaner to start from
init and let var-length shape attention from the start.

### Checkpoint registry

Added `CHECKPOINTS.md` listing named snapshots so we don't lose track
across runs. v4a best_qat is preserved at `/tmp/v4a_best_qat.pt`.

---

## 2026-04-30 09:43 — Trainer fix: save best_qat.pt separately

Caught an important bug while v4a was running: with bf16-then-QAT, the
trainer's `best.pt` always lands on the bf16 phase since bf16's val is
always lower than QAT's. **The QAT-mode best — what we actually deploy
to docs/tiny.bin — was never being saved as a single canonical file.**

Fix: track `best_qat_val` separately from `best_val`. Save
`best_qat.pt` whenever `qat_currently_on AND vloss < best_qat_val`.
Also fix plateau detector to compare against the phase-appropriate
best (in QAT phase, compare against best_qat_val so the detector
doesn't freeze at the bf16 floor).

Stopped v4a at step ~27000, fixed the trainer, resumed from
`ckpt_27000.pt`. v4a state is preserved (QAT mode active, optimizer
restored). Next val will produce the first `best_qat.pt`.

---

## 2026-04-30 09:20 — v4a launched: STE backward (production 130M default)

### Why STE not vote-backward

User pointed out that the production runs that "recovered in 4 vals
after QAT switch" were the **2B-scale** ones — which use very aggressive
integer routing (`go.sign() × sign(w_ternary)` then `.sign()` the
final gradient). The smaller production runs (130M PoC, qat_qwen35,
autoresearch 130M) all use **plain STE** (`return grad_output`).

We've been using vote-backward (`grad × sign(w)` with 0.1× zero rescue)
at 38M scale — neither of the two recipes that work in production.
That's likely why our QAT plateaus at PPL 11 instead of recovering to
the bf16 floor.

### v4 plan: A/B between STE and aggressive sign-routing

- **v4a (STE):** plain straight-through. 130M production default.
  Expected to recover to bf16 floor. Running now.
- **v4b (vote_sign):** if v4a doesn't recover, try 2B-style aggressive
  — sign(grad) × sign(w), final result quantized to ±1.

Also added `qat_backward_mode` config field to model + trainer so we
can A/B without code changes.

### v4a config

[`training/configs/tiny_hetmoe_v4a_ste.json`](training/configs/tiny_hetmoe_v4a_ste.json)
- Resume: v3 bf16 best (step 24000, val 1.5155)
- `qat_backward_mode: "ste"`
- `qat_start_step: 24050` (50 bf16 prime steps after resume, then QAT)
- `min_lr: 2e-4` (validated good in v3)
- `val_interval: 250` (so we measure recovery faster — production "4 vals" = 1000 steps)
- `max_steps: 40000` (16K post-QAT)

### Smoke test result (encouraging)

30 steps after QAT switch: val 2.98 (already below v3's plateau of 2.55+).
Loss descending fast: 5.74 → 4.49 → 3.91 → 3.58 → 3.30 → 3.20 (50 steps).
v3 with vote-backward at the same wall-time was at loss 4.5+.

### v3 ship in the meantime

v3 model already shipped at https://surajbhan.github.io/tinyhetmoe/
(15 MB packed HTMOE003). Demo is live. v4 will replace it if it
beats v3's QAT floor of 2.45.

---

## 2026-04-30 01:00 — v3 launched: min_lr fix after diagnosing v2

### v2 outcome

v2 hit bf16 PPL 4.67 (matching Eldan & Li 30M FP baseline) at step 17000.
Plateau-trigger fired at step 21000, QAT enabled. Loss jumped 1.54 → 4.93 in
100 steps and never recovered: 11K QAT steps later still at val ~2.55 / PPL
~12.7 with LR halved to 7.7e-5.

### Diagnosis

Read through `/data/training_kanam` production trainers (`train_poc.py`,
`qat_qwen35.py`, `autoresearch/train.py`, `2b_pipeline/train_2b.py`). Three
findings:

1. **Vote-backward math is identical** between us and them — `grad × sign(w)`
   with 0.1× rescue. Not the issue.
2. **Production uses `min_lr = 2e-4`** (autoresearch). We had `min_lr = 3e-5`.
   By the time QAT fires, our LR was already past the bf16-friendly range
   and continued cosine-decaying into QAT — vote-backward updates couldn't
   make big enough adjustments to recover.
3. **Production switches ternary much later in the schedule** (autoresearch:
   90% FP / 10% ternary) but they keep LR floored high so QAT phase has
   plenty of effective LR.

### v3 fix

One-line change: `min_lr: 0.0002` (was `0.00003`). Resumed from v2's bf16
best (step 17000, val 1.5418) — preserves all 4 hours of FP work, only
redoes QAT phase with corrected LR.

Config: [`training/configs/tiny_hetmoe_v3.json`](training/configs/tiny_hetmoe_v3.json)
- `resume_from: /tmp/v2_bf16_best_step17000.pt`
- `min_lr: 2e-4` (10× higher floor)
- `max_steps: 50000` (down from 60K — already 17K in)
- `qat_on_plateau: true` — should fire within 2-4K steps since model is
  already at bf16 floor
- All other recipe knobs unchanged

Live at 96K tok/s. ETA: plateau-triggered QAT in ~2K steps, then ~28K
steps of QAT-finetune at LR 2-2.7e-4. Total wall ~3h.

### What we expect

- QAT switch will still cause loss jump (vote-backward shock is unavoidable)
- BUT recovery should be much faster + reach lower floor because LR
  doesn't starve out
- Target: ternary val ≤ 2.0 (PPL ~7.5), beating v1 (PPL 21) and v2-QAT
  (PPL 12.7)
- Stretch: ternary val ≤ 1.8 (PPL ~6), close to bf16 floor

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
