# The dream, instantiated — six experts on six imaginary computers

*2026-05-03. The follow-up to [post 4](#0-the-dream).
The browser tab is now actually doing the thing.*

## What we're trying to do

The thesis of stitched-MoE, stated as plainly as I can:

> **Distributed training and distributed inference over a LAN. Six
> small ternary models, each trained independently on its domain on
> whatever hardware happens to be free. At inference time, a tiny
> classifier picks which model handles the next token. The bet is
> that 6 well-specialized 114M experts can match the perceived
> quality of one ~700M general model — not by being smarter
> individually, but by always running the *right* one.**

Three claims in there. Each is testable:

1. **Distributed training**: train each expert on a separate box, on
   whatever GPU you can grab when you can grab it. The 6 experts in
   the [v7 demo](stitch_v7.html) were trained sequentially on a
   single machine because that was easiest, but architecturally
   nothing stops six volunteers from each training their own expert
   in parallel. The bottleneck isn't coordination; it's just disk
   space.

2. **Distributed inference over LAN**: at chat speed (10–30 tok/s),
   the wire cost between router and expert is ~530 bytes per token
   (token id + 132-dim meaning vector). WiFi handles that. Your
   phone could be the router and your gaming PC could host code\_py.
   Per-expert compute stays local because each node maintains its
   own KV cache by replaying the shared token stream.

3. **Specialization can substitute for scale, *if you route well*.**
   This is the load-bearing one. A 114M-param ternary model on
   conversational text alone is bad. A 114M model on math problems
   alone is genuinely competitive with a 0.5B teacher (PPL ~4 on
   orca-math). The classifier picks which one runs. If the classifier
   is right, you get 0.5B-class output for ~120 MB of weights and
   ~50 MB of routing infrastructure.

Post 4 ended with **"speculation, not roadmap."** This post is from
the other side. We built (1) and (2). (3) is the part the demo lets
you eyeball — and it works for some domains, fails for others, and
the failure mode is exactly what you'd expect.

## What's running in the tab

The [stitched v7 demo](stitch_v7.html) loads six 114M ternary experts
into your browser, plus a frozen meaning embedding shared across all
of them, plus a 39 KB classifier MLP. Total bundle ~390 MB; the demo
loads it all upfront because lazy fetch is a v8 feature.

Type a prompt and watch the graph at the top: six nodes arranged in a
hexagon, a router in the middle. Per token, the router picks an
expert and that node lights up. At "WAN ~50ms" latency the dispatch
animation paces visibly — you can read the shape of how a *real*
distributed inference would feel.

## The cast

Six experts. Each is a 114M-param TinyHetMoE — same architecture as
v6 but at full Qwen 2.5 vocab (151,665 tokens, no `<unk>` problem) and
with a base-pretraining → distillation pipeline:

| node | domain      | teacher                  | sft / distill data                      |
|------|-------------|--------------------------|-----------------------------------------|
| 1    | general     | Qwen 2.5 0.5B (base)     | SlimOrca + UltraChat                     |
| 2    | thinker     | Qwen 2.5 Math 1.5B (base) | orca-math-word-problems                  |
| 3    | code\_py    | Qwen 2.5 Coder 0.5B      | Magicoder OSS-Instruct (Python)          |
| 4    | code\_js    | Qwen 2.5 Coder 0.5B      | Magicoder OSS-Instruct (JS)              |
| 5    | medical     | Qwen 2.5 0.5B (base)     | medical-qa-datasets                      |
| 6    | legal       | Qwen 2.5 0.5B (base)     | open-australian-legal-qa                 |

Each expert is the same architecture, **distilled** from a 0.5B–1.5B
class teacher with the *exact* same vocab. The pipeline:

1. Pretrain a 114M base model per domain (next-token, ternary QAT,
   bf16-then-QAT recipe).
2. Distill from the matched teacher: feed both teacher and student the
   same raw text, minimize a top-K cross-entropy distillation loss
   over the teacher's top-50 logits.
3. Quality on in-domain val drops 4–5×; on out-of-domain val it doesn't
   move (it's mostly style transfer; we measured this honestly).

Total bundle: ~390 MB. Per-tab first load (router + shared embedding +
one expert): ~99 MB. The shared meaning embedding (40 MB fp16) is
downloaded once and used by every expert — that's the architectural
benefit of "Recipe A" frozen meaning.

## What the graph in the tab is showing you

Up top there's an SVG with a router circle in the middle and six
expert nodes around it. Per token:

1. Your input arrives at the router.
2. The router runs a tiny MLP on the EMA of recent meaning vectors —
   `132 → 64 → 6` softmax.
3. The argmax expert "node" lights up. A dot animates from the router
   to that node. The node pulses, computes, returns logits.
4. We sample the next token from those logits. Repeat.

The latency selector in the controls (Off / LAN / WAN) controls how
long the dot takes to travel. At Off it's basically instant — the
demo is locally-WASM, no real network. At "WAN ~50ms" the dot
visibly travels and the node pulse feels paced. **It's a simulation
of what the architecture would feel like if you actually deployed one
expert per real machine.**

Watch what happens with the example prompts:

- "The capital of France is" → almost everything routes to *general*
  (blue underline, node-1 keeps pulsing). The classifier is correctly
  recognizing "this is general-domain text."
- "def fibonacci(n):" → most tokens route to *code\_py* (green, node-3).
- The math example mixes thinker (gold, node-2) for the actual numeric
  reasoning.

The classifier won't be perfect — it's an 87% val accuracy MLP, and
the code\_py ↔ code\_js boundary is fuzzy. You'll see flutter. That's
honest. Real distributed deployments have to deal with this anyway.

## Things this is *not* claiming

It's not claiming the model is good at facts. It will tell you Paris
is in Germany, sometimes. The 114M-param-per-expert ceiling is what
it is. The OOV val numbers say distillation gets us style transfer
on the trained distribution but doesn't add knowledge from the
teacher's pretraining. That's a separate problem.

It's not claiming the dispatch animation maps to a real protocol.
Real distributed inference would need actual TLS handshakes, KV-cache
warmup on cold experts, retry on dead nodes. The demo skips all of
that.

It *is* claiming that the **routing decision per token costs nothing
meaningful** (the classifier MLP is 39 KB, runs in microseconds), the
**wire cost between nodes is negligible** (one token id + meaning
vector ≈ 530 bytes), and the **per-expert compute is independent**
(no cross-expert attention coupling).

So if you replaced the in-tab WASM with six WebSockets pointed at six
different volunteer Raspberry Pis, the demo would *behave the same*,
just with the Off/LAN/WAN slider replaced by reality.

## What's still wrong

The two big honest problems remain:

1. **Quality**. 114M ternary experts, even distilled, are not a
   substitute for a single 7B model on broad text. Distillation
   tightens the distribution shape but doesn't add world knowledge.
   Some of the experts (thinker on math, code\_js on common patterns)
   work; others (medical, legal) are about as good as their training
   distribution allows, which isn't great.

2. **Bundle size**. 393 MB total — you wouldn't ship this for a
   public website. The architecture supports lazy expert fetch (the
   classifier picks an expert; if you don't have it yet, fetch it
   then), but the current build loads all six up front. That's a
   v8 todo.

If you fixed both, you'd have something genuinely useful: a
~50-MB-per-load swap-cartridge model that lives entirely in your
browser cache, can answer most things competently, and would still
work if half your network went down because every other node has a
copy of the router.

## The thing I keep coming back to

In the post 4 sketch, the "swarm" framing was a bit cute — let
volunteers run experts on whatever boxes they have. The reality of
distributed-MoE-as-deployed seems less swarmy and more *boring*: it's
just CDN-as-model-host. You ship six 50 MB blobs to the CDN, the
browser fetches the one it needs based on the prompt's first token,
and the user never knows.

That's not as romantic, but it's a real shipping option. And the demo
in this tab shows it works at 100M-param-per-expert scale, today,
with no infrastructure beyond a browser.

The romantic version — actually different machines, gossip protocol,
verifiable replays — is still on the table. The boring version is
already shipped.
