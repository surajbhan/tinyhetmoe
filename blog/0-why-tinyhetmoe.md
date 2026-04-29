# Why TinyHetMoE

*Draft — 2026-04-29.*

Most ML architecture papers end with a benchmark table and a link to a
GitHub repo. You read the table, you skim the repo, and you walk away
with a vague sense that "it works." You don't see the meaning vectors
moving. You don't see which expert fired on which token. You don't see
the ternary weights — the actual {−1, 0, +1} grid that produced the
answer.

I've spent the last few weeks building a ~418M-parameter language model
with a stack of architectural ideas that, individually, are non-obvious,
and collectively, are unique:

- **NoPE** — no positional encoding. Just a causal mask. The model
  learns position from the attention pattern itself.
- **Meaning + Intuition** — the input embedding is split. Half is
  trainable, like a normal embedding. The other half is anchored on 132
  named semantic axes (HAPPY, WANT, BIG, NOT, …) extracted from a
  bigger model's hidden states.
- **Highway expansion** — inside each block, the hidden dim expands 2×
  before attention/FFN, then compresses back. More compute per token
  for the same I/O cost.
- **HetMoE** — 4 *heterogeneous* expert FFNs per layer. Standard
  GLU, SwiGLU, DeepNarrow, Bottleneck. Top-2 routing. The router learns
  which token type each expert specializes in.
- **Ternary QAT** — every Linear weight, in the trained model, is one
  of three values: −α, 0, +α. The forward pass is ternary; the backward
  pass uses a "vote-backward" gradient (`grad × sign(w)`, with a 0.1×
  rescue at zero) instead of the usual straight-through estimator.

You can read about all of these. You cannot, easily, see them.

So the plan is to train a tiny version — 13M parameters, on TinyStories
— that ships in a browser as a WASM blob. ~1.5 MB compressed. Loads in
a second. You type a prompt, and as it generates each token you can:

- hover the token to see which of 32 meaning axes lit up,
- click any of the 4 layers × 4 heads to see the attention pattern,
- watch the routing weights for the 4 experts shift token by token,
- pull up the actual ternary weight grid, color-coded red/gray/green,
- pick a different top-K next token and branch the story.

The recipe is exactly the same as the 418M production model. The point
is to make the architecture *legible* — to anyone who can open a
browser tab. That's worth more than another arXiv preprint.

This blog is going to track the build. The journal is the boring
engineering log — daily notes, what broke, what I tried. The blog is
the narrative — a few longer posts about the interesting bits as they
come up.

Next post: probably about the tokenizer. The choice of 5K word-level
+ BPE + byte-fallback was the first interesting design call, and it
shaped a lot of what comes after.

— *S.*
