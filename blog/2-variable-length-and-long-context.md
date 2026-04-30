# Variable-length training and what "long context" actually means

*Draft — 2026-04-30.*

The model has no positional encoding. No RoPE, no sinusoidal embeddings,
no learned position vectors. Just a causal mask that says "you can only
look backward."

This was supposed to give it infinite context. The reasoning sounded
good: positional encodings are what break down past their training
length (they're trained on positions 0–N, you can't extrapolate to
position N+1 cleanly). If you don't have one, you have nothing to
extrapolate. Free lunch.

Then I ran the long-context evaluation on a model trained at fixed
sequence length 512 and watched perplexity go from 6 in-range to **237
at position 1500**. Somehow without a positional encoding the model
had still learned a length-512 attention pattern.

This post is about what was happening, why "no positional encoding"
isn't the same as "infinite context," and what actually fixed it.

## How a transformer learns position without being told

Consider what attention sees during training. At every position
1, 2, 3, ..., 512, the query attends to a *causal prefix* of those
positions. The softmax over scores has a normalization that depends on
prefix length: at position 5 it normalizes over 5 keys; at position
500 it normalizes over 500 keys.

Even with identical weights, attention at position 500 produces
*statistically different* outputs than attention at position 5 — there
are simply more options to spread weight over. The model learns to
exploit this. It develops attention patterns that are tuned to "what
attention to a prefix of length-X looks like." When it generates at
position 600, it's seeing an attention distribution wider than any it
saw in training. Small distributional shift, large effect on the
hidden state, eventually compounds across layers, output goes off the
rails.

So position-information was leaking in through attention statistics
even though we never explicitly encoded it.

## What variable-length training fixes

The recipe is simple: instead of always training on length-512
sequences, **per optimizer step, sample one of {256, 512, 1024, 2048}
weighted appropriately.** A single mini-batch is all the same length
(so it can be batched cleanly), but across batches the model sees a
range of lengths.

After 10K-20K steps of this, the model has learned to handle attention
distributions across different prefix sizes. Position-conditional
patterns can't form because there's no privileged length to specialize
on.

Concretely, the long-context evaluation went from this (fixed-length
training):

| Range | PPL |
|---|---|
| 256-512 (training) | 6.1 |
| 512-1024 | 38 |
| 1024-2048 | 237 |

To this (variable-length training):

| Range | PPL |
|---|---|
| 256-512 | 5.0 |
| 512-1024 | 5.1 |
| 1024-2048 | 5.8 |

That's the difference between "the model breaks past its training
length" and "the model holds its quality." 23× less degradation in the
1K-2K range.

## Where the data caps you

But variable-length training only takes you as far as your *longest
training sequence*. We trained up to 2048. Past 2048 the model still
degrades — at 3K-4K it's back to PPL 30+.

That's not a recipe failure. It's a *data* failure that no recipe can
fix. Specifically:

**TinyStories is the wrong dataset for testing long context.**

Each TinyStories document is 100-500 tokens — a short kid's tale about
Lily and the bee. When I sample a 4K-token val window, I'm not seeing
a single 4K-token narrative. I'm seeing **5-15 separate stories with
end-of-story markers between them.**

The "long context" we're evaluating is "can the model handle a sequence
of unrelated short stories." That's a different, much weirder skill
than "can it track a single narrative across 4K tokens." Because there's
no useful information past the most recent `<eos>` boundary, the
attention pattern past that point is mostly noise from previous
unrelated stories.

This is why I keep suggesting that data quality dominates architecture
for training quality. It's also why our claim is now narrower: **at
TinyStories scale, our recipe gives clean extrapolation to 2× the
longest training length.** A real long-context evaluation would need a
dataset where 4K-token windows contain coherent single documents.

## What this means for the architecture claim

The architecture **does** support long context. NoPE means no positional
extrapolation problem. Variable-length training means no length-conditional
attention specialization. Meaning protection means semantic anchors
don't drift even at long range. Together those *enable* arbitrary-length
generation.

What they don't do — what nothing can do — is give you long-context
*skill* if your training data doesn't have long-context content.

For the production version of this stack: train on PG-19 (full novels),
WikiText (full articles), or web crawl with document-level sequences. Then
the long-context recipe pays off for real. For TinyStories, what you
have is what we have: clean extrapolation up to 2K tokens, beyond
which there's nothing useful in the data anyway.

— *S.*
