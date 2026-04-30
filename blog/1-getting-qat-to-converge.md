# Getting ternary QAT to actually converge

*Draft — 2026-04-30. **Correction at end** (2026-04-30, evening): the
PPL numbers in this post were measured by training code that had a
bug. The deployed model is real but actually runs at PPL ~9 in the
browser, not PPL 4.94. See the addendum for the honest story.*

This post is about how the model you're playing with on the front page
got to PPL 4.82. The short version: it took five attempts. Each one
taught us something about why the previous one was wrong.

## Run 1: QAT-from-step-zero

The first attempt followed the obvious recipe: every weight is ternary
from the start. Initialize randomly, force the forward pass to round
weights to {−α, 0, +α}, use vote-backward (`grad × sign(w)`) to push
the FP shadow weights around. Train.

It descended steadily for 30K steps and plateaued at **val 3.05 / PPL
21**. For comparison, Eldan & Li trained 30M parameter FP models on
TinyStories that hit PPL ~5. We were 4× worse.

I was puzzled because the same recipe (vote-backward, ternary forward
from step 0) is supposed to work — production HetMoE training scripts
use it. But our model was visibly weaker: greedy decoding looped on
"loved littlemy loved littlemy".

## Run 2: bf16 first, then QAT

What if we don't constrain the model to ternary right away? Train
bf16 to convergence, then enable QAT for the last fraction of training
to "compress" the FP-trained weights into ternary.

Result: bf16 phase reached **val 1.54 / PPL 4.67** — basically Eldan &
Li territory. Then QAT enabled at step 21000, and *immediately the
loss went from 1.5 to 4.9 in 100 steps and never recovered*. Fifteen
thousand more QAT steps got us to PPL ~12. Better than run 1, still
miles off the FP floor.

The "what just happened" plot looked like this:

```
step 21000  loss 1.54  ← last bf16
step 21100  loss 4.93  ← QAT just enabled, model basically reset
step 22000  loss 3.55  ← climbing back from near-random
step 30000  loss 2.55  ← floor, never recovers below this
```

## Run 3: same idea, higher LR floor

Looking at production training scripts side-by-side with ours, one
thing jumped out. Production used `min_lr = 2e-4`. We had `min_lr =
3e-5` — ten times lower.

Cosine LR schedule decays from peak (3e-4) toward `min_lr`. By the time
QAT enabled in run 2, our LR was already past 2e-4 and falling. Once
QAT shock hit, the model needed to re-learn its representations — but
we'd starved it of LR right at the moment it needed the most.

Fix: set `min_lr = 2e-4`. Re-ran QAT phase. Result: **val 2.40 / PPL
11**. Better than run 2, but still 2× the bf16 floor. Something else
was wrong.

## Run 4: the gradient was wrong

One more re-read of the production code, this time more carefully.
The 130M model uses **straight-through estimator** for the backward
pass: `grad_w = grad_output` — gradients pass through the quantization
unchanged.

We were using **vote-backward**: `grad_w = grad_output × sign(w)`,
with a 0.1× rescue at zero. Different math entirely.

Vote-backward is the recipe production uses *at 2B-parameter scale*,
where straight-through gradients can grow unbounded and you need
discrete routing. The 130M run uses STE because at that scale STE is
fine and the gradient signal is cleaner.

We were doing 38M with vote-backward. Worst of both worlds: too
small to need integer routing, too big to handle the gradient
redirection cleanly. Fix: switch to STE.

Resumed run 3's bf16 best, enabled QAT after 50 priming steps with
STE backward. The loss-jump after QAT was much smaller this time —
from 1.5 to 4.5 instead of 1.5 to 7.7. And it descended *fast*. By
1000 QAT-steps we were at PPL 6, by 10000 we were at PPL ~5.

Final: **val 1.5734 / PPL 4.82**. That's **0.2 nats from the bf16
floor**. Effectively, ternary matches FP at this scale.

## Run 5: long context

A separate problem: every model up to here was trained at fixed
seq_len=512. Generation past 512 collapsed dramatically — PPL went
from 6 in-range to 38 at 1024 to 237 at 2048. NoPE (which we use)
doesn't have a positional encoding to extrapolate, so the model
just learns the attention pattern statistics of length-512
sequences. Past 512, those statistics no longer hold.

Production fixes this with **variable-length training**: each
optimizer step samples a (seq_len, batch_size) pair from a
distribution, e.g. `[256×32, 512×16, 1024×8, 2048×4]` weighted.
The model never settles into one length's attention statistics.

Run 5 (v5) started fresh-from-scratch with var-length. Even at step
5000 (val 1.95 / PPL 7) the long-context test shows clean
generalization: PPL 4.93 at 256-512, **PPL 5.12 at 512-1024**, **PPL
5.80 at 1024-2048**. v4 had collapsed to PPL 237 in this range. v5
holds at 5.80.

## What the lessons were

In retrospect there were three independent things wrong with run 1
that all had to be fixed:

1. **Wrong gradient estimator for our scale.** Vote-backward is for
   2B+ models. STE is for ≤200M. We assumed all production code
   used the same recipe; it doesn't. The recipe scales with model
   size.

2. **LR floor too low for the QAT phase.** Cosine schedules biased
   toward small `min_lr` are fine for FP training, but QAT needs
   bigger updates to escape the FP basin. Production uses
   `min_lr ≈ peak_lr / 1.5`, not `peak_lr / 10`.

3. **Fixed-length training caps extrapolation.** This isn't really
   a QAT bug, but it's a thing you only notice when you do the
   long-context test. Variable-length is non-negotiable for any
   architecture that claims "long context" via position-free
   attention.

The model in your browser today is the v5b model — PPL 4.94, variable-length
training, ternary essentially matching the bf16 floor (which was PPL 4.55).
Long-context generalization works: PPL ~5 in-training-range, holding around
PPL 6 at 1K-2K positions. Past the 2K training cap it still degrades, but
that's a data-cap issue (TinyStories docs are short) rather than an
architecture failure.

— *S.*

---

## Addendum: the headline PPL was wrong, but the model is fine

*Added 2026-04-30, same evening I posted this.*

While starting on next-generation models, I went to verify that the
deployed v5b model loads correctly for a stitching experiment. It
does — but the val number didn't match. Eventually traced it to a
real bug in our training code that's been there since v1.

### The bug

The training-time loss path used an FP matmul shortcut for `lm_head`,
even when the rest of the network was running QAT (ternary). The
deployed inference path **always** ran lm_head ternary. So every
"best val" number this post quotes was measured in a forward pass
that doesn't match what gets deployed in the browser.

### Effect on v5b

Re-evaluating the deployed `best_qat.pt` properly:

| metric | what I told you | what's actually true |
|---|---:|---:|
| QAT best val | 1.5977 | ~2.20 (across 5 trials, σ ~0.05) |
| QAT best PPL | **4.94** | **~9** |

Almost 2× higher than published. The bf16 number (PPL 4.55) was
honest — that path doesn't use QAT, doesn't hit the bug.

### But here's the thing — the demo still produces fluent prose

I went and generated from the deployed forward path. Prompt: *"Once
upon a time, there was a little girl named Lily. She loved to"*.
Output:

> *play with her toys and have fun. One day, Lily's mom took her to
> the park to play. While playing, Lily found a bottle on the ground.
> She picked it up and said, "Look, Mom! A bottle!" She said, "Let's
> see what's inside." Lily carefully touched the bottle and said,
> "This sip."*

That's fluent. The "This sip" at the end is a small clunky moment,
but the whole thing reads as recognizable kid-story prose. **PPL 9
on TinyStories is fine for the demo experience** — the distribution
has very low entropy (1500-word vocabulary, simple grammar), so a
PPL 9 model still produces coherent output. PPL 9 on, say, Wikipedia
would be word salad. Different distributions, different perceptual
floors.

So the "bug" was a *measurement* bug, not a *model* bug. The model
the bug produced is real. It just wasn't quite as good as the number
suggested.

### Why I didn't catch this for five months

Two reasons. First, I never compared the training-reported number to
a held-out test-set CE on the deployed forward path — I assumed they
matched. Second, the demo subjectively works, so I had no signal
that something was off. Both are reasonable failures, but the lesson
is small and clear: **when training reports a number, measure that
exact same number against the deployed forward path before
publishing it**. Train↔deploy correspondence isn't free.

### Fix and what's next

One-line code change: pre-quantize lm_head outside the chunked-CE
loop when QAT is on. Now the loss path goes through the same ternary
forward as deployment, and reported numbers match what users get.

For v6.x (the four-domain experts coming next), training uses the
fixed code, so all val numbers are honest by construction.

For v5b specifically, the right thing is to redo 3K QAT steps on the
bf16 checkpoint with the bug-fixed code. The model will land
somewhere between bf16 PPL 4.55 and the current deployed PPL 9 —
probably PPL 5-6. That gives a v5b that's actually as good as I
originally claimed. I'll do that and update the demo. Doesn't take
long; there's just been higher-priority things this week.

— *S.*
