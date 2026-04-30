# The trick where you turn it off and on again

*Draft — 2026-04-30.*

There's a debugging practice in software that's so universal it's a
joke: when something's broken, restart it. The IT-helpdesk question
"have you tried turning it off and on again?" works far more often than
it should.

Turns out it works on neural network training too. And I have data
now.

## The setup

I was training the v5 model. Variable-length sequences, the recipe
that finally gave us clean extrapolation. By step 11500 it had hit
val 1.6107 / PPL 5.01 — the bf16 floor for this run. After that:

```
step 12000:  val 1.7060
step 12250:  val 1.6948
step 12500:  val 1.7704
step 12750:  val 1.7037
step 13000:  val 1.7404
... (15 more vals around 1.65-1.85)
```

Bouncing in a band, never beating 1.6107, never plateau-detector-fire-able
because of a bug I had to fix later. The model was stuck in some basin
of attraction around the all-time best.

Standard ML practice when a model plateaus: halve the learning rate.
LR halving is supposed to help the model settle into a deeper minimum
near the current point. But sometimes it doesn't help — the model is
just *stuck*.

## The clean restart experiment

Stopped the run. Re-launched with **only one change**: the
`--skip-optimizer` flag. This means: load the model weights from
best.pt (so no quality lost), but throw away Adam's accumulated `m` and
`v` momentum estimates and start them fresh. Same recipe, same data,
same config, same RNG seed.

Within 2K steps:

```
step 13750 (v5b): val 1.6080 — beat v5's all-time best
step 15000 (v5b): val 1.6056 — new best, below PPL 5
```

The only thing that changed is the optimizer state. The model weights
were identical at the start. The data was the same. And it broke past
v5's wall in 2K steps — about 1/8 of how long v5 had been trying and
failing.

## Why does this work?

I'm honestly not sure I have a satisfying answer. Some plausible
candidates:

**1. Adam's momentum is stale.** The `m` and `v` accumulators are
exponential moving averages over gradients seen during training.
After tens of thousands of steps they reflect a long history. Bias
correction in Adam tries to handle this, but for a long run, the
estimates are still biased toward older parts of the loss landscape
that the model has now moved past. Restarting the optimizer aligns
its statistics to the *current* loss landscape.

**2. The cosine LR schedule resets to high LR.** When you restart,
your scheduler re-enters the warmup-then-peak-then-decay arc. So the
model gets another shot at exploring at a high LR before decay kicks
in. That's a real intervention — it's not just optimizer state, it's
an effective LR boost.

**3. RNG reset shifts the batch sequence.** Same seed but the run
started over, so the batches sampled in steps 11500-13000 of v5b are
different from those of v5. Maybe the original sequence had a run of
similar batches that biased the loss landscape.

**4. Numerical precision drift.** bf16 + Adam can develop tiny
denormalized values in optimizer state. After tens of thousands of
updates these accumulate without crashing but probably hurt subtly.
Restart drops them.

Probably some combination. I don't think there's One True Mechanism.

## What I now do differently

When a run plateaus, **the first thing I try is restart-from-best with
fresh optimizer.** Not change hyperparameters, not add a new trick,
not investigate the loss landscape. Just turn it off and on again.

It's free, it's fast, and based on the cleanish v5 → v5b experiment,
it works often enough to be the right default escape hatch.

If after a restart the model *still* plateaus — that's when you know
you have a real architectural or recipe issue, not just stale
optimizer state. So the restart is also a *diagnostic*: if it doesn't
help, you've ruled out a category of problem.

## A small philosophical aside

Many papers describe training as a deterministic process that
follows the loss landscape. Real training is much messier. Optimizer
state, learning rate schedule, batch order, numerical precision all
have small effects that accumulate. A long training run is more like
a chaotic system than a smooth optimization.

The restart trick exploits this. It's not solving a math problem;
it's perturbing a stuck dynamical system enough to find a different
trajectory. That's the kind of thing that doesn't make it into
papers but matters in practice.

Maybe the next paper title should be "Have You Tried Turning It Off
And On Again? An Empirical Study of Stochastic Training Dynamics."

— *S.*
