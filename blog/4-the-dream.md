# The dream — a thousand small experts on a global swarm

*Draft — 2026-04-30. Speculation, not roadmap. The runs in the next
room are still doing real work. But sometimes the runs give you
ideas.*

This started as a side conversation while two GPUs trained the v6
models. The plan was prosaic: train a PG-19 expert and a Wiki expert
on a unified vocab, then maybe add tool-calling and code experts,
then route between them per-token. A "hierarchical HetMoE" — outer
router picks which whole-domain-model handles the next token, inner
HetMoE routing handles per-layer expert selection as before.

That much is the actual plan. What follows is the place the plan
keeps trying to escape to.

## The thought that won't sit still

If routing between domain experts works at all, the experts are
**independent compute**. The outer router decides which one runs;
the others sit idle. That looks an awful lot like a dispatch problem,
not a model problem.

So: what if the experts didn't live on the same machine?

The first instinct is "attention is too coupled, you can't shard it
across a network." That's true if you shard *layers* across machines
(which is what Petals does). But hierarchical HetMoE shards *domains*,
not layers. Each domain expert is a complete model living on a
complete machine. The only thing that crosses the wire is the token
stream itself — 2 bytes per token at vocab=32K.

KV cache stays local. Each node maintains its own KV by replaying
the shared token stream through its own embedding + attention. The
cache is reproducible from the token sequence; it doesn't have to
move. The only LAN cost per token is the new token id, which fits in
a uint16, plus optionally the 132-dim meaning vector if the router
needs it (~530 bytes total).

At chat speed (10-30 tok/s), that's a few KB/sec. WiFi can do that.
Your phone can do that.

## The household chatbot, distributed

So one cleaner architecture for the offline-browser-chatbot product
might be: a small dispatcher node (could be a phone, could be a
Raspberry Pi) holds the tokenizer and outer router. Four other
machines on the same LAN — old laptops, mini PCs, whatever's lying
around — each hold one full ~50M domain expert. Code on the dev
laptop, literary on the Kindle-replacement, encyclopedic on the
desktop, function-calling on the Pi.

When you type a message, the dispatcher tokenizes, the router scores
the meaning vector against the four domains, and the winning machine
generates the response. The other three either idle (single-user
case, where catch-up cost is invisible to a human) or run attention
only to keep their KV synced for fast switches (multi-user case).

The whole thing is offline. Nothing leaves your house. If one node
goes down, you lose one domain — the rest keep working.

There's a real privacy story here too: your code never enters the
literary expert's RAM. Sensitive code stays on one machine. Different
nodes can have different security postures.

This is just a different way of doing what we'd already do. It's not
the dream yet.

## Where it gets weird

The dream starts when you stop asking "which machines do *I* own?"
and start asking "which machines does the *world* have idle?"

If a domain expert is 50M params at 2-bit ternary, it's 12 MB on
disk. It runs on a phone. It runs on a Raspberry Pi. There are
hundreds of millions of those.

So: a global network. Anyone can host an expert. The protocol pays
them per token served. A user's dispatcher routes their query to
whichever node is fastest, cheapest, and trusted for that domain.
The node serves the token, posts a tiny proof, gets paid, the chain
settles.

A thousand experts. Each one specialized for something narrow:
embedded C, Postgres internals, Tagalog poetry, 19th-century
botanical taxonomy, late-night commit-message generation. Together
they form a model that no single machine could ever hold —
effectively 50 billion parameters of capacity, but no node ever
needs more than 50 million.

The router is the auctioneer. The 132-axis meaning vector is the
public bid signal: this token has these semantic features, which
expert wants to serve it? Nodes self-select for the experts they
believe they're best at. Specialists win their niches. Generalists
get out-bid.

## The verification problem, and why ternary helps

The reason none of this exists yet is that you can't pay people for
work you can't verify. If a node returns "yeah I generated that
token, send the crypto," you have to be able to *prove* they
actually ran the model.

Today's verification options for ML inference are all bad:
- Trusted execution environments (closed hardware, walled gardens)
- ZK-proofs of inference (heavy, slow, expensive)
- Stake-and-slash with off-chain disputes (Bittensor's approach;
  works for training but flaky for chat-speed inference)

Here's where the architecture quietly helps: **ternary weights are
deterministic at inference**. There's no float nondeterminism, no
matmul rounding, no "well it depends on the cuDNN version." Given
the same token id, the same KV cache hash, and the same expert
weights, every node will compute *exactly the same* logits down to
the bit.

That means verification is just **replay**. A random validator
samples a small fraction of served tokens, replays them on their own
copy of the expert, checks the bits match. If they don't, the node
gets slashed. If they do, payment settles.

Replay is cheap — it's literally one forward pass through a 50M
model — and it composes. Ten validators each spot-checking 1% of a
node's claims gives 10% coverage. Cheating becomes expected-loss
negative very fast.

Float models can't do this. Ternary models can. That's a structural
advantage I didn't appreciate until thinking through this.

## Why a thousand and not ten or a million

There's a natural number. Below a hundred experts, domains overlap
too much — the outer router has nothing useful to disambiguate, and
"specialization" is a fiction. Above ten thousand, the routing
decision itself becomes harder than running an expert. You'd spend
more compute deciding who serves the token than serving it.

A thousand feels right. Enough room for a long tail (every
programming language gets its own expert; every literary tradition;
every academic subfield). Not so much room that you're routing into
noise.

A thousand is also roughly the number of natural specializations a
human civilization has. Not a coincidence.

## The empirical question that decides everything

None of this matters if specialization doesn't actually beat
generality at small scale. If a 50M PG-19 expert isn't meaningfully
better at literary continuation than a 50M general-purpose model
trained on the union of all four corpora, the swarm has no thesis.
You'd just train one bigger general model and ship it.

And the honest answer is: I don't know yet. The runs that finish in
the next forty minutes give us the first data point. If the PG-19
expert's val on PG-19 is meaningfully below what a unified-trained
model would hit on the same eval, the specialization story has
legs. If it isn't, it doesn't.

Either way it's a real number, and it's about to exist.

That's what makes this fun. The dream is downstream of an empirical
question I can answer with a 30-minute experiment. Most dreams
aren't.

## The economics if it works

The fun part of working through this honestly: the economics aren't
crazy.

A 50M expert running on a consumer GPU costs around $0.000001 per
token in electricity. Settling on a chain that charges $0.001 per
token leaves three orders of magnitude of margin to split between
the node operator and the validators and the protocol. That's
sustainable margin even after fees.

A user generating a thousand tokens a day spends a tenth of a cent.
A node operator with a single 4060 Ti serving a popular expert
might earn enough to cover their power bill. A few well-trained
specialists could earn meaningfully more.

This isn't going to disrupt OpenAI. It's an alternative shape of
the same supply curve: less centralized, more privacy-preserving,
cheaper at the long tail, slower at the head. Different niche, not
the same fight.

## The reason this is just a blog post and not a project

Two GPUs are training right now. The reasonable thing to do is
finish the four-domain experiment, see if specialization is real,
ship the offline-browser chatbot product, and then — only then —
think about whether any of this dream is worth pursuing for real.

But I wanted to write this down because the architecture decisions
we're making *right now* — ternary weights, frozen meaning axes,
domain-not-layer sharding, deterministic ternary inference — are
the exact decisions a future protocol like this would need.

Not by accident. The constraints that made TinyHetMoE work as a
small model (M+I priors, ternary efficiency, HetMoE specialization)
are the same constraints that would make it work as a swarm. Small,
deterministic, modular, specializable.

If the dream ever happens, it won't be because someone designed for
the dream. It'll be because the dream was a natural extension of
the boring choices that worked at the small scale.

That's how most good dreams go.

---

*If you're reading this and you're someone who actually works on
decentralized inference protocols — Bittensor, Petals, Akash,
Gensyn, anyone in this space — I'd love to compare notes. The
specific question I have is whether deterministic-replay
verification at the per-token granularity has been tried, and if
so, what blocked it. My intuition says ternary makes it newly
viable, but my intuition is often wrong.*
