//! Hierarchical-MoE stitched engine.
//!
//! Holds N domain experts (independent Model + KV cache + scratch each),
//! a frozen 132→hidden→K classifier, and a small EMA-of-meaning state.
//! Each `step()` call:
//!   1. Looks up the new token's meaning vector
//!   2. Updates the EMA
//!   3. Runs the classifier on the EMA → picks expert idx + confidence
//!   4. If the chosen expert's KV is behind the active position, replays
//!      tokens forward to catch it up (cold-cache warmup)
//!   5. Runs that expert's forward pass on the new token
//!   6. Returns logits + route info (chosen idx, confidence, probs)
//!
//! Constraints assumed:
//!   - All experts share the same vocab (same meaning_embed; passed to the
//!     classifier via the first expert's model).
//!   - Each expert may have its own `masked_tids` (tids the engine zeroes
//!     in logits before sampling). Used so prose experts never emit ChatML
//!     special tokens, etc.
//!   - Force-route mode: caller can pin a specific expert for the next
//!     N tokens (overrides classifier).
//!
//! The struct is `cfg(feature = "wasm")` neutral — used both natively
//! (CLI driver in main.rs / scripts) and from the wasm bindings.

use crate::{
    forward::{forward_token_with_trace, TokenTrace},
    model::{Config, KVCache, Model, ScratchBuffers},
};

/// Per-expert resources and metadata.
pub struct StitchExpert {
    pub name: String,
    pub model: Model,
    pub kv: KVCache,
    pub scratch: ScratchBuffers,
    pub masked_tids: Vec<u32>,
    /// History of tokens this expert has been advanced through. Lets us
    /// replay forward to catch up a cold expert without losing stuck-in-
    /// kv state. (We could read from the engine's global token log, but
    /// keeping per-expert position simpler.)
    pub synced_pos: usize,
}

impl StitchExpert {
    pub fn new(name: String, model: Model, masked_tids: Vec<u32>) -> Self {
        let kv = KVCache::new(&model.config);
        let scratch = ScratchBuffers::new(&model.config);
        Self {
            name, model, kv, scratch, masked_tids,
            synced_pos: 0,
        }
    }
    pub fn reset(&mut self) {
        self.kv.reset();
        self.synced_pos = 0;
    }
    pub fn config(&self) -> &Config { &self.model.config }
}

/// Frozen 132 → hidden → K classifier with per-feature z-score normalization
/// and a sliding-window mean over the last `window_size` meaning vectors.
///
/// Why a flat window instead of an EMA: an EMA with α=0.05 has effective
/// memory of ~20 tokens, which means after a short prompt the EMA drifts
/// to the running generation and the classifier follows the model's
/// hallucinations rather than the user's intent. A flat window of N=64
/// tokens keeps the prompt's domain signal load-bearing for many more
/// generation steps, while still allowing genuine domain drift when the
/// generation produces enough cross-domain content to flip the average.
///
/// We KEEP `ema_alpha` in the struct for back-compat with stitch.json
/// schemas that include it — but the math now uses a sliding window.
pub struct DomainClassifier {
    pub w1: Vec<f32>,         // hidden × 132
    pub b1: Vec<f32>,         // hidden
    pub w2: Vec<f32>,         // K × hidden
    pub b2: Vec<f32>,         // K
    pub feat_mu: Vec<f32>,    // 132
    pub feat_sd: Vec<f32>,    // 132
    pub hidden: usize,
    pub k: usize,
    pub meaning_dim: usize,
    /// Ring-buffer of the last `window_size` meaning vectors (each `meaning_dim` long).
    /// Stored flat: rows are window positions, cols are meaning dims.
    pub window: Vec<f32>,
    pub window_size: usize,
    /// How many of the window slots are filled. Caps at `window_size`.
    pub filled: usize,
    /// Index where the next push goes. Wraps modulo `window_size`.
    pub head: usize,
    pub ema_alpha: f32,       // unused at runtime; kept for stitch.json compat
    /// Cached classifier-input mean across the window. Updated on every push.
    pub ema: Vec<f32>,
    /// Blend weight: classifier feature = flat_blend·flat_mean + (1-flat_blend)·attention_pool.
    /// 0.0 = pure attention (most reactive, can over-respond to single
    /// hallucinated tokens); 1.0 = pure flat mean (most stable, equivalent
    /// to the old uniform-window behavior). Default 0.5.
    pub flat_blend: f32,
}

impl DomainClassifier {
    pub fn new(
        w1: Vec<f32>, b1: Vec<f32>,
        w2: Vec<f32>, b2: Vec<f32>,
        feat_mu: Vec<f32>, feat_sd: Vec<f32>,
        meaning_dim: usize,
        ema_alpha: f32,
    ) -> Self {
        let hidden = b1.len();
        let k = b2.len();
        // Flat sliding-window over the last 64 meaning vectors. 64 is
        // long enough to keep prompt context load-bearing during ~30
        // generated tokens, short enough that genuine domain drift
        // eventually flips the route.
        const WINDOW_SIZE: usize = 64;
        Self {
            w1, b1, w2, b2, feat_mu, feat_sd,
            hidden, k, meaning_dim,
            window: vec![0.0; meaning_dim * WINDOW_SIZE],
            window_size: WINDOW_SIZE,
            filled: 0,
            head: 0,
            ema_alpha,
            ema: vec![0.0; meaning_dim],
            flat_blend: 0.5,
        }
    }

    /// Push a new meaning vector into the sliding window and recompute
    /// the classifier input as a **blend of attention pool + flat mean**.
    ///
    /// Two pieces:
    ///   - **Flat mean**: average over all filled window slots. Anchors
    ///     decisions in the overall window content (e.g. a JS prompt
    ///     keeps voting JS even when the model emits math-flavored
    ///     hallucinations). Stable, slow to drift.
    ///   - **Attention pool**: query = current token's meaning;
    ///     keys/values = each PRIOR slot in the window (self-attention
    ///     masked). Captures "what kind of token is this, given similar
    ///     past tokens." Reactive, picks up genuine domain shifts.
    ///
    /// Final feature = FLAT_BLEND·flat + (1 - FLAT_BLEND)·attention.
    /// At 0.5 the two contribute equally; the prompt context never
    /// fully drops out, but the attention pool can still flag genuine
    /// drift when generation crosses domains.
    pub fn update_ema(&mut self, meaning: &[f32]) {
        let flat_blend = self.flat_blend;

        let new_slot = self.head;

        // Push the new vector into the head slot
        let off = new_slot * self.meaning_dim;
        for i in 0..self.meaning_dim {
            self.window[off + i] = meaning[i];
        }
        self.head = (self.head + 1) % self.window_size;
        if self.filled < self.window_size {
            self.filled += 1;
        }

        let dim = self.meaning_dim;

        // First-token corner case: just use the new vector directly.
        if self.filled == 1 {
            for d in 0..dim { self.ema[d] = meaning[d]; }
            return;
        }

        // ── Flat mean over ALL filled slots (includes new) ────────────
        let mut flat = vec![0.0_f32; dim];
        for slot in 0..self.filled {
            let s_off = slot * dim;
            for d in 0..dim { flat[d] += self.window[s_off + d]; }
        }
        let inv_n = 1.0 / (self.filled as f32);
        for d in 0..dim { flat[d] *= inv_n; }

        // ── Attention pool over PRIOR slots (mask self) ───────────────
        let scale = 1.0 / (dim as f32).sqrt();
        let mut scores = vec![0.0_f32; self.filled];
        let mut max_s = f32::NEG_INFINITY;
        for slot in 0..self.filled {
            if slot == new_slot { continue; }
            let s_off = slot * dim;
            let mut s = 0.0_f32;
            for d in 0..dim { s += meaning[d] * self.window[s_off + d]; }
            s *= scale;
            scores[slot] = s;
            if s > max_s { max_s = s; }
        }
        let mut sum = 0.0_f32;
        for slot in 0..self.filled {
            if slot == new_slot { scores[slot] = 0.0; continue; }
            scores[slot] = (scores[slot] - max_s).exp();
            sum += scores[slot];
        }
        let inv_sum = if sum > 1e-20 { 1.0 / sum } else { 0.0 };
        for slot in 0..self.filled { scores[slot] *= inv_sum; }
        let mut attn = vec![0.0_f32; dim];
        for slot in 0..self.filled {
            if slot == new_slot { continue; }
            let w = scores[slot];
            let s_off = slot * dim;
            for d in 0..dim { attn[d] += w * self.window[s_off + d]; }
        }

        // ── Blend ──────────────────────────────────────────────────────
        for d in 0..dim {
            self.ema[d] = flat_blend * flat[d] + (1.0 - flat_blend) * attn[d];
        }
    }

    /// Reset window to empty. Caller invokes between prompts.
    pub fn reset(&mut self) {
        for v in self.window.iter_mut() { *v = 0.0; }
        for v in self.ema.iter_mut() { *v = 0.0; }
        self.filled = 0;
        self.head = 0;
    }

    /// Predict expert. Returns (k probabilities, chosen_idx, confidence).
    /// Confidence = top1 - top2 (margin).
    pub fn predict(&self) -> (Vec<f32>, usize, f32) {
        // Normalize EMA features
        let mut x = vec![0.0_f32; self.meaning_dim];
        for i in 0..self.meaning_dim {
            x[i] = (self.ema[i] - self.feat_mu[i]) / self.feat_sd[i];
        }
        // h = ReLU(W1 @ x + b1)
        let mut h = vec![0.0_f32; self.hidden];
        for i in 0..self.hidden {
            let mut acc = self.b1[i];
            let row = &self.w1[i * self.meaning_dim .. (i + 1) * self.meaning_dim];
            for j in 0..self.meaning_dim {
                acc += row[j] * x[j];
            }
            if acc < 0.0 { acc = 0.0; }
            h[i] = acc;
        }
        // logits = W2 @ h + b2
        let mut logits = vec![0.0_f32; self.k];
        for i in 0..self.k {
            let mut acc = self.b2[i];
            let row = &self.w2[i * self.hidden .. (i + 1) * self.hidden];
            for j in 0..self.hidden {
                acc += row[j] * h[j];
            }
            logits[i] = acc;
        }
        // Softmax
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for v in logits.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in logits.iter_mut() { *v /= sum; }
        // Argmax + margin
        let mut top1 = 0;
        for i in 1..self.k {
            if logits[i] > logits[top1] { top1 = i; }
        }
        let mut top2_val = 0.0_f32;
        for i in 0..self.k {
            if i == top1 { continue; }
            if logits[i] > top2_val { top2_val = logits[i]; }
        }
        let margin = logits[top1] - top2_val;
        (logits, top1, margin)
    }
}

/// One step's output: logits (post-mask) + routing info.
pub struct StitchStep {
    pub chosen_expert: usize,
    pub classifier_probs: Vec<f32>,     // length K
    pub confidence: f32,                // top1 - top2 margin
    pub logits: Vec<f32>,                // length vocab_size, masked
    /// True if a cold-warmup happened this step (caller can show a "warming
    /// up Node N" toast).
    pub warmup_happened: bool,
    /// Number of replay tokens during warmup (zero if no warmup).
    pub warmup_tokens: usize,
    /// If the classifier WANTED an expert that isn't loaded yet, this is
    /// that expert's index; the engine fell back to `chosen_expert`. The
    /// caller (JS) should fetch + add_expert(idx) and re-step. Length of
    /// `classifier_probs` matches num expert SLOTS (= classifier K), not
    /// num loaded experts.
    pub pending_expert: Option<usize>,
}

/// Override mode for routing: classifier-driven (None) or forced.
pub enum RouteMode {
    Classifier,
    Forced(usize),
}

/// One slot in the engine's expert pool — either filled (model loaded)
/// or empty (lazy-load not yet completed). `step()` picks the classifier's
/// argmax slot; if it's empty, falls back to the first filled slot.
type ExpertSlot = Option<StitchExpert>;

pub struct StitchedEngine {
    /// Indexed by classifier-output idx (length = classifier.k). Each
    /// slot is None until the JS side fetches + add_expert_at()s it.
    pub experts: Vec<ExpertSlot>,
    /// Shared meaning_embed (frozen, same across all 6 experts in v7).
    /// Held at engine level so we don't depend on having any expert loaded
    /// to look up token meaning vectors for the classifier.
    pub meaning_embed: Vec<f32>,
    pub meaning_dim: usize,
    pub vocab_size: usize,
    pub classifier: DomainClassifier,
    /// Token history, used for cold-cache warmup replay.
    pub token_history: Vec<u32>,
    /// Current routing mode.
    pub mode: RouteMode,
    /// Last classifier-chosen expert (sticky-routing state). Set on first
    /// step; updated only when the new top-1 has confidence margin
    /// >= switch_threshold. Reset to None on engine.reset().
    pub last_chosen: Option<usize>,
    /// Minimum margin (top1 - top2) required to SWITCH expert away from
    /// last_chosen. Higher = stickier routing. 0.0 = always-follow-classifier.
    /// Default 0.3 — modest hysteresis that prevents flutter on ambiguous
    /// short prompts without dampening real domain transitions.
    pub switch_threshold: f32,
}

impl StitchedEngine {
    /// Construct with classifier + shared meaning embedding, but ZERO
    /// experts. Caller adds experts via `add_expert_at(idx, expert)` as
    /// they're fetched. v7 entry point.
    pub fn new_lazy(
        classifier: DomainClassifier,
        meaning_embed: Vec<f32>,
        meaning_dim: usize,
        vocab_size: usize,
    ) -> Self {
        assert_eq!(classifier.meaning_dim, meaning_dim,
            "classifier meaning_dim must match meaning_embed dim");
        assert_eq!(meaning_embed.len(), vocab_size * meaning_dim,
            "meaning_embed size {} != vocab*dim {}",
            meaning_embed.len(), vocab_size * meaning_dim);
        let k = classifier.k;
        let experts: Vec<ExpertSlot> = (0..k).map(|_| None).collect();
        Self {
            experts,
            meaning_embed, meaning_dim, vocab_size,
            classifier,
            token_history: Vec::new(),
            mode: RouteMode::Classifier,
            last_chosen: None,
            switch_threshold: 0.3,
        }
    }

    /// Legacy v6 entrypoint: takes a non-empty Vec<StitchExpert>, derives
    /// the shared meaning from experts[0]. Used by the old all-upfront
    /// constructor in wasm_api_stitch.rs.
    pub fn new(experts: Vec<StitchExpert>, classifier: DomainClassifier) -> Self {
        assert!(!experts.is_empty(), "need at least one expert");
        let cfg0 = experts[0].config();
        for e in &experts[1..] {
            let cfg = e.config();
            assert_eq!(cfg.vocab_size, cfg0.vocab_size,
                "expert {} vocab_size mismatch", e.name);
            assert_eq!(cfg.meaning_dim, cfg0.meaning_dim,
                "expert {} meaning_dim mismatch", e.name);
        }
        assert_eq!(classifier.meaning_dim, cfg0.meaning_dim,
            "classifier meaning_dim must match experts'");
        assert_eq!(classifier.k, experts.len(),
            "classifier output classes ({}) != num experts ({})",
            classifier.k, experts.len());

        let meaning_embed = experts[0].model.meaning_embed.clone();
        let meaning_dim = cfg0.meaning_dim;
        let vocab_size = cfg0.vocab_size;
        let slots: Vec<ExpertSlot> = experts.into_iter().map(Some).collect();
        Self {
            experts: slots,
            meaning_embed, meaning_dim, vocab_size,
            classifier,
            token_history: Vec::new(),
            mode: RouteMode::Classifier,
            last_chosen: None,
            switch_threshold: 0.3,
        }
    }

    /// Lazy-load: insert an expert at the given classifier-index slot.
    pub fn add_expert_at(&mut self, idx: usize, expert: StitchExpert) {
        assert!(idx < self.experts.len(),
            "add_expert_at idx {} out of range (k={})", idx, self.experts.len());
        let cfg = expert.config();
        assert_eq!(cfg.vocab_size, self.vocab_size, "expert vocab_size mismatch");
        assert_eq!(cfg.meaning_dim, self.meaning_dim, "expert meaning_dim mismatch");
        self.experts[idx] = Some(expert);
    }

    /// True if the classifier slot has a loaded expert.
    pub fn is_loaded(&self, idx: usize) -> bool {
        idx < self.experts.len() && self.experts[idx].is_some()
    }

    /// First loaded slot index, or None if no experts loaded yet.
    pub fn first_loaded(&self) -> Option<usize> {
        self.experts.iter().position(|e| e.is_some())
    }

    pub fn loaded_count(&self) -> usize {
        self.experts.iter().filter(|e| e.is_some()).count()
    }

    pub fn set_switch_threshold(&mut self, t: f32) {
        self.switch_threshold = t;
    }

    pub fn num_experts(&self) -> usize { self.experts.len() }

    pub fn reset(&mut self) {
        for slot in self.experts.iter_mut() {
            if let Some(e) = slot.as_mut() { e.reset(); }
        }
        self.classifier.reset();
        self.token_history.clear();
        self.mode = RouteMode::Classifier;
        self.last_chosen = None;
    }

    pub fn force_expert(&mut self, idx: usize) {
        assert!(idx < self.experts.len());
        self.mode = RouteMode::Forced(idx);
    }
    pub fn unforce(&mut self) { self.mode = RouteMode::Classifier; }

    /// Read a meaning embedding row for token t from the engine's shared
    /// meaning_embed. Works even when no experts are loaded yet — that's
    /// the point of holding meaning at engine level.
    fn get_meaning(&self, token: u32) -> &[f32] {
        let dim = self.meaning_dim;
        let off = token as usize * dim;
        &self.meaning_embed[off .. off + dim]
    }

    /// Catch up a cold expert (loaded slot) by replaying tokens it missed.
    fn warmup_expert(&mut self, idx: usize) -> usize {
        let target = self.token_history.len();
        let synced = match &self.experts[idx] {
            Some(e) => e.synced_pos,
            None => return 0,
        };
        if synced >= target {
            return 0;
        }
        let mut replayed = 0;
        for pos in synced..target {
            let tok = self.token_history[pos] as usize;
            if let Some(e) = self.experts[idx].as_mut() {
                let _ = forward_token_with_trace(
                    &e.model, tok, pos,
                    &mut e.kv, &mut e.scratch, false,
                );
                replayed += 1;
            }
        }
        if let Some(e) = self.experts[idx].as_mut() {
            e.synced_pos = target;
        }
        replayed
    }

    /// Peek the classifier's decision for `token` WITHOUT committing
    /// to history or running an expert forward. Updates a copy of the
    /// EMA, runs the classifier, applies sticky-routing rules, returns
    /// (intended_expert_idx, classifier_probs, margin).
    ///
    /// Caller pattern: peek → if not is_loaded(intended), fetch it,
    /// then call step() which performs the real EMA update + forward.
    /// This avoids the engine getting "stuck" on a fallback expert
    /// before the real one is loaded.
    pub fn peek_step(&self, token: u32) -> (usize, Vec<f32>, f32) {
        // Compute what update_ema would do, but on a local copy.
        let meaning: Vec<f32> = {
            let dim = self.meaning_dim;
            let off = token as usize * dim;
            self.meaning_embed[off .. off + dim].to_vec()
        };
        // Simulate a hypothetical update_ema: blend flat-mean and
        // self-masked attention pool. Don't mutate state.
        let flat_blend = self.classifier.flat_blend;
        let dim = self.meaning_dim;
        let win = &self.classifier.window;
        let ws = self.classifier.window_size;
        let head = self.classifier.head;
        let filled = self.classifier.filled;
        let evicted_slot = if filled == ws { head } else { ws };

        // Build the post-push K/V list (existing slots minus evicted)
        let mut kv: Vec<&[f32]> = Vec::with_capacity(filled);
        for slot in 0..filled {
            if slot == evicted_slot { continue; }
            let off = slot * dim;
            kv.push(&win[off .. off + dim]);
        }

        // First-token corner case
        let mut ema_clone = vec![0.0_f32; dim];
        if kv.is_empty() {
            for d in 0..dim { ema_clone[d] = meaning[d]; }
        } else {
            // Flat mean over (existing-minus-evicted) ∪ {new}
            let mut flat = vec![0.0_f32; dim];
            for v in &kv {
                for d in 0..dim { flat[d] += v[d]; }
            }
            for d in 0..dim { flat[d] += meaning[d]; }
            let inv_n = 1.0 / ((kv.len() + 1) as f32);
            for d in 0..dim { flat[d] *= inv_n; }

            // Attention over kv (which already excludes self)
            let scale = 1.0 / (dim as f32).sqrt();
            let mut scores = vec![0.0_f32; kv.len()];
            let mut max_s = f32::NEG_INFINITY;
            for (i, k) in kv.iter().enumerate() {
                let mut s = 0.0_f32;
                for d in 0..dim { s += meaning[d] * k[d]; }
                s *= scale;
                scores[i] = s;
                if s > max_s { max_s = s; }
            }
            let mut sum = 0.0_f32;
            for s in scores.iter_mut() { *s = (*s - max_s).exp(); sum += *s; }
            let inv_sum = if sum > 1e-20 { 1.0 / sum } else { 0.0 };
            for s in scores.iter_mut() { *s *= inv_sum; }
            let mut attn = vec![0.0_f32; dim];
            for (i, v) in kv.iter().enumerate() {
                let w = scores[i];
                for d in 0..dim { attn[d] += w * v[d]; }
            }

            // Blend
            for d in 0..dim {
                ema_clone[d] = flat_blend * flat[d] + (1.0 - flat_blend) * attn[d];
            }
        }
        // classifier.predict() reads self.ema; we replicate its math
        // against ema_clone instead.
        let mut x = vec![0.0_f32; self.meaning_dim];
        for i in 0..self.meaning_dim {
            x[i] = (ema_clone[i] - self.classifier.feat_mu[i]) / self.classifier.feat_sd[i];
        }
        let mut h = vec![0.0_f32; self.classifier.hidden];
        for i in 0..self.classifier.hidden {
            let mut acc = self.classifier.b1[i];
            let row = &self.classifier.w1[i * self.meaning_dim .. (i + 1) * self.meaning_dim];
            for j in 0..self.meaning_dim { acc += row[j] * x[j]; }
            if acc < 0.0 { acc = 0.0; }
            h[i] = acc;
        }
        let mut logits = vec![0.0_f32; self.classifier.k];
        for i in 0..self.classifier.k {
            let mut acc = self.classifier.b2[i];
            let row = &self.classifier.w2[i * self.classifier.hidden .. (i + 1) * self.classifier.hidden];
            for j in 0..self.classifier.hidden { acc += row[j] * h[j]; }
            logits[i] = acc;
        }
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for v in logits.iter_mut() { *v = (*v - max).exp(); sum += *v; }
        for v in logits.iter_mut() { *v /= sum; }
        let mut top1 = 0;
        for i in 1..self.classifier.k {
            if logits[i] > logits[top1] { top1 = i; }
        }
        let mut top2v = 0.0_f32;
        for i in 0..self.classifier.k {
            if i == top1 { continue; }
            if logits[i] > top2v { top2v = logits[i]; }
        }
        let margin = logits[top1] - top2v;
        // Apply sticky-routing rules + force-mode override
        let intended = match self.mode {
            RouteMode::Classifier => {
                match self.last_chosen {
                    Some(prev) if prev != top1 && margin < self.switch_threshold => prev,
                    _ => top1,
                }
            }
            RouteMode::Forced(idx) => idx,
        };
        (intended, logits, margin)
    }

    /// One step. `token` is the current input token; the engine will:
    ///   - Update EMA (using the meaning vector of `token`)
    ///   - Pick expert (via classifier or force)
    ///   - Catch up that expert's KV if needed
    ///   - Forward `token` through chosen expert at the new position
    ///   - Mask logits and return
    pub fn step(&mut self, token: u32) -> StitchStep {
        // 1. Append to history (this is the position the chosen expert
        //    will write into next). Note: history length BEFORE append =
        //    position of this new token.
        let pos = self.token_history.len();
        self.token_history.push(token);

        // 2. Update EMA
        let meaning_owned: Vec<f32> = self.get_meaning(token).to_vec();
        self.classifier.update_ema(&meaning_owned);

        // 3. Classify with sticky hysteresis (same as before)
        let (probs, classified_idx, margin) = self.classifier.predict();
        let intended = match self.mode {
            RouteMode::Classifier => {
                match self.last_chosen {
                    Some(prev) if prev != classified_idx
                        && margin < self.switch_threshold => prev,
                    _ => classified_idx,
                }
            }
            RouteMode::Forced(idx) => idx,
        };

        // 3b. Lazy-load fallback: if intended slot isn't loaded, fall
        // back to first loaded slot. JS will see `pending_expert` and
        // can fetch it then re-step.
        let pending = if !self.is_loaded(intended) { Some(intended) } else { None };
        let chosen = if pending.is_some() {
            self.first_loaded()
                .expect("no experts loaded; engine cannot serve any token")
        } else {
            intended
        };

        if matches!(self.mode, RouteMode::Classifier) {
            self.last_chosen = Some(chosen);
        }

        // 4. Catch up the chosen expert if cold
        let target_synced = pos;
        let cur_synced = self.experts[chosen].as_ref().map(|e| e.synced_pos).unwrap_or(0);
        let warmup_tokens = if cur_synced < target_synced {
            let mut n = 0;
            for replay_pos in cur_synced..target_synced {
                let rtok = self.token_history[replay_pos] as usize;
                if let Some(e) = self.experts[chosen].as_mut() {
                    let _ = forward_token_with_trace(
                        &e.model, rtok, replay_pos,
                        &mut e.kv, &mut e.scratch, false,
                    );
                    n += 1;
                }
            }
            if let Some(e) = self.experts[chosen].as_mut() {
                e.synced_pos = target_synced;
            }
            n
        } else {
            0
        };

        // 5. Forward the new token through the chosen expert
        let logits = if let Some(e) = self.experts[chosen].as_mut() {
            let trace = forward_token_with_trace(
                &e.model, token as usize, pos,
                &mut e.kv, &mut e.scratch, true,
            ).expect("trace requested");
            e.synced_pos = pos + 1;
            // 6. Mask logits at the masked_tids for this expert
            let mut logits = trace.logits;
            for &tid in &e.masked_tids {
                if (tid as usize) < logits.len() {
                    logits[tid as usize] = f32::NEG_INFINITY;
                }
            }
            logits
        } else {
            // Should never happen — first_loaded() already verified above.
            vec![0.0f32; self.vocab_size]
        };

        StitchStep {
            chosen_expert: chosen,
            classifier_probs: probs,
            confidence: margin,
            logits,
            warmup_happened: warmup_tokens > 0,
            warmup_tokens,
            pending_expert: pending,
        }
    }

    /// Borrowless trace export — the active expert's last trace.
    /// (Not exposed in the public step path because TokenTrace owns large
    /// buffers; we'd return them by-value if needed.)
    pub fn last_trace_for(&self, _expert_idx: usize) -> Option<&TokenTrace> {
        // Reserved for future debug viz — would require keeping the last
        // TokenTrace per expert. Currently we discard it after masking.
        None
    }
}
