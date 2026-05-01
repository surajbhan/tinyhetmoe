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
/// and an EMA over meaning vectors.
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
    pub ema: Vec<f32>,        // 132 — running EMA, reset on engine reset
    pub ema_alpha: f32,
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
        Self {
            w1, b1, w2, b2, feat_mu, feat_sd,
            hidden, k, meaning_dim,
            ema: vec![0.0; meaning_dim],
            ema_alpha,
        }
    }

    /// Update EMA with the new token's meaning vector.
    pub fn update_ema(&mut self, meaning: &[f32]) {
        let decay = 1.0 - self.ema_alpha;
        for i in 0..self.meaning_dim {
            self.ema[i] = decay * self.ema[i] + self.ema_alpha * meaning[i];
        }
    }

    /// Reset EMA to zero. Caller invokes between prompts.
    pub fn reset(&mut self) {
        for v in self.ema.iter_mut() { *v = 0.0; }
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
}

/// Override mode for routing: classifier-driven (None) or forced.
pub enum RouteMode {
    Classifier,
    Forced(usize),
}

pub struct StitchedEngine {
    pub experts: Vec<StitchExpert>,
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
    pub fn new(experts: Vec<StitchExpert>, classifier: DomainClassifier) -> Self {
        // Sanity: all experts must have same vocab + meaning_dim
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

        Self {
            experts, classifier,
            token_history: Vec::new(),
            mode: RouteMode::Classifier,
            last_chosen: None,
            switch_threshold: 0.3,
        }
    }

    pub fn set_switch_threshold(&mut self, t: f32) {
        self.switch_threshold = t;
    }

    pub fn num_experts(&self) -> usize { self.experts.len() }

    pub fn reset(&mut self) {
        for e in self.experts.iter_mut() { e.reset(); }
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

    /// Read a meaning embedding row for token t from the FIRST expert
    /// (all experts share identical meaning_embed since they share vocab).
    fn get_meaning(&self, token: u32) -> &[f32] {
        let m = &self.experts[0].model.meaning_embed;
        let dim = self.experts[0].model.config.meaning_dim;
        let off = token as usize * dim;
        &m[off .. off + dim]
    }

    /// Catch up a cold expert by replaying tokens it missed. Each replay
    /// step is one full forward without trace.
    fn warmup_expert(&mut self, idx: usize) -> usize {
        let target = self.token_history.len();
        let synced = self.experts[idx].synced_pos;
        if synced >= target {
            return 0;
        }
        let mut replayed = 0;
        for pos in synced..target {
            let tok = self.token_history[pos] as usize;
            // Take split borrows: forward needs &model + &mut kv + &mut scratch
            let e = &mut self.experts[idx];
            let _ = forward_token_with_trace(
                &e.model, tok, pos,
                &mut e.kv, &mut e.scratch, false,
            );
            replayed += 1;
        }
        self.experts[idx].synced_pos = target;
        replayed
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

        // 3. Classify (or use forced mode), with sticky-hysteresis
        // when in classifier mode: if we already have a last_chosen
        // expert and the new classifier top-1 differs from it, only
        // switch if (top1 - top2) margin >= switch_threshold. This
        // prevents flutter between near-tied classes (e.g. pg19 vs
        // wiki on short prose prompts).
        let (probs, classified_idx, margin) = self.classifier.predict();
        let chosen = match self.mode {
            RouteMode::Classifier => {
                match self.last_chosen {
                    Some(prev) if prev != classified_idx
                        && margin < self.switch_threshold => {
                        // Don't switch — stick with previous expert.
                        prev
                    }
                    _ => classified_idx,
                }
            }
            RouteMode::Forced(idx) => idx,
        };
        if matches!(self.mode, RouteMode::Classifier) {
            self.last_chosen = Some(chosen);
        }

        // 4. Catch up the chosen expert if cold
        // (Replay is up to but NOT including pos — at pos we want to do
        // the new forward.)
        let target_synced = pos;
        let cur_synced = self.experts[chosen].synced_pos;
        let warmup_tokens = if cur_synced < target_synced {
            // Need to replay tokens [cur_synced .. target_synced]
            let mut n = 0;
            for replay_pos in cur_synced..target_synced {
                let rtok = self.token_history[replay_pos] as usize;
                let e = &mut self.experts[chosen];
                let _ = forward_token_with_trace(
                    &e.model, rtok, replay_pos,
                    &mut e.kv, &mut e.scratch, false,
                );
                n += 1;
            }
            self.experts[chosen].synced_pos = target_synced;
            n
        } else {
            0
        };

        // 5. Forward the new token through the chosen expert
        let e = &mut self.experts[chosen];
        let trace = forward_token_with_trace(
            &e.model, token as usize, pos,
            &mut e.kv, &mut e.scratch, true,
        ).expect("trace requested");
        e.synced_pos = pos + 1;

        // 6. Mask logits at the masked_tids for this expert
        let mut logits = trace.logits;
        for &tid in &self.experts[chosen].masked_tids {
            if (tid as usize) < logits.len() {
                logits[tid as usize] = f32::NEG_INFINITY;
            }
        }

        StitchStep {
            chosen_expert: chosen,
            classifier_probs: probs,
            confidence: margin,
            logits,
            warmup_happened: warmup_tokens > 0,
            warmup_tokens,
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
