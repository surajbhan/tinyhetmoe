//! Model structs.

use crate::tensor::PackedTernaryWeight;

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub meaning_dim: usize,
    pub intuition_dim: usize,
    pub input_dim: usize,
    pub internal_dim: usize,
    pub new_intuition: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub ffn_mult: f32,
}

impl Config {
    pub fn ffn_hidden(&self) -> usize {
        (self.internal_dim as f32 * self.ffn_mult) as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExpertArch {
    Standard,    // up, down
    SwiGLU,      // w1, w2, down
    DeepNarrow,  // l1, l2, l3, l4
    Bottleneck,  // down_proj, up_proj, out_proj
}

impl ExpertArch {
    pub fn num_weights(&self) -> usize {
        match self {
            ExpertArch::Standard => 2,
            ExpertArch::SwiGLU => 3,
            ExpertArch::DeepNarrow => 4,
            ExpertArch::Bottleneck => 3,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            ExpertArch::Standard => "Standard",
            ExpertArch::SwiGLU => "SwiGLU",
            ExpertArch::DeepNarrow => "DeepNarrow",
            ExpertArch::Bottleneck => "Bottleneck",
        }
    }
}

#[derive(Clone)]
pub struct Expert {
    pub arch: ExpertArch,
    pub weights: Vec<PackedTernaryWeight>,
}

#[derive(Clone)]
pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,        // QK-Norm: head_dim
    pub k_norm: Vec<f32>,        // QK-Norm: head_dim
    pub ffn_norm: Vec<f32>,
    pub q_proj: PackedTernaryWeight,
    pub k_proj: PackedTernaryWeight,
    pub v_proj: PackedTernaryWeight,
    pub o_proj: PackedTernaryWeight,
    pub experts: Vec<Expert>,
    pub gate: Vec<f32>,           // num_experts × internal_dim
}

pub struct Model {
    pub config: Config,
    pub meaning_embed: Vec<f32>,    // vocab_size × meaning_dim
    pub intuition_embed: Vec<f32>,  // vocab_size × intuition_dim
    pub expand: PackedTernaryWeight, // new_intuition × intuition_dim
    pub layers: Vec<Layer>,
    pub compress: PackedTernaryWeight, // intuition_dim × (intuition_dim + new_intuition)
    pub final_norm: Vec<f32>,         // input_dim
    pub lm_head: PackedTernaryWeight, // vocab_size × input_dim
}

/// Reusable scratch buffers — sized for the largest dim each is used at.
/// All zeroed at start of each forward step.
pub struct ScratchBuffers {
    // Hidden state through the stack — internal_dim wide
    pub x_internal: Vec<f32>,
    // Layer-level scratch
    pub normed: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub o_out: Vec<f32>,
    pub moe_input: Vec<f32>,
    pub router_scores: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub expert_out: Vec<f32>,
    // FFN scratch (sized for max FFN hidden)
    pub ffn_h1: Vec<f32>,
    pub ffn_h2: Vec<f32>,
    pub ffn_h3: Vec<f32>,
    // Highway expand output (new_intuition wide)
    pub expand_out: Vec<f32>,
    // Compress input scratch (intuition_internal wide)
    pub compress_in: Vec<f32>,
    // Final hidden (input_dim) and logits (vocab_size)
    pub hidden_out: Vec<f32>,
    pub logits: Vec<f32>,
    // Per-attention scratch
    pub scores: Vec<f32>,
}

impl ScratchBuffers {
    pub fn new(cfg: &Config) -> Self {
        let dim = cfg.internal_dim;
        let ffn_h = cfg.ffn_hidden();
        // DeepNarrow uses dim*2 in middle
        let max_mid = ffn_h.max(dim * 2);
        Self {
            x_internal: vec![0.0; dim],
            normed: vec![0.0; dim],
            q: vec![0.0; dim],
            k: vec![0.0; dim],
            v: vec![0.0; dim],
            attn_out: vec![0.0; dim],
            o_out: vec![0.0; dim],
            moe_input: vec![0.0; dim],
            router_scores: vec![0.0; cfg.num_experts],
            ffn_out: vec![0.0; dim],
            expert_out: vec![0.0; dim],
            ffn_h1: vec![0.0; max_mid],
            ffn_h2: vec![0.0; max_mid],
            ffn_h3: vec![0.0; max_mid],
            expand_out: vec![0.0; cfg.new_intuition],
            compress_in: vec![0.0; cfg.intuition_dim + cfg.new_intuition],
            hidden_out: vec![0.0; cfg.input_dim],
            logits: vec![0.0; cfg.vocab_size],
            scores: vec![0.0; cfg.max_seq_len],
        }
    }
}

/// KV cache. One per layer; flat row-major (T × internal_dim).
/// Grows by one row per token.
pub struct KVCache {
    pub keys: Vec<Vec<f32>>,    // per layer
    pub values: Vec<Vec<f32>>,  // per layer
    pub pos: usize,             // current sequence length
}

impl KVCache {
    pub fn new(cfg: &Config) -> Self {
        let cap = cfg.max_seq_len * cfg.internal_dim;
        Self {
            keys: (0..cfg.num_layers).map(|_| Vec::with_capacity(cap)).collect(),
            values: (0..cfg.num_layers).map(|_| Vec::with_capacity(cap)).collect(),
            pos: 0,
        }
    }
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        self.keys[layer].extend_from_slice(k);
        self.values[layer].extend_from_slice(v);
    }
    pub fn advance(&mut self) {
        self.pos += 1;
    }
    pub fn reset(&mut self) {
        for k in self.keys.iter_mut() { k.clear(); }
        for v in self.values.iter_mut() { v.clear(); }
        self.pos = 0;
    }
}
