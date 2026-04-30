//! Forward pass for TinyHetMoE — gather-add edition.
//!
//! Gone vs the LUT version: no INT8 activation quantization, no
//! `input_int8` scratch slot. Hidden state stays FP32 throughout. The
//! ternary matvec is now `output[r] = scale * (sum(in[plus_i]) -
//! sum(in[minus_i]))` — pure FP32 input, ternary weight, zero
//! multiplications inside the inner loop.
//!
//! Also fixes the validation drift the LUT path showed at converged
//! checkpoints, since there's no per-tensor activation quantization
//! noise to compound across layers.

use crate::model::{Config, Expert, ExpertArch, KVCache, Layer, Model, ScratchBuffers};
use crate::tensor::ternary_matvec;

// ── Math helpers ────────────────────────────────────────────────────

#[inline]
pub fn rms_norm(out: &mut [f32], input: &[f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(out.len(), input.len());
    debug_assert_eq!(weight.len(), input.len());
    let n = input.len();
    let mut sumsq = 0.0f32;
    for &v in input { sumsq += v * v; }
    let rms = (sumsq / n as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    for i in 0..n {
        out[i] = input[i] * inv * weight[i];
    }
}

#[inline]
pub fn softmax_inplace(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() { if v > max { max = v; } }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() { *v *= inv; }
}

#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    // PyTorch nn.functional.gelu approximation (matches default 'none')
    0.5 * x * (1.0
        + ((std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2)
            * (x + 0.044715 * x * x * x))
            .tanh())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// PyTorch nn.RMSNorm with eps=None uses the FP32 epsilon (~1.19e-7).
// We use a tiny value here for numerical parity.
const RMS_EPS: f32 = 1.1920929e-7;

// ── Per-layer attention with QK-Norm ────────────────────────────────

fn attention_step(
    layer: &Layer,
    cfg: &Config,
    pos: usize,
    scratch: &mut ScratchBuffers,
    kv: &mut KVCache,
    layer_idx: usize,
    mut attn_rows_out: Option<&mut Vec<Vec<f32>>>,
) {
    let dim = cfg.internal_dim;
    let n_heads = cfg.num_heads;
    let head_dim = cfg.head_dim;

    ternary_matvec(&layer.q_proj, &scratch.normed[..dim], &mut scratch.q[..dim]);
    ternary_matvec(&layer.k_proj, &scratch.normed[..dim], &mut scratch.k[..dim]);
    ternary_matvec(&layer.v_proj, &scratch.normed[..dim], &mut scratch.v[..dim]);

    // QK-Norm: RMSNorm per-head on q and k, sharing one (head_dim,) weight
    // across all heads (matches PyTorch nn.RMSNorm(head_dim) applied to
    // a (B, H, T, head_dim) tensor — last-dim norm).
    for h in 0..n_heads {
        let off = h * head_dim;
        let q_head_in: Vec<f32> = scratch.q[off..off + head_dim].to_vec();
        rms_norm(&mut scratch.q[off..off + head_dim], &q_head_in, &layer.q_norm, RMS_EPS);
        let k_head_in: Vec<f32> = scratch.k[off..off + head_dim].to_vec();
        rms_norm(&mut scratch.k[off..off + head_dim], &k_head_in, &layer.k_norm, RMS_EPS);
    }

    kv.append(layer_idx, &scratch.k[..dim], &scratch.v[..dim]);
    let seq_len = pos + 1;

    // Grow scores buffer if we've exceeded max_seq_len (NoPE means there's
    // no positional limit on the model itself; just our static buffer).
    if scratch.scores.len() < seq_len {
        scratch.scores.resize(seq_len, 0.0);
    }

    for o in scratch.attn_out[..dim].iter_mut() { *o = 0.0; }

    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..n_heads {
        let q_off = h * head_dim;
        let q_head = &scratch.q[q_off..q_off + head_dim];

        for t in 0..seq_len {
            let k_off = t * dim + h * head_dim;
            scratch.scores[t] = dot_product(q_head, &kv.keys[layer_idx][k_off..k_off + head_dim]) * scale;
        }
        softmax_inplace(&mut scratch.scores[..seq_len]);

        if let Some(ref mut rows) = attn_rows_out {
            rows.push(scratch.scores[..seq_len].to_vec());
        }

        for t in 0..seq_len {
            let v_off = t * dim + h * head_dim;
            let w = scratch.scores[t];
            for i in 0..head_dim {
                scratch.attn_out[q_off + i] += w * kv.values[layer_idx][v_off + i];
            }
        }
    }
}

// ── Expert FFN ──────────────────────────────────────────────────────

fn expert_forward(
    expert: &Expert,
    cfg: &Config,
    input: &[f32],
    output: &mut [f32],
    ffn_h1: &mut [f32],
    ffn_h2: &mut [f32],
    ffn_h3: &mut [f32],
) {
    let dim = cfg.internal_dim;
    let ffn_h = cfg.ffn_hidden();

    match expert.arch {
        ExpertArch::Standard => {
            // up: (ffn_h, dim), down: (dim, ffn_h)
            let up = &expert.weights[0];
            let down = &expert.weights[1];
            ternary_matvec(up, input, &mut ffn_h1[..ffn_h]);
            for v in ffn_h1[..ffn_h].iter_mut() { *v = gelu(*v); }
            ternary_matvec(down, &ffn_h1[..ffn_h], output);
        }
        ExpertArch::SwiGLU => {
            let w1 = &expert.weights[0];
            let w2 = &expert.weights[1];
            let down = &expert.weights[2];
            ternary_matvec(w1, input, &mut ffn_h1[..ffn_h]);
            ternary_matvec(w2, input, &mut ffn_h2[..ffn_h]);
            for i in 0..ffn_h {
                ffn_h1[i] = silu(ffn_h1[i]) * ffn_h2[i];
            }
            ternary_matvec(down, &ffn_h1[..ffn_h], output);
        }
        ExpertArch::DeepNarrow => {
            let mid = dim * 2;
            let l1 = &expert.weights[0];
            let l2 = &expert.weights[1];
            let l3 = &expert.weights[2];
            let l4 = &expert.weights[3];
            ternary_matvec(l1, input, &mut ffn_h1[..mid]);
            for v in ffn_h1[..mid].iter_mut() { *v = gelu(*v); }
            ternary_matvec(l2, &ffn_h1[..mid], &mut ffn_h2[..ffn_h]);
            for v in ffn_h2[..ffn_h].iter_mut() { *v = gelu(*v); }
            ternary_matvec(l3, &ffn_h2[..ffn_h], &mut ffn_h3[..mid]);
            for v in ffn_h3[..mid].iter_mut() { *v = gelu(*v); }
            ternary_matvec(l4, &ffn_h3[..mid], output);
        }
        ExpertArch::Bottleneck => {
            // down_proj: (dim, dim), up_proj: (ffn_h, dim), out_proj: (dim, ffn_h)
            // Bottleneck "neck" is dim itself (matches Python).
            let down_proj = &expert.weights[0];
            let up_proj = &expert.weights[1];
            let out_proj = &expert.weights[2];
            ternary_matvec(down_proj, input, &mut ffn_h1[..dim]);
            for v in ffn_h1[..dim].iter_mut() { *v = gelu(*v); }
            ternary_matvec(up_proj, &ffn_h1[..dim], &mut ffn_h2[..ffn_h]);
            for v in ffn_h2[..ffn_h].iter_mut() { *v = gelu(*v); }
            ternary_matvec(out_proj, &ffn_h2[..ffn_h], output);
        }
    }
}

// ── MoE step ────────────────────────────────────────────────────────

fn moe_step(
    layer: &Layer,
    cfg: &Config,
    scratch: &mut ScratchBuffers,
    routing_out: Option<&mut Vec<f32>>,
) {
    let dim = cfg.internal_dim;
    let ne = cfg.num_experts;

    // Router: FP32 linear, gate has shape (num_experts, internal_dim)
    for e in 0..ne {
        scratch.router_scores[e] = dot_product(
            &layer.gate[e * dim..(e + 1) * dim],
            &scratch.moe_input[..dim],
        );
    }
    softmax_inplace(&mut scratch.router_scores[..ne]);

    if let Some(out) = routing_out {
        out.clear();
        out.extend_from_slice(&scratch.router_scores[..ne]);
    }

    let mut ranked: Vec<(usize, f32)> = scratch.router_scores[..ne]
        .iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let selected: Vec<(usize, f32)> = ranked[..cfg.top_k_experts].to_vec();

    let sum_w: f32 = selected.iter().map(|(_, w)| *w).sum();
    let inv_sum = if sum_w > 0.0 { 1.0 / sum_w } else { 0.0 };

    for v in scratch.ffn_out[..dim].iter_mut() { *v = 0.0; }

    let moe_input_snapshot: Vec<f32> = scratch.moe_input[..dim].to_vec();
    for (eidx, w) in selected {
        expert_forward(
            &layer.experts[eidx], cfg,
            &moe_input_snapshot,
            &mut scratch.expert_out[..dim],
            &mut scratch.ffn_h1, &mut scratch.ffn_h2, &mut scratch.ffn_h3,
        );
        let weight = w * inv_sum;
        for i in 0..dim {
            scratch.ffn_out[i] += scratch.expert_out[i] * weight;
        }
    }
}

// ── Top-level forward ───────────────────────────────────────────────

pub struct TokenTrace {
    pub attn_rows: Vec<Vec<Vec<f32>>>,
    pub routing: Vec<Vec<f32>>,
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub meaning: Vec<f32>,
    pub intuition: Vec<f32>,
}

pub fn forward_token_with_trace(
    model: &Model,
    token: usize,
    pos: usize,
    kv: &mut KVCache,
    scratch: &mut ScratchBuffers,
    record_trace: bool,
) -> Option<TokenTrace> {
    let cfg = &model.config;
    let dim = cfg.internal_dim;
    let m = cfg.meaning_dim;
    let i = cfg.intuition_dim;
    let new_i = cfg.new_intuition;

    // ── Build initial x from M+I + Highway expand ────────────────────
    let m_off = token * m;
    let i_off = token * i;
    let meaning_vec = &model.meaning_embed[m_off..m_off + m];
    let intuition_vec = &model.intuition_embed[i_off..i_off + i];

    scratch.x_internal[..m].copy_from_slice(meaning_vec);
    scratch.x_internal[m..m + i].copy_from_slice(intuition_vec);

    ternary_matvec(
        &model.expand,
        intuition_vec,
        &mut scratch.expand_out[..new_i],
    );
    scratch.x_internal[m + i..m + i + new_i].copy_from_slice(&scratch.expand_out[..new_i]);

    // ── Trace prep ───────────────────────────────────────────────────
    let mut attn_rows_per_layer: Vec<Vec<Vec<f32>>> = if record_trace {
        Vec::with_capacity(cfg.num_layers)
    } else { Vec::new() };
    let mut routing_per_layer: Vec<Vec<f32>> = if record_trace {
        Vec::with_capacity(cfg.num_layers)
    } else { Vec::new() };

    // ── Layer loop ───────────────────────────────────────────────────
    for (l_idx, layer) in model.layers.iter().enumerate() {
        // Attention
        rms_norm(&mut scratch.normed[..dim], &scratch.x_internal[..dim], &layer.attn_norm, RMS_EPS);
        let mut head_rows: Vec<Vec<f32>> = if record_trace {
            Vec::with_capacity(cfg.num_heads)
        } else { Vec::new() };
        attention_step(layer, cfg, pos, scratch, kv, l_idx,
                        if record_trace { Some(&mut head_rows) } else { None });
        ternary_matvec(&layer.o_proj, &scratch.attn_out[..dim], &mut scratch.o_out[..dim]);
        for k in 0..m { scratch.o_out[k] = 0.0; }
        for k in 0..dim { scratch.x_internal[k] += scratch.o_out[k]; }
        if record_trace {
            attn_rows_per_layer.push(head_rows);
        }

        // MoE
        rms_norm(&mut scratch.moe_input[..dim], &scratch.x_internal[..dim], &layer.ffn_norm, RMS_EPS);
        let mut routing_buf = if record_trace { Some(Vec::with_capacity(cfg.num_experts)) } else { None };
        moe_step(layer, cfg, scratch, routing_buf.as_mut());
        for k in 0..m { scratch.ffn_out[k] = 0.0; }
        for k in 0..dim { scratch.x_internal[k] += scratch.ffn_out[k]; }
        if let Some(r) = routing_buf {
            routing_per_layer.push(r);
        }
    }

    // ── Compress + final norm + lm_head ──────────────────────────────
    let int_internal_len = i + new_i;
    scratch.compress_in[..int_internal_len].copy_from_slice(&scratch.x_internal[m..dim]);

    scratch.hidden_out[..m].copy_from_slice(&scratch.x_internal[..m]);
    let mut intuition_out = vec![0.0f32; i];
    ternary_matvec(
        &model.compress,
        &scratch.compress_in[..int_internal_len],
        &mut intuition_out,
    );
    scratch.hidden_out[m..m + i].copy_from_slice(&intuition_out);

    let mut hidden_normed = vec![0.0f32; cfg.input_dim];
    rms_norm(&mut hidden_normed, &scratch.hidden_out[..cfg.input_dim], &model.final_norm, RMS_EPS);

    ternary_matvec(
        &model.lm_head,
        &hidden_normed,
        &mut scratch.logits[..cfg.vocab_size],
    );

    kv.advance();

    if record_trace {
        Some(TokenTrace {
            attn_rows: attn_rows_per_layer,
            routing: routing_per_layer,
            hidden: hidden_normed,
            logits: scratch.logits[..cfg.vocab_size].to_vec(),
            meaning: meaning_vec.to_vec(),
            intuition: intuition_vec.to_vec(),
        })
    } else {
        None
    }
}
