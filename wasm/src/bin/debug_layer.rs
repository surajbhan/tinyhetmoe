//! Walk through layer 0 step-by-step, dumping intermediates that match
//! scripts/dump_layer0.py. Used to find the first point of divergence.

use tiny_hetmoe_wasm::{
    forward_token_with_trace, load_model_from_bytes, KVCache, ScratchBuffers,
};
use tiny_hetmoe_wasm::forward::{rms_norm, dot_product, softmax_inplace};
use tiny_hetmoe_wasm::tensor::ternary_matvec;

fn main() {
    let bytes = std::fs::read("/tmp/tiny_snap.bin").expect("read bin");
    let model = load_model_from_bytes(bytes).expect("load");
    let cfg = model.config.clone();

    let m = cfg.meaning_dim;
    let i = cfg.intuition_dim;
    let new_i = cfg.new_intuition;
    let dim = cfg.internal_dim;
    let prompt = vec![1usize, 59, 63, 10, 40, 7, 42, 11, 10, 43, 41];

    // Run prompt through full forward to get position 10's KV state
    let mut kv = KVCache::new(&cfg);
    let mut scratch = ScratchBuffers::new(&cfg);
    for (pos, &tid) in prompt.iter().enumerate() {
        forward_token_with_trace(&model, tid, pos, &mut kv, &mut scratch, false);
    }
    // After this, scratch.x_internal holds layer-3 output's last x for position 10.
    // To dump layer-0 specifically, we re-run from scratch but stop early.

    // Re-run cleanly
    kv.reset();
    let mut scratch = ScratchBuffers::new(&cfg);
    for (pos, &tid) in prompt[..10].iter().enumerate() {
        forward_token_with_trace(&model, tid, pos, &mut kv, &mut scratch, false);
    }
    // Now position 10 is next. Manually walk layer 0 for position 10.
    let token = prompt[10];
    let pos = 10;

    // Build initial x at position 10 (M+I + Highway expand)
    let m_off = token * m;
    let i_off = token * i;
    let meaning_vec = &model.meaning_embed[m_off..m_off + m];
    let intuition_vec = &model.intuition_embed[i_off..i_off + i];

    let mut x = vec![0.0f32; dim];
    x[..m].copy_from_slice(meaning_vec);
    x[m..m + i].copy_from_slice(intuition_vec);
    let mut expand_out = vec![0.0f32; new_i];
    ternary_matvec(&model.expand, intuition_vec, &mut expand_out);
    x[m + i..].copy_from_slice(&expand_out);

    println!("============================================================");
    println!("AFTER M+I + EXPAND (input to layer 0)");
    println!("============================================================");
    println!("x[0..5]:          {:?}", &x[..5]);
    println!("x[132..137]:      {:?}", &x[132..137]);
    println!("x[264..269]:      {:?}", &x[264..269]);
    let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("x norm:           {:.4}", norm);

    // n1 (attn_norm)
    let layer0 = &model.layers[0];
    let mut n1_out = vec![0.0f32; dim];
    rms_norm(&mut n1_out, &x, &layer0.attn_norm, 1.1920929e-7);
    println!();
    println!("============================================================");
    println!("AFTER LAYER 0 n1 (attn_norm RMSNorm)");
    println!("============================================================");
    println!("n1[0..5]:         {:?}", &n1_out[..5]);
    println!("n1[132..137]:     {:?}", &n1_out[132..137]);
    let norm: f32 = n1_out.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("n1 norm:          {:.4}", norm);

    // Q, K, V projections
    let mut q = vec![0.0f32; dim];
    let mut k = vec![0.0f32; dim];
    let mut v = vec![0.0f32; dim];
    ternary_matvec(&layer0.q_proj, &n1_out, &mut q);
    ternary_matvec(&layer0.k_proj, &n1_out, &mut k);
    ternary_matvec(&layer0.v_proj, &n1_out, &mut v);
    println!();
    println!("============================================================");
    println!("AFTER Q/K/V PROJECTIONS (pre-QK-Norm)");
    println!("============================================================");
    let head_dim = cfg.head_dim;
    println!("q[head=0, 0..5]:   {:?}", &q[..5]);
    println!("k[head=0, 0..5]:   {:?}", &k[..5]);
    println!("v[head=0, 0..5]:   {:?}", &v[..5]);
    let norm: f32 = q[..head_dim].iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("q[head=0] norm:    {:.4}", norm);

    // QK-Norm per head
    let n_heads = cfg.num_heads;
    for h in 0..n_heads {
        let off = h * head_dim;
        let q_in: Vec<f32> = q[off..off + head_dim].to_vec();
        rms_norm(&mut q[off..off + head_dim], &q_in, &layer0.q_norm, 1.1920929e-7);
        let k_in: Vec<f32> = k[off..off + head_dim].to_vec();
        rms_norm(&mut k[off..off + head_dim], &k_in, &layer0.k_norm, 1.1920929e-7);
    }
    println!();
    println!("============================================================");
    println!("AFTER QK-NORM");
    println!("============================================================");
    println!("q[head=0, 0..5]:   {:?}", &q[..5]);
    println!("k[head=0, 0..5]:   {:?}", &k[..5]);
    let norm: f32 = q[..head_dim].iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("q[head=0] norm:    {:.4}", norm);

    // Run a full forward to get layer-0's output for position 10
    // (We need the KV cache populated from positions 0..9, which we did
    // above. Now do a full step at position 10 — capture x_internal after
    // layer 0 only.)
    //
    // Easiest: redo the forward with a full call but stop at layer 0 manually.
    // That requires accessing scratch.x_internal. Instead, replicate the rest:

    // SDPA (causal) — must integrate against all 11 KV positions.
    // The KV cache is from prompt[..10] but the current k/v haven't been
    // appended. So we have positions 0..9 in cache, current is position 10.
    let seq_len = pos + 1;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut attn_out = vec![0.0f32; dim];
    for h in 0..n_heads {
        let q_off = h * head_dim;
        let q_head = &q[q_off..q_off + head_dim];
        let mut scores = vec![0.0f32; seq_len];

        for t in 0..pos {
            // Read past K from kv cache
            let k_off = t * dim + h * head_dim;
            scores[t] = dot_product(q_head, &kv.keys[0][k_off..k_off + head_dim]) * scale;
        }
        // Current position: use the just-computed k
        scores[pos] = dot_product(q_head, &k[q_off..q_off + head_dim]) * scale;

        softmax_inplace(&mut scores);

        for t in 0..pos {
            let v_off = t * dim + h * head_dim;
            let w = scores[t];
            for i in 0..head_dim {
                attn_out[q_off + i] += w * kv.values[0][v_off + i];
            }
        }
        // Current position: use the just-computed v
        let w = scores[pos];
        for i in 0..head_dim {
            attn_out[q_off + i] += w * v[q_off + i];
        }
    }

    println!();
    println!("============================================================");
    println!("AFTER ATTENTION (pre-o_proj)");
    println!("============================================================");
    println!("attn_out[0..5]:    {:?}", &attn_out[..5]);
    let norm: f32 = attn_out.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("attn_out norm:     {:.4}", norm);

    // o_proj
    let mut o_out = vec![0.0f32; dim];
    ternary_matvec(&layer0.o_proj, &attn_out, &mut o_out);
    println!();
    println!("============================================================");
    println!("AFTER o_proj");
    println!("============================================================");
    println!("o_out[0..5]:       {:?}", &o_out[..5]);
    println!("o_out[132..137]:   {:?}", &o_out[132..137]);

    // Meaning protection
    for i in 0..m { o_out[i] = 0.0; }

    // Residual
    let mut x_after = x.clone();
    for i in 0..dim { x_after[i] += o_out[i]; }
    println!();
    println!("============================================================");
    println!("AFTER RESIDUAL (x + protected o_out)");
    println!("============================================================");
    println!("x_after[0..5]:     {:?}", &x_after[..5]);
    println!("x_after[132..137]: {:?}", &x_after[132..137]);

    // n2 (ffn_norm)
    let mut n2_out = vec![0.0f32; dim];
    rms_norm(&mut n2_out, &x_after, &layer0.ffn_norm, 1.1920929e-7);

    // Router
    let ne = cfg.num_experts;
    let mut router_scores = vec![0.0f32; ne];
    for e in 0..ne {
        router_scores[e] = dot_product(
            &layer0.gate[e * dim..(e + 1) * dim],
            &n2_out,
        );
    }
    softmax_inplace(&mut router_scores);
    println!();
    println!("============================================================");
    println!("ROUTING at position 10");
    println!("============================================================");
    println!("router probs:      {:?}", router_scores);

    // Top-K
    let mut ranked: Vec<(usize, f32)> = router_scores.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let selected: Vec<(usize, f32)> = ranked[..cfg.top_k_experts].to_vec();
    let sum_w: f32 = selected.iter().map(|(_, w)| *w).sum();
    let inv_sum = if sum_w > 0.0 { 1.0 / sum_w } else { 0.0 };

    // Run experts
    let ffn_h = (cfg.internal_dim as f32 * cfg.ffn_mult) as usize;
    let max_mid = ffn_h.max(dim * 2);
    let mut ffn_h1 = vec![0.0f32; max_mid];
    let mut ffn_h2 = vec![0.0f32; max_mid];
    let mut ffn_h3 = vec![0.0f32; max_mid];
    let mut ffn_out = vec![0.0f32; dim];
    let mut expert_out = vec![0.0f32; dim];

    use tiny_hetmoe_wasm::model::ExpertArch;
    let gelu = |x: f32| 0.5 * x * (1.0 + ((std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2) * (x + 0.044715 * x * x * x)).tanh());
    let silu = |x: f32| x / (1.0 + (-x).exp());

    for (eidx, w) in selected {
        let expert = &layer0.experts[eidx];
        match expert.arch {
            ExpertArch::Standard => {
                let up = &expert.weights[0];
                let down = &expert.weights[1];
                ternary_matvec(up, &n2_out, &mut ffn_h1[..ffn_h]);
                for v in ffn_h1[..ffn_h].iter_mut() { *v = gelu(*v); }
                ternary_matvec(down, &ffn_h1[..ffn_h], &mut expert_out);
            }
            ExpertArch::SwiGLU => {
                let w1 = &expert.weights[0];
                let w2 = &expert.weights[1];
                let down = &expert.weights[2];
                ternary_matvec(w1, &n2_out, &mut ffn_h1[..ffn_h]);
                ternary_matvec(w2, &n2_out, &mut ffn_h2[..ffn_h]);
                for i in 0..ffn_h {
                    ffn_h1[i] = silu(ffn_h1[i]) * ffn_h2[i];
                }
                ternary_matvec(down, &ffn_h1[..ffn_h], &mut expert_out);
            }
            ExpertArch::DeepNarrow => {
                let mid = dim * 2;
                let l1 = &expert.weights[0];
                let l2 = &expert.weights[1];
                let l3 = &expert.weights[2];
                let l4 = &expert.weights[3];
                ternary_matvec(l1, &n2_out, &mut ffn_h1[..mid]);
                for v in ffn_h1[..mid].iter_mut() { *v = gelu(*v); }
                ternary_matvec(l2, &ffn_h1[..mid], &mut ffn_h2[..ffn_h]);
                for v in ffn_h2[..ffn_h].iter_mut() { *v = gelu(*v); }
                ternary_matvec(l3, &ffn_h2[..ffn_h], &mut ffn_h3[..mid]);
                for v in ffn_h3[..mid].iter_mut() { *v = gelu(*v); }
                ternary_matvec(l4, &ffn_h3[..mid], &mut expert_out);
            }
            ExpertArch::Bottleneck => {
                let down_proj = &expert.weights[0];
                let up_proj = &expert.weights[1];
                let out_proj = &expert.weights[2];
                ternary_matvec(down_proj, &n2_out, &mut ffn_h1[..dim]);
                for v in ffn_h1[..dim].iter_mut() { *v = gelu(*v); }
                ternary_matvec(up_proj, &ffn_h1[..dim], &mut ffn_h2[..ffn_h]);
                for v in ffn_h2[..ffn_h].iter_mut() { *v = gelu(*v); }
                ternary_matvec(out_proj, &ffn_h2[..ffn_h], &mut expert_out);
            }
        }
        let weight = w * inv_sum;
        for i in 0..dim {
            ffn_out[i] += expert_out[i] * weight;
        }
    }

    // Meaning protection on MoE output
    for i in 0..m { ffn_out[i] = 0.0; }

    // Residual
    let mut x_l0 = x_after.clone();
    for i in 0..dim { x_l0[i] += ffn_out[i]; }

    println!();
    println!("============================================================");
    println!("AFTER FULL LAYER 0 (output to layer 1)");
    println!("============================================================");
    println!("x_l0[0..5]:        {:?}", &x_l0[..5]);
    println!("x_l0[130..137]:    {:?}", &x_l0[130..137]);
    println!("x_l0[264..269]:    {:?}", &x_l0[264..269]);
    let norm: f32 = x_l0.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("x_l0 norm:         {:.4}", norm);
}
