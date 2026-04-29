//! Run individual expert forwards on the same input as Python's
//! dump_expert.py — find which expert's output diverges.

use tiny_hetmoe_wasm::forward::{rms_norm, dot_product, softmax_inplace, gelu, silu};
use tiny_hetmoe_wasm::model::ExpertArch;
use tiny_hetmoe_wasm::tensor::ternary_matvec;
use tiny_hetmoe_wasm::{forward_token_with_trace, load_model_from_bytes, KVCache, ScratchBuffers};

fn main() {
    let bytes = std::fs::read("/tmp/tiny_snap.bin").expect("read bin");
    let model = load_model_from_bytes(bytes).expect("load");
    let cfg = model.config.clone();

    let m = cfg.meaning_dim;
    let i = cfg.intuition_dim;
    let new_i = cfg.new_intuition;
    let dim = cfg.internal_dim;
    let prompt = vec![1usize, 59, 63, 10, 40, 7, 42, 11, 10, 43, 41];

    // Re-run prompt 0..9 to populate KV
    let mut kv = KVCache::new(&cfg);
    let mut scratch = ScratchBuffers::new(&cfg);
    for (pos, &tid) in prompt[..10].iter().enumerate() {
        forward_token_with_trace(&model, tid, pos, &mut kv, &mut scratch, false);
    }

    // Position 10 manual layer-0 walk — replicating debug_layer.rs
    let token = prompt[10];
    let pos = 10;
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

    let layer0 = &model.layers[0];
    let mut n1_out = vec![0.0f32; dim];
    rms_norm(&mut n1_out, &x, &layer0.attn_norm, 1.1920929e-7);

    let mut q = vec![0.0f32; dim];
    let mut k = vec![0.0f32; dim];
    let mut v = vec![0.0f32; dim];
    ternary_matvec(&layer0.q_proj, &n1_out, &mut q);
    ternary_matvec(&layer0.k_proj, &n1_out, &mut k);
    ternary_matvec(&layer0.v_proj, &n1_out, &mut v);
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_heads;
    for h in 0..n_heads {
        let off = h * head_dim;
        let q_in: Vec<f32> = q[off..off + head_dim].to_vec();
        rms_norm(&mut q[off..off + head_dim], &q_in, &layer0.q_norm, 1.1920929e-7);
        let k_in: Vec<f32> = k[off..off + head_dim].to_vec();
        rms_norm(&mut k[off..off + head_dim], &k_in, &layer0.k_norm, 1.1920929e-7);
    }

    let seq_len = pos + 1;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; dim];
    for h in 0..n_heads {
        let q_off = h * head_dim;
        let q_head = &q[q_off..q_off + head_dim];
        let mut scores = vec![0.0f32; seq_len];
        for t in 0..pos {
            let k_off = t * dim + h * head_dim;
            scores[t] = dot_product(q_head, &kv.keys[0][k_off..k_off + head_dim]) * scale;
        }
        scores[pos] = dot_product(q_head, &k[q_off..q_off + head_dim]) * scale;
        softmax_inplace(&mut scores);
        for t in 0..pos {
            let v_off = t * dim + h * head_dim;
            let w = scores[t];
            for i in 0..head_dim {
                attn_out[q_off + i] += w * kv.values[0][v_off + i];
            }
        }
        let w = scores[pos];
        for i in 0..head_dim {
            attn_out[q_off + i] += w * v[q_off + i];
        }
    }

    let mut o_out = vec![0.0f32; dim];
    ternary_matvec(&layer0.o_proj, &attn_out, &mut o_out);
    for i in 0..m { o_out[i] = 0.0; }
    let mut x_after = x.clone();
    for i in 0..dim { x_after[i] += o_out[i]; }

    let mut moe_in = vec![0.0f32; dim];
    rms_norm(&mut moe_in, &x_after, &layer0.ffn_norm, 1.1920929e-7);
    let norm: f32 = moe_in.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("moe_input[0..10]:  {:?}", &moe_in[..10]);
    println!("moe_input norm:    {:.4}", norm);
    println!();

    // Run each expert standalone on moe_in
    let ffn_h = (cfg.internal_dim as f32 * cfg.ffn_mult) as usize;
    let max_mid = ffn_h.max(dim * 2);
    let mut ffn_h1 = vec![0.0f32; max_mid];
    let mut ffn_h2 = vec![0.0f32; max_mid];
    let mut ffn_h3 = vec![0.0f32; max_mid];
    let mut out = vec![0.0f32; dim];

    for (e_idx, expert) in layer0.experts.iter().enumerate() {
        for v in out.iter_mut() { *v = 0.0; }
        match expert.arch {
            ExpertArch::Standard => {
                let up = &expert.weights[0];
                let down = &expert.weights[1];
                ternary_matvec(up, &moe_in, &mut ffn_h1[..ffn_h]);
                for v in ffn_h1[..ffn_h].iter_mut() { *v = gelu(*v); }
                ternary_matvec(down, &ffn_h1[..ffn_h], &mut out);
            }
            ExpertArch::SwiGLU => {
                let w1 = &expert.weights[0];
                let w2 = &expert.weights[1];
                let down = &expert.weights[2];
                ternary_matvec(w1, &moe_in, &mut ffn_h1[..ffn_h]);
                ternary_matvec(w2, &moe_in, &mut ffn_h2[..ffn_h]);
                for i in 0..ffn_h { ffn_h1[i] = silu(ffn_h1[i]) * ffn_h2[i]; }
                ternary_matvec(down, &ffn_h1[..ffn_h], &mut out);
            }
            ExpertArch::DeepNarrow => {
                let mid = dim * 2;
                ternary_matvec(&expert.weights[0], &moe_in, &mut ffn_h1[..mid]);
                for v in ffn_h1[..mid].iter_mut() { *v = gelu(*v); }
                ternary_matvec(&expert.weights[1], &ffn_h1[..mid], &mut ffn_h2[..ffn_h]);
                for v in ffn_h2[..ffn_h].iter_mut() { *v = gelu(*v); }
                ternary_matvec(&expert.weights[2], &ffn_h2[..ffn_h], &mut ffn_h3[..mid]);
                for v in ffn_h3[..mid].iter_mut() { *v = gelu(*v); }
                ternary_matvec(&expert.weights[3], &ffn_h3[..mid], &mut out);
            }
            ExpertArch::Bottleneck => {
                ternary_matvec(&expert.weights[0], &moe_in, &mut ffn_h1[..dim]);
                for v in ffn_h1[..dim].iter_mut() { *v = gelu(*v); }
                ternary_matvec(&expert.weights[1], &ffn_h1[..dim], &mut ffn_h2[..ffn_h]);
                for v in ffn_h2[..ffn_h].iter_mut() { *v = gelu(*v); }
                ternary_matvec(&expert.weights[2], &ffn_h2[..ffn_h], &mut out);
            }
        }
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("Expert {} ({:?}):", e_idx, expert.arch);
        println!("  output[0..5]:      {:?}", &out[..5]);
        println!("  output[132..137]:  {:?}", &out[132..137]);
        println!("  norm:              {:.4}", norm);
    }
}
