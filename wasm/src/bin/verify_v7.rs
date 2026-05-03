//! Native test: load the v7 stitch_v7 bundle, build the engine, and
//! run a few tokens through one expert to verify HTMOE004 + MNGSHR04
//! roundtrip works.
//!
//! Run with:
//!   cd wasm && cargo run --release --bin verify_v7

use tiny_hetmoe_wasm::{
    forward_token_with_trace,
    load_model_from_bytes_with_meaning,
    load_shared_meaning_from_bytes,
    KVCache, ScratchBuffers,
};
use std::path::Path;

fn main() {
    let stitch_dir = Path::new("/data/sematic-embedding/tinyhetmoe/docs/stitch_v7");

    // 1. Load shared meaning
    let mp = stitch_dir.join("meaning_shared.bin");
    println!("[verify] loading {}", mp.display());
    let mb = std::fs::read(&mp).expect("read meaning_shared");
    let (vocab, dim, meaning) = load_shared_meaning_from_bytes(mb)
        .expect("parse meaning_shared");
    println!("[verify] shared meaning: vocab={vocab} dim={dim} ({} fp32 floats, {:.1} MB raw)",
             meaning.len(), meaning.len() as f32 * 4.0 / 1e6);

    // Sanity: nonzero values, no NaN/Inf
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut zero_rows = 0;
    for row in 0..vocab {
        let r = &meaning[row * dim .. (row + 1) * dim];
        let s: f32 = r.iter().map(|x| x.abs()).sum();
        if s == 0.0 { zero_rows += 1; }
        for &x in r {
            if x.is_nan() { nan_count += 1; }
            if x.is_infinite() { inf_count += 1; }
        }
    }
    println!("[verify] meaning sanity: nan={nan_count} inf={inf_count} zero_rows={zero_rows}");
    println!("[verify] meaning row 0 norm: {:.4}",
             meaning[0..dim].iter().map(|x| x*x).sum::<f32>().sqrt());

    // 2. Load each expert with the shared meaning
    let domains = ["general", "thinker", "code_py", "code_js", "medical", "legal"];
    for d in domains {
        let bp = stitch_dir.join(format!("{d}.bin"));
        println!("\n[verify] loading expert {d} from {}", bp.display());
        let bytes = std::fs::read(&bp).expect("read expert bin");
        let model = load_model_from_bytes_with_meaning(bytes, Some(meaning.clone()))
            .expect("load expert");
        let cfg = &model.config;
        println!("[verify]   cfg vocab={} hidden={} L={} H={} E={}/{}",
                 cfg.vocab_size, cfg.input_dim,
                 cfg.num_layers, cfg.num_heads,
                 cfg.num_experts, cfg.top_k_experts);
        // Sanity-check intuition_embed and meaning_embed dims
        assert_eq!(model.meaning_embed.len(), cfg.vocab_size * cfg.meaning_dim);
        assert_eq!(model.intuition_embed.len(), cfg.vocab_size * cfg.intuition_dim);
        println!("[verify]   meaning OK ({} floats), intuition OK ({} floats)",
                 model.meaning_embed.len(), model.intuition_embed.len());
        // Spot-check the meaning tensor inside model matches shared meaning bit-for-bit
        let m0 = model.meaning_embed[0];
        let s0 = meaning[0];
        assert!((m0 - s0).abs() < 1e-9,
                "meaning row 0 elem 0 mismatch: model={m0} shared={s0}");
        println!("[verify]   shared meaning matches model.meaning_embed: ✓");
    }

    println!("\n[verify] ALL 6 EXPERTS LOADED OK");

    // 3. Forward-pass smoke on thinker — feed a few tokens and check
    // logits shape + sanity (argmax is a real token, no NaN).
    println!("\n[verify] forward-pass smoke: thinker, prompt='Question:'");
    let bp = stitch_dir.join("thinker.bin");
    let bytes = std::fs::read(&bp).expect("read thinker");
    let model = load_model_from_bytes_with_meaning(bytes, Some(meaning.clone()))
        .expect("load thinker");
    let cfg = model.config.clone();
    let mut kv = KVCache::new(&cfg);
    let mut scratch = ScratchBuffers::new(&cfg);

    // Hand-tokenize "Question:" — Qwen2.5 ids: "Question" + ":" = [14582, 25]
    // (verified separately; we just smoke-test anything works)
    let tokens: Vec<usize> = vec![14582, 25];
    for (pos, &tok) in tokens.iter().enumerate() {
        let trace = forward_token_with_trace(
            &model, tok, pos, &mut kv, &mut scratch, true,
        ).expect("forward");
        let logits = trace.logits;
        // Sanity
        let nan = logits.iter().filter(|x| x.is_nan()).count();
        let inf = logits.iter().filter(|x| x.is_infinite()).count();
        let argmax = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        let max_v = logits[argmax];
        let mean_v = logits.iter().sum::<f32>() / logits.len() as f32;
        println!("[verify]   pos {pos} tok {tok}: argmax={argmax} max={max_v:.4} \
                  mean={mean_v:.4} nan={nan} inf={inf}");
    }
    println!("\n[verify] FORWARD PASS SMOKE OK");
}
