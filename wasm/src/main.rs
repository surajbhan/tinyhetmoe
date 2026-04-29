//! Native test harness — loads tiny.bin, runs inference, compares to
//! Python's known outputs. Used for development; the WASM build uses
//! lib.rs + wasm_api.rs.

use std::env;
use tiny_hetmoe_wasm::{
    forward_token_with_trace, load_model_from_bytes, KVCache, ScratchBuffers,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: tiny_hetmoe_native <path-to-tiny.bin> [tokens...]");
        std::process::exit(1);
    }
    let bin_path = &args[1];
    let prompt_tokens: Vec<usize> = if args.len() > 2 {
        args[2..].iter().map(|s| s.parse().expect("not a token id")).collect()
    } else {
        // Default: <bos> "Once" " upon" " a" " time" — for quick smoke test
        vec![1, 7454_usize.min(5966)]  // placeholder, may be wrong tiny ids
    };

    println!("[native] reading {}", bin_path);
    let bytes = std::fs::read(bin_path).expect("could not read bin file");
    let model = load_model_from_bytes(bytes).expect("could not parse model");

    let cfg = model.config.clone();
    println!("[native] config: vocab={} hidden={} L={} H={} head_dim={} E={}/{}",
        cfg.vocab_size, cfg.input_dim, cfg.num_layers, cfg.num_heads, cfg.head_dim,
        cfg.num_experts, cfg.top_k_experts);

    let mut kv = KVCache::new(&cfg);
    let mut scratch = ScratchBuffers::new(&cfg);

    println!("[native] running prompt forward");
    let mut last_trace = None;
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        let trace = forward_token_with_trace(
            &model, tok, pos, &mut kv, &mut scratch, true,
        );
        last_trace = trace;
        println!("[native]   pos {} token {} done", pos, tok);
    }

    if let Some(t) = last_trace {
        println!("[native] last token trace summary:");
        println!("  meaning[0..5]   = {:?}", &t.meaning[..5.min(t.meaning.len())]);
        println!("  intuition[0..5] = {:?}", &t.intuition[..5.min(t.intuition.len())]);
        println!("  hidden[0..5]    = {:?}", &t.hidden[..5.min(t.hidden.len())]);
        println!("  logits[0..5]    = {:?}", &t.logits[..5.min(t.logits.len())]);
        // Top-5
        let mut idx: Vec<usize> = (0..t.logits.len()).collect();
        idx.sort_by(|a, b| t.logits[*b].partial_cmp(&t.logits[*a]).unwrap());
        println!("  top-5 next:");
        for &i in &idx[..5] {
            println!("    id {} logit {:.3}", i, t.logits[i]);
        }
        println!("  routing layer 0 = {:?}", t.routing.get(0));
    }
}
