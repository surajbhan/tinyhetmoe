//! Debug: dump routing per layer for the same prompt as Python.
use tiny_hetmoe_wasm::{forward_token_with_trace, load_model_from_bytes, KVCache, ScratchBuffers};

fn main() {
    let bytes = std::fs::read("/tmp/tiny_snap.bin").expect("read bin");
    let model = load_model_from_bytes(bytes).expect("load");
    let cfg = model.config.clone();
    let mut kv = KVCache::new(&cfg);
    let mut scratch = ScratchBuffers::new(&cfg);
    let prompt = vec![1usize, 59, 63, 10, 40, 7, 42, 11, 10, 43, 41];

    let mut last_trace = None;
    for (pos, &tid) in prompt.iter().enumerate() {
        last_trace = forward_token_with_trace(&model, tid, pos, &mut kv, &mut scratch, true);
    }
    let t = last_trace.unwrap();

    println!("Rust routing per layer (last position):");
    for (l, r) in t.routing.iter().enumerate() {
        println!("  L{}: {:?}", l, r);
    }
    println!();
    println!("Hidden[0..10] = {:?}", &t.hidden[..10]);
    println!();
    println!("Top-5 logits:");
    let mut idx: Vec<usize> = (0..t.logits.len()).collect();
    idx.sort_by(|a, b| t.logits[*b].partial_cmp(&t.logits[*a]).unwrap());
    for i in 0..5 {
        println!("  {}: id={} logit={:.4}", i, idx[i], t.logits[idx[i]]);
    }
}
