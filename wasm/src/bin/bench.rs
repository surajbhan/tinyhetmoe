use std::time::Instant;
use tiny_hetmoe_wasm::{forward_token_with_trace, load_model_from_bytes, KVCache, ScratchBuffers};

fn main() {
    let bytes = std::fs::read("../runs/tiny_hetmoe/tiny.bin").expect("read tiny.bin");
    let t0 = Instant::now();
    let model = load_model_from_bytes(bytes).expect("load");
    println!("load+pack: {} ms", t0.elapsed().as_millis());

    let cfg = model.config.clone();
    let mut kv = KVCache::new(&cfg);
    let mut scratch = ScratchBuffers::new(&cfg);
    let prompt = vec![1usize, 59, 63, 10, 40, 7, 42, 11, 10, 43, 41];

    // Warmup
    forward_token_with_trace(&model, prompt[0], 0, &mut kv, &mut scratch, true);
    kv.reset();

    let t1 = Instant::now();
    for (pos, &tid) in prompt.iter().enumerate() {
        forward_token_with_trace(&model, tid, pos, &mut kv, &mut scratch, true);
    }
    let prompt_ms = t1.elapsed().as_millis();
    println!("prompt fill ({} tok): {} ms total, {:.1} ms/tok",
        prompt.len(), prompt_ms, prompt_ms as f32 / prompt.len() as f32);

    let n = 50;
    let t2 = Instant::now();
    for i in 0..n {
        let pos = prompt.len() + i;
        forward_token_with_trace(&model, 41, pos, &mut kv, &mut scratch, true);
    }
    let gen_ms = t2.elapsed().as_millis();
    println!("generation ({} tok):  {} ms total, {:.1} ms/tok",
        n, gen_ms, gen_ms as f32 / n as f32);
    println!("→ {:.1} tokens/sec", (n as f32) / (gen_ms as f32 / 1000.0));
}
