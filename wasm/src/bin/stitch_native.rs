//! Native CLI driver for the stitched-MoE engine.
//!
//! Loads a stitch bundle (directory containing stitch.json + N expert .bin
//! files) and runs forward token-by-token, printing the chosen expert and
//! confidence per step. Then continues with greedy generation from the
//! last argmax for `--gen N` more tokens.
//!
//! Usage:
//!   stitch_native <bundle_dir> <token-id-1> <token-id-2> ... [--gen N]
//!
//! The token IDs are the input prompt (decimal uint16 tids in the
//! unified vocab). Each token is fed through `step()`, the engine picks
//! an expert, returns logits + route decision. Greedy argmax sampling
//! produces the next token.

use std::env;
use std::path::Path;
use tiny_hetmoe_wasm::{
    load_model_from_bytes, DomainClassifier,
    StitchExpert, StitchedEngine,
};

#[derive(serde::Deserialize)]
struct StitchClassifier {
    hidden: usize,
    k: usize,
    domains: Vec<String>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    feat_mu: Vec<f32>,
    feat_sd: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct StitchExpertMeta {
    name: String,
    bin: String,
    masked_tids: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct StitchBundle {
    format_version: String,
    ema_alpha: f32,
    meaning_dim: usize,
    experts: Vec<StitchExpertMeta>,
    classifier: StitchClassifier,
}


fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut best_v = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Top-k sampling with temperature. Returns the sampled token id.
/// Uses a deterministic seed for repro; in production this would use
/// a proper PRNG.
fn sample_top_k(logits: &[f32], temperature: f32, top_k: usize, rng_state: &mut u64) -> usize {
    let mut scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();
    // Find top_k threshold
    let mut idx: Vec<usize> = (0..scaled.len()).collect();
    idx.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal));
    let kth = scaled[idx[top_k.min(scaled.len() - 1)]];
    for v in scaled.iter_mut() {
        if *v < kth { *v = f32::NEG_INFINITY; }
    }
    // Softmax
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in scaled.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    // xorshift64 PRNG
    *rng_state ^= *rng_state << 13;
    *rng_state ^= *rng_state >> 7;
    *rng_state ^= *rng_state << 17;
    let u = ((*rng_state as f32) / (u64::MAX as f32)).clamp(0.0, 1.0 - 1e-7);
    let target = u * sum;
    let mut acc = 0.0_f32;
    for (i, &v) in scaled.iter().enumerate() {
        acc += v;
        if acc >= target { return i; }
    }
    scaled.len() - 1
}


fn main() {
    let raw_args: Vec<String> = env::args().collect();
    if raw_args.len() < 3 {
        eprintln!("usage: stitch_native <bundle_dir> <token-id-1> [<token-id-2> ...] [--gen N]");
        std::process::exit(1);
    }

    // Parse: separate --gen N from positional token args
    let mut bundle_dir: Option<&str> = None;
    let mut prompt_tids: Vec<u32> = Vec::new();
    let mut gen_count: usize = 30;
    let mut temperature: f32 = 0.0;  // 0 = greedy
    let mut top_k: usize = 40;
    let mut seed: u64 = 42;
    let mut i = 1;
    while i < raw_args.len() {
        let a = &raw_args[i];
        if a == "--gen" {
            gen_count = raw_args[i + 1].parse().expect("--gen needs a number");
            i += 2;
        } else if a == "--temp" {
            temperature = raw_args[i + 1].parse().expect("--temp needs a float");
            i += 2;
        } else if a == "--top_k" {
            top_k = raw_args[i + 1].parse().expect("--top_k needs a number");
            i += 2;
        } else if a == "--seed" {
            seed = raw_args[i + 1].parse().expect("--seed needs a number");
            i += 2;
        } else if bundle_dir.is_none() {
            bundle_dir = Some(a);
            i += 1;
        } else {
            prompt_tids.push(a.parse().expect("not a u32 token id"));
            i += 1;
        }
    }
    let bundle_dir = Path::new(bundle_dir.expect("missing bundle_dir"));

    let stitch_path = bundle_dir.join("stitch.json");
    eprintln!("[stitch] loading {}", stitch_path.display());
    let stitch_bytes = std::fs::read(&stitch_path).expect("could not read stitch.json");
    let bundle: StitchBundle = serde_json::from_slice(&stitch_bytes)
        .expect("stitch.json parse failed");
    assert_eq!(bundle.format_version, "STITCH001");
    eprintln!("[stitch] format={}, meaning_dim={}, ema_alpha={}",
        bundle.format_version, bundle.meaning_dim, bundle.ema_alpha);
    eprintln!("[stitch] {} experts, classifier hidden={}, k={}, domains={:?}",
        bundle.experts.len(),
        bundle.classifier.hidden, bundle.classifier.k,
        bundle.classifier.domains);

    // Load each expert's .bin
    let mut experts = Vec::new();
    for ex in &bundle.experts {
        let bin_path = bundle_dir.join(&ex.bin);
        eprintln!("[stitch] loading expert '{}' from {} (masked_tids={:?})",
            ex.name, bin_path.display(), ex.masked_tids);
        let bytes = std::fs::read(&bin_path).expect("could not read expert .bin");
        let model = load_model_from_bytes(bytes).expect("could not parse model");
        experts.push(StitchExpert::new(
            ex.name.clone(),
            model,
            ex.masked_tids.clone(),
        ));
    }

    // Construct classifier
    let clf = DomainClassifier::new(
        bundle.classifier.w1, bundle.classifier.b1,
        bundle.classifier.w2, bundle.classifier.b2,
        bundle.classifier.feat_mu, bundle.classifier.feat_sd,
        bundle.meaning_dim,
        bundle.ema_alpha,
    );

    let mut engine = StitchedEngine::new(experts, clf);
    eprintln!("[stitch] engine ready. {} experts.", engine.num_experts());

    // Run prompt forward
    eprintln!();
    eprintln!("[run] feeding {} prompt tokens", prompt_tids.len());

    // CSV header
    println!("phase,pos,token,expert_idx,expert_name,confidence,prob_pg19,prob_wiki,next_argmax");

    let mut last_logits_argmax: Option<usize> = None;

    for &tok in &prompt_tids {
        let step = engine.step(tok);
        let pos = engine.token_history.len() - 1;
        let expert_name = engine.experts[step.chosen_expert]
            .as_ref().map(|e| e.name.clone()).unwrap_or_default();
        let next = argmax(&step.logits);
        last_logits_argmax = Some(next);
        let probs_str: Vec<String> = step.classifier_probs.iter()
            .map(|p| format!("{:.4}", p))
            .collect();
        eprintln!("  PROMPT  pos={:>3} tok={:>5} -> {} (idx {}, conf={:.3}, probs=[{}])  next={}",
            pos, tok, expert_name,
            step.chosen_expert, step.confidence,
            probs_str.join(", "),
            next);
        if step.warmup_happened {
            eprintln!("    (warmup: replayed {} tokens through expert {})",
                step.warmup_tokens, expert_name);
        }
        // CSV row — assume 2 experts for stable column layout
        let p0 = step.classifier_probs.get(0).copied().unwrap_or(0.0);
        let p1 = step.classifier_probs.get(1).copied().unwrap_or(0.0);
        println!("PROMPT,{},{},{},{},{:.4},{:.4},{:.4},{}",
            pos, tok, step.chosen_expert, expert_name,
            step.confidence, p0, p1, next);
    }

    // Continuation: feed sampled token and repeat. temperature=0 → greedy.
    if gen_count > 0 {
        let sampling_mode = if temperature <= 0.0 {
            "greedy".to_string()
        } else {
            format!("temp={:.2} top_k={}", temperature, top_k)
        };
        let mut rng_state: u64 = seed.max(1);
        let mut next_tok = last_logits_argmax.expect("no prompt processed?") as u32;
        eprintln!();
        eprintln!("[gen] {} continuation, {} tokens", sampling_mode, gen_count);
        for _ in 0..gen_count {
            let step = engine.step(next_tok);
            let pos = engine.token_history.len() - 1;
            let expert_name = engine.experts[step.chosen_expert]
            .as_ref().map(|e| e.name.clone()).unwrap_or_default();
            let next = if temperature <= 0.0 {
                argmax(&step.logits)
            } else {
                sample_top_k(&step.logits, temperature, top_k, &mut rng_state)
            };
            let probs_str: Vec<String> = step.classifier_probs.iter()
                .map(|p| format!("{:.4}", p))
                .collect();
            eprintln!("  GEN     pos={:>3} tok={:>5} -> {} (idx {}, conf={:.3}, probs=[{}])  next={}",
                pos, next_tok, expert_name,
                step.chosen_expert, step.confidence,
                probs_str.join(", "),
                next);
            let p0 = step.classifier_probs.get(0).copied().unwrap_or(0.0);
            let p1 = step.classifier_probs.get(1).copied().unwrap_or(0.0);
            println!("GEN,{},{},{},{},{:.4},{:.4},{:.4},{}",
                pos, next_tok, step.chosen_expert, expert_name,
                step.confidence, p0, p1, next);
            next_tok = next as u32;
        }
    }

    eprintln!();
    eprintln!("[done] processed {} tokens total ({} prompt + {} gen)",
        engine.token_history.len(), prompt_tids.len(), gen_count);
}
