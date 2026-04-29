//! wasm-bindgen API. Exposes Model, KVCache, and an InferStep call to JS.
//!
//! Build:
//!   cargo build --release --target wasm32-unknown-unknown --features wasm
//!
//! For a JS-friendly bundle:
//!   wasm-pack build --release --target web --features wasm

use wasm_bindgen::prelude::*;

use crate::{
    forward::{forward_token_with_trace, TokenTrace},
    load_model_from_bytes,
    model::{KVCache, Model, ScratchBuffers},
};

#[wasm_bindgen]
pub struct WasmModel {
    model: Model,
    kv: KVCache,
    scratch: ScratchBuffers,
}

#[wasm_bindgen]
impl WasmModel {
    /// Construct from raw bytes (call this with the contents of tiny.bin
    /// fetched as ArrayBuffer in JS).
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: Vec<u8>) -> Result<WasmModel, JsValue> {
        let model = load_model_from_bytes(bytes)
            .map_err(|e| JsValue::from_str(&format!("model load error: {}", e)))?;
        let kv = KVCache::new(&model.config);
        let scratch = ScratchBuffers::new(&model.config);
        Ok(WasmModel { model, kv, scratch })
    }

    /// Reset the KV cache. Call this before re-running on a new prompt.
    pub fn reset(&mut self) {
        self.kv.reset();
    }

    /// Forward pass for one token. Returns a flat `WasmStep` JSON-ish blob.
    /// `pos` is the current sequence position (0 for first token).
    /// Internally captures the trace so the UI can read attention/routing.
    pub fn step(&mut self, token: u32) -> Result<WasmStep, JsValue> {
        let pos = self.kv.pos;
        let trace = forward_token_with_trace(
            &self.model, token as usize, pos,
            &mut self.kv, &mut self.scratch, true,
        );
        let trace = trace.ok_or_else(|| JsValue::from_str("trace not produced"))?;
        Ok(WasmStep::from_trace(&self.model, trace))
    }

    /// Number of layers (for UI sanity check).
    #[wasm_bindgen(getter)]
    pub fn num_layers(&self) -> usize { self.model.config.num_layers }

    #[wasm_bindgen(getter)]
    pub fn num_heads(&self) -> usize { self.model.config.num_heads }

    #[wasm_bindgen(getter)]
    pub fn num_experts(&self) -> usize { self.model.config.num_experts }

    #[wasm_bindgen(getter)]
    pub fn vocab_size(&self) -> usize { self.model.config.vocab_size }

    #[wasm_bindgen(getter)]
    pub fn meaning_dim(&self) -> usize { self.model.config.meaning_dim }
}

/// One forward-pass result, packaged for JS.
/// Each multi-D array is flattened with explicit shape getters so JS
/// can recover the structure with one indexing pass.
#[wasm_bindgen]
pub struct WasmStep {
    /// Flat logits, length vocab_size.
    logits: Vec<f32>,
    /// Flat hidden_out (post-final-norm), length input_dim.
    hidden: Vec<f32>,
    /// Flat meaning embedding, length meaning_dim.
    meaning: Vec<f32>,
    /// Flat intuition embedding, length intuition_dim.
    intuition: Vec<f32>,
    /// Flattened attn rows: layer-major, head-major, then attn row of length pos+1.
    /// JS unpacks with `attn_lengths[layer * num_heads + head]` describing each row's length.
    attn_flat: Vec<f32>,
    attn_lengths: Vec<u32>,
    /// Routing per layer: (num_layers × num_experts).
    routing_flat: Vec<f32>,
}

impl WasmStep {
    fn from_trace(model: &Model, trace: TokenTrace) -> Self {
        let cfg = &model.config;
        let mut attn_flat = Vec::new();
        let mut attn_lengths = Vec::with_capacity(cfg.num_layers * cfg.num_heads);
        for head_rows in &trace.attn_rows {
            for row in head_rows {
                attn_lengths.push(row.len() as u32);
                attn_flat.extend_from_slice(row);
            }
        }
        let mut routing_flat = Vec::with_capacity(cfg.num_layers * cfg.num_experts);
        for r in &trace.routing {
            routing_flat.extend_from_slice(r);
        }
        WasmStep {
            logits: trace.logits,
            hidden: trace.hidden,
            meaning: trace.meaning,
            intuition: trace.intuition,
            attn_flat, attn_lengths,
            routing_flat,
        }
    }
}

#[wasm_bindgen]
impl WasmStep {
    #[wasm_bindgen(getter)]
    pub fn logits(&self) -> Vec<f32> { self.logits.clone() }
    #[wasm_bindgen(getter)]
    pub fn hidden(&self) -> Vec<f32> { self.hidden.clone() }
    #[wasm_bindgen(getter)]
    pub fn meaning(&self) -> Vec<f32> { self.meaning.clone() }
    #[wasm_bindgen(getter)]
    pub fn intuition(&self) -> Vec<f32> { self.intuition.clone() }
    #[wasm_bindgen(getter)]
    pub fn attn_flat(&self) -> Vec<f32> { self.attn_flat.clone() }
    #[wasm_bindgen(getter)]
    pub fn attn_lengths(&self) -> Vec<u32> { self.attn_lengths.clone() }
    #[wasm_bindgen(getter)]
    pub fn routing_flat(&self) -> Vec<f32> { self.routing_flat.clone() }
}
