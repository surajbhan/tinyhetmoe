//! wasm-bindgen API for the stitched-MoE engine.
//!
//! Exposed to JS as `WasmStitchedEngine`. The JS side fetches and
//! decompresses the per-expert .bin files and the stitch.json sidecar,
//! then constructs the engine and calls `step()` per token.
//!
//! Build:
//!   wasm-pack build --release --target web --features wasm

use wasm_bindgen::prelude::*;

use crate::{
    load_model_from_bytes,
    load_model_from_bytes_with_meaning,
    load_shared_meaning_from_bytes,
    stitch::{DomainClassifier, StitchExpert, StitchedEngine},
};

/// One step's output, mirrored for JS consumption.
#[wasm_bindgen]
pub struct WasmStitchStep {
    chosen_expert: u32,
    confidence: f32,
    classifier_probs: Vec<f32>,
    logits: Vec<f32>,
    warmup_happened: bool,
    warmup_tokens: u32,
    /// 0xFFFFFFFF = no pending; otherwise the classifier wanted this expert
    /// idx but it wasn't loaded; engine fell back to chosen_expert.
    pending_expert: u32,
}

#[wasm_bindgen]
impl WasmStitchStep {
    #[wasm_bindgen(getter)] pub fn chosen_expert(&self) -> u32 { self.chosen_expert }
    #[wasm_bindgen(getter)] pub fn confidence(&self) -> f32 { self.confidence }
    #[wasm_bindgen(getter)] pub fn classifier_probs(&self) -> Vec<f32> { self.classifier_probs.clone() }
    #[wasm_bindgen(getter)] pub fn logits(&self) -> Vec<f32> { self.logits.clone() }
    #[wasm_bindgen(getter)] pub fn warmup_happened(&self) -> bool { self.warmup_happened }
    #[wasm_bindgen(getter)] pub fn warmup_tokens(&self) -> u32 { self.warmup_tokens }
    #[wasm_bindgen(getter)] pub fn pending_expert(&self) -> i32 {
        // -1 = none, otherwise classifier idx. JS-friendly than u32::MAX.
        if self.pending_expert == u32::MAX { -1 } else { self.pending_expert as i32 }
    }
}

/// Stitched-engine handle exposed to JS.
#[wasm_bindgen]
pub struct WasmStitchedEngine {
    engine: StitchedEngine,
    expert_names: Vec<String>,
}

#[wasm_bindgen]
impl WasmStitchedEngine {
    /// Construct from N expert .bin byte buffers + the stitch.json text.
    ///
    /// Args:
    ///   expert_bins: Vec<Uint8Array> in JS — each is the contents of an
    ///                expert's .bin file (already decompressed).
    ///   stitch_json: stringified stitch.json contents.
    ///
    /// JS marshals Vec<Uint8Array> via wasm-bindgen as a single
    /// `Box<[Box<[u8]>]>`-shaped thing. We flatten with a simpler shape:
    /// take a flat byte array + a list of offsets+lengths.
    #[wasm_bindgen(constructor)]
    pub fn new(
        flat_bytes: Vec<u8>,
        bin_offsets: Vec<u32>,    // length N+1; bin i = [offsets[i], offsets[i+1])
        stitch_json: String,
    ) -> Result<WasmStitchedEngine, JsValue> {
        // Parse stitch.json minimally — we only need fields matching
        // our struct, but the wasm build may not have serde_json.
        // To keep the wasm bundle small, we parse via js-sys JSON and
        // pull fields by name. (The native CLI uses serde_json.)
        //
        // For simplicity here, we accept the same JSON as
        // build_stitch_bundle.py emits and parse it the slow but
        // dependency-free way: tiny hand-rolled JSON path lookups via
        // js-sys::JSON::parse. This avoids pulling serde into wasm.

        let parsed = js_sys::JSON::parse(&stitch_json)
            .map_err(|e| JsValue::from_str(&format!("stitch.json parse failed: {:?}", e)))?;
        let parsed_obj: js_sys::Object = parsed.into();

        let format_version = get_string(&parsed_obj, "format_version")?;
        if format_version != "STITCH001" && format_version != "STITCHV7_001" {
            return Err(JsValue::from_str(&format!(
                "unsupported stitch format_version: {}", format_version)));
        }
        // V7 (STITCHV7_001) uses a different classifier shape — see new_v7().
        // The legacy `new()` constructor still expects v6's STITCH001 layout.
        if format_version == "STITCHV7_001" {
            return Err(JsValue::from_str(
                "STITCHV7_001 stitch.json — call WasmStitchedEngine.new_v7() \
                 instead of constructor (different signature, requires \
                 meaning_shared bytes)"));
        }
        let ema_alpha = get_f32(&parsed_obj, "ema_alpha")?;
        let meaning_dim = get_u32(&parsed_obj, "meaning_dim")? as usize;

        // experts list
        let experts_arr = get_array(&parsed_obj, "experts")?;
        let n_experts = experts_arr.length() as usize;
        if bin_offsets.len() != n_experts + 1 {
            return Err(JsValue::from_str(&format!(
                "bin_offsets length {} != n_experts {}+1",
                bin_offsets.len(), n_experts)));
        }

        let mut experts = Vec::with_capacity(n_experts);
        let mut expert_names = Vec::with_capacity(n_experts);
        for i in 0..n_experts {
            let entry: js_sys::Object = experts_arr.get(i as u32).into();
            let name = get_string(&entry, "name")?;
            let masked_tids_arr = get_array(&entry, "masked_tids")?;
            let mut masked_tids = Vec::with_capacity(masked_tids_arr.length() as usize);
            for j in 0..masked_tids_arr.length() {
                let v = masked_tids_arr.get(j);
                let n = v.as_f64().ok_or_else(||
                    JsValue::from_str("masked_tid not a number"))? as u32;
                masked_tids.push(n);
            }
            // Slice the flat_bytes for this expert
            let lo = bin_offsets[i] as usize;
            let hi = bin_offsets[i + 1] as usize;
            let slice = flat_bytes[lo..hi].to_vec();
            let model = load_model_from_bytes(slice)
                .map_err(|e| JsValue::from_str(&format!(
                    "expert '{}' load error: {}", name, e)))?;
            experts.push(StitchExpert::new(name.clone(), model, masked_tids));
            expert_names.push(name);
        }

        // Classifier
        let clf_obj: js_sys::Object = js_sys::Reflect::get(&parsed_obj, &"classifier".into())
            .map_err(|_| JsValue::from_str("missing classifier in stitch.json"))?
            .into();
        let _hidden = get_u32(&clf_obj, "hidden")? as usize;
        let _k = get_u32(&clf_obj, "k")? as usize;
        let w1 = get_f32_array(&clf_obj, "w1")?;
        let b1 = get_f32_array(&clf_obj, "b1")?;
        let w2 = get_f32_array(&clf_obj, "w2")?;
        let b2 = get_f32_array(&clf_obj, "b2")?;
        let feat_mu = get_f32_array(&clf_obj, "feat_mu")?;
        let feat_sd = get_f32_array(&clf_obj, "feat_sd")?;
        let clf = DomainClassifier::new(
            w1, b1, w2, b2, feat_mu, feat_sd, meaning_dim, ema_alpha,
        );

        let engine = StitchedEngine::new(experts, clf);
        Ok(WasmStitchedEngine { engine, expert_names })
    }

    /// V7 constructor.
    ///
    /// Accepts a flat byte buffer containing meaning_shared.bin followed
    /// by N expert bins, with offsets describing slices.
    /// `bin_offsets` is length N+2:
    ///   - offsets[0] = 0
    ///   - offsets[1] = end of meaning_shared (= start of expert 0)
    ///   - offsets[2..N+2] = end of expert i (= start of expert i+1)
    ///
    /// Note: meaning is loaded ONCE and shared across all experts via
    /// `load_model_from_bytes_with_meaning`. Each expert .bin uses HTMOE004
    /// (fp16 intuition only).
    ///
    /// V7 stitch.json schema differs from v6:
    ///   - format_version: "STITCHV7_001"
    ///   - classifier.input_dim, classifier.hidden, classifier.output_dim,
    ///     classifier.domains, classifier.{w1,b1,w2,b2,feat_mu,feat_sd}
    ///   - experts[i].masked_tids may be absent (default empty)
    pub fn new_v7(
        flat_bytes: Vec<u8>,
        bin_offsets: Vec<u32>,    // length N+2
        stitch_json: String,
    ) -> Result<WasmStitchedEngine, JsValue> {
        let parsed = js_sys::JSON::parse(&stitch_json)
            .map_err(|e| JsValue::from_str(&format!("stitch.json parse failed: {:?}", e)))?;
        let parsed_obj: js_sys::Object = parsed.into();

        let format_version = get_string(&parsed_obj, "format_version")?;
        if format_version != "STITCHV7_001" {
            return Err(JsValue::from_str(&format!(
                "new_v7 requires STITCHV7_001 stitch.json, got {}", format_version)));
        }
        let ema_alpha = get_f32(&parsed_obj, "ema_alpha")?;
        let meaning_dim = get_u32(&parsed_obj, "meaning_dim")? as usize;

        let experts_arr = get_array(&parsed_obj, "experts")?;
        let n_experts = experts_arr.length() as usize;
        // bin_offsets layout: [0, end_of_meaning, end_of_expert0, ..., end_of_expert{N-1}]
        if bin_offsets.len() != n_experts + 2 {
            return Err(JsValue::from_str(&format!(
                "bin_offsets length {} != n_experts {}+2",
                bin_offsets.len(), n_experts)));
        }

        // Slice 0: shared meaning
        let meaning_lo = bin_offsets[0] as usize;
        let meaning_hi = bin_offsets[1] as usize;
        let meaning_slice = flat_bytes[meaning_lo..meaning_hi].to_vec();
        let (_v, _d, meaning_weights) = load_shared_meaning_from_bytes(meaning_slice)
            .map_err(|e| JsValue::from_str(&format!("meaning_shared load: {}", e)))?;

        // Each expert uses the shared meaning
        let mut experts = Vec::with_capacity(n_experts);
        let mut expert_names = Vec::with_capacity(n_experts);
        for i in 0..n_experts {
            let entry: js_sys::Object = experts_arr.get(i as u32).into();
            let name = get_string(&entry, "name")?;
            // masked_tids optional in v7 stitch.json
            let masked_tids = match js_sys::Reflect::get(&entry, &"masked_tids".into()) {
                Ok(v) if js_sys::Array::is_array(&v) => {
                    let arr: js_sys::Array = v.into();
                    let mut t = Vec::with_capacity(arr.length() as usize);
                    for j in 0..arr.length() {
                        let n = arr.get(j).as_f64().unwrap_or(0.0) as u32;
                        t.push(n);
                    }
                    t
                }
                _ => Vec::new(),
            };

            let lo = bin_offsets[i + 1] as usize;
            let hi = bin_offsets[i + 2] as usize;
            let slice = flat_bytes[lo..hi].to_vec();
            let model = load_model_from_bytes_with_meaning(slice, Some(meaning_weights.clone()))
                .map_err(|e| JsValue::from_str(&format!(
                    "expert '{}' load error: {}", name, e)))?;
            experts.push(StitchExpert::new(name.clone(), model, masked_tids));
            expert_names.push(name);
        }

        // V7 classifier: w1 is hidden×input_dim (132), w2 is K×hidden
        let clf_obj: js_sys::Object = js_sys::Reflect::get(&parsed_obj, &"classifier".into())
            .map_err(|_| JsValue::from_str("missing classifier in stitch.json"))?
            .into();
        let w1 = get_f32_array(&clf_obj, "w1")?;
        let b1 = get_f32_array(&clf_obj, "b1")?;
        let w2 = get_f32_array(&clf_obj, "w2")?;
        let b2 = get_f32_array(&clf_obj, "b2")?;
        let feat_mu = get_f32_array(&clf_obj, "feat_mu")?;
        let feat_sd = get_f32_array(&clf_obj, "feat_sd")?;
        let clf = DomainClassifier::new(
            w1, b1, w2, b2, feat_mu, feat_sd, meaning_dim, ema_alpha,
        );

        let engine = StitchedEngine::new(experts, clf);
        Ok(WasmStitchedEngine { engine, expert_names })
    }

    /// V7 LAZY constructor.
    ///
    /// Construct an engine with the shared meaning + classifier from
    /// stitch.json, but ZERO experts loaded. Caller fetches each expert
    /// .bin lazily and calls `add_expert_lazy(name, bytes)` to load it.
    /// Engine still produces `step()` results; if classifier picks an
    /// unloaded expert, `step.pending_expert` reports which idx, and the
    /// engine falls back to the first loaded expert for output.
    pub fn new_v7_lazy(
        meaning_bytes: Vec<u8>,
        stitch_json: String,
    ) -> Result<WasmStitchedEngine, JsValue> {
        let parsed = js_sys::JSON::parse(&stitch_json)
            .map_err(|e| JsValue::from_str(&format!("stitch.json parse: {:?}", e)))?;
        let parsed_obj: js_sys::Object = parsed.into();
        let format_version = get_string(&parsed_obj, "format_version")?;
        if format_version != "STITCHV7_001" {
            return Err(JsValue::from_str(&format!(
                "new_v7_lazy needs STITCHV7_001 stitch.json, got {}", format_version)));
        }
        let ema_alpha = get_f32(&parsed_obj, "ema_alpha")?;
        let meaning_dim = get_u32(&parsed_obj, "meaning_dim")? as usize;
        let vocab_size = get_u32(&parsed_obj, "vocab_size")? as usize;

        // Load shared meaning
        let (mv, md, meaning_weights) = load_shared_meaning_from_bytes(meaning_bytes)
            .map_err(|e| JsValue::from_str(&format!("meaning_shared: {}", e)))?;
        if mv != vocab_size || md != meaning_dim {
            return Err(JsValue::from_str(&format!(
                "meaning_shared shape ({},{}) doesn't match stitch.json (V={},D={})",
                mv, md, vocab_size, meaning_dim)));
        }

        // Classifier
        let clf_obj: js_sys::Object = js_sys::Reflect::get(&parsed_obj, &"classifier".into())
            .map_err(|_| JsValue::from_str("missing classifier"))?
            .into();
        let w1 = get_f32_array(&clf_obj, "w1")?;
        let b1 = get_f32_array(&clf_obj, "b1")?;
        let w2 = get_f32_array(&clf_obj, "w2")?;
        let b2 = get_f32_array(&clf_obj, "b2")?;
        let feat_mu = get_f32_array(&clf_obj, "feat_mu")?;
        let feat_sd = get_f32_array(&clf_obj, "feat_sd")?;
        let clf = DomainClassifier::new(
            w1, b1, w2, b2, feat_mu, feat_sd, meaning_dim, ema_alpha,
        );

        // Pull expert names from manifest in classifier-output order
        let domains_arr = get_array(&clf_obj, "domains")?;
        let mut expert_names = Vec::with_capacity(domains_arr.length() as usize);
        for i in 0..domains_arr.length() {
            let v = domains_arr.get(i).as_string()
                .ok_or_else(|| JsValue::from_str("classifier.domains[i] not string"))?;
            expert_names.push(v);
        }

        let engine = StitchedEngine::new_lazy(
            clf, meaning_weights, meaning_dim, vocab_size,
        );
        Ok(WasmStitchedEngine { engine, expert_names })
    }

    /// Lazy-load: insert an expert at the given classifier-index slot.
    /// `expert_idx` is the classifier output position (0..K). Bytes is
    /// the HTMOE004 .bin (without meaning_embed; engine supplies shared).
    pub fn add_expert_lazy(
        &mut self,
        expert_idx: u32,
        expert_bytes: Vec<u8>,
    ) -> Result<(), JsValue> {
        let idx = expert_idx as usize;
        if idx >= self.expert_names.len() {
            return Err(JsValue::from_str(&format!(
                "expert_idx {} >= num expert slots {}",
                idx, self.expert_names.len())));
        }
        let name = self.expert_names[idx].clone();
        // Reuse the engine's shared meaning_embed.
        let model = load_model_from_bytes_with_meaning(
            expert_bytes, Some(self.engine.meaning_embed.clone()),
        ).map_err(|e| JsValue::from_str(&format!(
            "expert '{}' load: {}", name, e)))?;
        let expert = StitchExpert::new(name, model, Vec::new());
        self.engine.add_expert_at(idx, expert);
        Ok(())
    }

    /// Is expert at classifier idx loaded?
    pub fn is_loaded(&self, idx: u32) -> bool {
        self.engine.is_loaded(idx as usize)
    }

    /// Peek what expert the classifier WOULD pick for `token` without
    /// committing the token to history or running a forward. Returns the
    /// intended (post-sticky) expert idx. Use to pre-fetch experts
    /// before calling step().
    pub fn peek_expert(&self, token: u32) -> u32 {
        let (idx, _probs, _margin) = self.engine.peek_step(token);
        idx as u32
    }

    /// Set the sticky-routing threshold. Higher = harder to switch experts
    /// once locked in. Default 0.3. Range [0, 1].
    pub fn set_switch_threshold(&mut self, t: f32) {
        self.engine.set_switch_threshold(t);
    }

    /// Set the classifier-input blend ratio between flat-window mean and
    /// attention pool. 0.0 = pure attention (reactive but can over-respond
    /// to surface noise), 1.0 = pure flat mean (stable but slow to drift).
    /// Default 0.5. Range [0, 1].
    pub fn set_flat_blend(&mut self, t: f32) {
        let v = t.clamp(0.0, 1.0);
        self.engine.classifier.flat_blend = v;
    }
    #[wasm_bindgen(getter)]
    pub fn flat_blend(&self) -> f32 {
        self.engine.classifier.flat_blend
    }

    // Note: there used to be `freeze_classifier()` here when we had an
    // EMA-based classifier. The classifier now uses a flat sliding-window
    // mean over the last 64 meaning vectors, which provides stickiness
    // naturally — no freeze hack needed. The window keeps the prompt's
    // domain signal load-bearing for ~30+ generation tokens before it
    // averages out, while still allowing genuine drift on cross-domain
    // generations.

    /// Number of currently-loaded experts (rest are lazy slots).
    #[wasm_bindgen(getter)]
    pub fn loaded_count(&self) -> u32 { self.engine.loaded_count() as u32 }

    pub fn step(&mut self, token: u32) -> WasmStitchStep {
        let s = self.engine.step(token);
        WasmStitchStep {
            chosen_expert: s.chosen_expert as u32,
            confidence: s.confidence,
            classifier_probs: s.classifier_probs,
            logits: s.logits,
            warmup_happened: s.warmup_happened,
            warmup_tokens: s.warmup_tokens as u32,
            pending_expert: s.pending_expert.map(|i| i as u32).unwrap_or(u32::MAX),
        }
    }

    pub fn reset(&mut self) {
        self.engine.reset();
    }

    /// Force routing to a specific expert idx for subsequent step() calls.
    /// Pass -1 (or any negative value) to release back to classifier mode.
    pub fn force_expert(&mut self, idx: i32) {
        if idx < 0 || (idx as usize) >= self.engine.num_experts() {
            self.engine.unforce();
        } else {
            self.engine.force_expert(idx as usize);
        }
    }

    #[wasm_bindgen(getter)]
    pub fn num_experts(&self) -> u32 { self.engine.num_experts() as u32 }

    /// Return expert names as a single comma-joined string. JS splits on `,`.
    /// Avoids the wasm-bindgen overhead of returning Vec<String>.
    #[wasm_bindgen(getter)]
    pub fn expert_names(&self) -> String {
        self.expert_names.join(",")
    }

    #[wasm_bindgen(getter)]
    pub fn vocab_size(&self) -> u32 {
        self.engine.vocab_size as u32
    }

    #[wasm_bindgen(getter)]
    pub fn position(&self) -> u32 {
        self.engine.token_history.len() as u32
    }
}


// ─── small JSON helpers (no serde in wasm) ───────────────────────────────

fn get_string(obj: &js_sys::Object, key: &str) -> Result<String, JsValue> {
    let v = js_sys::Reflect::get(obj, &key.into())
        .map_err(|_| JsValue::from_str(&format!("missing '{}' in JSON", key)))?;
    v.as_string().ok_or_else(||
        JsValue::from_str(&format!("'{}' not a string", key)))
}

fn get_f32(obj: &js_sys::Object, key: &str) -> Result<f32, JsValue> {
    let v = js_sys::Reflect::get(obj, &key.into())
        .map_err(|_| JsValue::from_str(&format!("missing '{}'", key)))?;
    v.as_f64().map(|f| f as f32).ok_or_else(||
        JsValue::from_str(&format!("'{}' not a number", key)))
}

fn get_u32(obj: &js_sys::Object, key: &str) -> Result<u32, JsValue> {
    let v = js_sys::Reflect::get(obj, &key.into())
        .map_err(|_| JsValue::from_str(&format!("missing '{}'", key)))?;
    v.as_f64().map(|f| f as u32).ok_or_else(||
        JsValue::from_str(&format!("'{}' not a number", key)))
}

fn get_array(obj: &js_sys::Object, key: &str) -> Result<js_sys::Array, JsValue> {
    let v = js_sys::Reflect::get(obj, &key.into())
        .map_err(|_| JsValue::from_str(&format!("missing '{}'", key)))?;
    if !js_sys::Array::is_array(&v) {
        return Err(JsValue::from_str(&format!("'{}' not an array", key)));
    }
    Ok(v.into())
}

fn get_f32_array(obj: &js_sys::Object, key: &str) -> Result<Vec<f32>, JsValue> {
    let arr = get_array(obj, key)?;
    let n = arr.length() as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..arr.length() {
        let v = arr.get(i).as_f64().ok_or_else(||
            JsValue::from_str(&format!("'{}'[{}] not a number", key, i)))?;
        out.push(v as f32);
    }
    Ok(out)
}
