/* tslint:disable */
/* eslint-disable */

export class WasmModel {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Construct from raw bytes (call this with the contents of tiny.bin
     * fetched as ArrayBuffer in JS).
     */
    constructor(bytes: Uint8Array);
    /**
     * Reset the KV cache. Call this before re-running on a new prompt.
     */
    reset(): void;
    /**
     * Forward pass for one token. Returns a flat `WasmStep` JSON-ish blob.
     * `pos` is the current sequence position (0 for first token).
     * Internally captures the trace so the UI can read attention/routing.
     */
    step(token: number): WasmStep;
    readonly meaning_dim: number;
    readonly num_experts: number;
    readonly num_heads: number;
    /**
     * Number of layers (for UI sanity check).
     */
    readonly num_layers: number;
    readonly vocab_size: number;
}

/**
 * One forward-pass result, packaged for JS.
 * Each multi-D array is flattened with explicit shape getters so JS
 * can recover the structure with one indexing pass.
 */
export class WasmStep {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly attn_flat: Float32Array;
    readonly attn_lengths: Uint32Array;
    readonly hidden: Float32Array;
    readonly intuition: Float32Array;
    readonly logits: Float32Array;
    readonly meaning: Float32Array;
    readonly routing_flat: Float32Array;
}

/**
 * One step's output, mirrored for JS consumption.
 */
export class WasmStitchStep {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly chosen_expert: number;
    readonly classifier_probs: Float32Array;
    readonly confidence: number;
    readonly logits: Float32Array;
    readonly pending_expert: number;
    readonly warmup_happened: boolean;
    readonly warmup_tokens: number;
}

/**
 * Stitched-engine handle exposed to JS.
 */
export class WasmStitchedEngine {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Lazy-load: insert an expert at the given classifier-index slot.
     * `expert_idx` is the classifier output position (0..K). Bytes is
     * the HTMOE004 .bin (without meaning_embed; engine supplies shared).
     */
    add_expert_lazy(expert_idx: number, expert_bytes: Uint8Array): void;
    /**
     * Force routing to a specific expert idx for subsequent step() calls.
     * Pass -1 (or any negative value) to release back to classifier mode.
     */
    force_expert(idx: number): void;
    /**
     * Is expert at classifier idx loaded?
     */
    is_loaded(idx: number): boolean;
    /**
     * Construct from N expert .bin byte buffers + the stitch.json text.
     *
     * Args:
     *   expert_bins: Vec<Uint8Array> in JS — each is the contents of an
     *                expert's .bin file (already decompressed).
     *   stitch_json: stringified stitch.json contents.
     *
     * JS marshals Vec<Uint8Array> via wasm-bindgen as a single
     * `Box<[Box<[u8]>]>`-shaped thing. We flatten with a simpler shape:
     * take a flat byte array + a list of offsets+lengths.
     */
    constructor(flat_bytes: Uint8Array, bin_offsets: Uint32Array, stitch_json: string);
    /**
     * V7 constructor.
     *
     * Accepts a flat byte buffer containing meaning_shared.bin followed
     * by N expert bins, with offsets describing slices.
     * `bin_offsets` is length N+2:
     *   - offsets[0] = 0
     *   - offsets[1] = end of meaning_shared (= start of expert 0)
     *   - offsets[2..N+2] = end of expert i (= start of expert i+1)
     *
     * Note: meaning is loaded ONCE and shared across all experts via
     * `load_model_from_bytes_with_meaning`. Each expert .bin uses HTMOE004
     * (fp16 intuition only).
     *
     * V7 stitch.json schema differs from v6:
     *   - format_version: "STITCHV7_001"
     *   - classifier.input_dim, classifier.hidden, classifier.output_dim,
     *     classifier.domains, classifier.{w1,b1,w2,b2,feat_mu,feat_sd}
     *   - experts[i].masked_tids may be absent (default empty)
     */
    static new_v7(flat_bytes: Uint8Array, bin_offsets: Uint32Array, stitch_json: string): WasmStitchedEngine;
    /**
     * V7 LAZY constructor.
     *
     * Construct an engine with the shared meaning + classifier from
     * stitch.json, but ZERO experts loaded. Caller fetches each expert
     * .bin lazily and calls `add_expert_lazy(name, bytes)` to load it.
     * Engine still produces `step()` results; if classifier picks an
     * unloaded expert, `step.pending_expert` reports which idx, and the
     * engine falls back to the first loaded expert for output.
     */
    static new_v7_lazy(meaning_bytes: Uint8Array, stitch_json: string): WasmStitchedEngine;
    /**
     * Peek what expert the classifier WOULD pick for `token` without
     * committing the token to history or running a forward. Returns the
     * intended (post-sticky) expert idx. Use to pre-fetch experts
     * before calling step().
     */
    peek_expert(token: number): number;
    reset(): void;
    /**
     * Set the classifier-input blend ratio between flat-window mean and
     * attention pool. 0.0 = pure attention (reactive but can over-respond
     * to surface noise), 1.0 = pure flat mean (stable but slow to drift).
     * Default 0.5. Range [0, 1].
     */
    set_flat_blend(t: number): void;
    /**
     * Set the sticky-routing threshold. Higher = harder to switch experts
     * once locked in. Default 0.3. Range [0, 1].
     */
    set_switch_threshold(t: number): void;
    step(token: number): WasmStitchStep;
    /**
     * Return expert names as a single comma-joined string. JS splits on `,`.
     * Avoids the wasm-bindgen overhead of returning Vec<String>.
     */
    readonly expert_names: string;
    readonly flat_blend: number;
    /**
     * Number of currently-loaded experts (rest are lazy slots).
     */
    readonly loaded_count: number;
    readonly num_experts: number;
    readonly position: number;
    readonly vocab_size: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmmodel_free: (a: number, b: number) => void;
    readonly __wbg_wasmstep_free: (a: number, b: number) => void;
    readonly __wbg_wasmstitchedengine_free: (a: number, b: number) => void;
    readonly __wbg_wasmstitchstep_free: (a: number, b: number) => void;
    readonly wasmmodel_meaning_dim: (a: number) => number;
    readonly wasmmodel_new: (a: number, b: number, c: number) => void;
    readonly wasmmodel_num_experts: (a: number) => number;
    readonly wasmmodel_num_heads: (a: number) => number;
    readonly wasmmodel_num_layers: (a: number) => number;
    readonly wasmmodel_reset: (a: number) => void;
    readonly wasmmodel_step: (a: number, b: number, c: number) => void;
    readonly wasmmodel_vocab_size: (a: number) => number;
    readonly wasmstep_attn_flat: (a: number, b: number) => void;
    readonly wasmstep_attn_lengths: (a: number, b: number) => void;
    readonly wasmstep_hidden: (a: number, b: number) => void;
    readonly wasmstep_intuition: (a: number, b: number) => void;
    readonly wasmstep_logits: (a: number, b: number) => void;
    readonly wasmstep_meaning: (a: number, b: number) => void;
    readonly wasmstep_routing_flat: (a: number, b: number) => void;
    readonly wasmstitchedengine_add_expert_lazy: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasmstitchedengine_expert_names: (a: number, b: number) => void;
    readonly wasmstitchedengine_flat_blend: (a: number) => number;
    readonly wasmstitchedengine_force_expert: (a: number, b: number) => void;
    readonly wasmstitchedengine_is_loaded: (a: number, b: number) => number;
    readonly wasmstitchedengine_loaded_count: (a: number) => number;
    readonly wasmstitchedengine_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly wasmstitchedengine_new_v7: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly wasmstitchedengine_new_v7_lazy: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasmstitchedengine_num_experts: (a: number) => number;
    readonly wasmstitchedengine_peek_expert: (a: number, b: number) => number;
    readonly wasmstitchedengine_position: (a: number) => number;
    readonly wasmstitchedengine_reset: (a: number) => void;
    readonly wasmstitchedengine_set_flat_blend: (a: number, b: number) => void;
    readonly wasmstitchedengine_set_switch_threshold: (a: number, b: number) => void;
    readonly wasmstitchedengine_step: (a: number, b: number) => number;
    readonly wasmstitchedengine_vocab_size: (a: number) => number;
    readonly wasmstitchstep_chosen_expert: (a: number) => number;
    readonly wasmstitchstep_classifier_probs: (a: number, b: number) => void;
    readonly wasmstitchstep_confidence: (a: number) => number;
    readonly wasmstitchstep_logits: (a: number, b: number) => void;
    readonly wasmstitchstep_pending_expert: (a: number) => number;
    readonly wasmstitchstep_warmup_happened: (a: number) => number;
    readonly wasmstitchstep_warmup_tokens: (a: number) => number;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
