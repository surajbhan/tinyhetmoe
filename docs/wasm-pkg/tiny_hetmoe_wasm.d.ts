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
     * Force routing to a specific expert idx for subsequent step() calls.
     * Pass -1 (or any negative value) to release back to classifier mode.
     */
    force_expert(idx: number): void;
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
    reset(): void;
    step(token: number): WasmStitchStep;
    /**
     * Return expert names as a single comma-joined string. JS splits on `,`.
     * Avoids the wasm-bindgen overhead of returning Vec<String>.
     */
    readonly expert_names: string;
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
    readonly wasmstitchedengine_expert_names: (a: number, b: number) => void;
    readonly wasmstitchedengine_force_expert: (a: number, b: number) => void;
    readonly wasmstitchedengine_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly wasmstitchedengine_num_experts: (a: number) => number;
    readonly wasmstitchedengine_position: (a: number) => number;
    readonly wasmstitchedengine_reset: (a: number) => void;
    readonly wasmstitchedengine_step: (a: number, b: number) => number;
    readonly wasmstitchedengine_vocab_size: (a: number) => number;
    readonly wasmstitchstep_chosen_expert: (a: number) => number;
    readonly wasmstitchstep_classifier_probs: (a: number, b: number) => void;
    readonly wasmstitchstep_confidence: (a: number) => number;
    readonly wasmstitchstep_logits: (a: number, b: number) => void;
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
