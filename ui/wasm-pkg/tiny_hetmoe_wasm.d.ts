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

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmmodel_free: (a: number, b: number) => void;
    readonly __wbg_wasmstep_free: (a: number, b: number) => void;
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
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number) => void;
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
