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
