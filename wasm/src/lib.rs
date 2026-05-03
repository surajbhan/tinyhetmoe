//! TinyHetMoE ternary inference — native + WASM.
//!
//! Forked from the production HetMoE engine
//! (`/data/training_kanam/ternary_inference_v3`) and adapted for the
//! TinyHetMoE architecture:
//!   - Highway expansion: input_dim (264) → internal_dim (528) → input_dim
//!   - M+I split embedding (meaning_dim 132 + intuition_dim 132)
//!   - NoPE (no rotary positional encoding, just causal mask)
//!   - QK-Norm (RMSNorm per-head on q and k)
//!   - Meaning protection (attn/MoE outputs zeroed on dims 0:meaning_dim)
//!   - Untied lm_head (separate ternary weight)
//!
//! Binary format consumed: HTMOE002. See `scripts/export_model.py`.
//!
//! Forward pass at fp32 with INT8 activation quantization in matvec
//! hotspots (same trick as v3).

use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Cursor, Read};

pub mod model;
pub mod tensor;
pub mod forward;
pub mod stitch;

pub use model::{Config, Model, ScratchBuffers, Layer, Expert, ExpertArch, KVCache};
pub use tensor::PackedTernaryWeight;
pub use forward::{forward_token_with_trace, TokenTrace};
pub use stitch::{StitchedEngine, StitchExpert, DomainClassifier, StitchStep, RouteMode};

#[cfg(feature = "wasm")]
pub mod wasm_api;

#[cfg(feature = "wasm")]
pub mod wasm_api_stitch;

/// Read a shared MNGSHR04 file → (vocab_size, meaning_dim, fp32 weights).
/// Browser downloads this once across all 6 experts since meaning_embed
/// is frozen and identical.
pub fn load_shared_meaning_from_bytes(file_data: Vec<u8>) -> std::io::Result<(usize, usize, Vec<f32>)> {
    let mut cursor = Cursor::new(file_data);
    let mut magic = [0u8; 8];
    cursor.read_exact(&mut magic)?;
    if &magic != b"MNGSHR04" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Expected magic MNGSHR04, got {:?}", magic),
        ));
    }
    let vocab_size = cursor.read_u32::<LittleEndian>()? as usize;
    let meaning_dim = cursor.read_u32::<LittleEndian>()? as usize;
    let count = vocab_size * meaning_dim;
    let mut weights = vec![0.0f32; count];
    for v in weights.iter_mut() {
        let h = cursor.read_u16::<LittleEndian>()?;
        *v = f16_to_f32(h);
    }
    Ok((vocab_size, meaning_dim, weights))
}

/// IEEE 754 binary16 → f32. Matches `numpy.float16.astype(np.float32)`.
fn f16_to_f32(h: u16) -> f32 {
    // Sign / exponent / mantissa from the 16-bit half
    let sign = (h >> 15) & 0x1;
    let exp  = (h >> 10) & 0x1f;
    let frac = h & 0x3ff;

    let bits = if exp == 0 {
        if frac == 0 {
            // signed zero
            (sign as u32) << 31
        } else {
            // subnormal half → normalize as f32
            let mut e: i32 = -14;
            let mut m: u32 = frac as u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let exp32 = ((e + 127) as u32) & 0xff;
            ((sign as u32) << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        // Inf / NaN
        ((sign as u32) << 31) | 0x7f800000 | ((frac as u32) << 13)
    } else {
        let exp32 = (exp as i32 - 15 + 127) as u32;
        ((sign as u32) << 31) | (exp32 << 23) | ((frac as u32) << 13)
    };
    f32::from_bits(bits)
}

/// Top-level model loader. Reads bytes (file contents in native, or
/// JS-supplied ArrayBuffer in WASM) and returns a Model ready to infer.
///
/// Supports magics:
///   HTMOE002: int8 ternary,    fp32 embeds, includes meaning_embed
///   HTMOE003: 2-bit ternary,   fp32 embeds, includes meaning_embed
///   HTMOE004: 2-bit ternary,   fp16 embeds, MAY skip meaning_embed
///             (in which case caller supplies it via shared_meaning)
///
/// `shared_meaning`: required when the file is HTMOE004 (which always
/// uses an external shared meaning). Ignored for HTMOE002/003.
pub fn load_model_from_bytes(file_data: Vec<u8>) -> std::io::Result<Model> {
    load_model_from_bytes_with_meaning(file_data, None)
}

pub fn load_model_from_bytes_with_meaning(
    file_data: Vec<u8>,
    shared_meaning: Option<Vec<f32>>,
) -> std::io::Result<Model> {
    let mut cursor = Cursor::new(file_data);

    let mut magic = [0u8; 8];
    cursor.read_exact(&mut magic)?;
    let (packed, fp16_embeds) = match &magic {
        b"HTMOE002" => (false, false),         // legacy: 1 byte / weight, fp32 embeds
        b"HTMOE003" => (true,  false),         // packed: 2 bits / weight, fp32 embeds
        b"HTMOE004" => (true,  true),          // packed + fp16 embeds, optional shared meaning
        _ => return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Expected magic HTMOE002/003/004, got {:?}", magic),
        )),
    };

    let config = Config {
        vocab_size:    cursor.read_u32::<LittleEndian>()? as usize,
        meaning_dim:   cursor.read_u32::<LittleEndian>()? as usize,
        intuition_dim: cursor.read_u32::<LittleEndian>()? as usize,
        input_dim:     cursor.read_u32::<LittleEndian>()? as usize,
        internal_dim:  cursor.read_u32::<LittleEndian>()? as usize,
        new_intuition: cursor.read_u32::<LittleEndian>()? as usize,
        num_layers:    cursor.read_u32::<LittleEndian>()? as usize,
        num_heads:     cursor.read_u32::<LittleEndian>()? as usize,
        head_dim:      cursor.read_u32::<LittleEndian>()? as usize,
        max_seq_len:   cursor.read_u32::<LittleEndian>()? as usize,
        num_experts:   cursor.read_u32::<LittleEndian>()? as usize,
        top_k_experts: cursor.read_u32::<LittleEndian>()? as usize,
        ffn_mult:      cursor.read_u32::<LittleEndian>()? as f32 / 100.0,
    };

    // Expert architecture types per layer
    let mut expert_archs = vec![vec![ExpertArch::Standard; config.num_experts]; config.num_layers];
    for l in 0..config.num_layers {
        for e in 0..config.num_experts {
            let arch_byte = cursor.read_u8()?;
            expert_archs[l][e] = match arch_byte {
                0 => ExpertArch::Standard,
                1 => ExpertArch::SwiGLU,
                2 => ExpertArch::DeepNarrow,
                3 => ExpertArch::Bottleneck,
                _ => return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Unknown expert arch byte {}", arch_byte),
                )),
            };
        }
    }

    // M+I embeddings.
    //   HTMOE002/003: fp32, both meaning + intuition in the file.
    //   HTMOE004:     fp16, meaning is supplied externally (shared file).
    //                       The file contains intuition only; reader gets
    //                       meaning from `shared_meaning` arg.
    let meaning_embed = if fp16_embeds {
        // HTMOE004: meaning is external
        let m = shared_meaning.ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "HTMOE004 file requires `shared_meaning` to be supplied; \
             load via load_shared_meaning_from_bytes() first",
        ))?;
        let expected = config.vocab_size * config.meaning_dim;
        if m.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "shared_meaning length {} does not match vocab*meaning_dim = {}",
                    m.len(), expected
                ),
            ));
        }
        m
    } else {
        read_fp32(&mut cursor, config.vocab_size * config.meaning_dim)?
    };
    let intuition_embed = if fp16_embeds {
        read_fp16_as_fp32(&mut cursor, config.vocab_size * config.intuition_dim)?
    } else {
        read_fp32(&mut cursor, config.vocab_size * config.intuition_dim)?
    };

    // Highway expand
    let expand = read_ternary(&mut cursor, packed)?;

    // Layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for l in 0..config.num_layers {
        let attn_norm = read_fp32_tensor(&mut cursor)?;
        let q_norm    = read_fp32_tensor(&mut cursor)?;
        let k_norm    = read_fp32_tensor(&mut cursor)?;
        let ffn_norm  = read_fp32_tensor(&mut cursor)?;

        let q_proj = read_ternary(&mut cursor, packed)?;
        let k_proj = read_ternary(&mut cursor, packed)?;
        let v_proj = read_ternary(&mut cursor, packed)?;
        let o_proj = read_ternary(&mut cursor, packed)?;

        let mut experts = Vec::with_capacity(config.num_experts);
        for e in 0..config.num_experts {
            let arch = expert_archs[l][e];
            let n_weights = arch.num_weights();
            let mut weights = Vec::with_capacity(n_weights);
            for _ in 0..n_weights {
                weights.push(read_ternary(&mut cursor, packed)?);
            }
            experts.push(Expert { arch, weights });
        }

        let gate = read_fp32_tensor(&mut cursor)?;

        layers.push(Layer {
            attn_norm, q_norm, k_norm, ffn_norm,
            q_proj, k_proj, v_proj, o_proj,
            experts, gate,
        });
    }

    // Highway compress
    let compress = read_ternary(&mut cursor, packed)?;
    // Final norm
    let final_norm = read_fp32_tensor(&mut cursor)?;
    // lm_head (ternary, untied)
    let lm_head = read_ternary(&mut cursor, packed)?;

    let mut remaining = Vec::new();
    cursor.read_to_end(&mut remaining)?;
    if !remaining.is_empty() {
        eprintln!("[load] WARN {} unexpected bytes at end", remaining.len());
    }

    Ok(Model {
        config,
        meaning_embed,
        intuition_embed,
        expand,
        layers,
        compress,
        final_norm,
        lm_head,
    })
}


/// HTMOE004 embedding: u32 count, count*f16 little-endian.
/// Cast to fp32 on read (model uses fp32 internally; ~2× memory blow-up
/// at load time, but storage on disk / over the wire is fp16).
fn read_fp16_as_fp32(
    cursor: &mut Cursor<Vec<u8>>, expected_count: usize,
) -> std::io::Result<Vec<f32>> {
    let count = cursor.read_u32::<LittleEndian>()? as usize;
    if count != expected_count {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("FP16 count mismatch: expected {}, got {}", expected_count, count),
        ));
    }
    let mut data = vec![0.0f32; count];
    for v in data.iter_mut() {
        let h = cursor.read_u16::<LittleEndian>()?;
        *v = f16_to_f32(h);
    }
    Ok(data)
}

fn read_fp32(cursor: &mut Cursor<Vec<u8>>, expected_count: usize) -> std::io::Result<Vec<f32>> {
    let count = cursor.read_u32::<LittleEndian>()? as usize;
    if count != expected_count {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("FP32 count mismatch: expected {}, got {}", expected_count, count),
        ));
    }
    let mut data = vec![0.0f32; count];
    for v in data.iter_mut() {
        *v = cursor.read_f32::<LittleEndian>()?;
    }
    Ok(data)
}

fn read_fp32_tensor(cursor: &mut Cursor<Vec<u8>>) -> std::io::Result<Vec<f32>> {
    let count = cursor.read_u32::<LittleEndian>()? as usize;
    let mut data = vec![0.0f32; count];
    for v in data.iter_mut() {
        *v = cursor.read_f32::<LittleEndian>()?;
    }
    Ok(data)
}

fn read_ternary(cursor: &mut Cursor<Vec<u8>>, packed: bool) -> std::io::Result<PackedTernaryWeight> {
    let scale = cursor.read_f32::<LittleEndian>()?;
    let ndim = cursor.read_u32::<LittleEndian>()? as usize;
    let mut shape = vec![0usize; ndim];
    for d in shape.iter_mut() {
        *d = cursor.read_u32::<LittleEndian>()? as usize;
    }
    let rows = shape[0];
    let cols = if ndim > 1 { shape[1] } else { 1 };
    let count = rows * cols;

    let data: Vec<i8> = if packed {
        // HTMOE003: 4 weights per byte (2 bits each).
        // Encoding: 00→0, 01→+1, 11→-1
        let n_bytes = (count + 3) / 4;
        let mut packed_buf = vec![0u8; n_bytes];
        cursor.read_exact(&mut packed_buf)?;
        let mut data = vec![0i8; count];
        for i in 0..count {
            let bits = (packed_buf[i / 4] >> ((i % 4) * 2)) & 0b11;
            data[i] = match bits {
                0b00 => 0,
                0b01 => 1,
                0b11 => -1,
                _ => 0,  // 0b10 reserved/unused
            };
        }
        data
    } else {
        // HTMOE002: 1 byte per weight
        let mut buf = vec![0u8; count];
        cursor.read_exact(&mut buf)?;
        buf.iter().map(|&b| b as i8).collect()
    };

    Ok(tensor::pack_ternary(&data, rows, cols, scale))
}
