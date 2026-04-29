//! Gather-add ternary weights — the v3.1 production layout.
//!
//! Core insight: with ternary weights {-1, 0, +1}, matmul is *gather*, not
//! multiply. For each output row we know upfront which input columns
//! contribute as +1 and which as -1; the rest are skipped (~47% of weights).
//!
//! Storage: two big flat arrays of u16 indices (`plus_data`, `minus_data`)
//! plus per-row offset tables. No pointer chasing, prefetcher-friendly.
//!
//! Forward: `output[r] = scale * (sum(input[plus_idx[r]]) - sum(input[minus_idx[r]]))`.
//! 8-way unrolled with `unsafe get_unchecked` after construction-time bounds
//! validation.

use std::convert::TryInto;

/// Compact ternary weight matrix in gather-add layout.
///
/// `plus_data[plus_offsets[r]..plus_offsets[r+1]]` lists the column
/// indices where row `r` has a +1 weight. Likewise for minus. Both
/// arrays are `u16` because cols < 65535 in any realistic LM.
#[derive(Clone)]
pub struct PackedTernaryWeight {
    pub plus_data: Vec<u16>,
    pub plus_offsets: Vec<u32>,
    pub minus_data: Vec<u16>,
    pub minus_offsets: Vec<u32>,
    pub scale: f32,
    pub rows: usize,
    pub cols: usize,
}

impl PackedTernaryWeight {
    pub fn memory_bytes(&self) -> usize {
        self.plus_data.len() * 2
            + self.plus_offsets.len() * 4
            + self.minus_data.len() * 2
            + self.minus_offsets.len() * 4
    }

    #[inline(always)]
    fn plus_row(&self, i: usize) -> &[u16] {
        let s = self.plus_offsets[i] as usize;
        let e = self.plus_offsets[i + 1] as usize;
        &self.plus_data[s..e]
    }

    #[inline(always)]
    fn minus_row(&self, i: usize) -> &[u16] {
        let s = self.minus_offsets[i] as usize;
        let e = self.minus_offsets[i + 1] as usize;
        &self.minus_data[s..e]
    }

    /// Return the dense ternary value at (r, c). Useful for rendering
    /// in the UI's "ternary weight grid" panel.
    pub fn dense_at(&self, r: usize, c: usize) -> i8 {
        for &j in self.plus_row(r) {
            if j as usize == c { return 1; }
        }
        for &j in self.minus_row(r) {
            if j as usize == c { return -1; }
        }
        0
    }
}

/// Build gather-add layout from dense int8 ternary weights.
pub fn pack_ternary(data: &[i8], rows: usize, cols: usize, scale: f32) -> PackedTernaryWeight {
    assert!(cols <= u16::MAX as usize, "cols={} too big for u16 indices", cols);

    // Pass 1: count plus/minus per row
    let mut plus_counts = vec![0u32; rows];
    let mut minus_counts = vec![0u32; rows];
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        for &v in row {
            match v {
                1 => plus_counts[r] += 1,
                -1 => minus_counts[r] += 1,
                _ => {}
            }
        }
    }

    // Build prefix-sum offset tables (length rows+1)
    let mut plus_offsets = vec![0u32; rows + 1];
    let mut minus_offsets = vec![0u32; rows + 1];
    for r in 0..rows {
        plus_offsets[r + 1] = plus_offsets[r] + plus_counts[r];
        minus_offsets[r + 1] = minus_offsets[r] + minus_counts[r];
    }

    let plus_total = plus_offsets[rows] as usize;
    let minus_total = minus_offsets[rows] as usize;

    let mut plus_data = vec![0u16; plus_total];
    let mut minus_data = vec![0u16; minus_total];

    // Pass 2: fill, using running cursors
    let mut plus_cursor = plus_offsets.clone();
    let mut minus_cursor = minus_offsets.clone();
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        for (c, &v) in row.iter().enumerate() {
            match v {
                1 => {
                    plus_data[plus_cursor[r] as usize] = c.try_into().unwrap();
                    plus_cursor[r] += 1;
                }
                -1 => {
                    minus_data[minus_cursor[r] as usize] = c.try_into().unwrap();
                    minus_cursor[r] += 1;
                }
                _ => {}
            }
        }
    }

    PackedTernaryWeight {
        plus_data, plus_offsets, minus_data, minus_offsets,
        scale, rows, cols,
    }
}

/// Pure FP32 gather-add matvec.
///
/// `output[r] = scale * (sum(input[plus_idx[r]]) - sum(input[minus_idx[r]]))`.
/// 8-way unrolled with `unsafe get_unchecked`. Inputs validated at pack
/// time so the get_unchecked is sound (column indices < cols == input.len()).
pub fn ternary_matvec(
    w: &PackedTernaryWeight,
    input: &[f32],
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), w.cols);
    debug_assert_eq!(output.len(), w.rows);

    let scale = w.scale;
    let rows = w.rows;
    let plus_data = &w.plus_data;
    let plus_offsets = &w.plus_offsets;
    let minus_data = &w.minus_data;
    let minus_offsets = &w.minus_offsets;

    for r in 0..rows {
        let mut sum = 0.0f32;

        // ── Plus indices ──────────────────────────────────────────
        let p_start = plus_offsets[r] as usize;
        let p_end = plus_offsets[r + 1] as usize;
        let plus = &plus_data[p_start..p_end];
        let chunks = plus.len() / 8;
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut s4, mut s5, mut s6, mut s7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for c in 0..chunks {
            let base = c * 8;
            unsafe {
                s0 += *input.get_unchecked(*plus.get_unchecked(base) as usize);
                s1 += *input.get_unchecked(*plus.get_unchecked(base + 1) as usize);
                s2 += *input.get_unchecked(*plus.get_unchecked(base + 2) as usize);
                s3 += *input.get_unchecked(*plus.get_unchecked(base + 3) as usize);
                s4 += *input.get_unchecked(*plus.get_unchecked(base + 4) as usize);
                s5 += *input.get_unchecked(*plus.get_unchecked(base + 5) as usize);
                s6 += *input.get_unchecked(*plus.get_unchecked(base + 6) as usize);
                s7 += *input.get_unchecked(*plus.get_unchecked(base + 7) as usize);
            }
        }
        sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
        for k in (chunks * 8)..plus.len() {
            sum += unsafe { *input.get_unchecked(*plus.get_unchecked(k) as usize) };
        }

        // ── Minus indices ─────────────────────────────────────────
        let m_start = minus_offsets[r] as usize;
        let m_end = minus_offsets[r + 1] as usize;
        let minus = &minus_data[m_start..m_end];
        let chunks = minus.len() / 8;
        s0 = 0.0; s1 = 0.0; s2 = 0.0; s3 = 0.0;
        s4 = 0.0; s5 = 0.0; s6 = 0.0; s7 = 0.0;
        for c in 0..chunks {
            let base = c * 8;
            unsafe {
                s0 += *input.get_unchecked(*minus.get_unchecked(base) as usize);
                s1 += *input.get_unchecked(*minus.get_unchecked(base + 1) as usize);
                s2 += *input.get_unchecked(*minus.get_unchecked(base + 2) as usize);
                s3 += *input.get_unchecked(*minus.get_unchecked(base + 3) as usize);
                s4 += *input.get_unchecked(*minus.get_unchecked(base + 4) as usize);
                s5 += *input.get_unchecked(*minus.get_unchecked(base + 5) as usize);
                s6 += *input.get_unchecked(*minus.get_unchecked(base + 6) as usize);
                s7 += *input.get_unchecked(*minus.get_unchecked(base + 7) as usize);
            }
        }
        sum -= s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
        for k in (chunks * 8)..minus.len() {
            sum -= unsafe { *input.get_unchecked(*minus.get_unchecked(k) as usize) };
        }

        output[r] = sum * scale;
    }
}

/// Stats summary, used by debug print at load time.
pub fn ternary_stats(w: &PackedTernaryWeight) -> (f32, f32, f32) {
    let plus = w.plus_data.len() as f32 / w.rows as f32;
    let minus = w.minus_data.len() as f32 / w.rows as f32;
    let zeros = (w.cols as f32) - plus - minus;
    (plus, minus, zeros)
}
