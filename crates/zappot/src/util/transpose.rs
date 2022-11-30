use crate::util::Block;
use bytemuck::cast_slice_mut;
use std::arch::x86_64::{__m128i, _mm_movemask_epi8, _mm_set_epi16, _mm_setr_epi8, _mm_slli_epi64};

pub fn transpose(bitmat: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    transpose_rs_sse(bitmat, rows, cols)
}

pub fn transpose_128(bitmat: &mut [Block; 128]) {
    // Todo there more efficient implementations. Look at libOte transpose128
    let bytes = cast_slice_mut(bitmat);
    let transposed = transpose(bytes, 128, 128);
    bytes.copy_from_slice(&transposed);
}

#[cfg(feature = "c_sse")]
pub fn transpose_c_sse(bitmat: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let mut out = vec![0_u8; rows * cols / 8];
    assert_eq!(
        bitmat.len(),
        rows * cols / 8,
        "Input matrix must have length of rows * cols / 8"
    );
    assert_eq!(rows % 8, 0, "rows must be divisible by 8");
    assert_eq!(cols % 8, 0, "rows must be divisible by 8");
    unsafe {
        sse_trans(out.as_mut_ptr(), bitmat.as_ptr(), rows as u64, cols as u64);
    }
    out
}

#[cfg(feature = "c_sse")]
#[link(name = "transpose")]
extern "C" {
    fn sse_trans(out: *mut u8, inp: *const u8, nrows: u64, ncols: u64);
}

// TODO: copied from swanky
#[repr(C)]
union __U128 {
    vector: __m128i,
    bytes: [u8; 16],
}

impl Default for __U128 {
    #[inline]
    fn default() -> Self {
        __U128 { bytes: [0u8; 16] }
    }
}

#[inline]
pub fn transpose_rs_sse(input: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    assert!(
        cfg!(target_feature = "sse2"),
        "SSE transpose is only available with target_feature = \"sse2\" enabled"
    );
    // TODO, why divisable by 16? This is different from c impl
    assert_eq!(nrows % 16, 0);
    assert_eq!(ncols % 8, 0);
    let mut output = vec![0u8; nrows * ncols / 8];

    let inp = |x: usize, y: usize| -> usize { x * ncols / 8 + y / 8 };
    let out = |x: usize, y: usize| -> usize { y * nrows / 8 + x / 8 };

    // TODO check unsafe in this code + compare with c version
    unsafe {
        let mut h = [0_u8; 2];
        let mut v: __m128i;
        let mut rr: usize = 0;
        let mut cc: usize;
        while rr <= nrows - 16 {
            cc = 0;
            while cc < ncols {
                v = _mm_setr_epi8(
                    *input.get_unchecked(inp(rr, cc)) as i8,
                    *input.get_unchecked(inp(rr + 1, cc)) as i8,
                    *input.get_unchecked(inp(rr + 2, cc)) as i8,
                    *input.get_unchecked(inp(rr + 3, cc)) as i8,
                    *input.get_unchecked(inp(rr + 4, cc)) as i8,
                    *input.get_unchecked(inp(rr + 5, cc)) as i8,
                    *input.get_unchecked(inp(rr + 6, cc)) as i8,
                    *input.get_unchecked(inp(rr + 7, cc)) as i8,
                    *input.get_unchecked(inp(rr + 8, cc)) as i8,
                    *input.get_unchecked(inp(rr + 9, cc)) as i8,
                    *input.get_unchecked(inp(rr + 10, cc)) as i8,
                    *input.get_unchecked(inp(rr + 11, cc)) as i8,
                    *input.get_unchecked(inp(rr + 12, cc)) as i8,
                    *input.get_unchecked(inp(rr + 13, cc)) as i8,
                    *input.get_unchecked(inp(rr + 14, cc)) as i8,
                    *input.get_unchecked(inp(rr + 15, cc)) as i8,
                );
                (0..8).rev().for_each(|i| {
                    h = (_mm_movemask_epi8(v) as i16).to_le_bytes();
                    // TODO maybe this can be optimized by directly writing the
                    // output of the movemask
                    *output.get_unchecked_mut(out(rr, cc + i)) = h[0];
                    *output.get_unchecked_mut(out(rr, cc + i) + 1) = h[1];

                    v = _mm_slli_epi64::<1>(v);
                });
                cc += 8;
            }
            rr += 16;
        }
        if rr == nrows {
            return output;
        }

        // panic!("Matrix dimensions must be divisible by 16");

        cc = 0;
        while cc <= ncols - 16 {
            let mut v = _mm_set_epi16(
                *input.get_unchecked(((rr + 7) * ncols / 8 + cc / 8) / 2) as i16,
                *input.get_unchecked(((rr + 6) * ncols / 8 + cc / 8) / 2) as i16,
                *input.get_unchecked(((rr + 5) * ncols / 8 + cc / 8) / 2) as i16,
                *input.get_unchecked(((rr + 4) * ncols / 8 + cc / 8) / 2) as i16,
                *input.get_unchecked(((rr + 3) * ncols / 8 + cc / 8) / 2) as i16,
                *input.get_unchecked(((rr + 2) * ncols / 8 + cc / 8) / 2) as i16,
                *input.get_unchecked(((rr + 1) * ncols / 8 + cc / 8) / 2) as i16,
                *input.get_unchecked((rr * ncols / 8 + cc / 8) / 2) as i16,
            );
            for i in (0..8).rev() {
                h = (_mm_movemask_epi8(v) as i16).to_le_bytes();
                *output.get_unchecked_mut((cc + i) * nrows / 8 + rr / 8) = h[0];
                *output.get_unchecked_mut((cc + i) * nrows / 8 + rr / 8 + 8) = h[1];
                v = _mm_slli_epi64::<1>(v);
            }
            cc += 16;
        }
        if cc == ncols {
            return output;
        }
        let mut tmp = __U128 {
            bytes: [
                *input.get_unchecked(rr * ncols / 8 + cc / 8),
                *input.get_unchecked((rr + 1) * ncols / 8 + cc / 8),
                *input.get_unchecked((rr + 2) * ncols / 8 + cc / 8),
                *input.get_unchecked((rr + 3) * ncols / 8 + cc / 8),
                *input.get_unchecked((rr + 4) * ncols / 8 + cc / 8),
                *input.get_unchecked((rr + 5) * ncols / 8 + cc / 8),
                *input.get_unchecked((rr + 6) * ncols / 8 + cc / 8),
                *input.get_unchecked((rr + 7) * ncols / 8 + cc / 8),
                0u8,
                0u8,
                0u8,
                0u8,
                0u8,
                0u8,
                0u8,
                0u8,
            ],
        };
        for i in (0..8).rev() {
            h = (_mm_movemask_epi8(tmp.vector) as i16).to_le_bytes();
            *output.get_unchecked_mut((cc + i) * nrows / 8 + rr / 8) = h[0];
            tmp.vector = _mm_slli_epi64::<1>(tmp.vector);
        }
    };
    output
}

#[cfg(test)]
mod tests {
    use crate::util::transpose::transpose_rs_sse;

    #[test]
    #[rustfmt::skip]
    fn sse_rust() {
        let data = vec![
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
            0b00000001,0b00000000,
        ];
        let expected: Vec<u8> = vec![
            0b11111111,0b11111111,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,
            0b00000000,0b00000000,

        ];
        let transposed = transpose_rs_sse(&data, 16, 16);
        assert_eq!(transposed, expected);
    }
}
