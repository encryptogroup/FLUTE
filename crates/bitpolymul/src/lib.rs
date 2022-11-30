use aligned_vec::typenum::U32;
use aligned_vec::AlignedVec;
use bitpolymul_sys::{
    bc_to_lch_2_unit256, bc_to_mono_2_unit256, btfy_128, decode_128, encode_128_half_input_zero,
    gf2ext128_mul_sse, i_btfy_128,
};
use std::cmp::max;

#[derive(Clone, Debug, Default)]
pub struct FftPoly {
    n: usize,
    n_pow2: usize,
    poly: AlignedVec<u64, U32>,
}

#[derive(Default, Clone)]
pub struct DecodeCache {
    temp: AlignedVec<u64, U32>,
}

impl FftPoly {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resize(&mut self, n: usize) {
        if self.n == n {
            // do nothing
        } else if n == 0 {
            self.n = 0;
            self.n_pow2 = 0;
            self.poly.clear();
        } else {
            self.n = n;
            let log_n = log2_ceil(self.n);
            self.n_pow2 = max(1 << log_n, 256);
            self.poly.resize(2 * self.n_pow2, 0);
        }
    }

    pub fn encode(&mut self, data: &[u64]) {
        self.resize(data.len());
        if self.n == 0 {
            return;
        }
        let log_n = log2_ceil(self.n_pow2);
        let mut temp: AlignedVec<_, U32> = AlignedVec::from(data);
        temp.resize(self.n_pow2, 0);
        let n_pow2 = self.n_pow2.try_into().expect("n_pow2 overflows u32");
        unsafe {
            bc_to_lch_2_unit256(temp.as_mut_ptr(), n_pow2);
            encode_128_half_input_zero(self.poly.as_mut_ptr(), temp.as_ptr(), n_pow2);
            btfy_128(self.poly.as_mut_ptr(), n_pow2, 64 + log_n + 1);
        }
    }

    pub fn encode_new(data: &[u64]) -> Self {
        let mut fft = Self::new();
        fft.encode(data);
        fft
    }

    pub fn mult_eq(&mut self, a: &FftPoly) {
        self.mult(&self.clone(), a)
    }

    pub fn mult(&mut self, a: &FftPoly, b: &FftPoly) {
        assert_eq!(a.n_pow2, b.n_pow2);
        self.resize(a.n);
        unsafe {
            for i in 0..self.n_pow2 {
                gf2ext128_mul_sse(
                    &mut self.poly[i * 2] as *mut _ as *mut _,
                    &a.poly[i * 2] as *const _ as *const _,
                    &b.poly[i * 2] as *const _ as *const _,
                );
            }
        }
    }
    pub fn add_eq(&mut self, a: &FftPoly) {
        self.add(&self.clone(), a)
    }

    pub fn add(&mut self, a: &FftPoly, b: &FftPoly) {
        assert_eq!(a.n_pow2, b.n_pow2);
        self.resize(a.n);
        self.poly
            .iter_mut()
            .zip(a.poly.iter())
            .zip(b.poly.iter())
            .for_each(|((this, a), b)| *this = *a ^ *b);
    }

    pub fn decode(self, dest: &mut [u64]) {
        let mut cache = DecodeCache::default();
        self.decode_with_cache(&mut cache, dest)
    }

    pub fn decode_with_cache(mut self, cache: &mut DecodeCache, dest: &mut [u64]) {
        assert_eq!(dest.len(), 2 * self.n);
        if cache.temp.len() < self.poly.len() {
            cache.temp.resize(self.poly.len(), 0);
        }
        let log_n = log2_ceil(self.n_pow2);
        let n_pow2 = self.n_pow2.try_into().expect("n_pow2 overflow");
        unsafe {
            i_btfy_128(self.poly.as_mut_ptr(), n_pow2, 64 + log_n + 1);
            decode_128(cache.temp.as_mut_ptr(), self.poly.as_ptr(), n_pow2);
            bc_to_mono_2_unit256(cache.temp.as_mut_ptr(), 2 * n_pow2);
        }
        dest.copy_from_slice(&cache.temp[..dest.len()])
    }
}

pub(crate) fn log2_floor(val: usize) -> u32 {
    assert!(val > 0);
    usize::BITS - val.leading_zeros() - 1
}

pub(crate) fn log2_ceil(val: usize) -> u32 {
    let floor = log2_floor(val);
    if val > (1 << floor) {
        floor + 1
    } else {
        floor
    }
}

#[cfg(test)]
mod tests {
    use crate::FftPoly;
    use rand::{thread_rng, Rng};

    pub(crate) fn bitpolymul(c: &mut [u64], a: &[u64], b: &[u64]) {
        assert_eq!(a.len(), b.len());
        let mut a = FftPoly::encode_new(a);
        let b = FftPoly::encode_new(b);
        debug_assert_eq!(a.poly.as_ptr() as usize % 32, 0);
        debug_assert_eq!(b.poly.as_ptr() as usize % 32, 0);
        a.mult_eq(&b);
        a.decode(c)
    }

    #[test]
    fn basic_test() {
        let len = 1 << 12;
        let mut poly1 = vec![0_u64; len];
        let mut poly2 = vec![0_u64; len];
        let mut rpoly1 = vec![0_u64; 2 * len];
        let mut rpoly2 = vec![0_u64; 2 * len];
        thread_rng().fill(&mut poly1[..]);
        thread_rng().fill(&mut poly2[..]);
        bitpolymul(&mut rpoly1, &poly1, &poly2);
        bitpolymul(&mut rpoly2, &poly2, &poly1);
        assert_eq!(rpoly1, rpoly2);
    }
}
