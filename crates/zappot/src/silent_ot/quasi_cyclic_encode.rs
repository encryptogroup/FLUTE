use std::cmp::{max, min};
use std::fmt::Debug;

use bitpolymul::{DecodeCache, FftPoly};
use bitvec::order::Lsb0;
use bitvec::slice::BitSlice;
use bytemuck::{cast_slice, cast_slice_mut};
use ndarray::Array2;
use num_integer::Integer;
use num_prime::nt_funcs::next_prime;
use rand::Rng;
use rand_core::SeedableRng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::time::Instant;

use crate::silent_ot::pprf::PprfConfig;
use crate::silent_ot::{get_partitions, ChoiceBitPacking};
use crate::util::aes_rng::AesRng;
use crate::util::transpose::transpose_128;
use crate::util::Block;

#[derive(Debug, Clone)]
pub struct QuasiCyclicEncoder {
    pub(crate) conf: QuasiCyclicConf,
}

impl QuasiCyclicEncoder {
    pub(crate) fn dual_encode(&self, rT: Array2<Block>) -> Vec<Block> {
        let conf = self.conf;
        let a = init_a_polynomials(conf);
        let mut c_mod_p1: Array2<Block> = Array2::zeros((QuasiCyclicConf::ROWS, conf.n_blocks()));
        let mut B = vec![Block::zero(); conf.N2];

        c_mod_p1
            .outer_iter_mut()
            .into_par_iter()
            .zip(rT.outer_iter())
            .for_each_init(
                || MultAddReducer::new(conf, &a),
                |reducer, (mut cmod_row, rt_row)| {
                    let cmod_row = cmod_row.as_slice_mut().unwrap();
                    let rt_row = rt_row.as_slice().unwrap();
                    reducer.reduce(cmod_row, rt_row);
                },
            );

        let num_blocks = Integer::next_multiple_of(&conf.requested_num_ots, &128);
        copy_out(&mut B[..num_blocks], &c_mod_p1);
        B.truncate(self.conf.requested_num_ots);
        B
    }

    pub(crate) fn dual_encode2(
        &self,
        rT: Array2<Block>,
        sb: &[Block],
        choice_bit_packing: ChoiceBitPacking,
    ) -> (Vec<Block>, Option<Vec<u8>>) {
        let conf = self.conf;
        let a = init_a_polynomials(conf);
        let mut c_mod_p1: Array2<Block> = Array2::zeros((QuasiCyclicConf::ROWS, conf.n_blocks()));
        let mut A = vec![Block::zero(); conf.N2];

        let end = match choice_bit_packing {
            ChoiceBitPacking::True => QuasiCyclicConf::ROWS,
            ChoiceBitPacking::False => QuasiCyclicConf::ROWS + 1,
        };
        // let mut sb: AlignedVec<u8, U16> = AlignedVec::new();
        // let sb_blocks = {
        //     assert_eq!(conf.N2 % 8, 0);
        //     let n2_bytes = conf.N2 / mem::size_of::<u8>();
        //     sb.resize(n2_bytes, 0);
        //
        //     let sb_bits: &mut BitSlice<u8, Lsb0> = BitSlice::from_slice_mut(sb.as_mut_slice());
        //     for noisy_idx in &self.conf.S {
        //         sb_bits.set(*noisy_idx, true);
        //     }
        //     cast_slice(sb.as_slice())
        // };

        let mut reducer = MultAddReducer::new(conf, &a);
        c_mod_p1
            .outer_iter_mut() // equivalent to .rows() but offers into_par_iter
            .into_par_iter()
            .take(end)
            .zip(rT.outer_iter())
            .enumerate()
            .for_each_init(
                || reducer.clone(),
                |reducer, (i, (mut cmod_row, rt_row))| {
                    let compute_c_vec = i == 0 && choice_bit_packing.packed();
                    let cmod_row = cmod_row.as_slice_mut().unwrap();
                    let rt_row = rt_row.as_slice().unwrap();

                    if compute_c_vec {
                        reducer.reduce(cmod_row, sb);
                    } else {
                        reducer.reduce(cmod_row, rt_row);
                    }
                },
            );

        let C = choice_bit_packing.unpacked().then(|| {
            let mut c128 = vec![Block::zero(); conf.n_blocks()];
            reducer.reduce(&mut c128, sb);

            let mut C = vec![0; conf.requested_num_ots];

            let c128_bits: &BitSlice<usize, Lsb0> = BitSlice::from_slice(cast_slice(&c128));
            C.iter_mut()
                .zip(c128_bits.iter().by_vals())
                .for_each(|(c, bit)| {
                    *c = bit as u8;
                });
            C
        });

        let num_blocks = Integer::next_multiple_of(&conf.requested_num_ots, &128);
        copy_out(&mut A[..num_blocks], &c_mod_p1);
        A.truncate(conf.requested_num_ots);

        (A, C)
    }
}

fn init_a_polynomials(conf: QuasiCyclicConf) -> Vec<FftPoly> {
    let mut temp = vec![0_u64; 2 * conf.n_blocks()];
    (0..conf.scaler - 1)
        .map(|s| {
            let mut fft_poly = FftPoly::new();
            let mut pub_rng = AesRng::from_seed((s + 1).into());
            pub_rng.fill(&mut temp[..]);
            fft_poly.encode(&temp);
            fft_poly
        })
        .collect()
}

fn copy_out(dest: &mut [Block], c_mod_p1: &Array2<Block>) {
    assert_eq!(dest.len() % 128, 0, "Dest must have a length of 128");
    dest.par_chunks_exact_mut(128)
        .enumerate()
        .for_each(|(i, chunk)| {
            chunk
                .iter_mut()
                .zip(c_mod_p1.column(i))
                .for_each(|(block, cmod)| *block = *cmod);
            transpose_128(chunk.try_into().unwrap());
        });
}

#[derive(Copy, Clone, Debug)]
/// Configuration options  for the quasi cyclic silent OT implementation. Is created by
/// calling the [configure()](`configure`) function.
pub struct QuasiCyclicConf {
    /// The prime for QuasiCyclic encoding
    pub(crate) P: usize,
    /// the number of OTs being requested.
    pub(crate) requested_num_ots: usize,
    /// The dense vector size, this will be at least as big as mRequestedNumOts.
    pub(crate) N: usize,
    /// The sparse vector size, this will be mN * mScaler.
    pub(crate) N2: usize,
    /// The scaling factor that the sparse vector will be compressed by.
    pub(crate) scaler: usize,
    /// The size of each regular section of the sparse vector.
    pub(crate) size_per: usize,
    /// The number of regular section of the sparse vector.
    pub(crate) num_partitions: usize,
}

impl QuasiCyclicConf {
    pub const ROWS: usize = 128;

    /// Create a new [QuasiCyclicConf](`QuasiCyclicConf`) given the provided values.
    pub fn configure(num_ots: usize, scaler: usize, sec_param: usize) -> Self {
        let P = next_prime(&max(num_ots, 128 * 128), None).unwrap();
        let num_partitions = get_partitions(scaler, P, sec_param);
        let ss = (P * scaler + num_partitions - 1) / num_partitions;
        let size_per = Integer::next_multiple_of(&ss, &8);
        let N2 = size_per * num_partitions;
        let N = N2 / scaler;
        Self {
            P,
            num_partitions,
            size_per,
            N2,
            N,
            scaler,
            requested_num_ots: num_ots,
        }
    }

    pub fn n_blocks(&self) -> usize {
        self.N / Self::ROWS
    }

    pub fn n2_blocks(&self) -> usize {
        self.N2 / Self::ROWS
    }

    pub fn n64(self) -> usize {
        self.n_blocks() * 2
    }

    pub fn P(&self) -> usize {
        self.P
    }
    pub fn requested_num_ots(&self) -> usize {
        self.requested_num_ots
    }
    pub fn N(&self) -> usize {
        self.N
    }
    pub fn N2(&self) -> usize {
        self.N2
    }
    pub fn scaler(&self) -> usize {
        self.scaler
    }
    pub fn size_per(&self) -> usize {
        self.size_per
    }
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }
    /// Returns the amount of base OTs needed for this configuration.
    pub fn base_ot_count(&self) -> usize {
        let pprf_conf = PprfConfig::from(*self);
        pprf_conf.base_ot_count()
    }
}

impl From<QuasiCyclicConf> for PprfConfig {
    fn from(conf: QuasiCyclicConf) -> Self {
        PprfConfig::new(conf.size_per, conf.num_partitions)
    }
}

#[derive(Clone)]
/// Helper struct which manages parameters and cached values for the mult_add_reduce operation
pub struct MultAddReducer<'a> {
    a_polynomials: &'a [FftPoly],
    conf: QuasiCyclicConf,
    b_poly: FftPoly,
    temp128: Vec<Block>,
    cache: DecodeCache,
}

impl<'a> MultAddReducer<'a> {
    pub(crate) fn new(conf: QuasiCyclicConf, a_polynomials: &'a [FftPoly]) -> Self {
        Self {
            a_polynomials,
            conf,
            b_poly: FftPoly::new(),
            temp128: vec![Block::zero(); 2 * conf.n_blocks()],
            cache: DecodeCache::default(),
        }
    }

    pub(crate) fn reduce(&mut self, dest: &mut [Block], b128: &[Block]) {
        let n64 = self.conf.n64();
        let mut c_poly = FftPoly::new();
        for s in 1..self.conf.scaler {
            let a_poly = &self.a_polynomials[s - 1];
            let b64 = &cast_slice(b128)[s * n64..(s + 1) * n64];
            let _now = Instant::now();
            self.b_poly.encode(b64);
            if s == 1 {
                c_poly.mult(a_poly, &self.b_poly);
            } else {
                self.b_poly.mult_eq(a_poly);
                c_poly.add_eq(&self.b_poly);
            }
        }
        c_poly.decode_with_cache(&mut self.cache, cast_slice_mut(&mut self.temp128));

        self.temp128
            .iter_mut()
            .zip(b128)
            .take(self.conf.n_blocks())
            .for_each(|(t, b)| *t ^= *b);

        modp(dest, &self.temp128, self.conf.P);
    }
}

pub fn modp(dest: &mut [Block], inp: &[Block], prime: usize) {
    let p: usize = prime;

    let p_blocks = (p + 127) / 128;
    let p_bytes = (p + 7) / 8;
    let dest_len = dest.len();
    assert!(dest_len >= p_blocks);
    assert!(inp.len() >= p_blocks);
    let count = (inp.len() * 128 + p - 1) / p;
    {
        let dest_bytes = cast_slice_mut::<_, u8>(dest);
        let inp_bytes = cast_slice::<_, u8>(inp);
        dest_bytes[..p_bytes].copy_from_slice(&inp_bytes[..p_bytes]);
    }

    for i in 1..count {
        let begin = i * p;
        let begin_block = begin / 128;
        let end_block = min(i * p + p, inp.len() * 128);
        let end_block = (end_block + 127) / 128;
        assert!(end_block <= inp.len());
        // TODO the above calculations seem redundant
        let in_i = &inp[begin_block..end_block];
        let shift = begin & 127;
        bit_shift_xor(dest, in_i, shift as u8);
    }
    let dest_bytes = cast_slice_mut::<_, u8>(dest);

    let offset = p & 7;
    if offset != 0 {
        let mask = ((1 << offset) - 1) as u8;
        let idx = p / 8;
        dest_bytes[idx] &= mask;
    }
    let rem = dest_len * 16 - p_bytes;
    if rem != 0 {
        dest_bytes[p_bytes..p_bytes + rem].fill(0);
    }
}

pub fn bit_shift_xor(dest: &mut [Block], inp: &[Block], bit_shift: u8) {
    assert!(bit_shift <= 127, "bit_shift must be less than 127");

    dest.iter_mut()
        .zip(inp)
        .zip(&inp[1..])
        .for_each(|((d, inp), inp_off)| {
            let mut shifted = *inp >> bit_shift;
            shifted |= *inp_off << (128 - bit_shift);
            *d ^= shifted;
        });
    if dest.len() >= inp.len() {
        let inp_last = *inp.last().expect("empty input");
        dest[inp.len() - 1] ^= inp_last >> bit_shift;
    }
}
