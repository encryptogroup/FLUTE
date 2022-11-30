//! SilentOT extension protocol.
#![allow(non_snake_case)]
use crate::silent_ot::pprf::{ChoiceBits, PprfConfig, PprfOutputFormat};
use crate::traits::{BaseROTReceiver, BaseROTSender};
use crate::util::aes_hash::FIXED_KEY_HASH;
use crate::util::aes_rng::AesRng;
use crate::util::tokio_rayon::AsyncThreadPool;
use crate::util::transpose::transpose_128;
use crate::util::Block;
use crate::{base_ot, BASE_OT_COUNT};
use aligned_vec::typenum::U16;
use aligned_vec::AlignedVec;
use bitpolymul::{DecodeCache, FftPoly};
use bitvec::order::Lsb0;
use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use bytemuck::{cast, cast_slice, cast_slice_mut};
use mpc_channel::CommunicationError;
use ndarray::Array2;
use num_integer::Integer;
use num_prime::nt_funcs::next_prime;
use rand::Rng;
use rand_core::{CryptoRng, RngCore, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use rayon::{ThreadPool, ThreadPoolBuilder};
use remoc::RemoteSend;
use serde::{Deserialize, Serialize};
use std::cmp::{max, min};
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use std::thread::available_parallelism;
use std::time::Instant;

pub mod pprf;

/// The chosen security parameter of 128 bits.
pub const SECURITY_PARAM: usize = 128;

/// The SilentOT sender.
pub struct Sender {
    /// The quasi cyclic configuration
    conf: QuasiCyclicConf,
    /// The ggm tree that's used to generate the sparse vectors.
    gen: pprf::Sender,
    /// ThreadPool which is used to spawn the compute heavy functions on
    thread_pool: Arc<ThreadPool>,
}

/// The SilentOT receiver.
pub struct Receiver {
    /// The quasi cyclic configuration
    conf: QuasiCyclicConf,
    /// The indices of the noisy locations in the sparse vector.
    S: Vec<usize>,
    /// The ggm tree thats used to generate the sparse vectors.
    gen: pprf::Receiver,
    /// ThreadPool which is used to spawn the compute heavy functions on
    thread_pool: Arc<ThreadPool>,
}

#[derive(Copy, Clone, Debug)]
/// Configuration options  for the quasi cyclic silent OT implementation. Is created by
/// calling the [configure()](`configure`) function.
pub struct QuasiCyclicConf {
    /// The prime for QuasiCyclic encoding
    P: usize,
    /// the number of OTs being requested.
    requested_num_ots: usize,
    /// The dense vector size, this will be at least as big as mRequestedNumOts.
    N: usize,
    /// The sparse vector size, this will be mN * mScaler.
    N2: usize,
    /// The scaling factor that the sparse vector will be compressed by.
    scaler: usize,
    /// The size of each regular section of the sparse vector.
    size_per: usize,
    /// The number of regular section of the sparse vector.
    num_partitions: usize,
}

#[derive(Serialize, Deserialize, Debug)]
/// Message sent during SilentOT evaluation.
pub enum Msg<BaseOTMsg: RemoteSend = base_ot::BaseOTMsg> {
    #[serde(bound = "")]
    BaseOTChannel(mpc_channel::Receiver<BaseOTMsg>),
    Pprf(mpc_channel::Receiver<pprf::Msg>),
}

pub enum ChoiceBitPacking {
    True,
    False,
}

#[derive(Clone)]
/// Helper struct which manages parameters and cached values for the mult_add_reduce operation
struct MultAddReducer<'a> {
    a_polynomials: &'a [FftPoly],
    conf: QuasiCyclicConf,
    b_poly: FftPoly,
    temp128: Vec<Block>,
    cache: DecodeCache,
}

impl Sender {
    #[tracing::instrument(skip(rng, sender, receiver))]
    pub async fn new<RNG: RngCore + CryptoRng + Send>(
        rng: &mut RNG,
        num_ots: usize,
        sender: &mut mpc_channel::Sender<Msg>,
        receiver: &mut mpc_channel::Receiver<Msg>,
    ) -> Self {
        let num_threads = available_parallelism()
            .expect("Unable to get parallelism")
            .get();
        Self::new_with_base_ot_sender(
            base_ot::Sender::new(),
            rng,
            num_ots,
            2,
            num_threads,
            sender,
            receiver,
        )
        .await
    }

    /// Create a new Sender with the provided base OT sender. This will execute the needed
    /// base OTs.
    #[tracing::instrument(skip(base_ot_sender, rng, sender, receiver))]
    pub async fn new_with_base_ot_sender<BaseOT, RNG>(
        mut base_ot_sender: BaseOT,
        rng: &mut RNG,
        num_ots: usize,
        scaler: usize,
        num_threads: usize,
        sender: &mut mpc_channel::Sender<Msg<BaseOT::Msg>>,
        receiver: &mut mpc_channel::Receiver<Msg<BaseOT::Msg>>,
    ) -> Self
    where
        BaseOT: BaseROTSender,
        BaseOT::Msg: RemoteSend + Debug,
        RNG: RngCore + CryptoRng + Send,
    {
        let conf = configure(num_ots, scaler, SECURITY_PARAM);
        let pprf_conf: PprfConfig = conf.into();
        let silent_base_ots = {
            let (sender, receiver) = base_ot_channel(sender, receiver)
                .await
                .expect("Establishing sub channel");
            base_ot_sender
                .send_random(pprf_conf.base_ot_count(), rng, sender, receiver)
                .await
                .expect("Failed to generate base ots")
        };
        Self::new_with_silent_base_ots(silent_base_ots, num_ots, scaler, num_threads)
    }

    /// Create a new Sender with the provided base OTs.
    ///
    /// # Panics
    /// If the number of provided base OTs is unequal to
    /// [`QuasiCyclicConf::base_ot_count()`](`QuasiCyclicConf::base_ot_count()`).
    pub fn new_with_silent_base_ots(
        silent_base_ots: Vec<[Block; 2]>,
        num_ots: usize,
        scaler: usize,
        num_threads: usize,
    ) -> Self {
        let conf = configure(num_ots, scaler, SECURITY_PARAM);
        let pprf_conf: PprfConfig = conf.into();
        assert_eq!(
            pprf_conf.base_ot_count(),
            silent_base_ots.len(),
            "Wrong number of silent base ots"
        );
        let gen = pprf::Sender::new(pprf_conf, silent_base_ots);
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Unable to initialize Sender threadpool")
            .into();

        Self {
            conf,
            gen,
            thread_pool,
        }
    }

    /// Perform the random silent send. Returns a vector of random OTs.
    pub async fn random_silent_send<RNG>(
        self,
        rng: &mut RNG,
        sender: mpc_channel::Sender<Msg>,
        receiver: mpc_channel::Receiver<Msg>,
    ) -> Vec<[Block; 2]>
    where
        RNG: RngCore + CryptoRng,
    {
        let delta = rng.gen();
        let conf = self.conf;
        let thread_pool = self.thread_pool.clone();
        let B = self
            .correlated_silent_send(delta, rng, sender, receiver)
            .await;

        thread_pool
            .spawn_install_compute(move || Sender::hash(conf, delta, &B))
            .await
    }

    /// Performs the correlated silent send. Outputs the correlated
    /// ot messages `b`. The outputs have the relation:
    /// `a[i] = b[i] + c[i] * delta`
    /// where, `a` and `c` are held by the receiver.
    pub async fn correlated_silent_send<RNG>(
        mut self,
        delta: Block,
        rng: &mut RNG,
        mut sender: mpc_channel::Sender<Msg>,
        mut receiver: mpc_channel::Receiver<Msg>,
    ) -> Vec<Block>
    where
        RNG: RngCore + CryptoRng,
    {
        let rT = {
            let (sender, _receiver) = pprf_channel(&mut sender, &mut receiver)
                .await
                .expect("Establishing pprf channel");
            self.gen
                .expand(sender, delta, rng, Some(Arc::clone(&self.thread_pool)))
                .await
        };
        let conf = self.conf;
        let mut B = self
            .thread_pool
            .clone()
            .spawn_install_compute(move || self.rand_mul_quasi_cyclic(rT))
            .await;
        B.truncate(conf.requested_num_ots);
        B
    }

    fn rand_mul_quasi_cyclic(&self, rT: Array2<Block>) -> Vec<Block> {
        let conf = self.conf;
        let rows = QuasiCyclicConf::ROWS;
        let a = init_a_polynomials(conf);
        let mut c_mod_p1: Array2<Block> = Array2::zeros((rows, conf.n_blocks()));
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
        B
    }

    fn hash(conf: QuasiCyclicConf, delta: Block, B: &[Block]) -> Vec<[Block; 2]> {
        let mask = Block::all_ones() ^ Block::constant::<1>();
        let d = delta & mask;
        let mut messages: Vec<_> = B
            .par_chunks(8)
            .flat_map(|chunk| {
                let mut messages = [Block::zero(); 2 * 8];
                chunk
                    .iter()
                    .zip(messages.chunks_exact_mut(2))
                    .for_each(|(blk, m)| {
                        let r = *blk & mask;
                        m[0] = r;
                        m[1] = r ^ d;
                    });
                FIXED_KEY_HASH.cr_hash_blocks(&messages);
                cast::<_, [[Block; 2]; 8]>(messages)
            })
            .collect();
        // Due to chunking, messages might be bigger than needed, so we truncate it
        messages.truncate(conf.requested_num_ots);
        messages
    }
}

impl Receiver {
    #[tracing::instrument(skip(rng, sender, receiver))]
    pub async fn new<RNG: RngCore + CryptoRng + Send>(
        rng: &mut RNG,
        num_ots: usize,
        sender: &mut mpc_channel::Sender<Msg>,
        receiver: &mut mpc_channel::Receiver<Msg>,
    ) -> Self {
        let num_threads = available_parallelism()
            .expect("Unable to get parallelism")
            .get();
        Self::new_with_base_ot_receiver(
            base_ot::Receiver::new(),
            rng,
            num_ots,
            2,
            num_threads,
            sender,
            receiver,
        )
        .await
    }

    /// Create a new Receiver with the provided base OT receiver. This will execute the needed
    /// base OTs.
    #[tracing::instrument(skip(base_ot_receiver, rng, sender, receiver))]
    pub async fn new_with_base_ot_receiver<BaseOT, RNG>(
        mut base_ot_receiver: BaseOT,
        rng: &mut RNG,
        num_ots: usize,
        scaler: usize,
        num_threads: usize,
        sender: &mut mpc_channel::Sender<Msg<BaseOT::Msg>>,
        receiver: &mut mpc_channel::Receiver<Msg<BaseOT::Msg>>,
    ) -> Self
    where
        BaseOT: BaseROTReceiver,
        BaseOT::Msg: RemoteSend + Debug,
        RNG: RngCore + CryptoRng + Send,
    {
        let conf = configure(num_ots, scaler, SECURITY_PARAM);
        let silent_choice_bits = Self::sample_base_choice_bits(conf, rng);
        let silent_base_ots = {
            let choices = silent_choice_bits.as_bit_vec();
            let (sender, receiver) = base_ot_channel(sender, receiver)
                .await
                .expect("Establishing Base OT channel");
            base_ot_receiver
                .receive_random(&choices, rng, sender, receiver)
                .await
                .expect("Failed to generate base ots")
        };
        Self::new_with_silent_base_ots(
            silent_base_ots,
            silent_choice_bits,
            num_ots,
            scaler,
            num_threads,
        )
    }

    /// Create a new Receiver with the provided base OTs and choice bits. The
    /// [`ChoiceBits`](`ChoiceBits`) need to be sampled by calling
    /// [`Receiver::sample_base_choice_bits()`](`Receiver::sample_base_choice_bits()`).
    ///
    /// # Panics
    /// If the number of provided base OTs is unequal to
    /// [`QuasiCyclicConf::base_ot_count()`](`QuasiCyclicConf::base_ot_count()`).
    pub fn new_with_silent_base_ots(
        base_ots: Vec<Block>,
        base_choices: ChoiceBits,
        num_ots: usize,
        scaler: usize,
        num_threads: usize,
    ) -> Self {
        let conf = configure(num_ots, scaler, 128);
        let pprf_conf = conf.into();
        let gen = pprf::Receiver::new(pprf_conf, base_ots, base_choices);
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Unable to initialize Sender threadpool")
            .into();

        Self {
            conf,
            S: gen.get_points(PprfOutputFormat::InterleavedTransposed),
            gen,
            thread_pool,
        }
    }

    /// Perform the random silent receive. Returns a vector `a` of random OTs and choices `c`
    /// corresponding to the OTs.
    ///
    /// Note that this is not the *usual* R-OT interface, as the choices are not provided by the
    /// user, but are the output.
    pub async fn random_silent_receive(
        self,
        sender: mpc_channel::Sender<Msg>,
        receiver: mpc_channel::Receiver<Msg>,
    ) -> (Vec<Block>, BitVec) {
        let conf = self.conf;
        let thread_pool = self.thread_pool.clone();
        let (A, _) = self
            .correlated_silent_receive(ChoiceBitPacking::True, sender, receiver)
            .await;

        thread_pool
            .spawn_install_compute(move || Self::hash(conf, &A))
            .await
    }

    /// Performs the correlated silent receive. Outputs the correlated
    /// ot messages `a` and choices `c`. The outputs have the relation:
    /// `a[i] = b[i] + c[i] * delta`
    /// where, `b` and `delta` are held by the sender.
    pub async fn correlated_silent_receive(
        mut self,
        choice_bit_packing: ChoiceBitPacking,
        mut sender: mpc_channel::Sender<Msg>,
        mut receiver: mpc_channel::Receiver<Msg>,
    ) -> (Vec<Block>, Option<Vec<u8>>) {
        let rT = {
            let (_sender, receiver) = pprf_channel(&mut sender, &mut receiver)
                .await
                .expect("Establishing pprf channel");
            self.gen
                .expand(receiver, Some(Arc::clone(&self.thread_pool)))
                .await
        };

        let conf = self.conf;
        let (mut A, C) = self
            .thread_pool
            .clone()
            .spawn_install_compute(move || self.rand_mul_quasi_cyclic(rT, choice_bit_packing))
            .await;

        A.truncate(conf.requested_num_ots);
        let C = C.map(|mut cvec| {
            cvec.truncate(conf.requested_num_ots);
            cvec
        });
        (A, C)
    }

    pub fn rand_mul_quasi_cyclic(
        &self,
        rT: Array2<Block>,
        choice_bit_packing: ChoiceBitPacking,
    ) -> (Vec<Block>, Option<Vec<u8>>) {
        let conf = self.conf;
        // assert_eq!(conf.n2_blocks(), rT.ncols());
        let a = init_a_polynomials(conf);
        let rows = QuasiCyclicConf::ROWS;
        let mut c_mod_p1: Array2<Block> = Array2::zeros((rows, conf.n_blocks()));
        let mut A = vec![Block::zero(); conf.N2];

        let end = match choice_bit_packing {
            ChoiceBitPacking::True => rows,
            ChoiceBitPacking::False => rows + 1,
        };
        let mut sb: AlignedVec<u8, U16> = AlignedVec::new();
        let sb_blocks = {
            assert_eq!(conf.N2 % 8, 0);
            let n2_bytes = conf.N2 / mem::size_of::<u8>();
            sb.resize(n2_bytes, 0);

            let sb_bits: &mut BitSlice<u8, Lsb0> = BitSlice::from_slice_mut(sb.as_mut_slice());
            for noisy_idx in &self.S {
                sb_bits.set(*noisy_idx, true);
            }
            cast_slice(sb.as_slice())
        };

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
                        reducer.reduce(cmod_row, sb_blocks);
                    } else {
                        reducer.reduce(cmod_row, rt_row);
                    }
                },
            );

        let C = choice_bit_packing.unpacked().then(|| {
            let mut c128 = vec![Block::zero(); conf.n_blocks()];
            reducer.reduce(&mut c128, sb_blocks);

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

        (A, C)
    }

    fn hash(conf: QuasiCyclicConf, A: &[Block]) -> (Vec<Block>, BitVec) {
        let mask = Block::all_ones() ^ Block::constant::<1>();
        let (mut messages, mut choices): (Vec<_>, Vec<_>) =
            A.par_chunks(8)
                .map(|chunk| {
                    let mut messages = [Block::zero(); 8];
                    let mut choices = [false; 8];
                    chunk.iter().zip(&mut messages).zip(&mut choices).for_each(
                        |((blk, m), choice)| {
                            *m = *blk & mask;
                            *choice = blk.lsb();
                        },
                    );
                    FIXED_KEY_HASH.cr_hash_blocks(&messages);
                    messages.into_iter().zip(choices)
                })
                .flatten_iter()
                .unzip();
        // Due to chunking, messages might be bigger than needed, so we truncate it
        messages.truncate(conf.requested_num_ots);
        choices.truncate(conf.requested_num_ots);
        let choices = BitVec::from_iter(choices);
        (messages, choices)
    }

    /// Sample the choice bits for the base OTs.
    pub fn sample_base_choice_bits<RNG: RngCore + CryptoRng>(
        conf: QuasiCyclicConf,
        rng: &mut RNG,
    ) -> ChoiceBits {
        let pprf_conf = conf.into();
        pprf::Receiver::sample_choice_bits(
            pprf_conf,
            conf.N2,
            PprfOutputFormat::InterleavedTransposed,
            rng,
        )
    }
}

impl QuasiCyclicConf {
    pub const ROWS: usize = 128;

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

impl ChoiceBitPacking {
    pub fn packed(&self) -> bool {
        matches!(self, Self::True)
    }

    pub fn unpacked(&self) -> bool {
        !self.packed()
    }
}

impl<'a> MultAddReducer<'a> {
    fn new(conf: QuasiCyclicConf, a_polynomials: &'a [FftPoly]) -> Self {
        Self {
            a_polynomials,
            conf,
            b_poly: FftPoly::new(),
            temp128: vec![Block::zero(); 2 * conf.n_blocks()],
            cache: DecodeCache::default(),
        }
    }

    fn reduce(&mut self, dest: &mut [Block], b128: &[Block]) {
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

/// Create a new [QuasiCyclicConf](`QuasiCyclicConf`) given the provided values.
pub fn configure(num_ots: usize, scaler: usize, sec_param: usize) -> QuasiCyclicConf {
    let P = next_prime(&max(num_ots, 128 * 128), None).unwrap();
    let num_partitions = get_partitions(scaler, P, sec_param);
    let ss = (P * scaler + num_partitions - 1) / num_partitions;
    let size_per = Integer::next_multiple_of(&ss, &8);
    let N2 = size_per * num_partitions;
    let N = N2 / scaler;
    QuasiCyclicConf {
        P,
        num_partitions,
        size_per,
        N2,
        N,
        scaler,
        requested_num_ots: num_ots,
    }
}

fn get_partitions(scaler: usize, prime: usize, sec_param: usize) -> usize {
    assert!(scaler >= 2, "scaler must be 2 or greater");
    let mut ret = 1;
    let mut ss = sec_level(scaler, prime, ret);
    while ss < sec_param {
        ret += 1;
        ss = sec_level(scaler, prime, ret);
        assert!(ret <= 1000, "failed to find silentOt parameters");
    }
    Integer::next_multiple_of(&ret, &8)
}

fn sec_level(scale: usize, p: usize, points: usize) -> usize {
    let x1 = ((scale * p) as f64 / p as f64).log2();
    let x2 = ((scale * p) as f64).log2() / 2.0;
    let sec_level = points as f64 * x1 + x2;
    sec_level as usize
}

fn modp(dest: &mut [Block], inp: &[Block], prime: usize) {
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

async fn base_ot_channel<BaseMsg: RemoteSend>(
    sender: &mut mpc_channel::Sender<Msg<BaseMsg>>,
    receiver: &mut mpc_channel::Receiver<Msg<BaseMsg>>,
) -> Result<(mpc_channel::Sender<BaseMsg>, mpc_channel::Receiver<BaseMsg>), CommunicationError> {
    mpc_channel::sub_channel_with(sender, receiver, BASE_OT_COUNT, Msg::BaseOTChannel, |msg| {
        match msg {
            Msg::BaseOTChannel(receiver) => Some(receiver),
            _ => None,
        }
    })
    .await
}

async fn pprf_channel<BaseMsg: RemoteSend>(
    sender: &mut mpc_channel::Sender<Msg<BaseMsg>>,
    receiver: &mut mpc_channel::Receiver<Msg<BaseMsg>>,
) -> Result<
    (
        mpc_channel::Sender<pprf::Msg>,
        mpc_channel::Receiver<pprf::Msg>,
    ),
    CommunicationError,
> {
    mpc_channel::sub_channel_with(sender, receiver, 128, Msg::Pprf, |msg| match msg {
        Msg::Pprf(receiver) => Some(receiver),
        _ => None,
    })
    .await
}

#[cfg(test)]
mod test {
    use crate::silent_ot::{bit_shift_xor, configure, modp, ChoiceBitPacking, Receiver, Sender};

    use crate::silent_ot::pprf::tests::fake_base;
    use crate::silent_ot::pprf::PprfOutputFormat;
    use crate::util::Block;
    use bitvec::order::Lsb0;
    use bitvec::slice::BitSlice;
    use bitvec::vec::BitVec;
    use rand::rngs::StdRng;
    use rand_core::SeedableRng;
    use std::cmp::min;

    fn check_correlated(A: &[Block], B: &[Block], choice: Option<&[u8]>, delta: Block) {
        let n = A.len();
        assert_eq!(B.len(), n);
        if let Some(choice) = choice {
            assert_eq!(choice.len(), n)
        }
        let mask = if choice.is_some() {
            // don't mask off lsb when not using choice packing
            Block::all_ones()
        } else {
            // mask to get lsb
            Block::all_ones() ^ Block::one()
        };

        for i in 0..n {
            let m1 = A[i];
            let c = if let Some(choice) = choice {
                choice[i] as usize
            } else {
                // extract choice bit from m1
                ((m1 & Block::from(1_u64)) == Block::from(1_u64)) as usize
            };
            let m1 = m1 & mask;
            let m2a = B[i] & mask;
            let m2b = (B[i] ^ delta) & mask;

            let eqq = [m1 == m2a, m1 == m2b];
            assert!(
                eqq[c] == true && eqq[c ^ 1] == false,
                "Blocks at {i} differ"
            );
            assert!(eqq[0] != false || eqq[1] != false);
        }
    }

    fn check_random(send_messages: &[[Block; 2]], recv_messages: &[Block], choice: &BitSlice) {
        let n = send_messages.len();
        assert_eq!(recv_messages.len(), n);
        assert_eq!(choice.len(), n);
        for i in 0..n {
            let m1 = recv_messages[i];
            let m2a = send_messages[i][0];
            let m2b = send_messages[i][1];
            let c = choice[i] as usize;

            let eqq = [m1 == m2a, m1 == m2b];

            assert!(eqq[c] ^ eqq[c ^ 1], "ROT Block {i} failed");
        }
    }

    #[test]
    fn basic_bit_shift_xor() {
        let dest = &mut [Block::zero(), Block::zero()];
        let inp = &[Block::all_ones(), Block::all_ones()];
        let bit_shift = 10;
        bit_shift_xor(dest, inp, bit_shift);
        assert_eq!(Block::all_ones(), dest[0]);
        let exp = Block::from(u128::MAX >> bit_shift);
        assert_eq!(exp, dest[1]);
    }

    #[test]
    fn basic_modp() {
        let i_bits = 1026;
        let n_bits = 223;
        let n = (n_bits + 127) / 128;
        let c = (i_bits + n_bits - 1) / n_bits;
        let mut dest = vec![Block::zero(); n];
        let mut inp = vec![Block::all_ones(); (i_bits + 127) / 128];
        let p = n_bits;
        let inp_bits: &mut BitSlice<usize, Lsb0> =
            BitSlice::from_slice_mut(bytemuck::cast_slice_mut(&mut inp));
        inp_bits[i_bits..].fill(false);
        let mut dv: BitVec<usize, Lsb0> = BitVec::repeat(true, p);
        let mut iv: BitVec<usize, Lsb0> = BitVec::new();
        for j in 1..c {
            let rem = min(p, i_bits - j * p);
            iv.clear();
            let inp = &inp_bits[j * p..(j * p) + rem];
            iv.extend_from_bitslice(inp);
            iv.resize(p, false);
            dv ^= &iv;
        }
        modp(&mut dest, &inp, p);
        let dest_bits: &BitSlice<usize, Lsb0> = BitSlice::from_slice(bytemuck::cast_slice(&dest));
        let dv2 = &dest_bits[..p];
        assert_eq!(dv, dv2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn correlated_silent_ot() {
        let num_ots = 1000;
        let scaler = 2;
        let num_threads = 2;
        let delta = Block::all_ones();
        let conf = configure(num_ots, scaler, 128);
        let (ch1, ch2) = mpc_channel::in_memory::new_pair(128);
        let mut rng = StdRng::seed_from_u64(42);
        let (sender_base_ots, receiver_base_ots, base_choices) = fake_base(
            conf.into(),
            conf.N2,
            PprfOutputFormat::InterleavedTransposed,
            &mut rng,
        );

        let send = tokio::spawn(async move {
            let sender =
                Sender::new_with_silent_base_ots(sender_base_ots, num_ots, scaler, num_threads);
            sender
                .correlated_silent_send(delta, &mut rng, ch1.0, ch1.1)
                .await
        });
        let receiver = Receiver::new_with_silent_base_ots(
            receiver_base_ots,
            base_choices,
            num_ots,
            scaler,
            num_threads,
        );
        let receive = tokio::spawn(async move {
            receiver
                .correlated_silent_receive(ChoiceBitPacking::False, ch2.0, ch2.1)
                .await
        });
        let (r_out, s_out) = futures::future::try_join(receive, send).await.unwrap();
        check_correlated(&r_out.0, &s_out, r_out.1.as_deref(), delta);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn random_silent_ot() {
        let num_ots = 1000;
        let scaler = 2;
        let num_threads = 2;
        let conf = configure(num_ots, scaler, 128);
        let (ch1, ch2) = mpc_channel::in_memory::new_pair(128);
        let mut rng = StdRng::seed_from_u64(42);
        let (sender_base_ots, receiver_base_ots, base_choices) = fake_base(
            conf.into(),
            conf.N2,
            PprfOutputFormat::InterleavedTransposed,
            &mut rng,
        );

        let send = tokio::spawn(async move {
            let sender =
                Sender::new_with_silent_base_ots(sender_base_ots, num_ots, scaler, num_threads);
            sender.random_silent_send(&mut rng, ch1.0, ch1.1).await
        });
        let receiver = Receiver::new_with_silent_base_ots(
            receiver_base_ots,
            base_choices,
            num_ots,
            scaler,
            num_threads,
        );
        let receive =
            tokio::spawn(async move { receiver.random_silent_receive(ch2.0, ch2.1).await });
        let (r_out, s_out) = futures::future::try_join(receive, send).await.unwrap();
        check_random(&s_out, &r_out.0, &r_out.1);
    }
}
