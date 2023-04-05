//! SilentOT extension protocol.
#![allow(non_snake_case)]
use crate::silent_ot::pprf::{ChoiceBits, PprfConfig, PprfOutputFormat};
use crate::traits::{BaseROTReceiver, BaseROTSender};
use crate::util::aes_hash::FIXED_KEY_HASH;

use crate::util::tokio_rayon::AsyncThreadPool;

use crate::util::Block;
use crate::{base_ot, BASE_OT_COUNT};
use aligned_vec::typenum::U16;
use aligned_vec::AlignedVec;

use bitvec::order::Lsb0;
use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use bytemuck::cast_slice;
use mpc_channel::CommunicationError;
use ndarray::Array2;
use num_integer::Integer;
use rand::Rng;
use rand_core::{CryptoRng, RngCore};

use crate::silent_ot::quasi_cyclic_encode::QuasiCyclicEncoder;
use crate::silent_ot::silver_encode::{SilverConf, SilverEncoder};
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use quasi_cyclic_encode::QuasiCyclicConf;
use rand::distributions::Standard;
use rayon::{ThreadPool, ThreadPoolBuilder};
use remoc::RemoteSend;
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use std::thread::available_parallelism;

pub mod pprf;
pub mod quasi_cyclic_encode;
pub mod silver_encode;

/// The chosen security parameter of 128 bits.
pub const SECURITY_PARAM: usize = 128;

/// The SilentOT sender.
pub struct Sender {
    enc: Encoder,
    gap_ots: Vec<[Block; 2]>,
    /// The ggm tree that's used to generate the sparse vectors.
    gen: pprf::Sender,
    /// ThreadPool which is used to spawn the compute heavy functions on
    thread_pool: Arc<ThreadPool>,
}

/// The SilentOT receiver.
pub struct Receiver {
    enc: Encoder,
    gap_ots: Vec<Block>,
    gap_choices: BitVec,
    /// The indices of the noisy locations in the sparse vector.
    S: Vec<usize>,
    /// The ggm tree thats used to generate the sparse vectors.
    gen: pprf::Receiver,
    /// ThreadPool which is used to spawn the compute heavy functions on
    thread_pool: Arc<ThreadPool>,
}

#[derive(Debug)]
pub enum Encoder {
    QuasiCyclic(QuasiCyclicEncoder),
    Silver(SilverEncoder),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MultType {
    QuasiCyclic { scaler: usize },
    Silver5,
    Silver11,
}

#[derive(Serialize, Deserialize, Debug)]
/// Message sent during SilentOT evaluation.
pub enum Msg<BaseOTMsg: RemoteSend = base_ot::BaseOTMsg> {
    #[serde(bound = "")]
    BaseOTChannel(mpc_channel::Receiver<BaseOTMsg>),
    Pprf(mpc_channel::Receiver<pprf::Msg>),
    GapValues(Vec<Block>),
}

pub enum ChoiceBitPacking {
    True,
    False,
}

#[derive(Debug, Clone)]
enum ArrayOrVec {
    Array(Array2<Block>),
    Vec(Vec<Block>),
}

impl Sender {
    #[tracing::instrument(skip(rng, sender, receiver))]
    pub async fn new<RNG: RngCore + CryptoRng + Send>(
        rng: &mut RNG,
        num_ots: usize,
        mult_type: MultType,
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
            mult_type,
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
        mult_type: MultType,
        num_threads: usize,
        sender: &mut mpc_channel::Sender<Msg<BaseOT::Msg>>,
        receiver: &mut mpc_channel::Receiver<Msg<BaseOT::Msg>>,
    ) -> Self
    where
        BaseOT: BaseROTSender,
        BaseOT::Msg: RemoteSend + Debug,
        RNG: RngCore + CryptoRng + Send,
    {
        let enc = Encoder::configure(num_ots, SECURITY_PARAM, mult_type);
        let silent_base_ots = {
            let (sender, receiver) = base_ot_channel(sender, receiver)
                .await
                .expect("Establishing sub channel");
            base_ot_sender
                .send_random(dbg!(enc.base_ot_count()), rng, sender, receiver)
                .await
                .expect("Failed to generate base ots")
        };
        Self::new_with_silent_base_ots(silent_base_ots, enc, num_threads)
    }

    /// Create a new Sender with the provided base OTs.
    ///
    /// # Panics
    /// If the number of provided base OTs is unequal to
    /// [`QuasiCyclicConf::base_ot_count()`](`QuasiCyclicConf::base_ot_count()`).
    pub fn new_with_silent_base_ots(
        mut silent_base_ots: Vec<[Block; 2]>,
        encoder: Encoder,
        num_threads: usize,
    ) -> Self {
        assert_eq!(
            encoder.base_ot_count(),
            silent_base_ots.len(),
            "Wrong number of silent base ots"
        );
        let gap_ots = silent_base_ots.split_off(encoder.base_ot_count() - encoder.gap());
        let gen = pprf::Sender::new(encoder.pprf_conf(), silent_base_ots);
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Unable to initialize Sender threadpool")
            .into();

        Self {
            enc: encoder,
            gap_ots,
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
        let thread_pool = self.thread_pool.clone();
        let B = self
            .correlated_silent_send(delta, rng, sender, receiver)
            .await;

        thread_pool
            .spawn_install_compute(move || Sender::hash(delta, &B))
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
        let pprf_format = self.enc.pprf_format();
        let rT = {
            let (sender, _receiver) = pprf_channel(&mut sender, &mut receiver)
                .await
                .expect("Establishing pprf channel");
            self.gen
                .expand(
                    sender,
                    delta,
                    pprf_format,
                    rng,
                    Some(Arc::clone(&self.thread_pool)),
                )
                .await
        };
        let rT = self.derandomize_gap(rT, delta, &mut sender).await;
        self.thread_pool
            .clone()
            .spawn_install_compute(move || self.enc.dual_encode(rT))
            .await
    }

    async fn derandomize_gap(
        &self,
        rT: Array2<Block>,
        delta: Block,
        sender: &mut mpc_channel::Sender<Msg>,
    ) -> ArrayOrVec {
        if self.gap_ots.is_empty() {
            assert!(matches!(self.enc, Encoder::QuasiCyclic(_)));
            return ArrayOrVec::Array(rT);
        }
        let mut rT = rT.into_raw_vec();
        let gap_vals: Vec<Block> = self
            .gap_ots
            .iter()
            .map(|&[gap_ot0, gap_ot1]| {
                rT.push(gap_ot0);
                let v = gap_ot0 ^ delta;
                let gap_val = Block::zero();
                Aes128::new(&gap_ot1.into()).encrypt_block(&mut gap_val.into());
                gap_val ^ v
            })
            .collect();
        sender.send(Msg::GapValues(gap_vals)).await.unwrap();
        ArrayOrVec::Vec(rT)
    }

    fn hash(delta: Block, B: &[Block]) -> Vec<[Block; 2]> {
        let mask = Block::all_ones() ^ Block::one();
        let d = delta & mask;
        let mut messages: Vec<_> = B
            .iter()
            .map(|block| {
                let masked = *block & mask;
                [masked, masked ^ d]
            })
            .collect();
        FIXED_KEY_HASH.cr_hash_slice_mut(bytemuck::cast_slice_mut(&mut messages));
        messages
    }
}

impl Receiver {
    #[tracing::instrument(skip(rng, sender, receiver))]
    pub async fn new<RNG: RngCore + CryptoRng + Send>(
        rng: &mut RNG,
        num_ots: usize,
        mult_type: MultType,
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
            mult_type,
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
        mult_type: MultType,
        num_threads: usize,
        sender: &mut mpc_channel::Sender<Msg<BaseOT::Msg>>,
        receiver: &mut mpc_channel::Receiver<Msg<BaseOT::Msg>>,
    ) -> Self
    where
        BaseOT: BaseROTReceiver,
        BaseOT::Msg: RemoteSend + Debug,
        RNG: RngCore + CryptoRng + Send,
    {
        let enc = Encoder::configure(num_ots, SECURITY_PARAM, mult_type);
        let silent_choice_bits = Self::sample_base_choice_bits(&enc, rng);
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
        Self::new_with_silent_base_ots(silent_base_ots, silent_choice_bits, enc, num_threads)
    }

    /// Create a new Receiver with the provided base OTs and choice bits. The
    /// [`ChoiceBits`](`ChoiceBits`) need to be sampled by calling
    /// [`Receiver::sample_base_choice_bits()`](`Receiver::sample_base_choice_bits()`).
    ///
    /// # Panics
    /// If the number of provided base OTs is unequal to
    /// [`QuasiCyclicConf::base_ot_count()`](`QuasiCyclicConf::base_ot_count()`).
    pub fn new_with_silent_base_ots(
        mut silent_base_ots: Vec<Block>,
        mut silent_base_choices: ChoiceBits,
        encoder: Encoder,
        num_threads: usize,
    ) -> Self {
        let gap_ots = silent_base_ots.split_off(encoder.base_ot_count() - encoder.gap());
        let gap_choices = silent_base_choices.take_gap_choices();
        let gen = pprf::Receiver::new(encoder.pprf_conf(), silent_base_ots, silent_base_choices);
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Unable to initialize Sender threadpool")
            .into();
        // TODO: THIS MUST BE INTERLEAVED for Silver
        let mut S = gen.get_points(encoder.pprf_format());
        for (idx, gap_choice) in
            ((encoder.num_partitions() * encoder.size_per())..).zip(gap_choices.iter())
        {
            if *gap_choice {
                S.push(idx)
            }
        }
        Self {
            enc: encoder,
            gap_ots,
            gap_choices,
            S,
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
        let thread_pool = self.thread_pool.clone();
        let (A, _) = self
            .correlated_silent_receive(ChoiceBitPacking::True, sender, receiver)
            .await;

        thread_pool
            .spawn_install_compute(move || Self::hash(A))
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
                .expand(
                    receiver,
                    self.enc.pprf_format(),
                    Some(Arc::clone(&self.thread_pool)),
                )
                .await
        };

        let rT = self.derandomize_gap(rT, &mut receiver).await;

        self.thread_pool
            .clone()
            .spawn_install_compute(move || self.enc.dual_encode2(rT, &self.S, choice_bit_packing))
            .await
    }

    async fn derandomize_gap(
        &self,
        rT: Array2<Block>,
        receiver: &mut mpc_channel::Receiver<Msg>,
    ) -> ArrayOrVec {
        if self.gap_ots.is_empty() {
            assert!(matches!(self.enc, Encoder::QuasiCyclic(_)));
            return ArrayOrVec::Array(rT);
        }

        let Msg::GapValues(gap_vals) = receiver.recv().await.unwrap().unwrap() else {
            panic!("Wrong message. Expected GapValues");
        };

        let mut rT = rT.into_raw_vec();
        let derand_iter = self
            .gap_ots
            .iter()
            .zip(self.gap_choices.iter().by_vals())
            .zip(gap_vals)
            .map(|((&gap_ot, gap_choice), gap_val)| {
                if gap_choice {
                    let t = Block::zero();
                    Aes128::new(&gap_ot.into()).encrypt_block(&mut t.into());
                    t ^ gap_val
                } else {
                    gap_ot
                }
            });
        rT.extend(derand_iter);
        ArrayOrVec::Vec(rT)
    }

    fn hash(mut A: Vec<Block>) -> (Vec<Block>, BitVec) {
        let mask = Block::all_ones() ^ Block::constant::<1>();
        let choices = A
            .iter_mut()
            .map(|block| {
                let choice = block.lsb();
                *block &= mask;
                choice
            })
            .collect();
        FIXED_KEY_HASH.cr_hash_slice_mut(&mut A);
        (A, choices)
    }

    /// Sample the choice bits for the base OTs.
    pub fn sample_base_choice_bits<RNG: RngCore + CryptoRng>(
        encoder: &Encoder,
        rng: &mut RNG,
    ) -> ChoiceBits {
        let mut base_choices = pprf::Receiver::sample_choice_bits(
            encoder.pprf_conf(),
            encoder.N2(),
            encoder.pprf_format(),
            rng,
        );
        base_choices
            .gap
            .extend(rng.sample_iter::<bool, _>(Standard).take(encoder.gap()));
        base_choices
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

impl Encoder {
    fn configure(num_ots: usize, sec_param: usize, mult_type: MultType) -> Self {
        match mult_type {
            MultType::QuasiCyclic { scaler } => Encoder::QuasiCyclic(QuasiCyclicEncoder {
                conf: QuasiCyclicConf::configure(num_ots, scaler, sec_param),
            }),
            MultType::Silver5 => {
                let conf = SilverConf::configure(num_ots, 5, sec_param);
                let enc =
                    libote_sys::SilverEncoder::new(libote_sys::SilverCode::Weight5, conf.N as u64);
                Encoder::Silver(SilverEncoder { enc, conf })
            }
            MultType::Silver11 => {
                let conf = SilverConf::configure(num_ots, 11, sec_param);
                let enc =
                    libote_sys::SilverEncoder::new(libote_sys::SilverCode::Weight11, conf.N as u64);
                Encoder::Silver(SilverEncoder { enc, conf })
            }
        }
    }

    fn dual_encode(&mut self, rT: ArrayOrVec) -> Vec<Block> {
        match (self, rT) {
            (Encoder::QuasiCyclic(enc), ArrayOrVec::Array(rT)) => enc.dual_encode(rT),
            (Encoder::Silver(enc), ArrayOrVec::Vec(mut c)) => {
                {
                    let c = bytemuck::cast_slice_mut(&mut c);
                    enc.enc.dual_encode(c);
                }
                let mut b = bytemuck::allocation::cast_vec(c);
                b.truncate(enc.conf.requested_num_ots);
                b
            }
            _ => panic!("called dual_encode with illegal combination of Encoder and ArrayOrVec"),
        }
    }

    fn dual_encode2(
        &mut self,
        rT: ArrayOrVec,
        S: &[usize],
        choice_bit_packing: ChoiceBitPacking,
    ) -> (Vec<Block>, Option<Vec<u8>>) {
        match (self, rT) {
            (Encoder::QuasiCyclic(enc), ArrayOrVec::Array(rT)) => {
                let mut sb: AlignedVec<u8, U16> = AlignedVec::new();
                let conf = enc.conf;
                let sb_blocks = {
                    assert_eq!(conf.N2 % 8, 0);
                    let n2_bytes = conf.N2 / mem::size_of::<u8>();
                    sb.resize(n2_bytes, 0);

                    let sb_bits: &mut BitSlice<u8, Lsb0> =
                        BitSlice::from_slice_mut(sb.as_mut_slice());
                    for noisy_idx in S {
                        sb_bits.set(*noisy_idx, true);
                    }
                    cast_slice(sb.as_slice())
                };
                enc.dual_encode2(rT, sb_blocks, choice_bit_packing)
            }
            (Encoder::Silver(enc), ArrayOrVec::Vec(mut c)) => {
                if choice_bit_packing.packed() {
                    let mask = Block::one() ^ Block::all_ones();
                    for block in c.iter_mut() {
                        *block = *block & mask;
                    }
                    for noisy_idx in S {
                        c[*noisy_idx] = c[*noisy_idx] | Block::one();
                    }
                    enc.enc.dual_encode(bytemuck::cast_slice_mut(&mut c));
                    c.truncate(enc.conf.requested_num_ots);
                    (c, None)
                } else {
                    let c0 = bytemuck::cast_slice_mut(&mut c);
                    let mut c1 = vec![0_u8; enc.conf.N2];
                    for noisy_idx in S {
                        c1[*noisy_idx] = 1;
                    }
                    enc.enc.dual_encode2(c0, &mut c1);
                    c.truncate(enc.conf.requested_num_ots);
                    (c, Some(c1))
                }
            }
            _ => panic!("called dual_encode with illegal combination of Encoder and ArrayOrVec"),
        }
    }

    fn base_ot_count(&self) -> usize {
        let pprf_conf = self.pprf_conf();
        match self {
            Encoder::QuasiCyclic(_) => pprf_conf.base_ot_count(),
            Encoder::Silver(enc) => pprf_conf.base_ot_count() + enc.conf.gap,
        }
    }

    fn gap(&self) -> usize {
        match self {
            Encoder::QuasiCyclic(_) => 0,
            Encoder::Silver(enc) => enc.conf.gap,
        }
    }

    fn pprf_conf(&self) -> PprfConfig {
        match self {
            Encoder::QuasiCyclic(enc) => enc.conf.into(),
            Encoder::Silver(enc) => enc.conf.into(),
        }
    }

    fn pprf_format(&self) -> PprfOutputFormat {
        match self {
            Encoder::QuasiCyclic(_) => PprfOutputFormat::InterleavedTransposed,
            Encoder::Silver(_) => PprfOutputFormat::Interleaved,
        }
    }

    #[allow(unused)]
    fn requested_num_ots(&self) -> usize {
        match self {
            Encoder::QuasiCyclic(enc) => enc.conf.requested_num_ots,
            Encoder::Silver(enc) => enc.conf.requested_num_ots,
        }
    }

    fn N2(&self) -> usize {
        match self {
            Encoder::QuasiCyclic(enc) => enc.conf.N2,
            Encoder::Silver(enc) => enc.conf.N2,
        }
    }

    fn num_partitions(&self) -> usize {
        match self {
            Encoder::QuasiCyclic(enc) => enc.conf.num_partitions,
            Encoder::Silver(enc) => enc.conf.num_partitions,
        }
    }

    fn size_per(&self) -> usize {
        match self {
            Encoder::QuasiCyclic(enc) => enc.conf.size_per,
            Encoder::Silver(enc) => enc.conf.size_per,
        }
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

fn get_reg_noise_weight(min_dist_ratio: f64, sec_param: usize) -> u64 {
    assert!(min_dist_ratio <= 0.5 && min_dist_ratio > 0.0);
    let d = (1.0 - 2.0 * min_dist_ratio).log2();
    let t = max(128, (-(sec_param as f64) / d) as u64);
    Integer::next_multiple_of(&t, &8)
}

fn sec_level(scale: usize, p: usize, points: usize) -> usize {
    let x1 = ((scale * p) as f64 / p as f64).log2();
    let x2 = ((scale * p) as f64).log2() / 2.0;
    let sec_level = points as f64 * x1 + x2;
    sec_level as usize
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
    use crate::silent_ot::{ChoiceBitPacking, Encoder, MultType, Receiver, Sender, SECURITY_PARAM};

    use crate::silent_ot::pprf::tests::fake_base;
    use crate::silent_ot::pprf::PprfOutputFormat;
    use crate::silent_ot::quasi_cyclic_encode::{
        bit_shift_xor, modp, QuasiCyclicConf, QuasiCyclicEncoder,
    };
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
        dbg!(&send_messages[..10]);
        dbg!(&recv_messages[..10]);
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
        let enc = Encoder::QuasiCyclic(QuasiCyclicEncoder {
            conf: QuasiCyclicConf::configure(num_ots, scaler, SECURITY_PARAM),
        });
        let enc_c = Encoder::QuasiCyclic(QuasiCyclicEncoder {
            conf: QuasiCyclicConf::configure(num_ots, scaler, SECURITY_PARAM),
        });
        let (ch1, ch2) = mpc_channel::in_memory::new_pair(128);
        let mut rng = StdRng::seed_from_u64(42);
        let (sender_base_ots, receiver_base_ots, base_choices) = fake_base(
            enc.pprf_conf(),
            enc.N2(),
            PprfOutputFormat::InterleavedTransposed,
            &mut rng,
        );

        let send = tokio::spawn(async move {
            let sender = Sender::new_with_silent_base_ots(sender_base_ots, enc_c, num_threads);
            sender
                .correlated_silent_send(delta, &mut rng, ch1.0, ch1.1)
                .await
        });
        let receiver =
            Receiver::new_with_silent_base_ots(receiver_base_ots, base_choices, enc, num_threads);
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
        let enc = Encoder::QuasiCyclic(QuasiCyclicEncoder {
            conf: QuasiCyclicConf::configure(num_ots, scaler, SECURITY_PARAM),
        });
        let enc_c = Encoder::QuasiCyclic(QuasiCyclicEncoder {
            conf: QuasiCyclicConf::configure(num_ots, scaler, SECURITY_PARAM),
        });
        let (ch1, ch2) = mpc_channel::in_memory::new_pair(128);
        let mut rng = StdRng::seed_from_u64(42);
        let (sender_base_ots, receiver_base_ots, base_choices) = fake_base(
            enc.pprf_conf(),
            enc.N2(),
            PprfOutputFormat::InterleavedTransposed,
            &mut rng,
        );

        let send = tokio::spawn(async move {
            let sender = Sender::new_with_silent_base_ots(sender_base_ots, enc_c, num_threads);
            sender.random_silent_send(&mut rng, ch1.0, ch1.1).await
        });
        let receiver =
            Receiver::new_with_silent_base_ots(receiver_base_ots, base_choices, enc, num_threads);
        let receive =
            tokio::spawn(async move { receiver.random_silent_receive(ch2.0, ch2.1).await });
        let (r_out, s_out) = futures::future::try_join(receive, send).await.unwrap();
        check_random(&s_out, &r_out.0, &r_out.1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn random_silent_ot_silver() {
        let num_ots = 10000;
        let _num_threads = 2;
        let _enc = Encoder::configure(num_ots, 128, MultType::Silver5);
        let _enc_c = Encoder::configure(num_ots, 128, MultType::Silver5);
        let (mut ch1, mut ch2) = mpc_channel::in_memory::new_pair(128);
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42 * 42);

        let send = tokio::spawn(async move {
            let sender = Sender::new(
                &mut rng1,
                num_ots,
                MultType::Silver5,
                &mut ch1.0,
                &mut ch1.1,
            )
            .await;
            sender.random_silent_send(&mut rng1, ch1.0, ch1.1).await
        });

        let receive = tokio::spawn(async move {
            let receiver = Receiver::new(
                &mut rng2,
                num_ots,
                MultType::Silver5,
                &mut ch2.0,
                &mut ch2.1,
            )
            .await;
            receiver.random_silent_receive(ch2.0, ch2.1).await
        });
        let (r_out, s_out) = futures::future::try_join(receive, send).await.unwrap();
        check_random(&s_out, &r_out.0, &r_out.1);
    }
}
