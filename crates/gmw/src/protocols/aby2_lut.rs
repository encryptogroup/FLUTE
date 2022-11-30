use crate::circuit::base_circuit::{BaseGate, Load};
use crate::circuit::{BaseCircuit, DefaultIdx, GateIdx};
use crate::common::BitVec;
use crate::errors::CircuitError;
use crate::executor::Executor;
use crate::mul_triple::{MTProvider, MulTriples};
use crate::parse::lut_circuit::Wire;
use crate::parse::{aby, lut_circuit};
use crate::protocols::boolean_gmw::BooleanGmw;
use crate::protocols::{
    boolean_gmw, FunctionDependentSetup, Gate, Protocol, SetupStorage, ShareStorage,
};
use crate::share_wrapper::ShareWrapper;
use crate::{bristol, BooleanGate, Circuit, CircuitBuilder, GateId, SubCircuitGate};
use ahash::AHashMap;
use async_trait::async_trait;
use bitvec::order::Lsb0;
use bitvec::view::BitView;
use bitvec::{bitvec, slice, vec};
use itertools::Itertools;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt::Debug;
use std::ops::{BitXor, Not};
use std::path::Path;
use tracing::{info, trace, warn};

pub struct LutAby2 {
    delta_sharing_state: DeltaSharing,
}

#[derive(Clone)]
pub struct DeltaSharing {
    private_rng: ChaChaRng,
    local_joint_rng: ChaChaRng,
    remote_joint_rng: ChaChaRng,
    // TODO ughh
    input_position_share_type_map: HashMap<usize, ShareType>,
}

#[derive(Copy, Clone)]
pub enum ShareType {
    Local,
    Remote,
}

#[derive(Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq, Debug, Default)]
pub struct Share {
    public: bool,
    private: bool,
}

#[derive(Clone, Debug, Default)]
pub struct DeltaShareStorage {
    public: BitVec,
    private: BitVec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Msg {
    Delta { delta: Vec<u8> },
}

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum LutGate {
    Base(BaseGate<Share>),
    Lut {
        output_mask: vec::BitVec<u8, Lsb0>,
        inputs: u8,
    },
    Xor,
    Not,
    Assign,
}

#[derive(Clone)]
/// Contains LutEvalShares in **reverse topological order**.
pub struct SetupData {
    lut_eval_shares: Vec<LutEvalShares>,
}

// TODO: this is currently 12 bytes big and could be reduced by just using BitArrays
#[derive(Clone)]
pub struct LutEvalShares {
    shares: BitVec,
    for_gate: SubCircuitGate<usize>,
}

impl LutAby2 {
    pub fn new(sharing_state: DeltaSharing) -> Self {
        Self {
            delta_sharing_state: sharing_state,
        }
    }
}

pub type LutSetupMsg = boolean_gmw::Msg;

pub struct LutSetupProvider<Mtp> {
    party_id: usize,
    mt_provider: Mtp,
    sender: mpc_channel::Sender<LutSetupMsg>,
    receiver: mpc_channel::Receiver<LutSetupMsg>,
}

impl Protocol for LutAby2 {
    type Msg = Msg;
    type Gate = LutGate;
    type ShareStorage = DeltaShareStorage;
    type SetupStorage = SetupData;

    fn compute_msg(
        &self,
        party_id: usize,
        interactive_gates: impl IntoIterator<Item = (LutGate, Share)>,
        inputs: impl IntoIterator<Item = Share>,
        preprocessing_data: &SetupData,
    ) -> Self::Msg {
        let mut inputs = inputs.into_iter();
        let interactive_gates: Vec<_> = interactive_gates.into_iter().collect();
        let inputs_per_gate: Vec<Vec<_>> = interactive_gates
            .iter()
            .map(|(gate, _)| inputs.by_ref().take(gate.input_size()).collect())
            .collect();
        let mut preprocessing_data = preprocessing_data.clone();
        preprocessing_data.lut_eval_shares.reverse();
        let mut delta: Vec<bool> = interactive_gates
            .into_par_iter()
            .zip(inputs_per_gate)
            .zip(preprocessing_data.lut_eval_shares)
            .map(|(((gate, output), inputs), eval_shares)| {
                gate.compute_delta_share(party_id, inputs.iter().copied(), eval_shares, output)
            })
            .collect();
        delta.shrink_to_fit();
        Msg::Delta {
            delta: BitVec::from_iter(delta).into_vec(),
        }
    }

    fn evaluate_interactive(
        &self,
        _party_id: usize,
        interactive_gates: impl IntoIterator<Item = (LutGate, Share)>,
        Msg::Delta { delta }: Msg,
        Msg::Delta { delta: other_delta }: Msg,
        _preprocessing_data: SetupData,
    ) -> Self::ShareStorage {
        let delta = BitVec::from_vec(delta);
        let other_delta = BitVec::from_vec(other_delta);
        interactive_gates
            .into_iter()
            .zip(delta)
            .zip(other_delta)
            .map(|(((_gate, mut out_share), my_delta), other_delta)| {
                out_share.public = my_delta ^ other_delta;
                out_share
            })
            .collect()
    }

    // TODO I think this needs a &self for access to seeded rng's
    fn setup_gate_outputs<Idx: GateIdx>(
        &mut self,
        _party_id: usize,
        circuit: &Circuit<Self::Gate, Idx>,
    ) -> Vec<Self::ShareStorage> {
        let mut storage: Vec<_> = circuit
            .circuits
            .iter()
            .map(|base_circ| DeltaShareStorage::repeat(Default::default(), base_circ.gate_count()))
            .collect();

        for (gate, sc_gate_id) in circuit.iter() {
            let gate_input_iter = circuit
                .parent_gates(sc_gate_id)
                .map(|parent| storage[parent.circuit_id as usize].get(parent.gate_id.as_usize()));
            let rng = match self
                .delta_sharing_state
                .input_position_share_type_map
                .get(&sc_gate_id.gate_id.as_usize())
            {
                None => &mut self.delta_sharing_state.private_rng,
                Some(ShareType::Local) => {
                    // dbg!(&sc_gate_id);
                    &mut self.delta_sharing_state.private_rng
                }
                Some(ShareType::Remote) => &mut self.delta_sharing_state.remote_joint_rng,
            };
            let output = gate.setup_output_share(gate_input_iter, rng);
            storage[sc_gate_id.circuit_id as usize].set(sc_gate_id.gate_id.as_usize(), output);
        }

        storage
    }
}

impl LutGate {
    /// output_share contains the previously randomly generated private share needed for the
    /// evaluation
    #[tracing::instrument(level = "debug", skip(input_shares, eval_share), ret)]
    // TODO reduce the amount of allocations in this function (before this I should probably
    //  benchmark if the allocs are indeed the problem)
    fn compute_delta_share(
        &self,
        party_id: usize,
        input_shares: impl Iterator<Item = Share>,
        eval_share: LutEvalShares,
        output_share: Share,
    ) -> bool {
        assert!(matches!(party_id, 0 | 1));
        match self {
            LutGate::Lut {
                output_mask,
                inputs,
            } => {
                let (input_masks, input_lambdas): (BitVec, BitVec) = input_shares
                    .map(|share| (share.public, share.private))
                    .unzip();
                trace!(?input_masks, ?input_lambdas);
                let x_vecs = expand(*inputs, output_mask, &input_masks);
                let mut lut_eval_shares = input_lambdas;
                trace!(for_gate = ?eval_share.for_gate);
                lut_eval_shares.extend(eval_share.shares);
                lut_eval_shares.reverse();

                let mut pset: Vec<_> = x_vecs.iter().powerset().collect();
                let whole_set = pset.pop().unwrap();
                // We want the powersets in the reverse order so we can omit the
                // delta share for the empty set case when output_mask.ones() is even
                pset.reverse();

                // assert_eq!(pset.len(), lut_eval_shares.len());

                let reduced_powerset = pset
                    .into_iter()
                    .map(|x_q| {
                        let and_x_q = bitvec_and_fold(output_mask.len(), &x_q);
                        match lut_eval_shares.pop() {
                            Some(eval_share) => {
                                trace!(eval_share);
                                and_x_q
                                    .into_iter()
                                    .map(|bit| bit & eval_share)
                                    .reduce(BitXor::bitxor)
                                     .expect("empty input")
                            },
                            None => {
                                assert!(
                                    x_q.is_empty(),
                                    "Missing eval_share must be the case for empty x_q set, Actual: {x_q:?}"
                                );
                                false
                            }
                        }
                    })
                    .reduce(BitXor::bitxor)
                    .expect("empty input");

                if party_id == 1 {
                    let whole_set_mask = bitvec_and_fold(whole_set[0].len(), &whole_set)
                        .into_iter()
                        .reduce(BitXor::bitxor)
                        .unwrap();
                    whole_set_mask ^ reduced_powerset ^ output_share.private
                } else {
                    reduced_powerset ^ output_share.private
                }
            }
            not_lut => panic!("Called compute_delta_share on {not_lut:?}"),
        }
    }

    fn setup_output_share(
        &self,
        mut inputs: impl Iterator<Item = Share>,
        mut rng: impl Rng,
    ) -> Share {
        match self {
            LutGate::Base(base_gate) => match base_gate {
                BaseGate::Input => {
                    // TODO this needs to randomly generate the private part of the share
                    //  however, there is a problem. This private part needs to match the private
                    //  part of the input which is supplied to Executor::execute
                    //  one option to ensure this would be to use two PRNGs seeded with the same
                    //  seed for this method and for the Sharing of the input
                    //  Or maybe the setup gate outputs of the input gates can be passed to
                    //  the share() method?
                    Share {
                        public: Default::default(),
                        private: rng.gen(),
                    }
                }
                BaseGate::Output | BaseGate::SubCircuitInput | BaseGate::SubCircuitOutput => {
                    inputs.next().expect("Empty input")
                }
                BaseGate::Constant(share) => *share,
            },
            LutGate::Lut { .. } => Share {
                public: Default::default(),
                private: thread_rng().gen(),
            },
            LutGate::Xor => {
                let a = inputs.next().expect("Empty input");
                let b = inputs.next().expect("Empty input");
                Share {
                    public: Default::default(),
                    private: a.private ^ b.private,
                }
            }
            LutGate::Not => !inputs.next().expect("Empty input"),
            LutGate::Assign => inputs.next().expect("Empty input"),
        }
    }

    fn setup_data_circ<'a>(
        &self,
        input_shares: impl Iterator<Item = &'a ShareWrapper>,
        setup_sub_circ_cache: &mut AHashMap<Vec<ShareWrapper>, ShareWrapper>,
    ) -> Vec<Option<ShareWrapper>> {
        // TODO return SmallVec here?
        match self {
            LutGate::Lut {
                output_mask: _,
                inputs,
            } => {
                let inputs = *inputs as usize;
                // skip the empty and single elem sets
                let inputs_pset = input_shares
                    .take(inputs)
                    .cloned()
                    .powerset()
                    .skip(inputs + 1);

                inputs_pset
                    .map(|set| match setup_sub_circ_cache.get(&set) {
                        None => match &set[..] {
                            [] => unreachable!("Empty set is filtered"),
                            [a, b] => {
                                let sh = a.clone() & b;
                                setup_sub_circ_cache.insert(set, sh.clone());
                                sh
                            }
                            [processed_subset @ .., last] => {
                                assert!(processed_subset.len() >= 2, "Smaller sets are filtered");
                                // We know we generated the mul triple for the smaller subset
                                let subset_out = setup_sub_circ_cache
                                    .get(processed_subset)
                                    .expect("Subset not present in cache");
                                let sh = last.clone() & subset_out;
                                setup_sub_circ_cache.insert(set, sh.clone());
                                sh
                            }
                        },
                        Some(processed_set) => processed_set.clone(),
                    })
                    .map(Some)
                    .collect()
            }
            non_interactive => {
                assert!(non_interactive.is_non_interactive());
                panic!("Called setup_data_circ on non_interactive gate")
            }
        }
    }

    fn and() -> Self {
        LutGate::Lut {
            // TODO is the order of bits correct?
            output_mask: bitvec![u8, Lsb0; 0,0,0,1],
            inputs: 2,
        }
    }
}

impl Gate for LutGate {
    type Share = Share;

    fn is_interactive(&self) -> bool {
        match self {
            LutGate::Lut { .. } => true,
            LutGate::Base(_) | LutGate::Xor | LutGate::Not | LutGate::Assign => false,
        }
    }

    fn input_size(&self) -> usize {
        match self {
            LutGate::Base(base_gate) => base_gate.input_size(),
            LutGate::Lut { inputs, .. } => *inputs as usize,
            LutGate::Xor => 2,
            LutGate::Not => 1,
            LutGate::Assign => 1,
        }
    }

    fn as_base_gate(&self) -> Option<&BaseGate<Self::Share>> {
        match self {
            LutGate::Base(base_gate) => Some(base_gate),
            _ => None,
        }
    }

    fn evaluate_non_interactive(
        &self,
        party_id: usize,
        inputs: impl IntoIterator<Item = Self::Share>,
    ) -> Self::Share {
        let mut input = inputs.into_iter();
        match self {
            LutGate::Base(base) => base.evaluate_non_interactive(party_id, input.by_ref()),
            LutGate::Xor => {
                let a = input.next().expect("Empty input");
                let b = input.next().expect("Empty input");
                // TODO change this
                a.xor(b)
            }
            LutGate::Assign => input.next().expect("Empty input"),
            LutGate::Not => !input.next().expect("Empty input"),
            interactive @ LutGate::Lut { .. } => panic!(
                "Called evaluate_non_interactive on interactive gate \"{:?}\"",
                interactive
            ),
        }
    }
}

impl From<BaseGate<Share>> for LutGate {
    fn from(base_gate: BaseGate<Share>) -> Self {
        Self::Base(base_gate)
    }
}

impl TryFrom<&aby::Gate> for LutGate {
    type Error = ();

    fn try_from(gate: &aby::Gate) -> Result<Self, Self::Error> {
        match gate {
            aby::Gate::And(data) => {
                assert_eq!(
                    data.input_wires.len(),
                    2,
                    "Only And gates with two inputs are supported"
                );
                Ok(LutGate::and())
            }
            aby::Gate::Xor(_) => Ok(LutGate::Xor),
            aby::Gate::Mux(_) => Err(()),
        }
    }
}

impl From<&bristol::Gate> for LutGate {
    fn from(gate: &bristol::Gate) -> Self {
        match gate {
            bristol::Gate::And(data) => {
                assert_eq!(
                    data.input_wires.len(),
                    2,
                    "Only And gates with two inputs are supported"
                );
                LutGate::and()
            }
            bristol::Gate::Xor(data) => {
                assert_eq!(
                    data.input_wires.len(),
                    2,
                    "Only Xor gates with two inputs are supported"
                );
                LutGate::Xor
            }
            bristol::Gate::Inv(data) => {
                assert_eq!(
                    data.input_wires.len(),
                    1,
                    "Only Inv gates with one input are supported"
                );
                LutGate::Not
            }
        }
    }
}

impl ShareStorage<Share> for DeltaShareStorage {
    fn len(&self) -> usize {
        debug_assert_eq!(self.private.len(), self.public.len());
        self.private.len()
    }

    fn repeat(val: Share, len: usize) -> Self {
        Self {
            private: BitVec::repeat(val.private, len),
            public: BitVec::repeat(val.public, len),
        }
    }

    fn set(&mut self, idx: usize, val: Share) {
        self.public.set(idx, val.public);
        self.private.set(idx, val.private);
    }

    fn get(&self, idx: usize) -> Share {
        Share {
            public: self.public[idx],
            private: self.private[idx],
        }
    }
}

pub struct ShareIter {
    public: <BitVec as IntoIterator>::IntoIter,
    private: <BitVec as IntoIterator>::IntoIter,
}

impl IntoIterator for DeltaShareStorage {
    type Item = Share;
    type IntoIter = ShareIter;

    fn into_iter(self) -> Self::IntoIter {
        ShareIter {
            public: self.public.into_iter(),
            private: self.private.into_iter(),
        }
    }
}

impl Iterator for ShareIter {
    type Item = Share;

    fn next(&mut self) -> Option<Self::Item> {
        let public = self.public.next()?;
        let private = self.private.next()?;
        Some(Share { public, private })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.public.size_hint()
    }
}

impl ExactSizeIterator for ShareIter {}

impl FromIterator<Share> for DeltaShareStorage {
    fn from_iter<T: IntoIterator<Item = Share>>(iter: T) -> Self {
        let (public, private) = iter
            .into_iter()
            .map(|share| (share.public, share.private))
            .unzip();
        Self { public, private }
    }
}

impl Extend<Share> for DeltaShareStorage {
    fn extend<T: IntoIterator<Item = Share>>(&mut self, iter: T) {
        for share in iter {
            self.private.push(share.private);
            self.public.push(share.public);
        }
    }
}

impl SetupData {
    // LutEvalShares in **topological order**.
    pub fn from_raw(mut lut_eval_shares: Vec<LutEvalShares>) -> Self {
        lut_eval_shares.reverse();
        Self { lut_eval_shares }
    }

    pub fn len(&self) -> usize {
        self.lut_eval_shares.len()
    }
}

impl SetupStorage for SetupData {
    fn split_off_last(&mut self, count: usize) -> Self {
        Self {
            lut_eval_shares: self.lut_eval_shares.split_off(self.len() - count),
        }
    }
}

impl Share {
    pub fn new(private: bool, public: bool) -> Self {
        Self { public, private }
    }

    fn xor(&self, other: Share) -> Share {
        Share {
            public: self.public ^ other.public,
            private: self.private ^ other.private,
        }
    }

    pub fn get_public(&self) -> bool {
        self.public
    }

    pub fn get_private(&self) -> bool {
        self.private
    }
}

impl Not for Share {
    type Output = Share;

    fn not(self) -> Self::Output {
        Self {
            public: !self.public,
            private: self.private,
        }
    }
}

impl DeltaSharing {
    pub fn new(
        priv_seed: [u8; 32],
        local_joint_seed: [u8; 32],
        remote_joint_seed: [u8; 32],
        input_position_share_type_map: HashMap<usize, ShareType>,
    ) -> Self {
        Self {
            private_rng: ChaChaRng::from_seed(priv_seed),
            local_joint_rng: ChaChaRng::from_seed(local_joint_seed),
            remote_joint_rng: ChaChaRng::from_seed(remote_joint_seed),
            input_position_share_type_map,
        }
    }

    /// # Warning - Insercure
    /// Insecurely initialize DeltaSharing RNGs with default value. No input_position_share_type_map
    /// is needed when all the RNGs are the same.
    pub fn insecure_default() -> Self {
        Self {
            private_rng: ChaChaRng::seed_from_u64(0),
            local_joint_rng: ChaChaRng::seed_from_u64(0),
            remote_joint_rng: ChaChaRng::seed_from_u64(0),
            input_position_share_type_map: HashMap::new(),
        }
    }

    pub fn share(&mut self, input: BitVec) -> (DeltaShareStorage, BitVec) {
        input
            .into_iter()
            .map(|bit| {
                let my_delta = self.private_rng.gen();
                let other_delta: bool = self.local_joint_rng.gen();
                let plain_delta = bit ^ my_delta ^ other_delta;
                let my_share = Share::new(my_delta, plain_delta);
                (my_share, plain_delta)
            })
            .unzip()
    }

    pub fn plain_delta_to_share(&mut self, plain_deltas: BitVec) -> DeltaShareStorage {
        plain_deltas
            .into_iter()
            .map(|plain_delta| Share::new(self.remote_joint_rng.gen(), plain_delta))
            .collect()
    }

    #[cfg(test)]
    pub fn test_share(input: BitVec) -> (DeltaShareStorage, DeltaShareStorage) {
        let mut sharing0 = Self::insecure_default();
        let mut sharing1 = Self::insecure_default();
        let (delta_storage0, plain_delta1) = sharing0.share(input);
        let delta_storage1 = sharing1.plain_delta_to_share(plain_delta1);
        (delta_storage0, delta_storage1)
    }
}

impl<Mtp> LutSetupProvider<Mtp> {
    pub fn new(
        party_id: usize,
        mt_provider: Mtp,
        sender: mpc_channel::Sender<LutSetupMsg>,
        receiver: mpc_channel::Receiver<LutSetupMsg>,
    ) -> Self {
        Self {
            party_id,
            mt_provider,
            sender,
            receiver,
        }
    }
}

#[async_trait]
impl<MtpErr, Mtp> FunctionDependentSetup<DeltaShareStorage, LutGate, usize>
    for LutSetupProvider<Mtp>
where
    MtpErr: Debug,
    Mtp: MTProvider<Output = MulTriples, Error = MtpErr> + Send,
{
    type Output = SetupData;
    type Error = Infallible;

    async fn setup(
        &mut self,
        shares: &[DeltaShareStorage],
        circuit: &Circuit<LutGate, usize>,
    ) -> Result<Self::Output, Self::Error> {
        let circ_builder: CircuitBuilder<BooleanGate> = CircuitBuilder::new();
        let old = circ_builder.install();
        let total_inputs = circuit
            .interactive_iter()
            .map(|(gate, _)| 2_usize.pow(gate.input_size() as u32))
            .sum();

        let mut circ_inputs = BitVec::with_capacity(total_inputs);
        // Block is needed as otherwise !Send types are held over .await
        let setup_outputs: Vec<Vec<_>> = {
            let mut input_sw_map: AHashMap<_, ShareWrapper> = AHashMap::with_capacity(total_inputs);
            let mut setup_outputs = Vec::with_capacity(circuit.interactive_count());
            let mut setup_sub_circ_cache = AHashMap::with_capacity(total_inputs);
            for (gate, gate_id) in circuit.interactive_iter() {
                let mut gate_input_shares = vec![];
                circuit
                    .parent_gates(gate_id)
                    .for_each(|parent| match input_sw_map.entry(parent) {
                        Entry::Vacant(vacant) => {
                            let sh = ShareWrapper::input(0);
                            gate_input_shares.push(sh.clone());
                            circ_inputs.push(
                                shares[parent.circuit_id as usize]
                                    .get(parent.gate_id.as_usize())
                                    .get_private(),
                            );
                            vacant.insert(sh);
                        }
                        Entry::Occupied(occupied) => {
                            gate_input_shares.push(occupied.get().clone());
                        }
                    });
                gate_input_shares.sort();

                let t = gate.setup_data_circ(gate_input_shares.iter(), &mut setup_sub_circ_cache);
                setup_outputs.push(t);
            }
            setup_outputs
                .into_iter()
                .map(|v| {
                    v.into_iter()
                        .map(|opt_sh| opt_sh.map(|sh| sh.output()))
                        .collect()
                })
                .collect()
        };

        let setup_data_circ: Circuit<BooleanGate> = CircuitBuilder::global_into_circuit();
        old.install();
        let mut executor: Executor<BooleanGmw, DefaultIdx> =
            Executor::new(&setup_data_circ, self.party_id, &mut self.mt_provider)
                .await
                .expect("Executor::new in LutSetupProvider");
        executor
            .execute(circ_inputs, &mut self.sender, &mut self.receiver)
            .await
            .unwrap();
        let executor_gate_outputs = &executor.gate_outputs()[0]; // not using sc's
        let lut_eval_shares = circuit
            .interactive_iter()
            .zip(setup_outputs)
            .map(|((gate, gate_id), setup_out)| match gate {
                LutGate::Lut { .. } => {
                    let shares = setup_out
                        .into_iter()
                        .map(|opt_out_id| {
                            let out_id =
                                opt_out_id.expect("Unneeded shares optimization not impled");
                            executor_gate_outputs.get(out_id.as_usize())
                        })
                        .collect();
                    LutEvalShares {
                        shares,
                        for_gate: gate_id,
                    }
                }
                _ => unreachable!("Lut are the only interactive gates"),
            })
            .collect();
        Ok(SetupData::from_raw(lut_eval_shares))

        // TODO this is an unfinished attempt at improving the setup phase and using sc's
        // let now = Instant::now();
        // let circ_builder: CircuitBuilder<BooleanGate> = CircuitBuilder::new();
        // let old = circ_builder.install();
        // let total_inputs = circuit
        //     .interactive_iter()
        //     .map(|(gate, _)| 2_usize.pow(gate.input_size() as u32))
        //     .sum();
        //
        // let mut circ_inputs = BitVec::with_capacity(total_inputs);
        // // Block is needed as otherwise !Send types are held over .await
        // let setup_outputs: Vec<Vec<_>> = {
        //     let base_circ = circuit
        //         .circuits
        //         .get(1)
        //         .expect("One sc is needed for this optimization");
        //     for other_sc in &circuit.circuits[1..] {
        //         assert!(
        //             Arc::ptr_eq(base_circ, other_sc),
        //             "This optimization only works if all sc are equal"
        //         );
        //     }
        //
        //     CircuitBuilder::<BooleanGate>::push_global_circuit(SharedCircuit::default());
        //     let total_inputs_for_circ: usize = base_circ
        //         .interactive_iter()
        //         .map(|(gate, _)| 2_usize.pow(gate.input_size() as u32))
        //         .sum();
        //     trace!(total_inputs_for_circ);
        //     let mut input_set = AHashMap::with_capacity(base_circ.interactive_count() * 4);
        //     let mut base_setup_outputs = Vec::with_capacity(base_circ.interactive_count());
        //     let mut setup_sub_circ_cache = AHashMap::with_capacity(total_inputs_for_circ);
        //     // parent ids of interactive gates in base_circ in iteration oder
        //     let mut parent_gate_ids = Vec::with_capacity(base_circ.interactive_count() * 4);
        //     let mut input_sw = Vec::with_capacity(base_circ.interactive_count() * 4);
        //     for (_, gate_id) in base_circ.interactive_iter() {
        //         base_circ
        //             .parent_gates(gate_id)
        //             .for_each(|parent| match input_set.entry(parent) {
        //                 Entry::Vacant(vacant) => {
        //                     parent_gate_ids.push(parent);
        //                     let sh = ShareWrapper::<BooleanGmw>::sub_circuit_input(
        //                         1,
        //                         BaseGate::SubCircuitInput.into(),
        //                     );
        //                     input_sw.push(sh.clone());
        //                     vacant.insert(sh);
        //                 }
        //                 Entry::Occupied(occupied) => {
        //                     input_sw.push(occupied.get().clone());
        //                 }
        //             });
        //     }
        //     let mut inputs_iter = input_sw.iter();
        //     for (gate, gate_id) in base_circ.interactive_iter() {
        //         base_circ.parent_gates(gate_id).for_each(|parent| {
        //             let t = gate.setup_data_circ(inputs_iter.by_ref(), &mut setup_sub_circ_cache);
        //             base_setup_outputs.push(t);
        //         });
        //     }
        //
        //     let setup_circ = CircuitBuilder::<BooleanGate, DefaultIdx>::get_global_circuit(1)
        //         .expect("No global setup circ");
        //     // skip the main circuit
        //     let flat_base_setup_outputs: Vec<_> =
        //         base_setup_outputs.iter().flatten().cloned().collect();
        //     let setup_outputs: Vec<Vec<Option<GateId>>> = (1..circuit.circuits.len())
        //         .map(|i| {
        //             let dupl_id = if i > 1 {
        //                 CircuitBuilder::push_global_circuit(setup_circ.clone())
        //             } else {
        //                 i as CircuitId
        //             };
        //             let flat_connected_to_main =
        //                 flat_base_setup_outputs.clone().connect_to_main(dupl_id);
        //             let mut idx = 0;
        //             let mut curr_grp_start_idx = 0;
        //             let mut base_setup_out_iter = base_setup_outputs.iter();
        //             let mut curr_grp_size = base_setup_out_iter.next().expect("no groups").len();
        //             // regroup the sharewrappers in main
        //             flat_connected_to_main
        //                 .into_iter()
        //                 .map(|opt| opt.map(ShareWrapper::output))
        //                 .group_by(move |_| {
        //                     if idx > curr_grp_start_idx + curr_grp_size {
        //                         curr_grp_start_idx += curr_grp_size;
        //                         curr_grp_size =
        //                             base_setup_out_iter.next().expect("no groups").len();
        //                     }
        //                     idx += 1;
        //                     curr_grp_start_idx
        //                 })
        //                 .into_iter()
        //                 .map(|(_, grouped)| grouped.collect())
        //                 .collect::<Vec<_>>()
        //         })
        //         .flatten()
        //         .collect();
        //
        //     // setup the main circuit inputs and connect main to sub circuits
        //     let mut main_inputs = vec![];
        //     for sc_id in 1..circuit.circuits.len() {
        //         main_inputs.clear();
        //         for parent_id in &parent_gate_ids {
        //             main_inputs.push(ShareWrapper::<BooleanGmw, DefaultIdx>::input(0));
        //             circ_inputs.push(shares[sc_id].get(parent_id.as_usize()).get_private());
        //         }
        //         CircuitBuilder::with_global(|builder| {
        //             builder.connect_sub_circuit(&main_inputs, sc_id as CircuitId);
        //         })
        //     }
        //
        //     setup_outputs
        //     // base_setup_outputs
        //     //     .into_iter()
        //     //     .map(|v| {
        //     //         v.into_iter()
        //     //             .map(|opt_sh| opt_sh.map(|sh| ))
        //     //             .collect()
        //     //     })
        //     //     .collect()
        // };
        //
        // // TODO input gates for main circuit must be created and correctly connected to sc's
        //
        // let setup_data_circ: Circuit<BooleanGate> = CircuitBuilder::global_into_circuit();
        // info!(construction_time_s = now.elapsed().as_secs_f64());
        // old.install();
        // let mut executor: Executor<BooleanGmw, DefaultIdx> =
        //     Executor::new(&setup_data_circ, self.party_id, &mut self.mt_provider)
        //         .await
        //         .expect("Executor::new in LutSetupProvider");
        // executor
        //     .execute(circ_inputs, &mut self.sender, &mut self.receiver)
        //     .await
        //     .unwrap();
        // let executor_gate_outputs = &executor.gate_outputs()[0]; // not using sc's
        // let lut_eval_shares = circuit
        //     .interactive_iter()
        //     .zip(setup_outputs)
        //     .map(|((gate, gate_id), setup_out)| match gate {
        //         LutGate::Lut { .. } => {
        //             let shares = setup_out
        //                 .into_iter()
        //                 .map(|opt_out_id| {
        //                     let out_id =
        //                         opt_out_id.expect("Unneeded shares optimization not impled");
        //                     executor_gate_outputs.get(out_id.as_usize())
        //                 })
        //                 .collect();
        //             LutEvalShares {
        //                 shares,
        //                 for_gate: gate_id,
        //             }
        //         }
        //         _ => unreachable!("Lut are the only interactive gates"),
        //     })
        //     .collect();
        // Ok(SetupData::from_raw(lut_eval_shares))
    }
}

fn expand(
    input_size: u8,
    lut_output: &slice::BitSlice<u8, Lsb0>,
    input: &slice::BitSlice<u8>,
) -> Vec<BitVec<u64>> {
    let lut_set_bits = lut_output.count_ones();
    let mut out_vecs = vec![BitVec::with_capacity(lut_set_bits); input_size as usize];
    for (i, (x_i, m_i)) in out_vecs.iter_mut().zip(input).enumerate() {
        for (j, out_bit) in lut_output.iter().enumerate() {
            if !*out_bit {
                continue;
            }
            let truth_table_bit = j.view_bits::<Lsb0>()[input_size as usize - i - 1];
            x_i.push(!*m_i ^ truth_table_bit);
        }
    }
    out_vecs
}

impl BaseCircuit<LutGate, usize> {
    #[tracing::instrument(skip(lut_circuit))]
    pub fn from_lut_circuit(
        lut_circuit: &lut_circuit::Circuit,
        load: Load,
    ) -> Result<Self, CircuitError> {
        info!(
            inputs = lut_circuit.inputs.len(),
            outputs = lut_circuit.outputs.len(),
            gates = lut_circuit.gates.len(),
            "Converting lut_circuit"
        );
        let mut circuit = Self::with_capacity(lut_circuit.gates.len(), 3 * lut_circuit.gates.len());

        let (input_gate, output_gate) = match load {
            Load::Circuit => (BaseGate::Input, BaseGate::Output),
            Load::SubCircuit => (BaseGate::SubCircuitInput, BaseGate::SubCircuitOutput),
        };

        let mut wire_mapping: HashMap<_, _> = lut_circuit
            .inputs
            .iter()
            .cloned()
            .map(|input| {
                (
                    lut_circuit::Wire::Input(input),
                    circuit.add_gate(input_gate.into()),
                )
            })
            .collect();

        for gate in lut_circuit.gates.clone() {
            match gate {
                lut_circuit::Gate::Lut(lut_circuit::Lut {
                    input_wires,
                    masked_luts,
                }) => {
                    for masked_lut in masked_luts {
                        let from_wires: Vec<_> = masked_lut
                            .wire_mask
                            .mask()
                            .iter()
                            .zip(&input_wires)
                            .filter_map(|(sel_bit, inp)| {
                                if *sel_bit {
                                    Some(
                                        *wire_mapping.get(inp).unwrap_or_else(|| panic!("{inp:?}")),
                                    )
                                } else {
                                    None
                                }
                            })
                            .rev()
                            .collect();
                        let out_id = circuit.add_wired_gate(
                            LutGate::Lut {
                                output_mask: masked_lut.output.unexpanded,
                                inputs: from_wires.len().try_into().unwrap(),
                            },
                            &from_wires,
                        );
                        wire_mapping.insert(masked_lut.out_wire, out_id);
                    }
                }
                lut_circuit::Gate::Xor(lut_circuit::Xor {
                    input: [a, b],
                    output,
                }) => {
                    let from_wires = [wire_mapping[&a], wire_mapping[&b]];
                    let out_id = circuit.add_wired_gate(LutGate::Xor, &from_wires);
                    wire_mapping.insert(output, out_id);
                }
                lut_circuit::Gate::Xnor(lut_circuit::Xnor {
                    input: [a, b],
                    output,
                }) => {
                    // Xnor is just  a !(a ^ b)
                    let from_wires = [wire_mapping[&a], wire_mapping[&b]];
                    let xor_id = circuit.add_wired_gate(LutGate::Xor, &from_wires);
                    let out_id = circuit.add_wired_gate(LutGate::Not, &[xor_id]);
                    wire_mapping.insert(output, out_id);
                }
                lut_circuit::Gate::Not(lut_circuit::Not { input, output }) => {
                    let from_wires = [wire_mapping[&input]];
                    let out_id = circuit.add_wired_gate(LutGate::Not, &from_wires);
                    wire_mapping.insert(output, out_id);
                }
                lut_circuit::Gate::Assign(lut_circuit::Assign::Constant { constant, output }) => {
                    let out_id = circuit.add_gate(LutGate::Base(BaseGate::Constant(Share::new(
                        false, constant,
                    ))));
                    wire_mapping.insert(output, out_id);
                }
                lut_circuit::Gate::Assign(lut_circuit::Assign::Wire { input, output }) => {
                    let from_wires = [wire_mapping[&input]];
                    let out_id = circuit.add_wired_gate(LutGate::Assign, &from_wires);
                    wire_mapping.insert(output, out_id);
                }
            }
        }

        for output in lut_circuit.outputs.clone() {
            match wire_mapping.get(&Wire::Output(output.clone())) {
                Some(mapped) => {
                    circuit.add_wired_gate(output_gate.into(), &[*mapped]);
                }
                None => {
                    warn!(
                        ?output,
                        "Output wire from header is not used in circuit. \
                        Output size of converted circuit will be smaller."
                    )
                }
            }
        }

        Ok(circuit)
    }

    #[tracing::instrument]
    pub fn load_lut_circuit(path: &Path, load: Load) -> Result<Self, CircuitError> {
        let lut_circ = lut_circuit::Circuit::load(path)?;
        Self::from_lut_circuit(&lut_circ, load)
    }
}

// TODO it's probably fine to use u32 for bristol circuits and this will reduce their in-memory size
impl BaseCircuit<LutGate, usize> {
    #[tracing::instrument(skip(aby_circ))]
    pub fn from_aby(aby_circ: aby::Circuit, load: Load) -> Result<Self, CircuitError> {
        info!(
            "Converting bristol circuit with header: {:?}",
            aby_circ.header
        );
        let mut circuit = Self::with_capacity(aby_circ.gates.len(), aby_circ.gates.len());
        let mut wire_mapping: HashMap<i64, GateId<usize>> =
            HashMap::with_capacity(aby_circ.gates.len());

        let (input_gate, output_gate) = match load {
            Load::Circuit => (BaseGate::Input, BaseGate::Output),
            Load::SubCircuit => (BaseGate::SubCircuitInput, BaseGate::SubCircuitOutput),
        };

        let mut input_wires = aby_circ.header.input_wires_server;
        input_wires.extend(aby_circ.header.input_wires_client);
        for inp_wire in input_wires {
            wire_mapping.insert(inp_wire, circuit.add_gate(input_gate.into()));
        }
        for (const_val, const_wire) in aby_circ.header.constant_wires {
            wire_mapping.insert(
                const_wire,
                circuit.add_gate(BaseGate::Constant(Share::new(false, const_val)).into()),
            );
        }

        for gate in &aby_circ.gates {
            trace!(?gate, "converting gate");
            let gate_data = gate.get_data();
            match gate {
                aby::Gate::And(_) | aby::Gate::Xor(_) => {
                    let lut_gate = gate.try_into().expect("Not a mux");
                    let gate_id = circuit.add_gate(lut_gate);
                    wire_mapping.insert(gate_data.output_wire, gate_id);
                    // TODO in which order do the gates need to be connected?
                    for inp_wire in &gate_data.input_wires {
                        circuit.add_wire(wire_mapping[inp_wire], gate_id);
                    }
                }
                aby::Gate::Mux(_) => {
                    assert_eq!(
                        gate_data.input_wires.len(),
                        3,
                        "Only Mux gates with 3 inputs are supported"
                    );

                    // Construct MUX a b s => a ^ s(a ^ b)

                    let inp_xor = circuit.add_wired_gate(
                        LutGate::Xor,
                        &[
                            wire_mapping[&gate_data.input_wires[0]],
                            wire_mapping[&gate_data.input_wires[1]],
                        ],
                    );
                    let and_sel_bit = circuit.add_wired_gate(
                        LutGate::and(),
                        &[wire_mapping[&gate_data.input_wires[2]], inp_xor],
                    );
                    let gate_id = circuit.add_wired_gate(
                        LutGate::Xor,
                        &[wire_mapping[&gate_data.input_wires[0]], and_sel_bit],
                    );
                    wire_mapping.insert(gate_data.output_wire, gate_id);
                }
            }
        }

        for out_wire in aby_circ.header.output_wires {
            trace!(out_wire, "mapping out_wire");
            circuit.add_wired_gate(output_gate.into(), &[wire_mapping[&out_wire]]);
        }
        Ok(circuit)
    }

    #[tracing::instrument(skip(path), fields(path = ?path.as_ref()))]
    pub fn load_aby(path: impl AsRef<Path>, load: Load) -> Result<Self, CircuitError> {
        let parsed = aby::Circuit::load(path.as_ref())?;
        BaseCircuit::from_aby(parsed, load)
    }
}

fn bitvec_and_fold(size: usize, inp: &[&BitVec<u64>]) -> BitVec<u64> {
    inp.iter()
        .fold(BitVec::<u64>::repeat(false, size), |mut acc, inp| {
            acc.as_raw_mut_slice()
                .iter_mut()
                .zip(inp.as_raw_slice())
                .for_each(|(acc, inp)| {
                    *acc &= *inp;
                });
            acc
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::order::{Lsb0, Msb0};
    use bitvec::{bits, bitvec};
    use std::iter;

    #[test]
    fn expand_output() {
        let expanded = expand(3, bits![u8, Lsb0; 0,1,1,0,1,1,0,1], bits![u8, Lsb0; 0;8]);
        let expected = vec![
            bitvec![u8, Msb0; 1,1,0,0,0],
            bitvec![u8, Msb0; 1,0,1,1,0],
            bitvec![u8, Msb0; 0,1,1,0,0],
        ];
        for (expected, expanded) in iter::zip(expected, expanded) {
            assert_eq!(expected, expanded)
        }
    }

    // #[test]
    // fn lut2() {
    //     /*
    //     +-----+-----+---+
    //     | x_1 | x_2 | y |
    //     +-----+-----+---+
    //     |   0 |   0 | 0 |
    //     |   0 |   1 | 1 |
    //     |   1 |   0 | 1 |
    //     |   1 |   1 | 0 |
    //     +-----+-----+---+
    //
    //     leads to input masks
    //
    //     +----------+----------+
    //     | mask x_1 | mask x_2 |
    //     +----------+----------+
    //     |        1 |        0 |
    //     |        0 |        1 |
    //     +----------+----------+
    //
    //     */
    //     let gate = LutGate::Inp2([
    //         bitarr!(u8, Msb0; 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
    //         bitarr!(u8, Msb0; 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
    //     ]);
    //
    //     // a = 0, b = 1
    //     let (input0, input1) = DeltaSharing::test_share(bitvec![u8, Lsb0; 0, 1]);
    //
    //     // reconstruct \delta_{ab} from the shared values
    //     let delta_ab = input0
    //         .private
    //         .iter()
    //         .zip(&input1.private)
    //         .map(|(priv0, priv1)| *priv0 ^ *priv1)
    //         .fold(true, BitAnd::bitand);
    //
    //     let mut setup_data_0 = SetupData {
    //         lut_eval_shares: vec![LutEvalShares::Inp2 { ab: Some(delta_ab) }],
    //     };
    //     let mut setup_data_1 = SetupData {
    //         lut_eval_shares: vec![LutEvalShares::Inp2 { ab: Some(false) }],
    //     };
    //
    //     let y0 =
    //         gate.compute_delta_share(0, input0.into_iter(), &mut setup_data_0, Share::default());
    //     let y1 =
    //         gate.compute_delta_share(1, input1.into_iter(), &mut setup_data_1, Share::default());
    //
    //     assert_eq!(true, y0 ^ y1);
    // }

    // #[test]
    // fn wire_out_convert() {
    //     let wire_out = WireOutput {
    //         unexpanded: bitarr![u8, Msb0; 0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0],
    //     };
    //     let expected: [_; 3] = [
    //         bitarr![u8, Msb0; 0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
    //         bitarr![u8, Msb0; 0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0],
    //         bitarr![u8, Msb0; 0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
    //     ];
    //     assert_eq!(expected, <[_; 3]>::from(wire_out));
    // }

    #[test]
    fn convert_lut_circuit() {
        let circ = BaseCircuit::load_lut_circuit(
            Path::new("test_resources/lut_circuits/Sample LUT file.lut"),
            Load::Circuit,
        )
        .unwrap();
        circ.save_dot("sample_lut").unwrap();
    }
}
