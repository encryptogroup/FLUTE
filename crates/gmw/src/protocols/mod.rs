use crate::circuit::base_circuit::BaseGate;
use crate::circuit::GateIdx;
use crate::common::{BitSlice, BitVec};
use crate::Circuit;
use async_trait::async_trait;
use remoc::RemoteSend;
use std::fmt::Debug;
use std::hash::Hash;

pub mod aby2_lut;
pub mod boolean_gmw;

pub trait Protocol {
    type Msg: RemoteSend + Clone;
    type Gate: Gate + From<BaseGate<<Self::Gate as Gate>::Share>>;
    type ShareStorage: ShareStorage<<Self::Gate as Gate>::Share>;
    /// The type which provides the data needed to evaluate interactive gate.
    /// In the case of normal GMW, this data is multiplication triples.
    type SetupStorage: SetupStorage;

    fn compute_msg(
        &self,
        party_id: usize,
        interactive_gates: impl IntoIterator<Item = (Self::Gate, <Self::Gate as Gate>::Share)>,
        inputs: impl IntoIterator<Item = <Self::Gate as Gate>::Share>,
        preprocessing_data: &Self::SetupStorage,
    ) -> Self::Msg;

    fn evaluate_interactive(
        &self,
        party_id: usize,
        interactive_gates: impl IntoIterator<Item = (Self::Gate, <Self::Gate as Gate>::Share)>,
        own_msg: Self::Msg,
        other_msg: Self::Msg,
        preprocessing_data: Self::SetupStorage,
    ) -> Self::ShareStorage;

    // TOODO i'm not sure if party_id is needed here
    fn setup_gate_outputs<Idx: GateIdx>(
        &mut self,
        _party_id: usize,
        circuit: &Circuit<Self::Gate, Idx>,
    ) -> Vec<Self::ShareStorage> {
        circuit
            .circuits
            .iter()
            .map(|base_circ| Self::ShareStorage::repeat(Default::default(), base_circ.gate_count()))
            .collect()
    }
}

pub trait Gate: Clone + Hash + Ord + PartialEq + Eq + Send + Sync + Debug + 'static {
    type Share: Copy + Clone + Default + Debug;

    fn is_interactive(&self) -> bool;

    fn input_size(&self) -> usize;

    fn as_base_gate(&self) -> Option<&BaseGate<Self::Share>>;

    fn evaluate_non_interactive(
        &self,
        party_id: usize,
        inputs: impl IntoIterator<Item = Self::Share>,
    ) -> Self::Share;

    fn is_non_interactive(&self) -> bool {
        !self.is_interactive()
    }
}

pub trait ShareStorage<Share>:
    IntoIterator<Item = Share> + FromIterator<Share> + Clone + Default + Debug
{
    fn len(&self) -> usize;
    fn repeat(val: Share, len: usize) -> Self;
    fn set(&mut self, idx: usize, val: Share);
    fn get(&self, idx: usize) -> Share;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ShareStorage<bool> for BitVec {
    fn len(&self) -> usize {
        BitVec::len(self)
    }

    fn repeat(val: bool, len: usize) -> Self {
        BitVec::repeat(val, len)
    }

    fn set(&mut self, idx: usize, val: bool) {
        BitSlice::set(self, idx, val)
    }

    fn get(&self, idx: usize) -> bool {
        self[idx]
    }
}

pub trait SetupStorage {
    /// Split of the last `count` mul triples.
    fn split_off_last(&mut self, count: usize) -> Self;
}

pub trait Sharing {
    type Plain: Copy + Clone + Default + Debug;
    type Shared: ShareStorage<Self::Plain>;

    fn share(&mut self, input: Self::Shared) -> [Self::Shared; 2];

    fn reconstruct(&mut self, shares: [Self::Shared; 2]) -> Self::Shared;
}

#[async_trait]
pub trait FunctionDependentSetup<ShareStorage, G, Idx> {
    type Output;
    type Error;

    async fn setup(
        &mut self,
        shares: &[ShareStorage],
        circuit: &Circuit<G, Idx>,
    ) -> Result<Self::Output, Self::Error>;
}
