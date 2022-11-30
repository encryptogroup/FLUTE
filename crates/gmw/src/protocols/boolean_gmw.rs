use crate::bristol;
use crate::circuit::base_circuit::BaseGate;
use crate::common::BitVec;
use crate::evaluate::and;
use crate::mul_triple::{MulTriple, MulTriples};
use crate::protocols::{Gate, Protocol, Sharing};
use crate::utils::rand_bitvec;
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};
use std::ops;

#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct BooleanGmw;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Msg {
    // TODO ser/de the BitVecs or Vecs? Or maybe a single Vec? e and d have the same length
    AndLayer { e: Vec<u8>, d: Vec<u8> },
}

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum BooleanGate {
    Base(BaseGate<bool>),
    And,
    Xor,
    Inv,
}

#[derive(Debug)]
pub struct XorSharing<R: CryptoRng + Rng> {
    rng: R,
}

impl Protocol for BooleanGmw {
    type Msg = Msg;
    type Gate = BooleanGate;
    type ShareStorage = BitVec;
    type SetupStorage = MulTriples;

    fn compute_msg(
        &self,
        _party_id: usize,
        interactive_gates: impl IntoIterator<Item = (BooleanGate, bool)>,
        inputs: impl IntoIterator<Item = bool>,
        mul_triples: &MulTriples,
    ) -> Self::Msg {
        let mut inputs = inputs.into_iter();
        let (d, e): (BitVec, BitVec) = interactive_gates
            .into_iter()
            .zip(mul_triples.iter())
            .map(|((gate, _), mt): ((BooleanGate, bool), MulTriple)| {
                // TODO debug_assert?
                assert!(matches!(gate, BooleanGate::And));
                let mut inputs = inputs.by_ref().take(gate.input_size());
                let (x, y) = (inputs.next().unwrap(), inputs.next().unwrap());
                debug_assert!(
                    inputs.next().is_none(),
                    "Currently only support AND gates with 2 inputs"
                );
                and::compute_shares(x, y, &mt)
            })
            .unzip();
        Msg::AndLayer {
            e: e.into_vec(),
            d: d.into_vec(),
        }
    }

    fn evaluate_interactive(
        &self,
        party_id: usize,
        _interactive_gates: impl IntoIterator<Item = (Self::Gate, bool)>,
        own_msg: Self::Msg,
        other_msg: Self::Msg,
        mul_triples: MulTriples,
    ) -> Self::ShareStorage {
        let Msg::AndLayer { d, e } = own_msg;
        let d = BitVec::from_vec(d);
        let e = BitVec::from_vec(e);
        let Msg::AndLayer {
            d: resp_d,
            e: resp_e,
        } = other_msg;
        d.into_iter()
            .zip(e)
            .zip(BitVec::from_vec(resp_d))
            .zip(BitVec::from_vec(resp_e))
            .zip(mul_triples.iter())
            .map(|((((d, e), d_resp), e_resp), mt)| {
                let d = [d, d_resp];
                let e = [e, e_resp];
                and::evaluate(d, e, mt, party_id)
            })
            .collect()
    }
}

impl Gate for BooleanGate {
    type Share = bool;

    fn is_interactive(&self) -> bool {
        matches!(self, BooleanGate::And)
    }

    fn input_size(&self) -> usize {
        match self {
            BooleanGate::Base(base_gate) => base_gate.input_size(),
            BooleanGate::Inv => 1,
            BooleanGate::And | BooleanGate::Xor => 2,
        }
    }

    fn as_base_gate(&self) -> Option<&BaseGate<Self::Share>> {
        match self {
            BooleanGate::Base(base_gate) => Some(base_gate),
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
            BooleanGate::Base(base) => base.evaluate_non_interactive(party_id, input.by_ref()),
            BooleanGate::And => panic!("Called evaluate_non_interactive on Gate::AND"),
            BooleanGate::Xor => input.take(2).fold(false, ops::BitXor::bitxor),
            BooleanGate::Inv => {
                let inp = input.next().expect("Empty input");
                if party_id == 0 {
                    !inp
                } else {
                    inp
                }
            }
        }
    }
}

impl From<&bristol::Gate> for BooleanGate {
    fn from(gate: &bristol::Gate) -> Self {
        match gate {
            bristol::Gate::And(_) => BooleanGate::And,
            bristol::Gate::Xor(_) => BooleanGate::Xor,
            bristol::Gate::Inv(_) => BooleanGate::Inv,
        }
    }
}

impl From<BaseGate<bool>> for BooleanGate {
    fn from(base_gate: BaseGate<bool>) -> Self {
        BooleanGate::Base(base_gate)
    }
}

impl<R: CryptoRng + Rng> XorSharing<R> {
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: CryptoRng + Rng> Sharing for XorSharing<R> {
    type Plain = bool;
    type Shared = BitVec;

    fn share(&mut self, input: Self::Shared) -> [Self::Shared; 2] {
        let rand = rand_bitvec(input.len(), &mut self.rng);
        let masked_input = input ^ &rand;
        [rand, masked_input]
    }

    fn reconstruct(&mut self, shares: [Self::Shared; 2]) -> Self::Shared {
        let [a, b] = shares;
        a ^ b
    }
}
