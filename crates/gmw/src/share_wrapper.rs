use crate::circuit::base_circuit::BaseGate;
use crate::circuit::builder::SharedCircuit;
use crate::circuit::{BooleanGate, CircuitId, DefaultIdx, GateId, GateIdx};
use crate::protocols::boolean_gmw::BooleanGmw;
use crate::protocols::{Gate, Protocol};
use crate::CircuitBuilder;
use itertools::Itertools;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

// TODO ShareWrappe can now implement Clone, but should it?
#[derive(Hash)]
pub struct ShareWrapper<P = BooleanGmw, Idx = DefaultIdx> {
    pub(crate) circuit_id: CircuitId,
    pub(crate) output_of: GateId<Idx>,
    // The current ShareWrapper API has some significant limitations when used in a multi-threaded
    // context. Better to forbid it for now so that we can maybe change to non-thread safe
    // primitives
    not_thread_safe: PhantomData<*const ()>,
    protocol: PhantomData<P>,
}

impl<P: Protocol, Idx: GateIdx> ShareWrapper<P, Idx> {
    pub(crate) fn new(circuit_id: CircuitId, output_of: GateId<Idx>) -> Self {
        Self {
            circuit_id,
            output_of,
            not_thread_safe: PhantomData,
            protocol: PhantomData,
        }
    }

    /// Note: Do not use while holding mutable borrow of self.circuit as it will panic!
    pub fn from_const(circuit_id: CircuitId, constant: <P::Gate as Gate>::Share) -> Self {
        let circuit = CircuitBuilder::<P::Gate, Idx>::get_global_circuit(circuit_id)
            .unwrap_or_else(|| {
                panic!("circuit_id {circuit_id} is not stored in global CircuitBuilder")
            });
        let output_of = {
            let mut circuit = circuit.lock();
            circuit.add_gate(BaseGate::Constant(constant).into())
        };
        Self::new(circuit_id, output_of)
    }

    // TODO remove circuit_id param as only circuit 0 can have inputs ?
    pub fn input(circuit_id: CircuitId) -> Self {
        let circuit = CircuitBuilder::<P::Gate, Idx>::get_global_circuit(circuit_id)
            .unwrap_or_else(|| {
                panic!("circuit_id {circuit_id} is not stored in global CircuitBuilder")
            });
        let output_of = circuit.lock().add_gate(BaseGate::Input.into());
        Self::new(circuit_id, output_of)
    }

    pub fn sub_circuit_input(circuit_id: CircuitId, gate: P::Gate) -> Self {
        let circuit = CircuitBuilder::<P::Gate, Idx>::get_global_circuit(circuit_id)
            .unwrap_or_else(|| {
                panic!("circuit_id {circuit_id} is not stored in global CircuitBuilder")
            });
        let output_of = {
            let mut circ = circuit.lock();
            circ.add_sc_input_gate(gate)
        };
        Self::new(circuit_id, output_of)
    }

    pub fn sub_circuit_output(&self) -> Self {
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        let output_of =
            circuit.add_wired_gate(BaseGate::SubCircuitOutput.into(), &[self.output_of]);
        Self::new(self.circuit_id, output_of)
    }

    /// Consumes this ShareWrapper and constructs a `Gate::Output` in the circuit with its value
    pub fn output(self) -> GateId<Idx> {
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        circuit.add_wired_gate(BaseGate::Output.into(), &[self.output_of])
    }

    pub fn connect_to_main_circuit(self) -> Self {
        assert_ne!(
            self.circuit_id, 0,
            "Can't connect ShareWrapper of main circuit to main circuit"
        );
        let out = self.sub_circuit_output();
        CircuitBuilder::<P::Gate, Idx>::with_global(|builder| {
            let input_to_main =
                ShareWrapper::sub_circuit_input(0, BaseGate::SubCircuitInput.into());
            builder.connect_circuits([(out, input_to_main.clone())]);
            input_to_main
        })
    }

    pub fn gate_id(&self) -> GateId<Idx> {
        self.output_of
    }

    pub fn get_circuit(&self) -> SharedCircuit<P::Gate, Idx> {
        CircuitBuilder::<P::Gate, Idx>::get_global_circuit(self.circuit_id)
            .expect("circuit_id is not stored in global CircuitBuilder")
    }
}

impl<Idx: GateIdx, Rhs: Borrow<Self>> BitXor<Rhs> for ShareWrapper<BooleanGmw, Idx> {
    type Output = Self;

    fn bitxor(mut self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "ShareWrapper operations are only defined on Wrappers for the same circuit"
        );
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            circuit.add_wired_gate(BooleanGate::Xor, &[self.output_of, rhs.output_of])
        };
        self
    }
}

impl<Idx: GateIdx> BitXor<bool> for ShareWrapper<BooleanGmw, Idx> {
    type Output = Self;

    fn bitxor(mut self, rhs: bool) -> Self::Output {
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            let const_gate = circuit.add_gate(BooleanGate::Base(BaseGate::Constant(rhs)));
            circuit.add_wired_gate(BooleanGate::Xor, &[self.output_of, const_gate])
        };
        self
    }
}

impl<Idx: GateIdx, Rhs: Borrow<Self>> BitXorAssign<Rhs> for ShareWrapper<BooleanGmw, Idx> {
    fn bitxor_assign(&mut self, rhs: Rhs) {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "ShareWrapper operations are only defined on Wrappers for the same circuit"
        );
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        self.output_of = circuit.add_wired_gate(BooleanGate::Xor, &[self.output_of, rhs.output_of]);
    }
}

impl<Idx: GateIdx> BitXorAssign<bool> for ShareWrapper<BooleanGmw, Idx> {
    fn bitxor_assign(&mut self, rhs: bool) {
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        let const_gate = circuit.add_gate(BooleanGate::Base(BaseGate::Constant(rhs)));
        self.output_of = circuit.add_wired_gate(BooleanGate::Xor, &[self.output_of, const_gate]);
    }
}

impl<Idx: GateIdx, Rhs: Borrow<Self>> BitAnd<Rhs> for ShareWrapper<BooleanGmw, Idx> {
    type Output = Self;

    fn bitand(mut self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "ShareWrapper operations are only defined on Wrappers for the same circuit"
        );
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            circuit.add_wired_gate(BooleanGate::And, &[self.output_of, rhs.output_of])
        };
        self
    }
}

impl<Idx: GateIdx> BitAnd<bool> for ShareWrapper<BooleanGmw, Idx> {
    type Output = Self;

    fn bitand(mut self, rhs: bool) -> Self::Output {
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            let const_gate = circuit.add_gate(BooleanGate::Base(BaseGate::Constant(rhs)));
            circuit.add_wired_gate(BooleanGate::And, &[self.output_of, const_gate])
        };
        self
    }
}

impl<Idx: GateIdx, Rhs: Borrow<Self>> BitAndAssign<Rhs> for ShareWrapper<BooleanGmw, Idx> {
    fn bitand_assign(&mut self, rhs: Rhs) {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "ShareWrapper operations are only defined on Wrappers for the same circuit"
        );
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        self.output_of = circuit.add_wired_gate(BooleanGate::And, &[self.output_of, rhs.output_of]);
    }
}

impl<Idx: GateIdx> BitAndAssign<bool> for ShareWrapper<BooleanGmw, Idx> {
    fn bitand_assign(&mut self, rhs: bool) {
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        let const_gate = circuit.add_gate(BooleanGate::Base(BaseGate::Constant(rhs)));
        self.output_of = circuit.add_wired_gate(BooleanGate::And, &[self.output_of, const_gate]);
    }
}

impl<Idx: GateIdx, Rhs: Borrow<Self>> BitOr<Rhs> for ShareWrapper<BooleanGmw, Idx> {
    type Output = Self;

    fn bitor(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "ShareWrapper operations are only defined on Wrappers for the same circuit"
        );
        // a | b <=> (a ^ b) ^ (a & b)
        self.clone() ^ rhs.clone() ^ (self & rhs)
    }
}

impl<Idx: GateIdx> BitOr<bool> for ShareWrapper<BooleanGmw, Idx> {
    type Output = Self;

    fn bitor(self, rhs: bool) -> Self::Output {
        let rhs = ShareWrapper::from_const(self.circuit_id, rhs);
        // a | b <=> (a ^ b) ^ (a & b)
        self.clone() ^ rhs.clone() ^ (self & rhs)
    }
}

impl<Idx: GateIdx, Rhs: Borrow<Self>> BitOrAssign<Rhs> for ShareWrapper<BooleanGmw, Idx> {
    fn bitor_assign(&mut self, rhs: Rhs) {
        let rhs = rhs.borrow();
        *self ^= rhs.clone() ^ (self.clone() & rhs);
    }
}

impl<Idx: GateIdx> BitOrAssign<bool> for ShareWrapper<BooleanGmw, Idx> {
    fn bitor_assign(&mut self, rhs: bool) {
        let rhs = ShareWrapper::from_const(self.circuit_id, rhs);
        *self ^= rhs.clone() ^ (self.clone() & rhs);
    }
}

impl<Idx: GateIdx> Not for ShareWrapper<BooleanGmw, Idx> {
    type Output = Self;

    fn not(mut self) -> Self::Output {
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            circuit.add_wired_gate(BooleanGate::Inv, &[self.output_of])
        };
        self
    }
}

impl<'a, Idx: GateIdx> Not for &'a ShareWrapper<BooleanGmw, Idx> {
    type Output = ShareWrapper<BooleanGmw, Idx>;

    fn not(self) -> Self::Output {
        let output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            circuit.add_wired_gate(BooleanGate::Inv, &[self.output_of])
        };
        ShareWrapper::new(self.circuit_id, output_of)
    }
}

impl<P, Idx: GateIdx + Debug> Debug for ShareWrapper<P, Idx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ShareWrapper for output of gate {:?}", self.output_of)
    }
}

impl<P, Idx: PartialEq> PartialEq for ShareWrapper<P, Idx> {
    fn eq(&self, other: &Self) -> bool {
        self.circuit_id == other.circuit_id && self.output_of == other.output_of
    }
}

impl<P, Idx: Eq> Eq for ShareWrapper<P, Idx> {}

impl<P, Idx: PartialOrd> PartialOrd for ShareWrapper<P, Idx> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.circuit_id.partial_cmp(&other.circuit_id) {
            Some(Ordering::Equal) => self.output_of.partial_cmp(&other.output_of),
            other => other,
        }
    }
}

impl<P, Idx: Ord> Ord for ShareWrapper<P, Idx> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).expect("Inconsistent partial_cmp")
    }
}

impl<P, Idx: Clone> Clone for ShareWrapper<P, Idx> {
    fn clone(&self) -> Self {
        Self {
            circuit_id: self.circuit_id,
            output_of: self.output_of.clone(),
            not_thread_safe: PhantomData,
            protocol: PhantomData,
        }
    }
}

// TODO placeholder function until I can think of a nice place to put this
// Creates inputs for the main circuit (id = 0)
pub fn inputs<P: Protocol, Idx: GateIdx>(inputs: usize) -> Vec<ShareWrapper<P, Idx>> {
    (0..inputs).map(|_| ShareWrapper::input(0)).collect()
}

// TODO placeholder function until I can think of a nice place to put this
// TODO this needs to have a generic return type to support more complex sub circuit input functions
pub(crate) fn sub_circuit_inputs<P: Protocol, Idx: GateIdx>(
    circuit_id: CircuitId,
    inputs: usize,
    gate: P::Gate,
) -> Vec<ShareWrapper<P, Idx>> {
    (0..inputs)
        .map(|_| ShareWrapper::sub_circuit_input(circuit_id, gate.clone()))
        .collect()
}

/// Reduce the slice of ShareWrappers with the provided operation. The operation can be a closure
/// or simply one of the operations implemented on [`ShareWrapper`]s, like [`std::ops::BitAnd`].  
/// The circuit will be constructed such that the depth is minimal.
///
/// ```rust
///# use std::sync::Arc;
///# use gmw::circuit::{BaseCircuit, DefaultIdx};
///# use gmw::share_wrapper::{inputs, low_depth_reduce};
///# use parking_lot::Mutex;
///# use gmw::{BooleanGate, Circuit, CircuitBuilder};
///#
/// let inputs = inputs::<DefaultIdx>(23);
/// low_depth_reduce(inputs, std::ops::BitAnd::bitand)
///     .unwrap()
///     .output();
/// // It is important that the Gate and Idx type of the circuit match up with those of the
/// // ShareWrappers, as otherwise an empty circuit will be returned
/// let and_tree: Circuit<BooleanGate, DefaultIdx> = CircuitBuilder::global_into_circuit();
/// assert_eq!(and_tree.interactive_count(), 22)
/// ```
///
pub fn low_depth_reduce<F, Idx: GateIdx>(
    shares: impl IntoIterator<Item = ShareWrapper<BooleanGmw, Idx>>,
    mut f: F,
) -> Option<ShareWrapper<BooleanGmw, Idx>>
where
    F: FnMut(
        ShareWrapper<BooleanGmw, Idx>,
        ShareWrapper<BooleanGmw, Idx>,
    ) -> ShareWrapper<BooleanGmw, Idx>,
{
    // Todo: This implementation is probably a little bit inefficient. It might be possible to use
    //  the lower level api to construct the circuit faster. This should be benchmarked however.
    let mut buf: Vec<_> = shares.into_iter().collect();
    let mut old_buf = Vec::with_capacity(buf.len() / 2);
    while buf.len() > 1 {
        mem::swap(&mut buf, &mut old_buf);
        let mut iter = old_buf.drain(..).tuples();
        for (s1, s2) in iter.by_ref() {
            buf.push(f(s1, s2));
        }
        for odd in iter.into_buffer() {
            buf.push(odd)
        }
    }
    debug_assert!(buf.len() <= 1);
    buf.pop()
}
