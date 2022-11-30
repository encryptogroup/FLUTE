#[cfg(debug_assertions)]
use std::collections::HashSet;
use std::fmt::Debug;
use std::iter;
use std::time::Instant;

use mpc_channel::{Receiver, Sender};
use tracing::{debug, info, trace};

use crate::circuit::base_circuit::BaseGate;
use crate::circuit::builder::SubCircuitGate;
use crate::circuit::{Circuit, CircuitLayerIter, DefaultIdx, GateIdx};
use crate::errors::{CircuitError, ExecutorError};

use crate::protocols::boolean_gmw::BooleanGmw;
use crate::protocols::{FunctionDependentSetup, Gate, Protocol, SetupStorage, ShareStorage};

pub type BoolGmwExecutor<'c> = Executor<'c, BooleanGmw, DefaultIdx>;

pub struct Executor<'c, P: Protocol, Idx> {
    circuit: &'c Circuit<P::Gate, Idx>,
    protocol_state: P,
    gate_outputs: Vec<P::ShareStorage>,
    party_id: usize,
    setup_storage: P::SetupStorage,
    // Used as a sanity check in debug builds. Stores for which gates we have set the output,
    // so that we can check if an output is set before accessing it.
    #[cfg(debug_assertions)]
    output_set: HashSet<SubCircuitGate<Idx>>,
}

impl<'c, P: Protocol + Default, Idx: GateIdx> Executor<'c, P, Idx> {
    pub async fn new<
        FDSetup: FunctionDependentSetup<P::ShareStorage, P::Gate, Idx, Output = P::SetupStorage>,
    >(
        circuit: &'c Circuit<P::Gate, Idx>,
        party_id: usize,
        setup: &mut FDSetup,
    ) -> Result<Executor<'c, P, Idx>, CircuitError>
    where
        FDSetup::Error: Debug,
    {
        Self::new_with_state(P::default(), circuit, party_id, setup).await
    }
}

impl<'c, P: Protocol, Idx: GateIdx> Executor<'c, P, Idx> {
    pub async fn new_with_state<
        FDSetup: FunctionDependentSetup<P::ShareStorage, P::Gate, Idx, Output = P::SetupStorage>,
    >(
        mut protocol_state: P,
        circuit: &'c Circuit<P::Gate, Idx>,
        party_id: usize,
        setup: &mut FDSetup,
    ) -> Result<Executor<'c, P, Idx>, CircuitError>
    where
        FDSetup::Error: Debug,
    {
        let gate_outputs = protocol_state.setup_gate_outputs(party_id, circuit);

        let setup_storage = setup.setup(&gate_outputs, circuit).await.unwrap();
        Ok(Self {
            circuit,
            protocol_state,
            gate_outputs,
            party_id,
            setup_storage,
            #[cfg(debug_assertions)]
            output_set: HashSet::new(),
        })
    }

    #[tracing::instrument(skip_all, fields(party_id = self.party_id), err)]
    pub async fn execute(
        &mut self,
        inputs: P::ShareStorage,
        sender: &mut Sender<P::Msg>,
        receiver: &mut Receiver<P::Msg>,
    ) -> Result<P::ShareStorage, ExecutorError> {
        info!(
            inputs = self.circuit.input_count(),
            outputs = self.circuit.output_count(),
            "Executing circuit"
        );
        let now = Instant::now();
        assert_eq!(
            self.circuit.input_count(),
            inputs.len(),
            "Length of inputs must be equal to circuit input size"
        );
        let mut layer_count = 0;
        let mut interactive_count = 0;
        // TODO provide the option to calculate next layer  during and communication
        //  take care to not block tokio threads -> use tokio rayon
        for layer in CircuitLayerIter::new(self.circuit) {
            for (gate, sc_gate_id) in layer.non_interactive_iter() {
                let output = if let Some(BaseGate::Input) = gate.as_base_gate() {
                    assert_eq!(
                        sc_gate_id.circuit_id, 0,
                        "Input gate in SubCircuit. Use SubCircuitInput"
                    );
                    // TODO, ugh log(n) in loop... and i'm not even sure if this is correct
                    let inp_idx = self.circuit.circuits[0]
                        .input_gates()
                        .binary_search(&sc_gate_id.gate_id)
                        .expect("Input gate not contained in input_gates");
                    gate.evaluate_non_interactive(self.party_id, iter::once(inputs.get(inp_idx)))
                } else {
                    let inputs = self.gate_inputs(sc_gate_id);
                    gate.evaluate_non_interactive(self.party_id, inputs)
                };
                // debug!(
                //     before = ?self.gate_outputs[sc_gate_id.circuit_id as usize]
                //         .get(sc_gate_id.gate_id.as_usize())
                // );
                trace!(
                    ?output,
                    sc_gate_id = %sc_gate_id,
                    "Evaluated {:?} gate",
                    gate
                );

                self.set_gate_output(sc_gate_id, output);
            }

            // TODO count() there should be a more efficient option
            let layer_int_cnt = layer.interactive_iter().count();
            if layer_int_cnt == 0 {
                // If the layer does not contain and gates we continue
                continue;
            }
            // Only count layers with and gates
            layer_count += 1;
            interactive_count += layer_int_cnt;
            let layer_mts = self.setup_storage.split_off_last(layer_int_cnt);

            let input_iter = layer
                .interactive_iter()
                .flat_map(|(_, sc_gate_id)| self.gate_inputs(sc_gate_id));
            let gate_iter = layer.interactive_iter().map(|(gate, sc_gate_id)| {
                (
                    gate,
                    self.gate_outputs[sc_gate_id.circuit_id as usize]
                        .get(sc_gate_id.gate_id.as_usize()),
                )
            });
            let msg = self.protocol_state.compute_msg(
                self.party_id,
                gate_iter.clone(),
                input_iter,
                &layer_mts,
            );

            // TODO unnecessary clone
            debug!("Sending interactive gates layer");
            let (send_res, recv_res) = tokio::join!(sender.send(msg.clone()), receiver.recv());
            send_res.ok().unwrap();
            let response = recv_res.ok().unwrap().unwrap();

            let interactive_outputs = self.protocol_state.evaluate_interactive(
                self.party_id,
                gate_iter,
                msg,
                response,
                layer_mts,
            );
            layer
                .interactive_iter()
                .zip(interactive_outputs)
                .for_each(|((_, id), out)| {
                    // debug!(
                    //     before = ?self.gate_outputs[id.circuit_id as usize]
                    //         .get(id.gate_id.as_usize())
                    // );
                    self.set_gate_output(id, out);
                    trace!(?out, gate_id = %id, "Evaluated interactive gate");
                });
        }
        info!(
            layer_count,
            interactive_count,
            execution_time_s = now.elapsed().as_secs_f32()
        );
        let output_iter = self.circuit.circuits[0].output_gates().iter().map(|id| {
            #[cfg(debug_assertions)]
            assert!(
                self.output_set.contains(&SubCircuitGate::new(0, *id)),
                "output gate with id {id:?} is not set",
            );
            self.gate_outputs[0].get(id.as_usize())
        });
        Ok(FromIterator::from_iter(output_iter))
    }

    pub fn gate_outputs(&self) -> &[P::ShareStorage] {
        &self.gate_outputs
    }

    fn gate_inputs(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = <P::Gate as Gate>::Share> + '_ {
        self.circuit
            .parent_gates(id)
            // TODO sorted allocates
            // .sorted()
            .map(move |parent_id| {
                trace!(?parent_id, of = ?id);
                #[cfg(debug_assertions)]
                assert!(
                    self.output_set.contains(&parent_id),
                    "parent {} of {} not set",
                    parent_id,
                    id
                );
                self.gate_outputs[parent_id.circuit_id as usize].get(parent_id.gate_id.as_usize())
            })
    }

    fn set_gate_output(&mut self, id: SubCircuitGate<Idx>, output: <P::Gate as Gate>::Share) {
        let sc_outputs = &mut self.gate_outputs[id.circuit_id as usize];
        sc_outputs.set(id.gate_id.as_usize(), output);
        #[cfg(debug_assertions)]
        self.output_set.insert(id);
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::base_circuit::BaseGate;
    use anyhow::Result;
    use bitvec::{bitvec, prelude::Lsb0};
    use tracing::debug;

    use crate::circuit::{BaseCircuit, DefaultIdx};
    use crate::common::BitVec;
    use crate::private_test_utils::{create_and_tree, execute_circuit, init_tracing, TestChannel};
    use crate::share_wrapper::{inputs, ShareWrapper};
    use crate::{BooleanGate, Circuit, CircuitBuilder};

    #[tokio::test]
    async fn execute_simple_circuit() -> Result<()> {
        let _guard = init_tracing();
        use crate::protocols::boolean_gmw::BooleanGate::*;
        let mut circuit: BaseCircuit = BaseCircuit::new();
        let in_1 = circuit.add_gate(Base(BaseGate::Input));
        let in_2 = circuit.add_gate(Base(BaseGate::Input));
        let and_1 = circuit.add_wired_gate(And, &[in_1, in_2]);
        let xor_1 = circuit.add_wired_gate(Xor, &[in_2, and_1]);
        let and_2 = circuit.add_wired_gate(And, &[and_1, xor_1]);
        circuit.add_wired_gate(Base(BaseGate::Output), &[and_2]);

        let inputs = (BitVec::repeat(true, 2), BitVec::repeat(false, 2));
        let out = execute_circuit(&circuit.into(), inputs, TestChannel::InMemory).await?;
        assert_eq!(1, out.len());
        assert_eq!(false, out[0]);
        Ok(())
    }

    #[tokio::test]
    async fn eval_and_tree() -> Result<()> {
        let _guard = init_tracing();
        let and_tree = create_and_tree(10);
        let inputs_0 = {
            let mut bits = BitVec::new();
            bits.resize(and_tree.input_count(), true);
            bits
        };
        let inputs_1 = !inputs_0.clone();
        let out = execute_circuit(
            &and_tree.into(),
            (inputs_0, inputs_1),
            TestChannel::InMemory,
        )
        .await?;
        assert_eq!(1, out.len());
        assert_eq!(true, out[0]);
        Ok(())
    }

    #[tokio::test]
    async fn eval_2_bit_adder() -> Result<()> {
        let _guard = init_tracing();
        debug!("Test start");
        let inputs = inputs(4);
        debug!("Inputs");
        let [a0, a1, b0, b1]: [ShareWrapper; 4] = inputs.try_into().unwrap();
        let xor1 = a0.clone() ^ b0.clone();
        let and1 = a0 & b0;
        let xor2 = a1.clone() ^ b1.clone();
        let and2 = a1 & b1;
        let xor3 = xor2.clone() ^ and1.clone();
        let and3 = xor2 & and1;
        let or = and2 | and3;
        for share in [xor1, xor3, or] {
            share.output();
        }

        debug!("End Sharewrapper ops");
        let inputs_0 = bitvec![u8, Lsb0; 1, 1, 0, 0];
        let inputs_1 = bitvec![u8, Lsb0; 0, 1, 0, 1];
        let exp_output = bitvec![u8, Lsb0; 1, 1, 0];
        let adder: Circuit<BooleanGate, DefaultIdx> = CircuitBuilder::global_into_circuit();
        debug!("Into circuit");
        let out = execute_circuit(&adder, (inputs_0, inputs_1), TestChannel::InMemory).await?;
        debug!("Executed");
        assert_eq!(exp_output, out);
        Ok(())
    }
}
