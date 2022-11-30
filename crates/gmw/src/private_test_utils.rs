use std::fmt::Debug;
use std::path::Path;

use anyhow::Result;
use bitvec::field::BitField;
use bitvec::order::Lsb0;
use bitvec::vec;
use funty::Integral;
use itertools::Itertools;
use mpc_channel::sub_channel;
use tokio::task::spawn_blocking;
use tokio::time::Instant;
use tracing::info;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use crate::circuit::base_circuit::{BaseGate, Load};
use crate::circuit::Circuit;
use crate::circuit::{BaseCircuit, BooleanGate, GateIdx};
use crate::common::BitVec;
use crate::executor::Executor;
use crate::mul_triple::insecure_provider::InsecureMTProvider;
use crate::protocols::boolean_gmw::BooleanGmw;

pub fn create_and_tree(depth: u32) -> BaseCircuit {
    let total_nodes = 2_u32.pow(depth);
    let mut layer_count = total_nodes / 2;
    let mut circuit = BaseCircuit::new();

    let mut previous_layer: Vec<_> = (0..layer_count)
        .map(|_| circuit.add_gate(BooleanGate::Base(BaseGate::Input)))
        .collect();
    while layer_count > 1 {
        layer_count /= 2;
        previous_layer = previous_layer
            .into_iter()
            .tuples()
            .map(|(from_a, from_b)| circuit.add_wired_gate(BooleanGate::And, &[from_a, from_b]))
            .collect();
    }
    debug_assert_eq!(1, previous_layer.len());
    circuit.add_wired_gate(BooleanGate::Base(BaseGate::Output), &[previous_layer[0]]);
    circuit
}

/// Initializes tracing subscriber with EnvFilter for usage in tests. This should be the first call
/// in each test, with the returned value being assigned to a variable to prevent dropping.
/// Output can be configured via RUST_LOG env variable as explained
/// [here](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/struct.EnvFilter.html)
///
/// ```ignore
/// use gmw::private_test_utils::init_tracing;
/// fn some_test() {
///     let _guard = init_tracing();
/// }
/// ```
pub fn init_tracing() -> tracing::dispatcher::DefaultGuard {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_test_writer()
        .set_default()
}

#[derive(Debug)]
pub enum TestChannel {
    InMemory,
    Tcp,
}

pub trait IntoShares {
    fn into_shares(self) -> (BitVec, BitVec)
    where
        bitvec::slice::BitSlice<u8, Lsb0>: BitField;
}

pub trait IntoInput {
    fn into_input(self) -> (BitVec, BitVec);
}

impl<T: Integral> IntoShares for T {
    fn into_shares(self) -> (BitVec, BitVec)
    where
        bitvec::slice::BitSlice<u8, Lsb0>: BitField,
    {
        // TODO use pseudo random sharing (with fixed seed)
        let a = vec::BitVec::repeat(false, T::BITS as usize);
        let mut b = a.clone();
        b.store(self);
        (a, b)
    }
}

impl<T: IntoShares> IntoInput for (T,) {
    fn into_input(self) -> (BitVec, BitVec) {
        self.0.into_shares()
    }
}

impl<T1: IntoShares, T2: IntoShares> IntoInput for (T1, T2) {
    fn into_input(self) -> (BitVec, BitVec) {
        let (mut p1, mut p2) = self.0.into_shares();
        let mut second_input = self.1.into_shares();
        p1.append(&mut second_input.0);
        p2.append(&mut second_input.1);
        (p1, p2)
    }
}

/// This is kind of cursed...
impl IntoInput for [BitVec; 2] {
    fn into_input(self) -> (BitVec, BitVec) {
        let [a, b] = self;
        (a, b)
    }
}

#[tracing::instrument(skip(inputs))]
pub async fn execute_bristol<I: IntoInput>(
    bristol_file: impl AsRef<Path> + Debug,
    inputs: I,
    channel: TestChannel,
) -> Result<BitVec> {
    let path = bristol_file.as_ref().to_path_buf();
    let now = Instant::now();
    let circuit = spawn_blocking(move || BaseCircuit::load_bristol(path, Load::Circuit)).await??;
    info!(
        parsing_time = %now.elapsed().as_millis(),
        "Parsing bristol time (ms)"
    );
    let inputs = inputs.into_input();
    execute_circuit(&circuit.into(), inputs, channel).await
}

#[tracing::instrument(skip(circuit, input_a, input_b))]
pub async fn execute_circuit<Idx: GateIdx>(
    circuit: &Circuit<BooleanGate, Idx>,
    (input_a, input_b): (BitVec, BitVec),
    channel: TestChannel,
) -> Result<BitVec> {
    let mut mt_provider = InsecureMTProvider::default();
    let mut ex1: Executor<BooleanGmw, Idx> = Executor::new(circuit, 0, &mut mt_provider.clone())
        .await
        .unwrap();
    let mut ex2: Executor<BooleanGmw, Idx> =
        Executor::new(circuit, 1, &mut mt_provider).await.unwrap();
    let now = Instant::now();
    let (out1, out2) = match channel {
        TestChannel::InMemory => {
            let (mut t1, mut t2) = mpc_channel::in_memory::new_pair(2);
            let h1 = ex1.execute(input_a, &mut t1.0, &mut t1.1);
            let h2 = ex2.execute(input_b, &mut t2.0, &mut t2.1);
            futures::try_join!(h1, h2)?
        }
        TestChannel::Tcp => {
            let (mut t1, mut t2) =
                mpc_channel::tcp::new_local_pair::<mpc_channel::Receiver<_>>(None).await?;
            let (mut sub_t1, mut sub_t2) = tokio::try_join!(
                sub_channel(&mut t1.0, &mut t1.2, 2),
                sub_channel(&mut t2.0, &mut t2.2, 2)
            )?;
            let h1 = ex1.execute(input_a, &mut sub_t1.0, &mut sub_t1.1);
            let h2 = ex2.execute(input_b, &mut sub_t2.0, &mut sub_t2.1);
            let out = futures::try_join!(h1, h2)?;
            info!(
                bytes_sent = t1.1.get(),
                bytes_received = t1.3.get(),
                "Tcp communication"
            );
            out
        }
    };
    info!(exec_time = %now.elapsed().as_millis(), "Execution time (ms)");

    Ok(out1 ^ out2)
}
