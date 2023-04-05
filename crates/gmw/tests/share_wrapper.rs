use anyhow::Result;
use gmw::circuit::DefaultIdx;

use gmw::common::BitVec;
use gmw::private_test_utils::{execute_circuit, init_tracing, TestChannel};
use gmw::protocols::boolean_gmw::BooleanGmw;
use gmw::share_wrapper::{inputs, low_depth_reduce};
use gmw::{BooleanGate, Circuit, CircuitBuilder};

#[tokio::test]
async fn and_tree() -> Result<()> {
    // TODO assert depth
    let _guard = init_tracing();
    let input_count = 23;
    let inputs = inputs::<BooleanGmw, DefaultIdx>(input_count);
    low_depth_reduce(inputs, std::ops::BitAnd::bitand)
        .unwrap()
        .output();
    // and_tree
    //     .lock()
    //     .save_dot("tests/circuit-graphs/and_tree.dot")
    //     .unwrap();

    let inputs_0 = BitVec::repeat(false, input_count);
    let inputs_1 = BitVec::repeat(true, input_count);

    let exp_output: BitVec = BitVec::repeat(true, 1);
    let and_tree: Circuit<BooleanGate, DefaultIdx> = CircuitBuilder::global_into_circuit();
    let out = execute_circuit(&and_tree, (inputs_0, inputs_1), TestChannel::Tcp).await?;
    assert_eq!(exp_output, out);
    Ok(())
}
