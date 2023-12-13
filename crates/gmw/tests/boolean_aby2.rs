use bitvec::order::Lsb0;
use bitvec::{bits, bitvec};
use gmw::circuit::base_circuit::Load;
use gmw::circuit::BaseCircuit;
use gmw::common::BitVec;
use gmw::executor::Executor;
use gmw::mul_triple::insecure_provider::InsecureMTProvider;
use gmw::parse::lut_circuit;
use gmw::private_test_utils::init_tracing;
use gmw::protocols::aby2_lut::{DeltaSharing, LutAby2, LutSetupProvider, ShareType};
use gmw::protocols::ShareStorage;
use std::path::Path;
use tracing::{info, trace};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "issue #6"]
async fn eval_sample_lut() -> anyhow::Result<()> {
    let test_lut_path = "test_resources/lut_circuits/minimal.lut";
    let _guard = init_tracing();
    let circ = BaseCircuit::load_lut_circuit(
        Path::new(test_lut_path),
        Load::Circuit,
    )?
    .into();

    // let priv_seed1 = thread_rng().gen();
    // let priv_seed2 = thread_rng().gen();
    // let joint_seed1 = thread_rng().gen();
    // let joint_seed2 = thread_rng().gen();

    let priv_seed1 = [
        225, 119, 29, 227, 179, 6, 133, 227, 30, 225, 15, 11, 58, 141, 22, 128, 124, 81, 70, 134,
        77, 191, 112, 25, 225, 226, 62, 148, 13, 151, 84, 234,
    ];
    let priv_seed2 = [
        106, 84, 187, 99, 98, 157, 150, 116, 47, 255, 82, 170, 125, 36, 181, 59, 251, 161, 249, 46,
        190, 235, 173, 197, 209, 174, 43, 45, 217, 47, 243, 47,
    ];
    let joint_seed1 = [
        35, 250, 55, 198, 214, 74, 135, 233, 136, 213, 142, 26, 48, 117, 34, 29, 192, 61, 193, 252,
        29, 38, 65, 49, 185, 118, 110, 72, 92, 166, 45, 120,
    ];
    let joint_seed2 = [
        211, 192, 166, 219, 177, 36, 39, 149, 154, 171, 88, 213, 68, 19, 131, 70, 149, 179, 185,
        143, 72, 58, 132, 236, 13, 13, 22, 224, 154, 126, 0, 236,
    ];

    info!(
        ?priv_seed1,
        ?priv_seed2,
        ?joint_seed1,
        ?joint_seed2,
        "Execution seeds"
    );

    let share_map1 = (0..64).map(|pos| (pos, ShareType::Local)).collect();
    let share_map2 = (0..64).map(|pos| (pos, ShareType::Remote)).collect();
    let mut sharing_state1 = DeltaSharing::new(priv_seed1, joint_seed1, joint_seed2, share_map1);
    let mut sharing_state2 = DeltaSharing::new(priv_seed2, joint_seed2, joint_seed1, share_map2);
    // let mut sharing_state1 = DeltaSharing::insecure_default();
    // let mut sharing_state2 = DeltaSharing::insecure_default();
    let state1 = LutAby2::new(sharing_state1.clone());
    let state2 = LutAby2::new(sharing_state2.clone());

    let (ch1, ch2) = mpc_channel::in_memory::new_pair(16);
    let mut delta_provider1 = LutSetupProvider::new(0, InsecureMTProvider::default(), ch1.0, ch1.1);
    let mut delta_provider2 = LutSetupProvider::new(1, InsecureMTProvider::default(), ch2.0, ch2.1);

    let (mut ex1, mut ex2): (Executor<LutAby2, usize>, Executor<LutAby2, usize>) =
        tokio::try_join!(
            Executor::new_with_state(state1, &circ, 0, &mut delta_provider1),
            Executor::new_with_state(state2, &circ, 1, &mut delta_provider2)
        )
        .unwrap();
    let mut inp = bitvec![u8, Lsb0; 1; 18];
    inp.extend_from_bitslice(bits![0; 64 - 18]);

    let (shared_0, plain_delta_0) = sharing_state1.share(inp.clone());

    let inp1 = shared_0;
    let inp2 = sharing_state2.plain_delta_to_share(plain_delta_0);

    let reconstruct: BitVec = inp1
        .clone()
        .into_iter()
        .zip(inp2.clone())
        .enumerate()
        .map(|(idx, (sh1, sh2))| {
            assert_eq!(sh1.get_public(), sh2.get_public());
            assert_eq!(
                sh1.get_private(),
                ex1.gate_outputs()[0].get(idx).get_private()
            );
            assert_eq!(
                sh2.get_private(),
                ex2.gate_outputs()[0].get(idx).get_private()
            );

            sh1.get_public() ^ sh1.get_private() ^ sh2.get_private()
        })
        .collect();
    assert_eq!(inp, reconstruct);

    let (mut ch1, mut ch2) = mpc_channel::in_memory::new_pair(16);

    let (out1, out2) = tokio::try_join!(
        ex1.execute(inp1, &mut ch1.0, &mut ch1.1),
        ex2.execute(inp2, &mut ch2.0, &mut ch2.1),
    )?;
    let out_bits: BitVec = out1
        .into_iter()
        .zip(out2)
        .map(|(sh1, sh2)| {
            assert_eq!(sh1.get_public(), sh2.get_public());
            sh1.get_public() ^ sh1.get_private() ^ sh2.get_private()
        })
        .collect();

    let lut_circ =
        lut_circuit::Circuit::load(Path::new(test_lut_path))?;
    // let mut inp = bitvec![u8, Msb0; 1;8];
    // inp[0..32].store_be(0_u32);
    // inp[32..64].store_be(0_u32);
    let expected = lut_circ.execute(&inp);
    trace!(?expected);
    assert_eq!(expected, out_bits);

    Ok(())
}
