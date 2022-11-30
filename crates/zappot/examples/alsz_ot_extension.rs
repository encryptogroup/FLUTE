//! Example - ALSZ13 OT extension
//!
//! This example shows how to use the zappot crate to execute the ALSZ13 OT extension protocol
//! to generate random OTs.
use bitvec::vec::BitVec;
use bitvec::{bitvec, order::Lsb0};
use clap::Parser;
use mpc_channel::sub_channel;
use rand::Rng;
use rand_core::OsRng;
use std::time::{Duration, Instant};
use tracing_subscriber::EnvFilter;
use zappot::base_ot;
use zappot::ot_ext::{Receiver, Sender};
use zappot::traits::{ExtROTReceiver, ExtROTSender};
use zappot::util::Block;

#[derive(Parser, Debug, Clone)]
struct Args {
    /// Number of OTs to execute
    #[clap(short, long, default_value_t = 1000)]
    num_ots: usize,
    /// The port to bind to on localhost
    #[clap(short, long, default_value_t = 8066)]
    port: u16,
}

/// Example of the sender side
async fn sender(args: Args) -> (Vec<[Block; 2]>, usize, usize) {
    // Create a secure RNG to use in the protocol
    let mut rng = OsRng::default();
    // Create the ot extension sender. A base OT **receiver** is passed as an argument and used
    // to create the base_ots
    let mut sender = Sender::new(base_ot::Receiver);
    // Create a channel by listening on a socket address. Once another party connect, this
    // returns the channel
    let (mut base_sender, send_cnt, mut base_receiver, recv_cnt) =
        mpc_channel::tcp::listen::<mpc_channel::Receiver<_>>(("127.0.0.1", args.port))
            .await
            .expect("Error listening for channel connection");
    let (ch_sender, ch_receiver) = sub_channel(&mut base_sender, &mut base_receiver, 128)
        .await
        .expect("Establishing sub channel");

    // Perform the random ots
    let ots = sender
        .send_random(args.num_ots, &mut rng, ch_sender, ch_receiver)
        .await
        .expect("Failed to generate ROTs");
    (ots, send_cnt.get(), recv_cnt.get())
}

/// Example of the receiver side
async fn receiver(args: Args) -> (Vec<Block>, BitVec) {
    // Create a secure RNG to use in the protocol
    let mut rng = OsRng::default();
    // Create the ot extension receiver. A base OT **sender** is passed as an argument and used
    // to create the base_ots
    let mut receiver = Receiver::new(base_ot::Sender);
    let (mut base_sender, _, mut base_receiver, _) =
        mpc_channel::tcp::connect::<mpc_channel::Receiver<_>>(("127.0.0.1", args.port))
            .await
            .expect("Error listening for channel connection");
    let (ch_sender, ch_receiver) = sub_channel(&mut base_sender, &mut base_receiver, 128)
        .await
        .expect("Establishing sub channel");

    // Randomly choose one of the blocks
    let choices: BitVec = {
        let mut bv = bitvec![usize, Lsb0; 0; args.num_ots];
        rng.fill(bv.as_raw_mut_slice());
        bv
    };

    // Perform the random ot extension
    let ots = receiver
        .receive_random(&choices, &mut rng, ch_sender, ch_receiver)
        .await
        .expect("Failed to generate ROTs");
    (ots, choices)
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args: Args = Args::parse();
    let now = Instant::now();
    // Spawn the sender future
    let sender_fut = tokio::spawn(sender(args.clone()));
    // Ensure that the sender is listening for connections, in a real setting, the receiver
    // might try to reconnect if the sender is not listening yet
    tokio::time::sleep(Duration::from_millis(50)).await;
    // Spawn the receiver future
    let (receiver_ots, choices) = tokio::spawn(receiver(args.clone()))
        .await
        .expect("Error await receiver");
    let (sender_ots, send_cnt, recv_cnt) = sender_fut.await.expect("Error awaiting sender");
    println!(
        "Executed {} ots in {} ms. Sent bytes: {}, Recv bytes: {}",
        args.num_ots,
        now.elapsed().as_millis(),
        send_cnt,
        recv_cnt
    );

    // Assert that the random OTs have been generated correctly
    for ((recv, choice), [send1, send2]) in receiver_ots.into_iter().zip(choices).zip(sender_ots) {
        let [chosen, not_chosen] = if choice {
            [send2, send1]
        } else {
            [send1, send2]
        };
        assert_eq!(recv, chosen);
        assert_ne!(recv, not_chosen);
    }
}
