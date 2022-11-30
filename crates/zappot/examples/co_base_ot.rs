//! Example - Chou Orlandi base OT
//!
//! This example shows how to use the zappot crate to execute the Chou Orlandi base OT protocol.
use bitvec::vec::BitVec;
use clap::Parser;
use mpc_channel::sub_channel;
use rand::{distributions, Rng};
use rand_core::OsRng;
use std::time::Duration;
use tokio::time::Instant;
use zappot::base_ot::{Receiver, Sender};
use zappot::traits::{BaseROTReceiver, BaseROTSender};
use zappot::util::Block;

#[derive(Parser, Debug, Clone)]
struct Args {
    /// Number of OTs to execute
    #[clap(short, long, default_value_t = 128)]
    num_ots: usize,
    /// The port to bind to on localhost
    #[clap(short, long, default_value_t = 8066)]
    port: u16,
}

/// Example of the sender side
async fn sender(args: Args) -> Vec<[Block; 2]> {
    // Create a secure RNG to use in the protocol
    let mut rng = OsRng::default();
    let mut sender = Sender::new();
    // Create a channel by listening on a socket address. Once another party connect, this
    // returns the channel
    let (mut base_sender, _, mut base_receiver, _) =
        mpc_channel::tcp::listen::<mpc_channel::Receiver<_>>(("127.0.0.1", args.port))
            .await
            .expect("Error listening for channel connection");
    let (ch_sender, ch_receiver) = sub_channel(&mut base_sender, &mut base_receiver, 8)
        .await
        .expect("Establishing sub channel");
    // Perform the random ots
    sender
        .send_random(args.num_ots, &mut rng, ch_sender, ch_receiver)
        .await
        .expect("Failed to generate ROTs")
}

/// Example of the receiver side
async fn receiver(args: Args) -> (Vec<Block>, BitVec) {
    // Create a secure RNG to use in the protocol
    let mut rng = OsRng::default();
    // Create the receiver. The struct holds no state
    let mut receiver = Receiver::new();
    // Connect to the sender on the listened on port
    let (mut base_sender, _, mut base_receiver, _) =
        mpc_channel::tcp::connect::<mpc_channel::Receiver<_>>(("127.0.0.1", args.port))
            .await
            .expect("Error listening for channel connection");
    let (ch_sender, ch_receiver) = sub_channel(&mut base_sender, &mut base_receiver, 8)
        .await
        .expect("Establishing sub channel");

    // Randomly choose one of the blocks
    let choices: BitVec = rng
        .sample_iter::<bool, _>(distributions::Standard)
        .take(args.num_ots)
        .collect();

    // Perform the random ots
    let ots = receiver
        .receive_random(&choices, &mut rng, ch_sender, ch_receiver)
        .await
        .expect("Failed to generate ROTs");
    (ots, choices)
}

#[tokio::main]
async fn main() {
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
    let sender_ots = sender_fut.await.expect("Error awaiting sender");
    println!(
        "Executed {} ots in {} ms",
        args.num_ots,
        now.elapsed().as_millis()
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
