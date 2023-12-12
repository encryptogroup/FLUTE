//! Chou Orlandi base OT protocol.
use crate::traits::{BaseROTReceiver, BaseROTSender, Error};
use crate::util::Block;
use crate::{DefaultRom, Rom128};
use async_trait::async_trait;
use bitvec::macros::internal::funty::Fundamental;
use bitvec::slice::BitSlice;
use blake2::digest::Output;
use blake2::Digest;
use curve25519_dalek::constants::RISTRETTO_BASEPOINT_TABLE;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use rand::{CryptoRng, Rng, RngCore};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, Default, Clone)]
pub struct Sender;

#[derive(Debug, Default, Clone)]
pub struct Receiver;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum BaseOTMsg {
    First(RistrettoPoint, Output<DefaultRom>),
    Second(Vec<RistrettoPoint>),
    Third(Block),
}

impl Sender {
    pub fn new() -> Self {
        Sender
    }
}

impl Receiver {
    pub fn new() -> Self {
        Receiver
    }
}

#[async_trait]
impl BaseROTSender for Sender {
    type Msg = BaseOTMsg;

    #[allow(non_snake_case)]
    #[tracing::instrument(level = "debug", skip(self, rng, sender, receiver))]
    async fn send_random<RNG>(
        &mut self,
        count: usize,
        rng: &mut RNG,
        sender: mpc_channel::Sender<Self::Msg>,
        mut receiver: mpc_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<[Block; 2]>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send,
    {
        let a = Scalar::random(rng);
        let mut A = RISTRETTO_BASEPOINT_TABLE * &a;
        let seed: Block = rng.gen();
        // TODO: libOTE uses fixedKeyAES hash here, using Blake should be fine and not really
        //  impact performance
        let seed_comm = seed.rom_hash();
        sender
            .send(BaseOTMsg::First(A, seed_comm))
            .await
            .map_err(Error::Send)?;
        tracing::trace!("Send first msg");
        let msg = receiver
            .recv()
            .await
            .map_err(Error::Receive)?
            .ok_or(Error::UnexpectedTermination)?;
        tracing::trace!("Received second msg");
        let points = match msg {
            BaseOTMsg::Second(points) => points,
            msg => return Err(Error::WrongOrder(msg)),
        };
        if count != points.len() {
            return Err(Error::UnexpectedTermination);
        }
        sender
            .send(BaseOTMsg::Third(seed))
            .await
            .map_err(Error::Send)?;
        tracing::trace!("Send third msg");
        A *= a;
        let ots = points
            .into_iter()
            .enumerate()
            .map(|(i, mut B)| {
                B *= a;
                let k0 = rom_hash_point(&B, i, seed);
                B -= A;
                let k1 = rom_hash_point(&B, i, seed);
                [k0, k1]
            })
            .collect();
        Ok(ots)
    }
}

#[async_trait]
impl BaseROTReceiver for Receiver {
    type Msg = BaseOTMsg;

    #[allow(non_snake_case)]
    #[tracing::instrument(level = "debug", skip(self, rng, sender, receiver))]
    async fn receive_random<RNG>(
        &mut self,
        choices: &BitSlice,
        rng: &mut RNG,
        sender: mpc_channel::Sender<Self::Msg>,
        mut receiver: mpc_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<Block>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send,
    {
        let msg = receiver
            .recv()
            .await
            .map_err(Error::Receive)?
            .ok_or(Error::UnexpectedTermination)?;
        tracing::trace!("Received first msg");
        let (A, comm) = match msg {
            BaseOTMsg::First(A, comm) => (A, comm),
            msg => return Err(Error::WrongOrder(msg)),
        };
        let (bs, Bs): (Vec<_>, Vec<_>) = choices
            .iter()
            .map(|choice| {
                let b = Scalar::random(rng);
                let B_0 = RISTRETTO_BASEPOINT_TABLE * &b;
                let B = [B_0, A + B_0];
                (b, B[choice.as_usize()])
            })
            .unzip();
        sender
            .send(BaseOTMsg::Second(Bs))
            .await
            .map_err(Error::Send)?;
        tracing::trace!("Sent second msg");
        let msg = receiver
            .recv()
            .await
            .map_err(Error::Receive)?
            .ok_or(Error::UnexpectedTermination)?;
        let seed = match msg {
            BaseOTMsg::Third(seed) => seed,
            msg => return Err(Error::WrongOrder(msg)),
        };
        tracing::trace!("Received third msg");
        if comm != seed.rom_hash() {
            return Err(Error::ProtocolDeviation);
        }
        let ots = bs
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                let B = A * b;
                rom_hash_point(&B, i, seed)
            })
            .collect();
        Ok(ots)
    }
}

/// Hash a point and counter using the ROM.
fn rom_hash_point(point: &RistrettoPoint, counter: usize, seed: Block) -> Block {
    let mut rom = Rom128::new();
    rom.update(point.compress().as_bytes());
    rom.update(counter.to_le_bytes());
    rom.update(seed.to_le_bytes());
    let out = rom.finalize();
    Block::from_le_bytes(out.into())
}

#[cfg(test)]
mod tests {
    use crate::base_ot::{Receiver, Sender};
    use crate::traits::{BaseROTReceiver, BaseROTSender};
    use bitvec::bitvec;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[tokio::test]
    async fn base_rot() {
        let (ch1, ch2) = mpc_channel::in_memory::new_pair(128);
        let mut rng_send = StdRng::seed_from_u64(42);
        let mut rng_recv = StdRng::seed_from_u64(42 * 42);
        let mut sender = Sender;
        let mut receiver = Receiver;
        let send = sender.send_random(128, &mut rng_send, ch1.0, ch1.1);
        let choices = bitvec![0;128];
        let receive = receiver.receive_random(&choices, &mut rng_recv, ch2.0, ch2.1);

        let (sender_out, receiver_out) = tokio::try_join!(send, receive).unwrap();
        for (recv, [send, _]) in receiver_out.into_iter().zip(sender_out) {
            assert_eq!(recv, send);
        }
    }
}
