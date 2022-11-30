use crate::common::BitVec;
use crate::mul_triple::{MTProvider, MulTriples};
use async_trait::async_trait;
use num_integer::Integer;
use rand::{CryptoRng, RngCore, SeedableRng};
use remoc::RemoteSend;
use zappot::traits::{ExtROTReceiver, ExtROTSender};
use zappot::util::aes_rng::AesRng;

use crate::utils::rand_bitvec;
use std::fmt::Debug;

pub struct OtMTProvider<RNG, S: ExtROTSender, R: ExtROTReceiver> {
    rng: RNG,
    ot_sender: S,
    ot_receiver: R,
    ch_sender: mpc_channel::Sender<mpc_channel::Receiver<S::Msg>>,
    ch_receiver: mpc_channel::Receiver<mpc_channel::Receiver<S::Msg>>,
}

impl<RNG: RngCore + CryptoRng + Send, S: ExtROTSender, R: ExtROTReceiver> OtMTProvider<RNG, S, R> {
    pub fn new(
        rng: RNG,
        ot_sender: S,
        ot_receiver: R,
        ch_sender: mpc_channel::Sender<mpc_channel::Receiver<S::Msg>>,
        ch_receiver: mpc_channel::Receiver<mpc_channel::Receiver<S::Msg>>,
    ) -> Self {
        Self {
            rng,
            ot_sender,
            ot_receiver,
            ch_sender,
            ch_receiver,
        }
    }
}

#[async_trait]
impl<RNG, S, R> MTProvider for OtMTProvider<RNG, S, R>
where
    RNG: RngCore + CryptoRng + Send,
    S: ExtROTSender<Msg = R::Msg> + Send,
    S::Msg: RemoteSend + Debug,
    R: ExtROTReceiver + Send,
    R::Msg: RemoteSend + Debug,
{
    type Output = MulTriples;
    type Error = ();

    async fn request_mts(&mut self, amount: usize) -> Result<MulTriples, Self::Error> {
        let mut sender_rng = AesRng::from_rng(&mut self.rng).unwrap();
        let mut receiver_rng = AesRng::from_rng(&mut self.rng).unwrap();

        let amount = Integer::next_multiple_of(&amount, &8);

        let (ch_sender1, ch_receiver1) =
            mpc_channel::sub_channel(&mut self.ch_sender, &mut self.ch_receiver, 128)
                .await
                .unwrap();
        let (ch_sender2, ch_receiver2) =
            mpc_channel::sub_channel(&mut self.ch_sender, &mut self.ch_receiver, 128)
                .await
                .unwrap();

        let send = self
            .ot_sender
            .send_random(amount, &mut sender_rng, ch_sender1, ch_receiver2);

        let a_i = rand_bitvec(amount, &mut receiver_rng);
        let receive = self.ot_receiver.receive_random(
            a_i.as_bitslice(),
            &mut receiver_rng,
            ch_sender2,
            ch_receiver1,
        );

        let (send_ots, recv_ots) = tokio::try_join!(send, receive).unwrap();

        let mut b_i = BitVec::with_capacity(amount);
        let mut v_i: BitVec<usize> = BitVec::with_capacity(amount);

        send_ots
            .into_iter()
            .map(|arr| arr.map(|b| b.lsb()))
            .for_each(|[m0, m1]| {
                b_i.push(m0 ^ m1);
                v_i.push(m0);
            });
        let u_i = recv_ots.into_iter().map(|b| b.lsb());
        let c_i = a_i
            .iter()
            .by_vals()
            .zip(b_i.iter().by_vals())
            .zip(u_i)
            .zip(v_i)
            .map(|(((a, b), u), v)| a & b ^ u ^ v)
            .collect();

        Ok(MulTriples::from_raw(a_i, b_i, c_i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::private_test_utils::init_tracing;
    use rand::rngs::OsRng;
    use zappot::ot_ext;

    #[tokio::test]
    async fn ot_ext_provider() {
        let _guard = init_tracing();
        let ((ch_sender1, ch_receiver1), (ch_sender2, ch_receiver2)) =
            mpc_channel::in_memory::new_pair(8);

        let party = |ch_sender, ch_receiver| async {
            let ot_sender = ot_ext::Sender::default();
            let ot_receiver = ot_ext::Receiver::default();

            let mut mtp = OtMTProvider::new(OsRng, ot_sender, ot_receiver, ch_sender, ch_receiver);
            mtp.request_mts(1024).await.unwrap()
        };

        let (mts1, mts2) = tokio::join!(
            party(ch_sender1, ch_receiver1),
            party(ch_sender2, ch_receiver2)
        );

        assert_eq!(mts1.c ^ mts2.c, (mts1.a ^ mts2.a) & (mts1.b ^ mts2.b))
    }
}
