//! ALSZ13 OT extension protocol.
use crate::traits::{BaseROTReceiver, BaseROTSender, Error, ExtROTReceiver, ExtROTSender};
use crate::util::aes_hash::FIXED_KEY_HASH;
use crate::util::aes_rng::AesRng;
use crate::util::tokio_rayon::spawn_compute;
use crate::util::transpose::transpose;
use crate::util::Block;
use crate::{base_ot, BASE_OT_COUNT};
use async_trait::async_trait;
use bitvec::bitvec;
use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use bytemuck::cast_slice;
use mpc_channel::channel;
use rand::{CryptoRng, Rng, RngCore};
use rand_core::SeedableRng;
use rayon::prelude::*;
use remoc::RemoteSend;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub struct Sender<BaseOT> {
    base_ot: BaseOT,
}

pub struct Receiver<BaseOT> {
    base_ot: BaseOT,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum ExtOTMsg<BaseOTMsg: RemoteSend = base_ot::BaseOTMsg> {
    // Workaround for compiler bug,
    // see https://github.com/serde-rs/serde/issues/1296#issuecomment-394056188
    #[serde(bound = "")]
    BaseOTChannel(mpc_channel::Receiver<BaseOTMsg>),
    URow(usize, Vec<u8>),
}

#[async_trait]
impl<BaseOT> ExtROTSender for Sender<BaseOT>
where
    BaseOT: BaseROTReceiver + Send,
    BaseOT::Msg: RemoteSend + Debug,
{
    type Msg = ExtOTMsg<BaseOT::Msg>;

    #[allow(non_snake_case)]
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
        assert_eq!(
            count % 8,
            0,
            "Number of OT extensions must be multiple of 8"
        );
        let (base_ots, choices) = {
            let (base_sender, base_remote_receiver) = channel(BASE_OT_COUNT);
            sender
                .send(ExtOTMsg::BaseOTChannel(base_remote_receiver))
                .await?;
            let msg = receiver.recv().await?.ok_or(Error::UnexpectedTermination)?;
            let base_receiver = match msg {
                ExtOTMsg::BaseOTChannel(receiver) => receiver,
                _ => return Err(Error::WrongOrder(msg)),
            };
            let rand_choices: BitVec = {
                let mut bv = bitvec![0; BASE_OT_COUNT];
                rng.fill(bv.as_raw_mut_slice());
                bv
            };
            let base_ots = self
                .base_ot
                .receive_random(&rand_choices, rng, base_sender, base_receiver)
                .await
                .map_err(|err| Error::BaseOT(Box::new(err)))?;
            (base_ots, rand_choices)
        };

        let delta: Block = (&choices)
            .try_into()
            .expect("BASE_OT_COUNT must be size of a Block");
        let rows = BASE_OT_COUNT;
        let cols = count / 8; // div by 8 because of u8
        let mut v_mat = spawn_compute(move || {
            let mut v_mat = vec![0_u8; rows * cols];
            v_mat
                .chunks_exact_mut(cols)
                .zip(base_ots)
                .for_each(|(row, seed)| {
                    let mut prg = AesRng::from_seed(seed);
                    prg.fill_bytes(row);
                });
            v_mat
        })
        .await;
        let mut rows_received = 0;
        while let Some(msg) = receiver.recv().await.transpose() {
            let (idx, mut u_row) = match msg.map_err(Error::Receive)? {
                ExtOTMsg::URow(idx, row) => (idx, row),
                msg => return Err(Error::WrongOrder(msg)),
            };
            let r = choices[idx];
            let v_row = &mut v_mat[idx * cols..(idx + 1) * cols];
            for el in &mut u_row {
                // computes r_j * u_j
                // TODO cleanup, also const time?
                *el = if r { *el } else { 0 };
            }
            v_row.iter_mut().zip(u_row).for_each(|(v, u)| {
                *v ^= u;
            });
            rows_received += 1;
            if rows_received == rows {
                break;
            }
        }

        let ots = spawn_compute(move || {
            let v_mat = transpose(&v_mat, rows, count);
            v_mat
                // TODO benchmark parallelization
                .par_chunks_exact(BASE_OT_COUNT / u8::BITS as usize)
                .map(|row| {
                    let block = row
                        .try_into()
                        .expect("message size must be block length (128 bits)");
                    let x_0 = FIXED_KEY_HASH.cr_hash_block(block);
                    let x_1 = FIXED_KEY_HASH.cr_hash_block(block ^ delta);
                    [x_0, x_1]
                })
                .collect()
        })
        .await;
        Ok(ots)
    }
}

// fn assert_static<T: 'static>(val: &T) {}

#[async_trait]
impl<BaseOT> ExtROTReceiver for Receiver<BaseOT>
where
    BaseOT: BaseROTSender + Send,
    BaseOT::Msg: RemoteSend + Debug,
{
    type Msg = ExtOTMsg<BaseOT::Msg>;

    #[allow(non_snake_case)]
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
        assert_eq!(
            choices.len() % 8,
            0,
            "Number of OT extensions must be multiple of 8"
        );
        let count = choices.len();
        let base_ots = {
            let (base_sender, base_remote_receiver) = channel(BASE_OT_COUNT);
            sender
                .send(ExtOTMsg::BaseOTChannel(base_remote_receiver))
                .await?;
            let msg = receiver.recv().await?.ok_or(Error::UnexpectedTermination)?;
            let base_receiver = match msg {
                ExtOTMsg::BaseOTChannel(receiver) => receiver,
                _ => return Err(Error::WrongOrder(msg)),
            };
            self.base_ot
                .send_random(BASE_OT_COUNT, rng, base_sender, base_receiver)
                .await
                .map_err(|err| Error::BaseOT(Box::new(err)))?
        };

        let rows = BASE_OT_COUNT;
        let cols = count / 8; // div by 8 because of u8

        let choices = choices.to_bitvec();
        let sender = sender.clone();
        let t_mat = spawn_compute(move || {
            let choices = cast_slice::<_, u8>(choices.as_raw_slice());
            let mut t_mat = vec![0_u8; rows * cols];
            t_mat
                .par_chunks_exact_mut(cols)
                .enumerate()
                .zip(base_ots)
                .for_each(|((idx, t_row), [s0, s1])| {
                    let mut prg0 = AesRng::from_seed(s0);
                    let mut prg1 = AesRng::from_seed(s1);
                    prg0.fill_bytes(t_row);
                    let u_row = {
                        let mut row = vec![0_u8; cols];
                        prg1.fill_bytes(&mut row);
                        row.iter_mut().zip(t_row).zip(choices).for_each(
                            |((val, rand_val), choice)| {
                                *val ^= *rand_val ^ choice;
                            },
                        );
                        row
                    };
                    sender
                        .blocking_send(ExtOTMsg::URow(idx, u_row))
                        .expect("URow send failed");
                });
            t_mat
        })
        .await;

        let ots = spawn_compute(move || {
            let t_mat = transpose(&t_mat, rows, count);
            t_mat
                // TODO parallelize this code
                .par_chunks_exact(BASE_OT_COUNT / u8::BITS as usize)
                .map(|rows| {
                    let block = rows
                        .try_into()
                        .expect("message size must be block length (128 bits)");
                    FIXED_KEY_HASH.cr_hash_block(block)
                })
                .collect()
        })
        .await;
        Ok(ots)
    }
}

impl<BaseOt> Sender<BaseOt> {
    pub fn new(base_ot_receiver: BaseOt) -> Self {
        Self {
            base_ot: base_ot_receiver,
        }
    }
}

impl<BaseOt> Receiver<BaseOt> {
    pub fn new(base_ot_sender: BaseOt) -> Self {
        Self {
            base_ot: base_ot_sender,
        }
    }
}

impl Default for Sender<base_ot::Receiver> {
    fn default() -> Self {
        Sender::new(base_ot::Receiver)
    }
}

impl Default for Receiver<base_ot::Sender> {
    fn default() -> Self {
        Receiver::new(base_ot::Sender)
    }
}

#[cfg(test)]
mod tests {
    use crate::base_ot;
    use crate::ot_ext::{Receiver, Sender};
    use crate::traits::{ExtROTReceiver, ExtROTSender};
    use bitvec::bitvec;
    use bitvec::order::Lsb0;

    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use tokio::time::Instant;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ot_ext() {
        let (ch1, ch2) = mpc_channel::in_memory::new_pair(128);
        let num_ots: usize = 1000;
        let now = Instant::now();
        let send = tokio::spawn(async move {
            let mut sender = Sender::new(base_ot::Receiver {});
            let mut rng_send = StdRng::seed_from_u64(42);
            sender
                .send_random(num_ots, &mut rng_send, ch1.0, ch1.1)
                .await
                .unwrap()
        });
        let choices = bitvec![usize, Lsb0; 0;num_ots];
        let receive = tokio::spawn(async move {
            let mut receiver = Receiver::new(base_ot::Sender {});
            let mut rng_recv = StdRng::seed_from_u64(42 * 42);
            receiver
                .receive_random(&choices, &mut rng_recv, ch2.0, ch2.1)
                .await
                .unwrap()
        });
        let (recv, sent) = tokio::try_join!(receive, send).unwrap();
        println!("Total time: {}", now.elapsed().as_secs_f32());
        for (r, [s, _]) in recv.into_iter().zip(sent) {
            assert_eq!(r, s)
        }
    }
}
