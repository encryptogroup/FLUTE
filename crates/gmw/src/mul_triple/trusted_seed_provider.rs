//! Trusted Seed MT Provider.
//!
//! This module implements a trusted third party MT provider according to the
//! [Chameleon paper](https://dl.acm.org/doi/pdf/10.1145/3196494.3196522). The third party
//! generates two random seeds and derives the MTs from them. The first party gets the first seed,
//! while the second party receives the second seed and the `c` values for their MTs.
//! For a visualization of the protocol, look at Figure 1 of the linked paper.
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use rand::{random, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use tokio::net::ToSocketAddrs;
use tokio::sync::Mutex;
use tracing::error;

use crate::common::BitVec;
use crate::errors::MTProviderError;
use crate::mul_triple::{compute_c_owned, MTProvider, MulTriples};
use crate::utils::rand_bitvecs;
use mpc_channel::{BaseReceiver, BaseSender};

pub struct TrustedMTProviderClient {
    id: String,
    sender: BaseSender<Message>,
    receiver: BaseReceiver<Message>,
}

// TODO: Which prng to choose? Context: https://github.com/rust-random/rand/issues/932
//  using ChaCha with 8 rounds is likely to be secure enough and would provide a little more
//  performance. This should be benchmarked however
type MtRng = ChaCha12Rng;
pub type MtRngSeed = <MtRng as SeedableRng>::Seed;

pub struct TrustedMTProviderServer {
    sender: BaseSender<Message>,
    receiver: BaseReceiver<Message>,
    seeds: Arc<Mutex<HashMap<String, MtRngSeed>>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Message {
    RequestTriples { id: String, amount: usize },
    Seed(MtRngSeed),
    SeedAndC { seed: MtRngSeed, c: BitVec<usize> },
}

impl TrustedMTProviderClient {
    pub fn new(id: String, sender: BaseSender<Message>, receiver: BaseReceiver<Message>) -> Self {
        Self {
            id,
            sender,
            receiver,
        }
    }
}

#[async_trait]
impl MTProvider for TrustedMTProviderClient {
    type Output = MulTriples;
    type Error = MTProviderError<Message>;

    async fn request_mts(&mut self, amount: usize) -> Result<MulTriples, Self::Error> {
        self.sender
            .send(Message::RequestTriples {
                id: self.id.clone(),
                amount,
            })
            .await?;
        let msg: Message = self
            .receiver
            .recv()
            .await?
            .ok_or(MTProviderError::RemoteClosed)?;
        match msg {
            Message::Seed(seed) => {
                let mut rng = MtRng::from_seed(seed);
                Ok(MulTriples::random(amount, &mut rng))
            }
            Message::SeedAndC { seed, c } => {
                let mut rng = MtRng::from_seed(seed);
                Ok(MulTriples::random_with_fixed_c(c, &mut rng))
            }
            _ => Err(MTProviderError::IllegalMessage),
        }
    }
}

impl TrustedMTProviderServer {
    pub fn new(sender: BaseSender<Message>, receiver: BaseReceiver<Message>) -> Self {
        Self {
            sender,
            receiver,
            seeds: Default::default(),
        }
    }
}

impl TrustedMTProviderServer {
    #[tracing::instrument(skip(self), err(Debug))]
    async fn handle_request(
        &mut self,
        id: String,
        amount: usize,
    ) -> Result<(), MTProviderError<Message>> {
        let mut seeds = self.seeds.lock().await;
        match seeds.entry(id) {
            Entry::Vacant(vacant) => {
                let seed1 = random();
                let seed2 = random();
                vacant.insert(seed1);
                let mut rng1 = MtRng::from_seed(seed1);
                let mut rng2 = MtRng::from_seed(seed2);
                let mts = MulTriples::random(amount, &mut rng1);
                let [a, b] = rand_bitvecs(amount, &mut rng2);
                let c = compute_c_owned(mts, a, b);
                self.sender
                    .send(Message::SeedAndC { seed: seed2, c })
                    .await?;
            }
            Entry::Occupied(occupied) => {
                let seed = occupied.remove();
                self.sender.send(Message::Seed(seed)).await?;
            }
        };
        Ok(())
    }

    async fn handle_conn(mut self) {
        loop {
            match self.receiver.recv().await {
                Ok(Some(Message::RequestTriples { id, amount })) => {
                    if let Err(err) = self.handle_request(id, amount).await {
                        error!(%err, "Error handling request");
                    }
                }
                Ok(None) => break,
                Ok(_other) => error!("Server received illegal msg"),
                Err(err) => {
                    error!(%err, "Error handling connection");
                }
            }
        }
    }

    #[tracing::instrument]
    pub async fn start(addr: impl ToSocketAddrs + Debug) -> Result<(), io::Error> {
        let data = Default::default();
        mpc_channel::tcp::server(addr)
            .await?
            .for_each(|channel| async {
                let (sender, receiver) = match channel {
                    Err(err) => {
                        error!(%err, "Encountered error when establishing connection");
                        return;
                    }
                    Ok((sender, _, receiver, _)) => (sender, receiver),
                };

                let data = Arc::clone(&data);
                let mt_server = Self {
                    seeds: data,
                    sender,
                    receiver,
                };
                tokio::spawn(async {
                    mt_server.handle_conn().await;
                });
            })
            .await;
        Ok(())
    }
}
