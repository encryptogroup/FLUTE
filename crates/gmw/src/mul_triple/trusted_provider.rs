//! Trusted Multiplication Triple Provider.
//!
//! This module implements a very basic trusted multiplication provider client/server.
//! The [`TrustedMTProviderClient`] is used to connect to a [`TrustedMTProviderServer`]. When
//! [`MTProvider::request_mts`] is called on the client, a request is sent to the server. Upon
//! receiving it, the server generates random multiplication triples by calling
//! [`MulTriples::random_pair`] and returns one [`MulTriples`] struct to each party.
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use tokio::net::ToSocketAddrs;
use tokio::sync::Mutex;
use tracing::error;

use crate::errors::MTProviderError;
use crate::mul_triple::{MTProvider, MulTriples};
use mpc_channel::{BaseReceiver, BaseSender};

pub struct TrustedMTProviderClient {
    id: String,
    sender: BaseSender<Message>,
    receiver: BaseReceiver<Message>,
}

pub struct TrustedMTProviderServer {
    sender: BaseSender<Message>,
    receiver: BaseReceiver<Message>,
    mts: Arc<Mutex<HashMap<String, MulTriples>>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Message {
    RequestTriples { id: String, amount: usize },
    MulTriples(MulTriples),
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
            Message::MulTriples(mts) => Ok(mts),
            _ => Err(MTProviderError::IllegalMessage),
        }
    }
}

impl TrustedMTProviderServer {
    pub fn new(sender: BaseSender<Message>, receiver: BaseReceiver<Message>) -> Self {
        Self {
            sender,
            receiver,
            mts: Default::default(),
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
        let mut mts = self.mts.lock().await;
        let mt = match mts.entry(id) {
            Entry::Vacant(vacant) => {
                // TODO `random` call might be blocking, better use rayon here
                //  Note: It's fine for the moment, as the Server is not really able to utilize
                //  parallelization anyway, due to the lock
                let [mt1, mt2] = MulTriples::random_pair(amount, &mut thread_rng());
                vacant.insert(mt1);
                mt2
            }
            Entry::Occupied(occupied) => occupied.remove(),
        };
        self.sender.send(Message::MulTriples(mt)).await?;
        Ok(())
    }

    #[tracing::instrument(skip(self))]
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
                    mts: data,
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
