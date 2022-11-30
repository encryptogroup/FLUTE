//! Channel abstraction for communication
use crate::util::Counter;
use async_trait::async_trait;
use remoc::rch::{base, mpsc};
use remoc::{codec, RemoteSend};

pub use mpc_channel_macros::sub_channels_for;

pub mod in_memory;
pub mod tcp;
pub mod util;

pub type BaseSender<T> = base::Sender<T, codec::Bincode>;
pub type BaseReceiver<T> = base::Receiver<T, codec::Bincode>;

pub type Sender<T> = mpsc::Sender<T, codec::Bincode, 128>;
pub type Receiver<T> = mpsc::Receiver<T, codec::Bincode, 128>;

pub type TrackingChannel<T> = (BaseSender<T>, Counter, BaseReceiver<T>, Counter);
pub type Channel<T> = (Sender<T>, Receiver<T>);

#[async_trait]
pub trait SenderT<T, E> {
    async fn send(&mut self, item: T) -> Result<(), E>;
}

#[async_trait]
pub trait ReceiverT<T, E> {
    async fn recv(&mut self) -> Result<Option<T>, E>;
}

#[derive(thiserror::Error, Debug)]
pub enum CommunicationError {
    #[error("Error sending initial value")]
    BaseSend(base::SendErrorKind),
    #[error("Error receiving value on base channel")]
    BaseRecv(#[from] base::RecvError),
    #[error("Error sending value on mpsc channel")]
    Send(mpsc::SendError<()>),
    #[error("Error receiving value on mpsc channel")]
    Recv(#[from] mpsc::RecvError),
    #[error("Unexpected termination. Remote is closed.")]
    RemoteClosed,
    #[error("Received out of order message")]
    UnexpectedMessage,
}

pub fn channel<T: RemoteSend, const BUFFER: usize>(
    local_buffer: usize,
) -> (
    mpsc::Sender<T, remoc::codec::Bincode, BUFFER>,
    mpsc::Receiver<T, remoc::codec::Bincode, BUFFER>,
) {
    let (sender, receiver) = mpsc::channel(local_buffer);
    let sender = sender.set_buffer::<BUFFER>();
    let receiver = receiver.set_buffer::<BUFFER>();
    (sender, receiver)
}

#[tracing::instrument(skip_all)]
pub async fn sub_channel<Msg, SubMsg, SendErr, RecvErr>(
    sender: &mut impl SenderT<Msg, SendErr>,
    receiver: &mut impl ReceiverT<Msg, RecvErr>,
    local_buffer: usize,
) -> Result<(Sender<SubMsg>, Receiver<SubMsg>), CommunicationError>
where
    Receiver<SubMsg>: Into<Msg>,
    Msg: Into<Option<Receiver<SubMsg>>> + RemoteSend,
    SubMsg: RemoteSend,
    CommunicationError: From<SendErr> + From<RecvErr>,
{
    tracing::debug!("Establishing new sub_channel");
    let (sub_sender, remote_sub_receiver) = channel(local_buffer);
    sender.send(remote_sub_receiver.into()).await?;
    tracing::debug!("Sent remote_sub_receiver");
    let msg = receiver
        .recv()
        .await?
        .ok_or(CommunicationError::RemoteClosed)?;
    let sub_receiver = msg.into().ok_or(CommunicationError::UnexpectedMessage)?;
    tracing::debug!("Received sub_receiver");
    Ok((sub_sender, sub_receiver))
}

#[tracing::instrument(skip_all)]
pub async fn sub_channel_with<Msg, SubMsg, SendErr, RecvErr>(
    sender: &mut impl SenderT<Msg, SendErr>,
    receiver: &mut impl ReceiverT<Msg, RecvErr>,
    local_buffer: usize,
    wrap_fn: impl FnOnce(Receiver<SubMsg>) -> Msg,
    extract_fn: impl FnOnce(Msg) -> Option<Receiver<SubMsg>>,
) -> Result<(Sender<SubMsg>, Receiver<SubMsg>), CommunicationError>
where
    Msg: RemoteSend,
    SubMsg: RemoteSend,
    CommunicationError: From<SendErr> + From<RecvErr>,
{
    tracing::debug!("Establishing new sub_channel");
    let (sub_sender, remote_sub_receiver) = channel(local_buffer);
    sender.send(wrap_fn(remote_sub_receiver)).await?;
    tracing::debug!("Sent remote_sub_receiver");
    let msg = receiver
        .recv()
        .await?
        .ok_or(CommunicationError::RemoteClosed)?;
    let sub_receiver = extract_fn(msg).ok_or(CommunicationError::UnexpectedMessage)?;
    tracing::debug!("Received sub_receiver");
    Ok((sub_sender, sub_receiver))
}

#[async_trait]
impl<T, Codec> SenderT<T, base::SendError<T>> for base::Sender<T, Codec>
where
    T: RemoteSend,
    Codec: codec::Codec,
{
    async fn send(&mut self, item: T) -> Result<(), base::SendError<T>> {
        base::Sender::send(self, item).await
    }
}

#[async_trait]
impl<T, Codec> ReceiverT<T, base::RecvError> for base::Receiver<T, Codec>
where
    T: RemoteSend,
    Codec: codec::Codec,
{
    async fn recv(&mut self) -> Result<Option<T>, base::RecvError> {
        base::Receiver::recv(self).await
    }
}

#[async_trait]
impl<T, Codec, const BUFFER: usize> SenderT<T, mpsc::SendError<T>>
    for mpsc::Sender<T, Codec, BUFFER>
where
    T: RemoteSend,
    Codec: codec::Codec,
{
    async fn send(&mut self, item: T) -> Result<(), mpsc::SendError<T>> {
        mpsc::Sender::send(self, item).await
    }
}

#[async_trait]
impl<T, Codec, const BUFFER: usize> ReceiverT<T, mpsc::RecvError>
    for mpsc::Receiver<T, Codec, BUFFER>
where
    T: RemoteSend,
    Codec: codec::Codec,
{
    async fn recv(&mut self) -> Result<Option<T>, mpsc::RecvError> {
        mpsc::Receiver::recv(self).await
    }
}

impl<T> From<base::SendError<T>> for CommunicationError {
    fn from(err: base::SendError<T>) -> Self {
        CommunicationError::BaseSend(err.kind)
    }
}

impl<T> From<mpsc::SendError<T>> for CommunicationError {
    fn from(err: mpsc::SendError<T>) -> Self {
        CommunicationError::Send(err.without_item())
    }
}
