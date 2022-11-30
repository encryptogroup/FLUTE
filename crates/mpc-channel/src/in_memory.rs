use crate::{channel, Channel};
use remoc::RemoteSend;

pub fn new_pair<T: RemoteSend>(local_buffer: usize) -> (Channel<T>, Channel<T>) {
    let (sender1, receiver1) = channel(local_buffer);
    let (sender2, receiver2) = channel(local_buffer);

    ((sender1, receiver2), (sender2, receiver1))
}
