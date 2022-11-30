//! Networking utilities.
use bytes::Bytes;
use futures::{Sink, Stream};
use pin_project::pin_project;
use std::io::{Error, IoSlice};
use std::ops::AddAssign;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::{io, mem};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

/// [AsyncWriter](`AsyncWrite`) that tracks the number of bytes written.
#[pin_project]
pub struct TrackingWriter<AsyncWriter> {
    #[pin]
    writer: AsyncWriter,
    bytes_written: Counter,
}

/// [AsyncReader](`AsyncRead`) that tracks the number of bytes read.
#[pin_project]
pub struct TrackingReader<AsyncReader> {
    #[pin]
    reader: AsyncReader,
    bytes_read: Counter,
}

#[derive(Clone, Default, Debug)]
pub struct Counter(Arc<AtomicUsize>);

impl<AsyncWriter> TrackingWriter<AsyncWriter> {
    pub fn new(writer: AsyncWriter) -> Self {
        Self {
            writer,
            bytes_written: Counter::default(),
        }
    }

    #[inline]
    pub fn bytes_written(&self) -> Counter {
        self.bytes_written.clone()
    }

    pub fn reset(&mut self) {
        self.bytes_written.reset();
    }
}

impl<AsyncReader> TrackingReader<AsyncReader> {
    pub fn new(reader: AsyncReader) -> Self {
        Self {
            reader,
            bytes_read: Counter::default(),
        }
    }

    #[inline]
    pub fn bytes_read(&self) -> Counter {
        self.bytes_read.clone()
    }

    pub fn reset(&mut self) {
        self.bytes_read.reset();
    }
}

impl<AW: AsyncWrite> AsyncWrite for TrackingWriter<AW> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, Error>> {
        let this = self.project();
        let poll = this.writer.poll_write(cx, buf);
        if let Poll::Ready(Ok(bytes_written)) = &poll {
            *this.bytes_written += *bytes_written;
        }
        poll
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
        let this = self.project();
        this.writer.poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
        let this = self.project();
        this.writer.poll_shutdown(cx)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<Result<usize, Error>> {
        let this = self.project();
        let poll = this.writer.poll_write_vectored(cx, bufs);
        if let Poll::Ready(Ok(bytes_written)) = &poll {
            *this.bytes_written += *bytes_written;
        }
        poll
    }

    fn is_write_vectored(&self) -> bool {
        self.writer.is_write_vectored()
    }
}

impl<AR: AsyncRead> AsyncRead for TrackingReader<AR> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let bytes_before = buf.filled().len();
        let this = self.project();
        let poll = this.reader.poll_read(cx, buf);
        *this.bytes_read += buf.filled().len() - bytes_before;
        poll
    }
}

impl<S: Sink<Bytes>> Sink<Bytes> for TrackingWriter<S> {
    type Error = S::Error;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.writer.poll_ready(cx)
    }

    fn start_send(self: Pin<&mut Self>, item: Bytes) -> Result<(), Self::Error> {
        // The size_of<u32> adds the size of the length tag which we'd use when actually
        // using a framed transport
        let this = self.project();
        *this.bytes_written += item.len() + mem::size_of::<u32>();
        this.writer.start_send(item)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.writer.poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.writer.poll_close(cx)
    }
}

impl<S: Stream<Item = Bytes>> Stream for TrackingReader<S> {
    type Item = Bytes;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        let poll = this.reader.poll_next(cx);
        if let Poll::Ready(Some(bytes)) = &poll {
            // The size_of<u32> adds the size of the length tag which we'd use when actually
            // using a framed transport
            *this.bytes_read += bytes.len() + mem::size_of::<u32>();
        }
        poll
    }
}

impl Counter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }

    pub fn reset(&self) -> usize {
        self.0.swap(0, Ordering::SeqCst)
    }
}

impl AddAssign<usize> for Counter {
    fn add_assign(&mut self, rhs: usize) {
        self.0.fetch_add(rhs, Ordering::SeqCst);
    }
}
