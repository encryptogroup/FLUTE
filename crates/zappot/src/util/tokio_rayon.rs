//! Tokio + Rayon compatibility
//!
//! This implementation is adapted from <https://github.com/andybarron/tokio-rayon>
use rayon::ThreadPool;
use std::future::Future;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread;
use tokio::sync::oneshot;

/// Spawn a compute intensive function into the global rayon threadpool.
pub fn spawn_compute<F, R>(func: F) -> AsyncRayonHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = oneshot::channel();
    rayon::spawn(move || {
        let ret = catch_unwind(AssertUnwindSafe(func));
        // Ignore error as this means the receiver has been dropped and the result is not needed
        // anymore
        let _res = tx.send(ret);
    });

    AsyncRayonHandle { rx }
}

/// Async handle for a blocking task running in a Rayon thread pool.
///
/// If the spawned task panics, `poll()` will propagate the panic.
#[must_use]
#[derive(Debug)]
pub struct AsyncRayonHandle<T> {
    pub(crate) rx: oneshot::Receiver<thread::Result<T>>,
}

impl<T> Future for AsyncRayonHandle<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let rx = Pin::new(&mut self.rx);
        rx.poll(cx).map(|result| {
            result
                // The expect should never fail, because panics in the compute function are caught and the
                // Sender should never be dropped before sending a value
                .expect("Unreachable error: Tokio channel closed")
                .unwrap_or_else(|err| resume_unwind(err))
        })
    }
}

pub trait AsyncThreadPool: private::Sealed {
    /// Asynchronous wrapper around Rayon's
    /// [`ThreadPool::spawn`](rayon::ThreadPool::spawn).
    ///
    /// Runs a function on the global Rayon thread pool with LIFO priority,
    /// produciing a future that resolves with the function's return value.
    ///
    /// # Panics
    /// If the task function panics, the panic will be propagated through the
    /// returned future. This will NOT trigger the Rayon thread pool's panic
    /// handler.
    ///
    /// If the returned handle is dropped, and the return value of `func` panics
    /// when dropped, that panic WILL trigger the thread pool's panic
    /// handler.
    fn spawn_compute<F, R>(&self, func: F) -> AsyncRayonHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;

    /// This is a combination of [`ThreadPool::spawn`](`ThreadPool::spawn`) and
    /// [`ThreadPool::install`](ThreadPool::install). This means, that not only the passed closure
    /// is run on the threadpool, but also all calls to parallel iterators which are performed by
    /// the closure.
    ///
    /// In order to support this, the ThreadPool must be inside an [`Arc`](`Arc`).
    fn spawn_install_compute<F, R>(self: Arc<Self>, func: F) -> AsyncRayonHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;
}

impl AsyncThreadPool for ThreadPool {
    fn spawn_compute<F, R>(&self, func: F) -> AsyncRayonHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        self.spawn(move || {
            let ret = catch_unwind(AssertUnwindSafe(func));
            // Ignore error as this means the receiver has been dropped and the result is not needed
            // anymore
            let _res = tx.send(ret);
        });

        AsyncRayonHandle { rx }
    }

    fn spawn_install_compute<F, R>(self: Arc<Self>, func: F) -> AsyncRayonHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let this = Arc::clone(&self);
        let (tx, rx) = oneshot::channel();
        self.spawn(move || {
            this.install(move || {
                let ret = catch_unwind(AssertUnwindSafe(func));
                // Ignore error as this means the receiver has been dropped and the result is not needed
                // anymore
                let _res = tx.send(ret);
            });
        });

        AsyncRayonHandle { rx }
    }
}

mod private {
    use rayon::ThreadPool;

    pub trait Sealed {}

    impl Sealed for ThreadPool {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::{ThreadPool, ThreadPoolBuilder};

    fn build_thread_pool() -> ThreadPool {
        ThreadPoolBuilder::new().num_threads(1).build().unwrap()
    }

    #[tokio::test]
    #[should_panic(expected = "Task failed successfully")]
    async fn test_poll_propagates_panic() {
        let panic_err = catch_unwind(|| {
            panic!("Task failed successfully");
        })
        .unwrap_err();

        let (tx, rx) = oneshot::channel::<thread::Result<()>>();
        let handle = AsyncRayonHandle { rx };
        tx.send(Err(panic_err)).unwrap();
        handle.await;
    }

    #[tokio::test]
    #[should_panic(expected = "Unreachable error: Tokio channel closed")]
    async fn test_unreachable_channel_closed() {
        let (_, rx) = oneshot::channel::<thread::Result<()>>();
        let handle = AsyncRayonHandle { rx };
        handle.await;
    }

    #[tokio::test]
    async fn test_spawn_compute_works() {
        let pool = build_thread_pool();
        let result = pool
            .spawn_compute(|| {
                let thread_index = rayon::current_thread_index();
                assert_eq!(thread_index, Some(0));
                1337_usize
            })
            .await;
        assert_eq!(result, 1337);
        let thread_index = rayon::current_thread_index();
        assert_eq!(thread_index, None);
    }

    #[tokio::test]
    #[should_panic(expected = "Task failed successfully")]
    async fn test_spawn_compute_propagates_panic() {
        let pool = build_thread_pool();
        let handle = pool.spawn_compute(|| {
            panic!("Task failed successfully");
        });

        handle.await;
    }
}
