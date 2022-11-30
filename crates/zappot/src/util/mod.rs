//! Cryptographic utilities.
pub mod aes_hash;
pub mod aes_rng;
pub mod block;
pub mod tokio_rayon;
pub mod transpose;

pub use block::Block;

#[cfg(any(feature = "silent_ot", test))]
pub(crate) fn log2_floor(val: usize) -> u32 {
    assert!(val > 0);
    usize::BITS - val.leading_zeros() - 1
}

#[cfg(any(feature = "silent_ot", test))]
pub(crate) fn log2_ceil(val: usize) -> u32 {
    let floor = log2_floor(val);
    if val > (1 << floor) {
        floor + 1
    } else {
        floor
    }
}

#[cfg(test)]
mod test {
    use crate::util::{log2_ceil, log2_floor};

    #[test]
    fn log2() {
        assert_eq!(log2_floor(2_usize.pow(24)), 24);
        assert_eq!(log2_floor(2_usize.pow(24) + 42), 24);
        assert_eq!(log2_ceil(2_usize.pow(24)), 24);
        assert_eq!(log2_ceil(2_usize.pow(24) + 42), 25);
    }
}
