//! Correlation robust AES hash.
//!
//! This implementation of a correlation robust AES hash function
//! is based on the findings of <https://eprint.iacr.org/2019/074>.
use crate::util::Block;
use aes::cipher::{BlockEncrypt, Key, KeyInit};
use aes::Aes128;
use once_cell::sync::Lazy;

pub struct AesHash {
    aes: Aes128,
}

impl AesHash {
    /// Create a new `AesHash` with the given key.
    pub fn new(key: &Key<Aes128>) -> Self {
        Self {
            aes: Aes128::new(key),
        }
    }

    /// Compute the correlation robust hash of a block.
    ///
    /// # Warning: only secure in semi-honest setting!
    /// See <https://eprint.iacr.org/2019/074> for details.
    pub fn cr_hash_block(&self, x: Block) -> Block {
        let mut x_enc = x.into();
        self.aes.encrypt_block(&mut x_enc);
        x ^ x_enc.into()
    }

    /// Compute the correlation robust hashes of multiple blocks.
    ///
    /// Warning: only secure in semi-honest setting!
    /// See <https://eprint.iacr.org/2019/074> for details.
    pub fn cr_hash_blocks<const N: usize>(&self, x: &[Block; N]) -> [Block; N] {
        let mut blocks = x.map(|blk| blk.into());
        self.aes.encrypt_blocks(&mut blocks);

        let mut blocks = blocks.map(|enc_blk| enc_blk.into());
        for (enc_x, x) in blocks.iter_mut().zip(x) {
            *enc_x ^= *x;
        }
        blocks
    }
}

/// An `AesHash` with a fixed key.
pub static FIXED_KEY_HASH: Lazy<AesHash> = Lazy::new(|| {
    // TODO: Is it sufficient to just choose some random key? This one was generated
    //  by just using `rand::thread_rng().gen()`
    let key = 193502124791825095790518994062991136444_u128
        .to_le_bytes()
        .into();
    AesHash::new(&key)
});
