//! # ZappOT
//!
//! This library provides an implementation of various oblivious transfer protocols.
//!
//! Currently implemented are:
//! - The [Chou Orlandi](`base_ot`) base OT protocol
//! - The [ALSZ13](`ot_ext`) OT extension protocol
//! - The [SilentOT](`silent_ot`) extension protocol

use blake2::{
    digest::consts::{U16, U20},
    Blake2b,
};

pub mod base_ot;
pub mod ot_ext;
#[cfg(feature = "silent_ot")]
pub mod silent_ot;
pub mod traits;
pub mod util;

pub mod bitvec {
    pub use bitvec::order::Lsb0;
    pub use bitvec::slice::BitSlice;
    pub use bitvec::vec::BitVec;
}

/// The default random oracle. Blake2b with an output of 160 bits.
pub type DefaultRom = Blake2b<U20>;
/// Blake2b random oracle with an output of 128 bits.
pub type Rom128 = Blake2b<U16>;

pub const BASE_OT_COUNT: usize = 128;
