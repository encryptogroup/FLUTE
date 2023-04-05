//! 128 bit Block
use crate::DefaultRom;
use bitvec::order::Lsb0;
use bitvec::store::BitStore;
use bitvec::vec::BitVec;
use blake2::digest::Output;
use blake2::Digest;
use bytemuck::{Pod, Zeroable};
use generic_array::{typenum::U16, GenericArray};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Shl, Shr};
use std::{array, mem};

#[derive(Pod, Zeroable, Debug, Default, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[repr(C, align(16))]
pub struct Block {
    data: u128,
}

impl Block {
    /// Block with all bits set to 0.
    pub const fn zero() -> Self {
        Self { data: 0 }
    }
    /// Block with the least significant bit set to 0.
    pub const fn one() -> Self {
        Self { data: 1 }
    }

    /// Block where every bit is set to `1`.
    pub const fn all_ones() -> Self {
        Self { data: u128::MAX }
    }

    /// Block for a constant number `N`.
    pub const fn constant<const N: u128>() -> Self {
        Self { data: N }
    }

    /// Least significant bit of the block.
    pub fn lsb(&self) -> bool {
        *self & Block::one() == Block::one()
    }

    /// Compute a hash of the Block using the [`DefaultRom`](`DefaultRom`) random oracle.
    pub fn rom_hash(&self) -> Output<DefaultRom> {
        DefaultRom::digest(self.data.to_le_bytes())
    }

    /// Convert the block to bytes in little-endian order.
    pub fn to_le_bytes(self) -> [u8; mem::size_of::<u128>()] {
        self.data.to_le_bytes()
    }

    /// Convert bytes in little-endian order into a block.
    pub fn from_le_bytes(bytes: [u8; mem::size_of::<u128>()]) -> Self {
        Self {
            data: u128::from_le_bytes(bytes),
        }
    }

    /// Cast a mutable slice of Blocks into a mutable slice of `GenericArray<u8, U16>`.
    /// Intended to pass Blocks directly to the [aes](https://docs.rs/aes/) encryption methods.
    #[cfg(feature = "silent_ot")]
    pub(crate) fn cast_slice_mut(slice: &mut [Block]) -> &mut [GenericArray<u8, U16>] {
        // Safety: GenericArray<u8, U16> works like a [u8; 16]. Since Block is a repr(C)
        // struct with a u128 field, with an alignment greater than [u8; 16], the cast is legal.
        unsafe { &mut *(slice as *mut _ as *mut [GenericArray<u8, U16>]) }
    }

    /// Cast a slice of Blocks into a slice of `GenericArray<u8, U16>`. Intended to pass Blocks
    /// directly to the [aes](https://docs.rs/aes/) encryption methods.
    pub(crate) fn cast_slice(slice: &[Block]) -> &[GenericArray<u8, U16>] {
        // See cast_slice_mut
        unsafe { &*(slice as *const _ as *const [GenericArray<u8, U16>]) }
    }
}

impl Distribution<Block> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Block {
        Block { data: rng.gen() }
    }
}

impl From<u32> for Block {
    fn from(val: u32) -> Self {
        Self { data: val.into() }
    }
}

impl From<u64> for Block {
    fn from(val: u64) -> Self {
        Self { data: val.into() }
    }
}

impl From<usize> for Block {
    fn from(val: usize) -> Self {
        Self {
            data: val
                .try_into()
                .expect("This library only works on platforms with a pointer size <= 128 bits"),
        }
    }
}

impl From<u128> for Block {
    fn from(val: u128) -> Self {
        Self { data: val }
    }
}

impl<'a, T: BitStore + Pod> TryFrom<&'a BitVec<T, Lsb0>> for Block {
    type Error = array::TryFromSliceError;

    fn try_from(value: &'a BitVec<T, Lsb0>) -> Result<Self, Self::Error> {
        let bytes = bytemuck::cast_slice(value.as_raw_slice()).try_into()?;
        Ok(Block::from_le_bytes(bytes))
    }
}

impl TryFrom<&[u8]> for Block {
    type Error = array::TryFromSliceError;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        let arr = value.try_into()?;
        Ok(Block::from_le_bytes(arr))
    }
}

impl From<GenericArray<u8, U16>> for Block {
    fn from(arr: GenericArray<u8, U16>) -> Self {
        Block::from_le_bytes(arr.into())
    }
}

impl From<Block> for GenericArray<u8, U16> {
    fn from(block: Block) -> Self {
        block.to_le_bytes().into()
    }
}

impl From<Block> for (u64, u64) {
    fn from(block: Block) -> Self {
        let bytes = block.to_le_bytes();
        let lower = bytes[0..mem::size_of::<u64>()].try_into().unwrap();
        let higher = bytes[mem::size_of::<u64>()..].try_into().unwrap();
        (u64::from_le_bytes(lower), u64::from_le_bytes(higher))
    }
}

impl From<Block> for u128 {
    fn from(block: Block) -> Self {
        block.data
    }
}

impl BitXor for Block {
    type Output = Block;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Block {
            data: self.data ^ rhs.data,
        }
    }
}

impl BitXorAssign for Block {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.data ^= rhs.data;
    }
}

impl BitOr for Block {
    type Output = Block;

    fn bitor(self, rhs: Self) -> Self::Output {
        Block {
            data: self.data | rhs.data,
        }
    }
}

impl BitOrAssign for Block {
    fn bitor_assign(&mut self, rhs: Self) {
        self.data |= rhs.data;
    }
}

impl BitAnd for Block {
    type Output = Block;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data & rhs.data,
        }
    }
}

impl BitAndAssign for Block {
    fn bitand_assign(&mut self, rhs: Self) {
        self.data &= rhs.data;
    }
}

impl<T> Shl<T> for Block
where
    u128: Shl<T, Output = u128>,
{
    type Output = Block;

    fn shl(self, rhs: T) -> Self::Output {
        Self {
            data: self.data << rhs,
        }
    }
}

impl<T> Shr<T> for Block
where
    u128: Shr<T, Output = u128>,
{
    type Output = Block;

    fn shr(self, rhs: T) -> Self::Output {
        Self {
            data: self.data >> rhs,
        }
    }
}

impl AsMut<[u8]> for Block {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        // TODO Safety
        unsafe { &mut *(self as *mut Block as *mut [u8; 16]) }
    }
}

impl num_traits::Zero for Block {
    fn zero() -> Self {
        Block::zero()
    }

    fn is_zero(&self) -> bool {
        self.data == 0
    }
}

impl Add for Block {
    type Output = Block;

    fn add(self, rhs: Self) -> Self::Output {
        Block {
            data: self.data + rhs.data,
        }
    }
}
