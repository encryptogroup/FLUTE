use crate::common::BitVec;
use bitvec::array::BitArray;
use bitvec::order::Msb0;
use bitvec::prelude::BitStore;

use num_integer::div_ceil;
use rand::{CryptoRng, Fill, Rng};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut, RangeInclusive};
use std::{array, mem};

pub struct ByAddress<'a, T: ?Sized>(pub &'a T);

impl<'a, T: ?Sized> Hash for ByAddress<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(self.0, state)
    }
}

impl<'a, T: ?Sized> PartialEq for ByAddress<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl<'a, T: ?Sized> Eq for ByAddress<'a, T> {}

//
// RangeInclusive start wrapper
//

#[derive(Eq, Debug, Clone)]
pub struct RangeInclusiveStartWrapper<T> {
    pub range: RangeInclusive<T>,
}

impl<T> RangeInclusiveStartWrapper<T> {
    pub fn new(range: RangeInclusive<T>) -> RangeInclusiveStartWrapper<T> {
        RangeInclusiveStartWrapper { range }
    }
}

impl<T> PartialEq for RangeInclusiveStartWrapper<T>
where
    T: Eq,
{
    #[inline]
    fn eq(&self, other: &RangeInclusiveStartWrapper<T>) -> bool {
        self.range.start() == other.range.start() && self.range.end() == other.range.end()
    }
}

impl<T> Ord for RangeInclusiveStartWrapper<T>
where
    T: Ord,
{
    #[inline]
    fn cmp(&self, other: &RangeInclusiveStartWrapper<T>) -> Ordering {
        match self.range.start().cmp(other.range.start()) {
            Ordering::Equal => self.range.end().cmp(other.range.end()),
            not_eq => not_eq,
        }
    }
}

impl<T> PartialOrd for RangeInclusiveStartWrapper<T>
where
    T: Ord,
{
    #[inline]
    fn partial_cmp(&self, other: &RangeInclusiveStartWrapper<T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// `N` is the number of bytes in the backing storage. A BitMask can at most store `u8::MAX` bits.
pub struct BitMask<const N: usize> {
    arr: BitArray<[u8; N], Msb0>,
    len: u8,
}

impl<const N: usize> BitMask<N> {
    pub fn zeros(len: u8) -> Self {
        assert!(len as usize <= N * mem::size_of::<u8>());
        Self {
            arr: BitArray::ZERO,
            len,
        }
    }
}

impl<const N: usize> Deref for BitMask<N> {
    type Target = bitvec::slice::BitSlice<u8, Msb0>;

    fn deref(&self) -> &Self::Target {
        &self.arr[..self.len as usize]
    }
}

impl<const N: usize> DerefMut for BitMask<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.arr[..self.len as usize]
    }
}

/// Helper method to quickly create an array of random BitVecs.
pub fn rand_bitvecs<R: CryptoRng + Rng, const N: usize>(
    size: usize,
    rng: &mut R,
) -> [BitVec<usize>; N] {
    array::from_fn(|_| rand_bitvec(size, rng))
}

pub fn rand_bitvec<T, R>(size: usize, rng: &mut R) -> BitVec<T>
where
    T: BitStore + Copy,
    [T]: Fill,
    R: CryptoRng + Rng,
{
    let bitstore_items = div_ceil(size, mem::size_of::<T>());
    let mut buf = vec![T::ZERO; bitstore_items];
    rng.fill(&mut buf[..]);
    let mut bv = BitVec::from_vec(buf);
    bv.truncate(size);
    bv
}
