use std::alloc::{alloc, dealloc, realloc, Layout};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;
use std::{fmt, mem, ptr, slice};
use typenum::{PowerOfTwo, Unsigned};

pub use typenum;

/// AlignedVec is an over-aligned vector where the element is aligned on a `ALIGN` byte boundary.
pub struct AlignedVec<T, ALIGN: Unsigned + PowerOfTwo> {
    ptr: NonNull<T>,
    cap: usize,
    len: usize,
    _marker: PhantomData<T>,
    _align: PhantomData<ALIGN>,
}

impl<T, ALIGN: Unsigned + PowerOfTwo> AlignedVec<T, ALIGN> {
    pub fn new() -> Self {
        assert_ne!(mem::size_of::<T>(), 0, "AlignedVec doesn't support ZSTs");
        assert!(
            ALIGN::to_usize() >= mem::align_of::<T>(),
            "ALIGN is smaller than alignment of T"
        );
        Self {
            ptr: NonNull::dangling(),
            cap: 0,
            len: 0,
            _marker: PhantomData,
            _align: PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        assert_ne!(mem::size_of::<T>(), 0, "AlignedVec doesn't support ZSTs");
        assert!(
            ALIGN::to_usize() >= mem::align_of::<T>(),
            "ALIGN is smaller than alignment of T"
        );
        if capacity == 0 {
            Self::new()
        } else {
            let size = capacity
                .checked_mul(mem::size_of::<T>())
                .expect("Size overflow");
            unsafe {
                let ptr = alloc(
                    Layout::from_size_align(size, ALIGN::to_usize()).expect("Illegal layout"),
                );
                Self {
                    ptr: NonNull::new(ptr as *mut T).expect("Allocation failed"),
                    cap: capacity,
                    len: 0,
                    _marker: PhantomData,
                    _align: PhantomData,
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn clear(&mut self) {
        self.truncate(0)
    }

    pub fn truncate(&mut self, len: usize) {
        // This is safe because:
        //
        // * the slice passed to `drop_in_place` is valid; the `len > self.len`
        //   case avoids creating an invalid slice, and
        // * the `len` of the vector is shrunk before calling `drop_in_place`,
        //   such that no value will be dropped twice in case `drop_in_place`
        //   were to panic once (if it panics twice, the program aborts).
        unsafe {
            // Note: It's intentional that this is `>` and not `>=`.
            //       Changing it to `>=` has negative performance
            //       implications in some cases. See #78884 for more.
            if len > self.len {
                return;
            }
            let remaining_len = self.len - len;
            let s = ptr::slice_from_raw_parts_mut(self.as_mut_ptr().add(len), remaining_len);
            self.len = len;
            ptr::drop_in_place(s);
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        let new_cap = self.len + additional;
        if new_cap > self.cap {
            let new_cap = new_cap
                .checked_next_power_of_two()
                .expect("cannot reserve a larger AlignedVec");
            let new_size = new_cap
                .checked_mul(mem::size_of::<T>())
                .expect("Size overflow");
            assert_ne!(new_size, 0, "AlignedVec doesn't support ZSTs");
            if self.cap == 0 {
                unsafe {
                    let layout = Layout::from_size_align(new_size, ALIGN::to_usize())
                        .expect("Illegal Layout");
                    self.ptr = NonNull::new(alloc(layout) as *mut T).expect("Allocation failed");
                    self.cap = new_cap;
                }
            } else {
                unsafe {
                    let new_ptr =
                        realloc(self.ptr.as_ptr() as *mut u8, self.layout(), new_size) as *mut T;
                    self.ptr = NonNull::new(new_ptr).expect("Reallocation failed");
                    self.cap = new_cap;
                }
            }
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe { Some(ptr::read(self.ptr.as_ptr().add(self.len))) }
        }
    }

    pub fn push(&mut self, elem: T) {
        unsafe {
            self.reserve(1);
            self.as_mut_ptr().add(self.len).write(elem);
            self.len += 1;
        }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }

    fn layout(&self) -> Layout {
        let size = self
            .cap
            .checked_mul(mem::size_of::<T>())
            .expect("Size overflow");
        Layout::from_size_align(size, ALIGN::to_usize()).expect("Illegal Layout")
    }
}

impl<T: Clone, ALIGN: Unsigned + PowerOfTwo> AlignedVec<T, ALIGN> {
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len();

        if new_len > len {
            self.extend_with(new_len - len, value)
        } else {
            self.truncate(new_len);
        }
    }

    fn extend_with(&mut self, n: usize, value: T) {
        self.reserve(n);

        unsafe {
            let mut ptr = self.as_mut_ptr().add(self.len());
            // Use SetLenOnDrop to work around bug where compiler
            // might not realize the store through `ptr` through self.set_len()
            // don't alias.
            let mut local_len = SetLenOnDrop::new(&mut self.len);

            // Write all elements except the last one
            for _ in 1..n {
                ptr::write(ptr, value.clone());
                ptr = ptr.offset(1);
                // Increment the length in every step in case next() panics
                local_len.increment_len(1);
            }

            if n > 0 {
                // We can write the last element directly without cloning needlessly
                ptr::write(ptr, value);
                local_len.increment_len(1);
            }

            // len set by scope guard
        }
    }
}

impl<T, ALIGN: Unsigned + PowerOfTwo> Drop for AlignedVec<T, ALIGN> {
    #[inline]
    fn drop(&mut self) {
        let elem_size = mem::size_of::<T>();

        if self.cap != 0 && elem_size != 0 {
            // drop all elements by popping them
            while self.pop().is_some() {}
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, self.layout());
            }
        }
    }
}

impl<T: Clone, ALIGN: Unsigned + PowerOfTwo> Clone for AlignedVec<T, ALIGN> {
    #[inline]
    fn clone(&self) -> Self {
        unsafe {
            let mut result = Self::with_capacity(self.len);
            result.len = self.len;
            ptr::copy_nonoverlapping(self.as_ptr(), result.as_mut_ptr(), self.len);
            result
        }
    }
}

impl<T, ALIGN: Unsigned + PowerOfTwo> Deref for AlignedVec<T, ALIGN> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, ALIGN: Unsigned + PowerOfTwo> DerefMut for AlignedVec<T, ALIGN> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T, ALIGN: Unsigned + PowerOfTwo> From<&[T]> for AlignedVec<T, ALIGN>
where
    T: Clone,
{
    fn from(slice: &[T]) -> Self {
        let mut v = AlignedVec::with_capacity(slice.len());
        slice.iter().cloned().for_each(|elem| v.push(elem));
        v
    }
}

impl<T, I: slice::SliceIndex<[T]>, ALIGN: Unsigned + PowerOfTwo> Index<I> for AlignedVec<T, ALIGN> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&**self, index)
    }
}

impl<T, I: slice::SliceIndex<[T]>, ALIGN: Unsigned + PowerOfTwo> IndexMut<I>
    for AlignedVec<T, ALIGN>
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut **self, index)
    }
}

impl<T: fmt::Debug, ALIGN: Unsigned + PowerOfTwo> fmt::Debug for AlignedVec<T, ALIGN> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl<T, ALIGN: Unsigned + PowerOfTwo> Default for AlignedVec<T, ALIGN> {
    fn default() -> Self {
        Self::new()
    }
}

// Set the length of the vec when the `SetLenOnDrop` value goes out of scope.
//
// The idea is: The length field in SetLenOnDrop is a local variable
// that the optimizer will see does not alias with any stores through the Vec's data
// pointer. This is a workaround for alias analysis issue #32155
struct SetLenOnDrop<'a> {
    len: &'a mut usize,
    local_len: usize,
}

impl<'a> SetLenOnDrop<'a> {
    #[inline]
    fn new(len: &'a mut usize) -> Self {
        SetLenOnDrop {
            local_len: *len,
            len,
        }
    }

    #[inline]
    fn increment_len(&mut self, increment: usize) {
        self.local_len += increment;
    }
}

impl Drop for SetLenOnDrop<'_> {
    #[inline]
    fn drop(&mut self) {
        *self.len = self.local_len;
    }
}

unsafe impl<T: Send, ALIGN: Unsigned + PowerOfTwo> Send for AlignedVec<T, ALIGN> {}
unsafe impl<T: Sync, ALIGN: Unsigned + PowerOfTwo> Sync for AlignedVec<T, ALIGN> {}

#[cfg(test)]
mod tests {
    use crate::AlignedVec;
    use std::arch::x86_64::{__m128i, _mm_load_si128};
    use typenum::U32;

    #[test]
    fn alignment() {
        let av: AlignedVec<_, U32> = AlignedVec::from(&[0_64; 64][..]);
        assert_eq!(av.as_ptr() as usize % 32, 0)
    }

    #[test]
    fn alignment_after_realloc() {
        let mut av: AlignedVec<_, U32> = AlignedVec::new();
        for i in 0..64 {
            av.push(i);
        }
        assert_eq!(av.as_ptr() as usize % 32, 0)
    }

    #[test]
    fn slice_is_legal() {
        let av: AlignedVec<_, U32> = AlignedVec::from(&[0_64; 64][..]);
        dbg!(av[63]);
    }

    #[test]
    #[should_panic]
    fn index_panics() {
        let av: AlignedVec<_, U32> = AlignedVec::from(&[0_64; 64][..]);
        dbg!(av[64]);
    }

    #[test]
    fn load_si128_from_vec() {
        let av: AlignedVec<u64, U32> = (&[42; 2][..]).into();
        let bits = unsafe { _mm_load_si128(&av[0] as *const _ as *const __m128i) };
        dbg!(bits);
    }
}
