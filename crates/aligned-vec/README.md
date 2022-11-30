# AlignedVec

This crate provides an implementation of an aligned vector. For example, the first element of an `AlignedVec<u64, 32>` is aligned on a 32 byte boundary.

## Motivation
Over-aligning values is sometimes needed to make use of SIMD intrinsics like `avx2` for allocated buffers of primitive values instead of types such as [__m128i](https://doc.rust-lang.org/core/arch/x86_64/struct.__m128i.html). This is especially useful when dealing with C libraries which expect over-aligned pointers.