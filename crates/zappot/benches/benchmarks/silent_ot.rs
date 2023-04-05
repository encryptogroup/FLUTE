use bitvec::order::Lsb0;
use bitvec::slice::BitSlice;
use bytemuck::cast_slice_mut;
use criterion::{criterion_group, BenchmarkId, Criterion};
use rand::thread_rng;
use rand_core::RngCore;
use zappot::silent_ot;
use zappot::silent_ot::quasi_cyclic_encode::bit_shift_xor;

use zappot::util::Block;

fn bench_bit_shift_xor(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_shift_xor");
    let conf = silent_ot::configure(1 << 24, 2, 128);

    let n_blocks = conf.n_blocks();
    let mut dest = rand_blocks(n_blocks);
    let inp = rand_blocks(n_blocks);

    group.bench_function(
        BenchmarkId::new("baseline", format!("{n_blocks} blocks")),
        |b| b.iter(|| baseline(&mut dest, &inp, 26)),
    );

    group.bench_function(
        BenchmarkId::new("optimized", format!("{n_blocks} blocks")),
        |b| b.iter(|| bit_shift_xor(&mut dest, &inp, 26)),
    );
}

fn baseline(dest: &mut [Block], inp: &[Block], bit_shift: u8) {
    assert!(bit_shift <= 127);
    let in_bits: &BitSlice<usize, Lsb0> = BitSlice::from_slice(bytemuck::cast_slice(inp));
    let shifted = &in_bits[bit_shift as usize..];
    let dest_bits: &mut BitSlice<usize, Lsb0> = BitSlice::from_slice_mut(cast_slice_mut(dest));
    *dest_bits ^= shifted;
}

fn rand_blocks(count: usize) -> Vec<Block> {
    let mut blocks = vec![Block::zero(); count];
    let bytes = cast_slice_mut(&mut blocks);
    thread_rng()
        .try_fill_bytes(bytes)
        .expect("Filling vec with random blocks");
    blocks
}

criterion_group!(benches, bench_bit_shift_xor);
