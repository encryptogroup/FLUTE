use criterion::{criterion_group, BenchmarkId, Criterion};
use rand::thread_rng;
use rand_core::RngCore;
#[cfg(feature = "c_sse")]
use zappot::util::transpose::transpose_c_sse;
use zappot::util::transpose::transpose_rs_sse;

fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");
    group.sample_size(10);
    let rows = 128;
    let cols = 2_usize.pow(24);
    let mut input = vec![0_u8; rows * cols / 8];
    thread_rng().fill_bytes(&mut input);

    group.bench_with_input(
        BenchmarkId::new("c sse", "128 * 2^24 bits"),
        &input,
        |b, input| b.iter(|| transpose_c_sse(input, rows, cols)),
    );

    group.bench_with_input(
        BenchmarkId::new("rs sse", "128 * 2^24 bits"),
        &input,
        |b, input| b.iter(|| transpose_rs_sse(input, rows, cols)),
    );
}

criterion_group!(benches, bench_transpose);
