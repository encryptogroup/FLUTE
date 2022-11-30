use criterion::{criterion_group, BenchmarkId, Criterion};

use rand::{thread_rng, Rng};
use rand_core::{RngCore, SeedableRng};
use rand_core_5_1::RngCore as OldRngCore;

use zappot::util::aes_rng::AesRng;

fn bench_aes_rng(c: &mut Criterion) {
    let mut group = c.benchmark_group("aes rng");
    let mut buf = vec![0; 2_usize.pow(24)];

    let mut prg = scuttlebutt::AesRng::new();
    group.bench_function(BenchmarkId::new("scuttlebutt", "2^24 bytes"), |b| {
        b.iter(|| {
            prg.fill_bytes(&mut buf);
        });
    });

    let mut prg = AesRng::from_seed(thread_rng().gen());
    group.bench_function(BenchmarkId::new("zappot", "2^24 bytes"), |b| {
        b.iter(|| {
            prg.fill_bytes(&mut buf);
        });
    });
}

criterion_group!(benches, bench_aes_rng);
