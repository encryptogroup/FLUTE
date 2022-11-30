use criterion::criterion_main;

mod benchmarks;
use benchmarks::*;

criterion_main!(
    transpose::benches,
    ot_ext::benches,
    aes_rng::benches,
    // silent_ot::benches
);
