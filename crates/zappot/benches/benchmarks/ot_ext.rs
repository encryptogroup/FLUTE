use bitvec::{bitvec, order::Lsb0};
use criterion::{criterion_group, BenchmarkId, Criterion};
use rand::rngs::StdRng;

use rand_core::SeedableRng;
use remoc::rch::mpsc::channel;
use std::io::{BufReader, BufWriter};
use std::os::unix::net::UnixStream;
use zappot::traits::{ExtROTReceiver, ExtROTSender};

use zappot::{base_ot, ot_ext};

fn bench_ot_ext(c: &mut Criterion) {
    let mut group = c.benchmark_group("ot extension");
    group.sample_size(10);

    let choices = bitvec![usize, Lsb0; 0;2_usize.pow(24)];

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    group.bench_with_input(
        BenchmarkId::new("alsz in_memory", "2^24 OTs"),
        &choices,
        |b, choices| {
            b.to_async(&runtime).iter(|| async {
                let ((sender1, receiver1), (sender2, receiver2)) =
                    mpc_channel::in_memory::new_pair(128);
                let send = tokio::spawn(async move {
                    let mut sender = ot_ext::Sender::new(base_ot::Receiver {});
                    let mut rng_send = StdRng::seed_from_u64(42);
                    sender
                        .send_random(2_usize.pow(24), &mut rng_send, sender1, receiver1)
                        .await
                });
                let choices = choices.clone();
                let receive = tokio::spawn(async move {
                    let mut receiver = ot_ext::Receiver::new(base_ot::Sender {});
                    let mut rng_recv = StdRng::seed_from_u64(42 * 42);
                    receiver
                        .receive_random(&choices, &mut rng_recv, sender2, receiver2)
                        .await
                });
                futures::future::join(send, receive).await
            });
        },
    );

    let bs = rand_bool_vec(2_usize.pow(24));

    group.bench_with_input(
        BenchmarkId::new("ocelot", "2^24 OTs"),
        &choices,
        |b, _choices| {
            b.iter(|| {
                _bench_ocelot_block_rot::<ocelot::ot::AlszSender, ocelot::ot::AlszReceiver>(&bs)
            });
        },
    );
}

fn rand_bool_vec(size: usize) -> Vec<bool> {
    (0..size).map(|_| rand::random::<bool>()).collect()
}

fn _bench_ocelot_block_rot<
    OTSender: ocelot::ot::RandomSender<Msg = scuttlebutt::Block>,
    OTReceiver: ocelot::ot::RandomReceiver<Msg = scuttlebutt::Block>,
>(
    bs: &[bool],
) {
    let (sender, receiver) = UnixStream::pair().unwrap();
    let m = bs.len();
    let handle = std::thread::spawn(move || {
        let mut rng = scuttlebutt::AesRng::new();
        let reader = BufReader::new(sender.try_clone().unwrap());
        let writer = BufWriter::new(sender);
        let mut channel = scuttlebutt::Channel::new(reader, writer);
        let mut ot = OTSender::init(&mut channel, &mut rng).unwrap();
        ot.send_random(&mut channel, m, &mut rng).unwrap();
    });
    let mut rng = scuttlebutt::AesRng::new();
    let reader = BufReader::new(receiver.try_clone().unwrap());
    let writer = BufWriter::new(receiver);
    let mut channel = scuttlebutt::Channel::new(reader, writer);
    let mut ot = OTReceiver::init(&mut channel, &mut rng).unwrap();
    ot.receive_random(&mut channel, &bs, &mut rng).unwrap();
    handle.join().unwrap();
}

criterion_group!(benches, bench_ot_ext);
