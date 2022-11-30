use chrono::Local;
use clap::Parser;
use flute_benchmark::load_circuits;
use gmw::circuit::base_circuit::Load;
use gmw::circuit::BaseCircuit;
use gmw::common::BitVec;
use gmw::executor::Executor;
use gmw::mul_triple::insecure_provider::InsecureMTProvider;
use gmw::mul_triple::silent_ot::{self, SilentMtProvider};
use gmw::parse::lut_circuit::Gate;
use gmw::parse::{aby, lut_circuit};
use gmw::protocols::aby2_lut::{
    self, DeltaSharing, LutAby2, LutGate, LutSetupMsg, LutSetupProvider, ShareType,
};
use gmw::share_wrapper::inputs;
use gmw::{bristol, Circuit, CircuitBuilder, SubCircuitOutput};
use mpc_channel::sub_channels_for;
use rand::distributions::Standard;
use rand::{rngs::OsRng, thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
struct Args {
    #[arg(long, value_parser = clap::value_parser!(u16).range(0..=1))]
    id: u16,
    #[arg(long, default_value = "127.0.0.1:7721")]
    server: SocketAddr,
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    #[arg(long, value_delimiter = ',', default_value = "1")]
    batch_sizes: Vec<usize>,
    /// Default output is bench_results_TIMESTAMP.[jsonl|csv]
    #[arg(short, long)]
    out: Option<PathBuf>,
    #[arg(long, default_value = "to_eval")]
    circuits: PathBuf,
    #[arg(long, default_value = "Baseline/aby")]
    aby_baseline: PathBuf,
    #[arg(long, default_value = "Baseline/bristol")]
    bristol_baseline: PathBuf,
    #[arg(long)]
    exclude: Vec<PathBuf>,
    #[arg(short, long)]
    depth: Option<usize>,
    #[arg(long, default_value_t = 2_000_000)]
    ots: usize,
    #[arg(long, default_value = "lan")]
    net: String,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
struct BenchResult {
    file: PathBuf,
    repeat: usize,
    network_setting: String,
    ots_generated: usize,
    ots_used: usize,
    comm_base_ots: usize,
    comm_silent_ots: usize,
    comm_fd_preprocessing: usize,
    comm_input_sharing: usize,
    comm_online: usize,
    time_base_ots_ms: u128,
    time_silent_ots_ms: u128,
    time_fd_preprocessing_ms: u128,
    time_input_sharing_ms: u128,
    time_online_ms: u128,
    theoretical_setup_bits: u64,
    theoretical_online_bits: u64,
    theoretical_ots: u64,
    theoretical_splut_bits: u64,
    theoretical_splut_ots: u64,
    theoretical_ottt_ands: u64,
    batch_size: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum InputSharing {
    Seed([u8; 32]),
    InputMasks(BitVec),
}

// enum Synchronize

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // let log_writer = BufWriter::new(
    //     File::options()
    //         .create(true)
    //         .append(true)
    //         .open("flute_benchmark.log")?,
    // );
    // let (non_blocking, _appender_guard) = tracing_appender::non_blocking(log_writer);
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        // .with_writer(non_blocking)
        .init();

    let mut args = Args::parse();
    if let None = &args.out {
        args.out = Some(PathBuf::from(format!(
            "bench_results_{}",
            Local::now().to_rfc3339()
        )));
    }

    let (mut sender, bw, mut receiver, br) = match args.id {
        0 => mpc_channel::tcp::listen(args.server).await?,
        1 => mpc_channel::tcp::connect(args.server).await?,
        _ => unreachable!("Checked by args parser"),
    };

    configure_net(&args.net)?;

    let lut_circuits = load_circuits(
        &args.circuits,
        &args.exclude,
        args.depth,
        lut_circuit::Circuit::load,
    )?;

    let base_aby_circuits = load_circuits(
        &args.aby_baseline,
        &args.exclude,
        args.depth,
        aby::Circuit::load,
    )?;

    let base_bristol_circuits =
        load_circuits(&args.bristol_baseline, &args.exclude, args.depth, |path| {
            bristol::Circuit::load(path)
        })?;

    let circuits: Vec<_> = lut_circuits
        .into_iter()
        .map(|(path, circ)| {
            info!(?path, "Converting");
            let converted = BaseCircuit::from_lut_circuit(&circ, Load::SubCircuit).unwrap();
            (path, converted, Some(circ))
        })
        .chain(base_aby_circuits.into_iter().map(|(path, circ)| {
            info!(?path, "Converting");
            let converted = BaseCircuit::from_aby(circ, Load::SubCircuit).unwrap();
            (path, converted, None)
        }))
        .chain(base_bristol_circuits.into_iter().map(|(path, circ)| {
            info!(?path, "Converting");
            let converted = BaseCircuit::from_bristol(circ, Load::SubCircuit).unwrap();
            (path, converted, None)
        }))
        .collect();

    // needs to be in scope for the macro
    #[allow(unused_assignments)]
    let mut res = BenchResult::default();

    macro_rules! record {
        ($time:ident, $comm:ident, $e:expr) => {{
            bw.reset();
            br.reset();
            let $time = Instant::now();
            let ret = $e;
            res.$time += $time.elapsed().as_millis();
            // sleep a little to ensure(well, not really...) data has been sent. Unfortunately
            // it seems there is no flush() operation in remoc
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            res.$comm += (bw.reset() + br.reset()) * 8;
            ret
        }};
    }

    for &batch_size in &args.batch_sizes {
        for (idx, (circ_path, circuit, plain_circuit)) in circuits.iter().enumerate() {
            let circuit: Circuit<LutGate, usize> =
                build_batched_circuit(circuit.clone(), batch_size);
            info!(curr = idx, total = circuits.len(), ?circ_path);
            res = BenchResult::default();
            res.batch_size = batch_size;
            res.repeat = args.repeat;
            res.network_setting = args.net.clone();
            res.file = circ_path.clone();
            if let Some(plain_circuit) = plain_circuit {
                calc_theoretical_numbers(&mut res, &plain_circuit, batch_size as u64);
            }

            for iteration in 0..args.repeat {
                info!(iteration, ?circ_path);
                let (ot_ch1, ot_ch2, setup_ch, mut input_ch, mut exec_ch) = sub_channels_for!(
                    &mut sender,
                    &mut receiver,
                    64,
                    silent_ot::Msg,
                    silent_ot::Msg,
                    LutSetupMsg,
                    InputSharing,
                    aby2_lut::Msg,
                )
                .await?;

                let ots_per_party = args.ots;
                let mut mtp = record!(time_base_ots_ms, comm_base_ots, {
                    if args.id == 0 {
                        SilentMtProvider::new(ots_per_party, OsRng, ot_ch1, ot_ch2).await
                    } else {
                        SilentMtProvider::new(ots_per_party, OsRng, ot_ch2, ot_ch1).await
                    }
                });
                // times two because both parties generate this amount
                res.ots_generated = 2 * ots_per_party;

                record!(
                    time_silent_ots_ms,
                    comm_silent_ots,
                    mtp.precompute_mts().await
                );

                let mut mtp = InsecureMTProvider::default();

                let mut lut_setup =
                    LutSetupProvider::new(args.id.into(), &mut mtp, setup_ch.0, setup_ch.1);

                let (inp, state) = record!(time_input_sharing_ms, comm_input_sharing, {
                    let priv_seed = thread_rng().gen();
                    let joint_seed = thread_rng().gen();
                    let remote_joint_seed = match tokio::join!(
                        input_ch.0.send(InputSharing::Seed(joint_seed)),
                        input_ch.1.recv()
                    ) {
                        (Ok(_), Ok(Some(InputSharing::Seed(seed)))) => seed,
                        other => panic!("{:#?}", other),
                    };
                    let input_size = circuit.input_count();

                    match args.id {
                        0 => {
                            let share_map =
                                (0..input_size).map(|id| (id, ShareType::Local)).collect();
                            let mut sharing = DeltaSharing::new(
                                priv_seed,
                                joint_seed,
                                remote_joint_seed,
                                share_map,
                            );
                            let state = LutAby2::new(sharing.clone());
                            let inp = BitVec::from_iter(
                                thread_rng()
                                    .sample_iter::<bool, _>(Standard)
                                    .take(input_size),
                            );
                            let (shared, inp_mask) = sharing.share(inp.clone());
                            input_ch.0.send(InputSharing::InputMasks(inp_mask)).await?;
                            (shared, state)
                        }
                        1 => {
                            let share_map =
                                (0..input_size).map(|id| (id, ShareType::Remote)).collect();
                            let mut sharing = DeltaSharing::new(
                                priv_seed,
                                joint_seed,
                                remote_joint_seed,
                                share_map,
                            );
                            let state = LutAby2::new(sharing.clone());
                            let shared = match input_ch.1.recv().await {
                                Ok(Some(InputSharing::InputMasks(mask))) => {
                                    sharing.plain_delta_to_share(mask)
                                }
                                other => panic!("{other:#?}"),
                            };
                            (shared, state)
                        }
                        _ => unreachable!(),
                    }
                });

                let mut executor = record!(
                    time_fd_preprocessing_ms,
                    comm_fd_preprocessing,
                    Executor::new_with_state(state, &circuit, args.id.into(), &mut lut_setup)
                        .await?
                );

                // times two because each party uses 1 OT per MT
                res.ots_used += 2 * mtp.count();

                let _result = record!(
                    time_online_ms,
                    comm_online,
                    executor
                        .execute(inp, &mut exec_ch.0, &mut exec_ch.1)
                        .await?
                );
            }
            res.calc_mean();
            info!(?res);
            if args.id == 0 {
                write_results(&args, &[res])?;
            }
        }
    }
    Ok(())
}

fn build_batched_circuit(
    base_circuit: BaseCircuit<LutGate, usize>,
    batch_size: usize,
) -> Circuit<LutGate, usize> {
    let input_sw = inputs::<LutAby2, usize>(base_circuit.sub_circuit_input_gates().len());
    let sc = base_circuit.into_shared();
    for _ in 0..batch_size {
        let (output, id) = CircuitBuilder::<LutGate, usize>::with_global(|builder| {
            let id = builder.push_circuit(sc.clone());
            let output = builder.connect_sub_circuit(&input_sw, id);
            (output, id)
        });
        output.connect_to_main(id).into_iter().for_each(|share| {
            share.output();
        });
    }
    CircuitBuilder::<LutGate, usize>::global_into_circuit()
}

/// For every p->q LUT, add the following:
/// Setup : 4 * (2^p - p - 1) bits
/// Online: 2q bits
/// #OTs : 2 * (2^p - p - 1)
/// theoretical_splut_bits = (2^p) * q + p
/// theoretical_splut_ots = p
/// theoretical_ottt_ands = (p-1) * 2^p * q
fn calc_theoretical_numbers(res: &mut BenchResult, circ: &lut_circuit::Circuit, batch_size: u64) {
    for gate in &circ.gates {
        let Gate::Lut(lut) = gate else {
            continue;
        };
        let p = lut.input_wires.len() as u64;
        let two_p = 2_u64.pow(p as u32);
        let q = lut.masked_luts.len() as u64;
        res.theoretical_setup_bits += (4 * (two_p - p - 1)) * batch_size;
        res.theoretical_online_bits += (2 * q) * batch_size;
        res.theoretical_ots += (2 * (two_p - p - 1)) * batch_size;
        res.theoretical_splut_bits += (two_p * q + p) * batch_size;
        res.theoretical_splut_ots += p * batch_size;
        res.theoretical_ottt_ands += ((p - 1) * two_p * q) * batch_size;
    }
}

fn write_results(args: &Args, results: &[BenchResult]) -> anyhow::Result<()> {
    let mut open_options = File::options();
    open_options.create(true).append(true);
    let out = args.out.clone().unwrap();
    let mut json_file = BufWriter::new(open_options.open(out.with_extension("jsonl"))?);
    let csv_file = open_options.open(out.with_extension("csv"))?;
    let write_header = csv_file.metadata()?.len() == 0;
    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(write_header)
        .from_writer(BufWriter::new(csv_file));
    for res in results {
        csv_writer.serialize(res)?;
        serde_json::to_writer(&mut json_file, res)?;
        write!(&mut json_file, "\n")?;
    }
    Ok(())
}

fn configure_net(net_setting: &str) -> anyhow::Result<()> {
    let mut child = match net_setting {
        "lan" => Command::new("sudo").arg("tc_lan10").spawn()?,
        "wan" => Command::new("sudo").arg("tc_wan").spawn()?,
        "none" => return Ok(()),
        other => anyhow::bail!("unsupported network setting: {}", other),
    };
    child.wait()?;
    Ok(())
}

impl BenchResult {
    fn calc_mean(&mut self) {
        self.ots_used /= self.repeat;
        self.comm_base_ots /= self.repeat;
        self.comm_silent_ots /= self.repeat;
        self.comm_fd_preprocessing /= self.repeat;
        self.comm_input_sharing /= self.repeat;
        self.comm_online /= self.repeat;
        self.time_base_ots_ms /= self.repeat as u128;
        self.time_silent_ots_ms /= self.repeat as u128;
        self.time_fd_preprocessing_ms /= self.repeat as u128;
        self.time_input_sharing_ms /= self.repeat as u128;
        self.time_online_ms /= self.repeat as u128;
    }
}
