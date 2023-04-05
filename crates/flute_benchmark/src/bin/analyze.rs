use anyhow::Context;
use clap::Parser;
use flute_benchmark::load_circuits;
use gmw::parse::lut_circuit::{Circuit, Gate};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use tracing::warn;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
struct Args {
    /// The output path of the analysis results. Results will be written as json and csv file
    #[arg(short, long, default_value = "analysis_results")]
    out: PathBuf,
    /// Directory of circuits. Sub directories are analyzed recursively.
    #[arg(long, default_value = "Circuits")]
    circuits: PathBuf,
    /// Paths (and their children) to exclude from the analysis. `--exclude` can be provided
    /// multiple times.
    #[arg(long)]
    exclude: Vec<PathBuf>,
    /// Restrict the depth of the file traversal. By default, all circuits are analyzed recursively.
    #[arg(short, long)]
    depth: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
/// This struct contains the analysis information of a single circuit located at the `file` path.
/// Additional fields can be added here and computed in the `analyze` function.
struct Analysis {
    file: PathBuf,
    theoretical_setup_bits: u64,
    theoretical_online_bits: u64,
    theoretical_ots: u64,
    // maps the input size to a map containing output sizes and a count for the respective
    // input/output size tuple
    lut_sizes: BTreeMap<u64, BTreeMap<u64, u64>>,
    non_interactive_gates: BTreeMap<OtherGate, u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
enum OtherGate {
    Xor,
    Xnor,
    Not,
    Assign,
}

impl TryFrom<&Gate> for OtherGate {
    type Error = ();

    fn try_from(value: &Gate) -> Result<Self, Self::Error> {
        match value {
            Gate::Lut(_) => Err(()),
            Gate::Xor(_) => Ok(Self::Xor),
            Gate::Xnor(_) => Ok(Self::Xnor),
            Gate::Not(_) => Ok(Self::Not),
            Gate::Assign(_) => Ok(Self::Assign),
        }
    }
}

fn main() -> anyhow::Result<()> {
    // Set the default logging level to INFO. This can be configured via the RUST_LOG environment
    // variable. E.g. RUST_LOG=error to get rid of the warnings.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(
                    LevelFilter::INFO.into())
                .from_env()
                .context("Invalid logging filter. Refer to \
                https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html#directives \
                for an explanation of the RUST_LOG env variable syntax.")?)
        .init();

    let args = Args::parse();
    let circuits: Vec<(PathBuf, Circuit)> =
        load_circuits(&args.circuits, &args.exclude, args.depth, Circuit::load)
            .context("Unable to load circuits")?;

    let results = circuits
        .into_iter()
        .map(|(path, circuit)| analyze(path, circuit))
        .collect();

    write_results(&args.out, results).context("Writing analysis results")?;
    Ok(())
}

/// Here is the place to add analysis for circuits! The `Circuit` struct is defined in
/// `crates/gmw/src/parse/lut_circuit`.
///
/// For every p->q LUT, add the following:
/// Setup : 4 * (2^p - p - 1) bits
/// Online: 2q bits
/// #OTs : 2 * (2^p - p - 1)
fn analyze(file: PathBuf, circuit: Circuit) -> Analysis {
    let mut analysis = Analysis {
        file: file.clone(),
        ..Default::default()
    };

    // Iterate over all gates in a circuit.
    for gate in &circuit.gates {
        match gate {
            Gate::Lut(lut) => {
                let p = lut.input_wires.len() as u64;
                let q = lut.masked_luts.len() as u64;
                analysis.theoretical_setup_bits += 4 * (2_u64.pow(p as u32) - p - 1);
                analysis.theoretical_online_bits += 2 * q;
                analysis.theoretical_ots += 2 * (2_u64.pow(p as u32) - p - 1);
                *analysis
                    .lut_sizes
                    .entry(p)
                    .or_default()
                    .entry(q)
                    .or_default() += 1;
                if p > 8 {
                    warn!(
                        "Circuit at {} contains LUT with more tha 8 inputs. LUT {} {}",
                        file.display(),
                        p,
                        q
                    );
                }
            }
            other @ (Gate::Xor(_) | Gate::Xnor(_) | Gate::Not(_) | Gate::Assign(_)) => {
                *analysis
                    .non_interactive_gates
                    .entry(other.try_into().unwrap())
                    .or_default() += 1;
            }
        }

        // The concrete LUTs can be accessed via
        // `lut.masked_luts` or iterated via `for masked_lut in &lut.masked_luts { // code }`
        // A masked LUT has a `wire_mask` which specifies which of the `lut.input_wires()` are
        // actually used for this lut, an `masked_lut.output` field which contains the
        // `output.unexpanded` output of the lut and a `masked_lut.out_wire`.
    }

    analysis
}

fn write_results(out_path: &Path, results: Vec<Analysis>) -> anyhow::Result<()> {
    let json_file = BufWriter::new(File::create(out_path.with_extension("json"))?);
    serde_json::to_writer_pretty(json_file, &results).context("Writing results as json")?;
    // let csv_file = BufWriter::new(File::create(out_path.with_extension("csv"))?);
    // let mut csv_writer = csv::Writer::from_writer(csv_file);
    // for res in results {
    //     csv_writer
    //         .serialize(res)
    //         .context("Writing results as csv")?;
    // }
    Ok(())
}
