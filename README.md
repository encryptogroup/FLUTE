# FLUTE

Implementation of the FLUTE protocol.

## Running the protocol

### Installing Rust

The project is implemented in the [Rust](https://www.rust-lang.org/) programming language. To compile it, the latest stable toolchain is needed (older toolchains might work but are not guaranteed). The recommended way to install it, is via the toolchain manager [rustup](https://rustup.rs/).

One way of installing `rustup`:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Rustup is also available via most operating system package managers.


### Compiling the benchmarking binary
To compile the benchmarking binary, use the command:
```shell
cargo build --release --bin bench
```
The binary will be placed at `target/release/bench`.

Note that Flute currently only works with AVX2 support which means it is limited to x86. It will not compile on e.g. Apple Silicon.

### Running the benchmarks
To list the available options of the benchmarking binary.
```shell
./target/release/bench --help
```

A simple execution of a single circuit can be done via the following command:

```shell
 RUST_LOG=info ./target/release/bench --id 0 --circuits crates/flute_benchmark/to_eval/sbox8.lut --net none
```
and in another terminal
```shell
 RUST_LOG=info ./target/release/bench --id 1 --circuits crates/flute_benchmark/to_eval/sbox8.lut --net none
```
