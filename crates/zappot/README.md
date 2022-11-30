# ZappOT

A high-performance, extensible oblivious transfer library written in Rust.

## Implemented protocols
- 1-out-of-2 Base OT [CO15]
- 1-out-of-2 OT Extension [ALSZ13]
- 1-out-of-2 SilentOT [BCG+19]

## Installing Rust
The recommended way of installing the Rust stable toolchain is via the installer
[rustup](https://rustup.rs/). It can either be downloaded from its website, or via various Linux package managers.

## Using this library in another project
Currently, this crate is not released on [crates.io](https://crates.io). Until then,
the easiest way of using this library is adding it as a git submodule and specifying it as a dependency in your `Cargo.toml` like this:
```toml
zappot = {path = "<path to ZappOT>", features = ["silent_ot"]} # features can be omitted
```

## Building the library
```shell
cargo build --all-features
```

## Running the tests
Tests can be executed with the following command:
```shell
cargo test --all-features
```

## Running the benchmarks
Benchmarks are done via the awesome [criterion](https://github.com/bheisler/criterion.rs) crate and can be run with the following command:
```shell
cargo bench --all-features
```

## Viewing the documentation
Documentation can be build and viewed by issuing:
```shell
cargo doc --all-features --no-deps --open
```

## Running the examples
This crate contains commented and executable examples on how to use the provided
protocols in the examples [directory](./examples). The examples can be run via Cargo.
```shell
cargo run --release --example co_base_ot -- --help
cargo run --release --example alsz_ot_extension -- --help
cargo run --release --example silent_ot -- --help
```
Omitting the `--help` flag will run the example with default values.

## Cargo Features
This library provides cargo feature flags to optionally enable additional functionality. Usage: `cargo test --features <features...>`

- `silent_ot` This feature enables the SilentOT implementation. Only compiles with the `avx2` target feature enabled.