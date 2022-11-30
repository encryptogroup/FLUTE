# FLUTE Benchmark


## Installing Rust
Rust is best installed via the `rustup` toolchain manager which is available in most package repositories or from [https://rustup.rs/](https://rustup.rs/).

The latest stable version (1.65) is needed to compile FLUTE.

## Running the analysis

The help output displays the available options.
```shell
cargo run --release --bin analyze -- --help 
```

## Adding analysis

- Add fields to the `Analysis` struct in `src/bin/analyze.rs`
- change the `analyze` function accordingly
