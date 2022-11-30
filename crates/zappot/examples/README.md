# Examples

Thi folder contains examples for executing the implemented OT protocols in
a real-world-ish setting. Communication is done via a TCP socket bound to localhost.

## Chou Orlandi Base OT
[Implementation](./co_base_ot.rs)

Execute via:
```shell
cargo run -r --example co_base_ot -- --num_ots 128 --port 8066
```

##  ALSZ13 OT Extension
[Implementation](./alsz_ot_extension.rs)

Execute via:
```shell
cargo run -r --example alsz_ot_extension -- --num_ots 1000 --port 8066
```

##  SilentOT Extension
[Implementation](./silent_ot.rs)

Execute via:
```shell
cargo run -r --example silent_ot -- --num_ots 1000 --port 8066 --threads 1 --scaler 2
```