# bitpolymul-sys

This crate provides low-level bindings to [bitpolymul](https://github.com/fast-crypto-lab/bitpolymul2) which implements fast multiplication of long binary polynomials. 

As the implementation uses the `avx2` instruction set, it must be enabled as a `target_feature`. This can be done by setting the `RUSTFLAGS` environment variable.
```shell
export RUSTFLAGS="-C target-feature=+avx2"
```
If the `target_feature` is not enabled, the crate won't build.