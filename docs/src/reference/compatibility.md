# Compatibility policy

## Minimum supported Rust version

The library's MSRV is **Rust 1.87**, declared as `rust-version` in `Cargo.toml`.
The MSRV applies to the library across all features. Benchmarks and integration
tests are not bound by it and may use newer toolchains.

CI checks the library at the MSRV with no features and with all features. A
change that requires a newer language or standard-library feature than 1.87
provides must either avoid it or come with a deliberate MSRV bump.

The development toolchain is pinned separately in `rust-toolchain.toml` and is
newer than the MSRV; it is what formatting, linting, and generated output are
produced with.

## Platforms

Syren is portable Rust and builds on the major desktop platforms (Linux, macOS,
Windows). The `gpu` feature depends on wgpu and therefore on a platform graphics
backend; building only needs wgpu to compile, while running a GPU system needs a
working adapter.

## Semantic versioning

Syren is pre-1.0. Version numbers follow the pre-1.0 Cargo convention:

- **Patch** (`0.y.z` → `0.y.(z+1)`): no public API breakage.
- **Minor** (`0.y` → `0.(y+1)`): may break the public API, with migration notes
  in the changelog.

See [API status](api-status.md) for which surfaces carry which stability, and the
[contributing guide](../contributing/releases.md) for the release process.

## What counts as a breaking change

A breaking change is one that can stop dependent code from compiling or change a
documented behaviour: removing or renaming a public item, changing a signature,
tightening a bound, or changing a documented guarantee. A change to a model's
numerical trajectory for a fixed seed is a behavioural change and is called out in
the changelog even though it does not break compilation.
