# Development setup

This section documents the engine internals and the development process. It
complements the repository's
[`CONTRIBUTING.md`](https://github.com/AshvinPerera/Syren-ABM-Framework/blob/master/CONTRIBUTING.md),
which is the authoritative entry point for setup, checks, and the pull-request
process.

## Toolchain

The development toolchain is pinned in `rust-toolchain.toml` to Rust 1.91.1 with
`rustfmt` and `clippy`. With `rustup` installed, the pinned toolchain is selected
automatically the first time you run `cargo` in the repository. The library's
MSRV is 1.87; see the [compatibility policy](../reference/compatibility.md).

## The check loop

Before opening a pull request, run the same checks CI runs:

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --no-default-features
cargo test --features "model messaging"
cargo test --all-features --lib
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

See [tests and the feature matrix](tests.md) for the full matrix and
[documentation](documentation.md) for building the guide.

## Where to start reading

- [Engine architecture](architecture.md) — how the core fits together.
- [Repository layout](layout.md) — where each part lives.
- The `first_model` example — the smallest complete model.
- The macroeconomy example — a large, fully documented model.
