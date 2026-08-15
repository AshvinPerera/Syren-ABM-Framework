# Contributing to Syren

Thank you for your interest in Syren. This document is the entry point for
development setup, the checks a change must pass, documentation expectations,
and the pull-request process.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

Syren pins its development toolchain in [`rust-toolchain.toml`](rust-toolchain.toml)
to Rust 1.91.1 with `rustfmt` and `clippy`. With `rustup` installed, the correct
toolchain and components are selected automatically the first time you run
`cargo` in the repository.

The library's minimum supported Rust version (MSRV) is **1.87**, declared as
`rust-version` in [`Cargo.toml`](Cargo.toml). The MSRV applies to the library
only; benchmarks and integration tests may use newer language features.

## Checks

The following mirror the continuous-integration jobs in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). A change should pass all
of them before review.

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --no-default-features
cargo test --features "model messaging"
cargo test --all-features --lib
cargo test --all-features --no-run
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

The feature matrix is checked explicitly for each supported combination (no
features, `agents`, `environment`, `messaging`, `model`, `model messaging`,
`gpu`, `model messaging_gpu`, and `all`), and the library is checked at the MSRV
with no features and all features.

GPU **execution** tests need a real adapter. Where no adapter is present they
report a skip, so they do not block CI; the GPU paths are exercised on demand
through the manual `GPU tests` workflow on hardware.

## Determinism

Reproducibility is a core guarantee. When adding or changing model behaviour:

- Draw randomness through `DetRng::from_context`, keyed on the run context and a
  salt, rather than a thread-local generator. Work stealing must not change
  results.
- Set the model seed with `ModelBuilder::with_seed`; it reaches systems as
  `RunContext::simulation_seed`.
- Keep parallel accumulations order-independent (for example, sum per worker and
  combine in worker order).

A change that affects a model trajectory for a fixed seed is a behavioural
change: call it out in the pull request and the changelog.

## Documentation

- **rustdoc** is the reference for the public Rust API. Document changed public
  items with their behaviour, errors, and required Cargo features.
- The **mdBook guide** (under `docs/`) is the user and contributor guide. Build
  it with `mdbook build docs`.

## Pull requests

- Base your work on a branch and open the pull request against `master`.
- Fill in the [pull-request template](.github/PULL_REQUEST_TEMPLATE.md): API
  impact, determinism, numerical behaviour, memory layout, GPU behaviour,
  features, tests, documentation, and benchmarks.
- Add a changelog entry under the current unreleased heading in
  [`CHANGELOG.md`](CHANGELOG.md).
- Keep pull requests focused. A single reviewable change is easier to verify and
  revert than a bundle.

Review is by the code owner listed in [`CODEOWNERS`](CODEOWNERS).

## API stability policy

Syren is pre-1.0. Until 1.0:

- Patch releases (`0.y.z` → `0.y.(z+1)`) do not break the public API.
- Minor releases (`0.y` → `0.(y+1)`) may break the public API. Breaking changes
  come with migration notes in the changelog.
- Items reachable through the `advanced` module and the GPU API are lower-level
  and are marked separately; they may change with less notice.
- Where practical, an item is deprecated for one minor release before removal.

## Release policy

A release follows these steps:

1. Update the version in `Cargo.toml` and keep `CITATION.cff` in sync with it.
2. Finalise the changelog entry for the version.
3. Inspect the package: `cargo package --locked` and review `cargo package
   --list`.
4. Build the documentation (rustdoc and the mdBook guide).
5. Run the full CI suite from a clean checkout.
6. Tag the release and create the GitHub release.
7. Publish to crates.io with `cargo publish`.
8. For a public, non-candidate release, mint a DOI.

Release-candidate versions (`-rc.N`) stop before tagging, publishing, and DOI
creation.
