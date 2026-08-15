# Tests and the feature matrix

## Kinds of test

- **Unit tests** live beside the code in `src/` and cover individual types and
  invariants.
- **Integration tests** in `tests/` exercise whole subsystems — the scheduler
  graph, boundary lifecycle, entity-aware iteration, memory layout, and the GPU
  dispatch path.
- **Example tests** verify the examples. The macroeconomy suite, run as an
  integration test, covers scheduler and market ordering, named equations,
  parameter defaults, CSV schema shape, and determinism.

## The feature matrix

Because features gate whole layers, tests must run under the relevant
combinations. CI checks each of: no features, `agents`, `environment`,
`messaging`, `model`, `model messaging`, `gpu`, `model messaging_gpu`, and `all`.
Test execution covers the no-features build, `model messaging`, and the
all-features library; integration tests are compiled with all features.

Run the common combinations locally:

```bash
cargo test --no-default-features
cargo test --features "model messaging"
cargo test --all-features --lib
cargo test --all-features --no-run
```

## Determinism tests

Determinism is tested explicitly: a model runs at several thread counts and the
trajectories are compared bit for bit, and distinct seeds are asserted to
diverge. When you change anything that touches iteration order, accumulation, or
randomness, keep these green — they are the guard against silently introducing
order sensitivity.

## GPU tests

GPU **execution** tests need an adapter. Where none is present they report a skip,
so they do not block CI. The manual `GPU tests` workflow runs them on a
self-hosted runner with real hardware.

## MSRV

The library is checked at the MSRV (Rust 1.87) with no features and all features.
Avoid language or standard-library features newer than the MSRV in library code,
or bump the MSRV deliberately.
