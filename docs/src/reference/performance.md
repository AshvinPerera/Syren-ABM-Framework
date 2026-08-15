# Performance methodology

A performance claim is only meaningful with its context. This chapter describes
how to measure and how to report.

## Measure in release

Always measure with an optimised build. Debug builds are for correctness, not
timing. The benchmarks use [Criterion], which handles warm-up, sampling, and
statistics.

## Run the benchmarks

The benchmark targets live in `benches/`. Compile them without running to check
they build:

```bash
cargo bench --no-run --all-features
```

Run a benchmark (or all of them) with the features it needs:

```bash
cargo bench --all-features
cargo bench --all-features --bench iterate
```

Some benchmarks require specific features (for example, the GPU and messaging
benchmarks); their targets declare the features they need.

## Report the full context

A number without its conditions is not reproducible. Report, with every
performance figure:

- the **hardware** (CPU model and core count; GPU adapter if relevant),
- the **compiler** version and target,
- the **features** enabled,
- the **build profile** (release),
- the **population** and problem size, and
- the exact **command**.

This is the same provenance discipline as for results; see [run
provenance](../reproducibility/provenance.md).

## Thread scaling

Because a correct model is thread-count invariant in *outcome*, thread count only
affects *timing*. When reporting scaling, hold everything else fixed and vary only
the worker count, and state the population — small populations do not have enough
work to scale across many cores.

## Where time goes

Use the [profiler](../concepts/profiling.md) to attribute time within a tick. It
shows whether time is spent in one system, in a stage that parallelises poorly,
or in structural mutation at a scheduler boundary.

[Criterion]: https://docs.rs/criterion
