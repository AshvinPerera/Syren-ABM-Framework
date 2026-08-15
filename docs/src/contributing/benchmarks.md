# Benchmarks

The benchmarks in `benches/` use [Criterion] and cover the hot paths: spawning,
iteration, ticking, reduction, query matching, scheduler packing, structural
mutation, parallel scaling, and the GPU paths.

## Running

Compile all benchmarks without running (this is what CI checks):

```bash
cargo bench --no-run --all-features
```

Run a benchmark with the features it needs:

```bash
cargo bench --all-features --bench iterate
```

Some benchmark targets declare required features (for example, the environment,
messaging, model, and GPU benchmarks); Cargo only builds a target when its
features are enabled.

## Measuring a change

When a change is meant to affect performance:

1. Measure the relevant benchmark before and after, on the same machine, in
   release.
2. Report the numbers with their full context — hardware, features, build
   profile, population, and command — as described in [performance
   methodology](../reference/performance.md).
3. Confirm the change did not alter a trajectory for a fixed seed unless that was
   the intent; a performance change should not be a behavioural change by
   accident.

## Attributing time

Use the [profiler](../concepts/profiling.md) to see where a tick spends its time
before optimising. Measure first; a benchmark tells you whether a change helped,
and the profiler tells you where to aim it.

[Criterion]: https://docs.rs/criterion
