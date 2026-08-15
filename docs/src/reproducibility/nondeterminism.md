# Sources of nondeterminism

Reproducibility can be lost in a few specific ways. This chapter lists them.

## Seeds

Two runs diverge if their seeds differ. Conversely, if two runs with *different*
seeds produce the *same* trajectory, the seed is not reaching the draw sites —
check that the model is built with `with_seed` and that draws go through
`DetRng::from_context`.

## Thread count

A correct model is thread-count invariant. If a trajectory changes with the
number of workers, some step depends on visitation order. The usual culprits are
a shared mutable RNG, a floating-point sum accumulated in completion order, or a
collected set whose order was not stabilised. See
[reproducibility](guarantees.md).

## Floating-point behaviour

Floating-point results are reproducible on the same target but are **not**
guaranteed identical across:

- different CPU architectures or compilers,
- fused-multiply-add contraction settings, or
- different math-library versions.

Report the target and toolchain alongside numerical results, and compare
trajectories on the same platform. Within one platform, order-independent
accumulation keeps results exact.

## External input and output

Anything the model reads from outside — a data file, the clock, the environment —
is part of its inputs. A run is only reproducible if those inputs are fixed. Pin
input data by content, avoid reading wall-clock time into model state, and record
which inputs a run used.

## GPU execution

GPU results depend on the adapter, its driver, and the shader compiler. A GPU run
is reproducible on the same adapter and driver, but not necessarily across
different hardware, and not necessarily bit-identical to the CPU path. Where
cross-platform reproducibility matters, run on the CPU or fix the GPU
environment, and validate the CPU/GPU equality tests on your hardware.

## Summary

| Source | Reproducible when |
| --- | --- |
| Seed | Same seed, reaching draws via `DetRng` |
| Thread count | Model is order-independent (always, if written correctly) |
| Floating point | Same architecture, compiler, and math library |
| External I/O | Inputs are pinned |
| GPU | Same adapter and driver |
