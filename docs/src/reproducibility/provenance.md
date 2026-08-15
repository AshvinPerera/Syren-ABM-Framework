# Run provenance

A result is reproducible only if what produced it is recorded. Record enough to
reconstruct the run.

## Provenance checklist

For each reported run, record:

- **Crate version** — the Syren version (or commit hash for an unreleased build).
- **Feature set** — the Cargo features the model was built with.
- **Toolchain and target** — the Rust version and the platform the run executed
  on. This matters for floating-point comparability.
- **Build profile** — debug or release. Performance numbers are only meaningful
  in release.
- **Seed** — the model seed.
- **Configuration** — every parameter and option, including the config file or
  scenario name if one was used.
- **Input data** — the identity (a content hash or a versioned path) of any
  external data the run read.
- **Command** — the exact command line, including arguments.
- **Thread count** — if reporting timing. Results should not depend on it, but a
  timing number does.

## Why each item

The trajectory is a function of version, features, seed, configuration, and input
data. Fix those and the trajectory is fixed. The toolchain, target, build
profile, and thread count do not change a correct trajectory, but they do change
timing and floating-point comparability, so record them whenever numbers are
compared across machines.

## Making it routine

Have the model print its provenance at startup, or write it into the output
alongside the results, so a stored result carries the information needed to
reproduce it.
