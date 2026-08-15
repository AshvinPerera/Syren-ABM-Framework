# Reproducibility

Reproducibility is a first-class guarantee in Syren: a model run with the same
inputs produces the same outputs, bit for bit, regardless of how many threads it
runs on.

## What the framework guarantees

Given the same crate version, the same feature set, the same seed, and the same
initial state:

- **Thread-count invariance.** The trajectory is identical whether the model runs
  on one worker or many. Work stealing decides which worker processes which rows,
  so nothing whose result depends on visitation order may leak into the outcome.
- **Deterministic scheduling.** The scheduler produces the same stages and the
  same activation order every run.
- **Deterministic randomness.** Draws taken through `DetRng::from_context` depend
  only on `(seed, tick, system_id, salt)`.

The macroeconomy example's test suite verifies this directly: the same seed
produces an identical trajectory at one and eight threads, and distinct seeds
diverge.

## What the model must do

The guarantee holds only if model code cooperates. Your obligations:

- **Draw randomness through `DetRng`**, keyed on the run context. Never use a
  thread-local or shared mutable generator; its draw order changes with thread
  count.
- **Salt per-agent draws with the agent's identity**, not with the loop index, so
  the draw does not depend on the order the agent is visited in.
- **Keep parallel accumulations order-independent.** Sum per worker and combine in
  a fixed worker order; floating-point addition is not associative, so summing in
  completion order drifts with thread count. Order-independent operations (maxima,
  for instance) are safe as they are.
- **Collect order-sensitive sets by a stable key.** If a system gathers rows and
  their order matters, sort by a model identifier rather than relying on the order
  they were produced in.

## Setting the seed

Set the seed once, on the builder:

```rust,ignore
let model = ModelBuilder::new().with_seed(config.seed) /* ... */ .build()?;
```

The seed reaches every system as `RunContext::simulation_seed`. A run is fully
described by its seed together with the version, features, and initial state; see
[run provenance](provenance.md).
