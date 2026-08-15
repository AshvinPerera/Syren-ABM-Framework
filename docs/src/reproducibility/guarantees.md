# Reproducibility

A Syren model run with the same inputs produces the same outputs, bit for bit,
regardless of the number of threads.

## What the framework guarantees

Given the same crate version, feature set, seed, and initial state:

- **Thread-count invariance.** The trajectory is identical on one worker or many.
  Work stealing assigns rows to workers, so no result may depend on the order
  rows are visited.
- **Deterministic scheduling.** The scheduler produces the same stages and the
  same activation order every run.
- **Deterministic randomness.** Draws taken through `DetRng::from_context` depend
  only on `(seed, tick, system_id, salt)`.

The macroeconomy example's test suite checks this directly: the same seed
produces an identical trajectory at one and eight threads, and distinct seeds
diverge.

## What the model must do

The guarantee holds only if model code follows these rules:

- **Draw randomness through `DetRng`**, keyed on the run context. A thread-local
  or shared mutable generator produces draws in an order that changes with the
  thread count.
- **Salt per-agent draws with the agent's identity**, not with the loop index, so
  a draw does not depend on the order the agent is visited in.
- **Keep parallel accumulations order-independent.** Sum per worker and combine in
  a fixed worker order; floating-point addition is not associative, so summing in
  completion order drifts with the thread count. Order-independent operations,
  such as maxima, are already safe.
- **Collect order-sensitive sets by a stable key.** When a system gathers rows
  whose order matters, sort by a model identifier rather than the order they were
  produced in.

## Setting the seed

Set the seed once, on the builder:

```rust,ignore
let model = ModelBuilder::new().with_seed(config.seed) /* ... */ .build()?;
```

The seed reaches every system as `RunContext::simulation_seed`. A run is
described by its seed together with the version, features, and initial state; see
[run provenance](provenance.md).
