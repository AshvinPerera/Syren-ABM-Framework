# Your first model

This chapter builds the `first_model` example: a population of walkers on an
integer number line. Each tick, every walker takes one deterministic step left
or right; a reduction then reports the population's spread.

Run it with:

```bash
cargo run --example first_model --features model
```

The code below is included from the compiled example. The full source is
[`examples/first_model.rs`](https://github.com/AshvinPerera/Syren-ABM-Framework/blob/master/examples/first_model.rs).

## The component

A component is a plain `Copy` struct. Each walker holds one integer position:

```rust
{{#include ../../../examples/first_model.rs:component}}
```

## Building the model

Register the component, freeze the registry, then describe the population and
the system with [`ModelBuilder`]. The system's access set is derived from the
query it runs, and the model seed is set with
[`with_seed`][ModelBuilder::with_seed]:

```rust
{{#include ../../../examples/first_model.rs:build}}
```

This build shows three points:

- `FnSystem::from_queries` derives the system's read/write set from the queries
  it runs, so the declared access matches what the system touches.
- `DetRng::from_context` is salted with the entity's identity, so a walker's step
  does not depend on the order in which worker threads visit rows. The same seed
  produces the same walk at any thread count.
- `with_agent_population` materialises the whole component column at once rather
  than spawning agents individually.

## Running and summarising

Run a fixed number of ticks, then reduce over the population. [`Welford`]
accumulates count, mean, and variance in a single pass and combines partials
from parallel chunks:

```rust
{{#include ../../../examples/first_model.rs:run}}
```

```rust
{{#include ../../../examples/first_model.rs:reduce}}
```

For a symmetric ±1 walk over 50 ticks, the mean is near zero and the variance is
near the number of steps.

Next: [running ticks](running-ticks.md) and [reading and recording
results](results.md).

[`ModelBuilder`]: https://docs.rs/syren/latest/syren/model/struct.ModelBuilder.html
[ModelBuilder::with_seed]: https://docs.rs/syren/latest/syren/model/struct.ModelBuilder.html#method.with_seed
[`Welford`]: https://docs.rs/syren/latest/syren/struct.Welford.html
