# Use environment values

_Requires the `environment` feature (implied by `model`)._

The environment holds model-wide state, keyed by name and type. Use it for
values that are not per-agent: a price index, a policy rate, a set of parameters,
or a per-tick audit record.

## Register a key

Register each environment value on the builder before building the model. You get
back a typed [`EnvKey`]:

```rust,ignore
let prices_key = builder.register_environment::<Prices>("prices", Prices::default())?;
```

The key carries the name and a channel id. Registering also sets the default that
the value holds before any system writes it.

## Read and write from a system

Inside a system, read and write the environment through the world reference by
name and type:

```rust,ignore
let prices: Prices = ecs.environment().get::<Prices>("prices")?;
// compute...
ecs.environment().set::<Prices>("prices", updated)?;
```

Read the final value after a tick through the model:

```rust,ignore
let prices: Prices = model.environment().get::<Prices>("prices")?;
```

## Ordering around environment values

Each environment key owns a channel, so the scheduler orders a system that reads
a value after the systems that write it — you do not have to wire that ordering
by hand. Because the environment is a boundary, writes made during a parallel
stage are finalised deterministically at the stage edge.

## Aggregates and audits

A common pattern is to compute a per-tick aggregate (say, a price index) into an
environment value in one system, and read it in later systems and after the tick.
The macroeconomy example keeps its whole aggregate and audit state in one
environment value updated each quarter.

[`EnvKey`]: https://docs.rs/syren/latest/syren/environment/struct.EnvKey.html
