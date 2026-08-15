# Reading and recording results

There are two places a model keeps state you will want to read: the component
columns (per-agent data) and the environment (model-wide values). You read
per-agent data with a **reduction**, and model-wide data from the
**environment**.

## Reducing over the population

A reduction runs a query over every matching component column, folds each value
into an accumulator, and combines the per-chunk accumulators. Run one against
the world reference the model exposes:

```rust,ignore
let query = QueryBuilder::with_registry(registry)
    .read::<Position>()?
    .build()?;

let stats = model.ecs().world_ref().reduce_read::<Position, Welford>(
    query,
    Welford::default,
    |acc, pos| acc.push(pos.x as f64),
    |acc, other| acc.combine(other),
)?;
println!("count={} mean={:.3} variance={:.3}", stats.n, stats.mean, stats.variance());
```

The built-in accumulators are [`Count`], [`Sum`], [`MinMax`], and [`Welford`]
(count, mean, and variance). [`Welford::combine`] merges partials stably, so a
reduction returns the same result regardless of how the population is
partitioned across threads. For two-component folds, use
[`reduce_read2`][reduce_read2].

## Reading the environment

Model-wide values live in the environment, keyed by name and type. Read the
current value with the model's [`environment`][Model::environment] handle:

```rust,ignore
let value: MyState = model.environment().get::<MyState>("my_state")?;
```

Environment values are the natural home for aggregates a system computes once
per tick (for example, a price index) and for audit records. See [use
environment values](../how-to/environment.md).

## Recording a time series

To record a trajectory, read after each tick and append a row. Give each output
schema a single source of truth — define the column names once, next to the code
that formats the row, so the header and the values cannot drift apart. The
macroeconomy example demonstrates this pattern; see [collect results and
profiles](../how-to/results-profiles.md).

[`Count`]: https://docs.rs/syren/latest/syren/struct.Count.html
[`Sum`]: https://docs.rs/syren/latest/syren/struct.Sum.html
[`MinMax`]: https://docs.rs/syren/latest/syren/struct.MinMax.html
[`Welford`]: https://docs.rs/syren/latest/syren/struct.Welford.html
[`Welford::combine`]: https://docs.rs/syren/latest/syren/struct.Welford.html#method.combine
[reduce_read2]: https://docs.rs/syren/latest/syren/struct.ECSReference.html#method.reduce_read2
[Model::environment]: https://docs.rs/syren/latest/syren/model/struct.Model.html#method.environment
