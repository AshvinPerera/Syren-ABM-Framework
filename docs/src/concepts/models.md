# Models, sub-schedulers, and nesting

_Requires the `model` feature._

The [`ModelBuilder`] assembles a whole model — components, agent templates,
environment, messaging, systems, and scheduling — in one fluent chain and
validates it in [`build`][build]. A built [`Model`] owns its world and advances
with `tick`.

## The builder

A typical build registers a component registry and shards, sets the seed,
registers environment keys and message types, adds agent templates and
populations, adds systems, and calls `build`:

```rust,ignore
let model = ModelBuilder::new()
    .with_seed(42)
    .with_component_registry(registry)
    .with_shards(shards)
    .with_agent_template(template)?
    .with_agent_population("walker", position_id, population)?
    .with_system(system)
    .build()?;
```

`build` freezes the component registry, validates that sub-scheduler names are
unique and that channels are used within a consistent scope, and constructs the
world.

## Sub-schedulers

A [`SubScheduler`] is a named scope with its own systems and stages that **shares
the model's world and boundaries**. Sub-schedulers run before the root scheduler
each tick, in the order they were added. Use one to group a phase of the tick
that should complete before the rest runs.

The model seed is applied to the root scheduler and every shared sub-scheduler,
so a system in a sub-scheduler sees the same `RunContext::simulation_seed` as the
root.

## Nested models

A [`NestedModel`] is a fully separate child `Model` — its own world, environment,
agents, and seed — with an optional **bridge** closure. Each tick, a nested child
completes its own `tick`, then its bridge writes parent-facing effects, and only
then does the root scheduler run. Because a nested model is isolated, it keeps
the seed configured by its own builder rather than the parent's.

Use a sub-scheduler to phase work within one world; use a nested model to compose
a separate world whose interaction with the parent is limited to what the bridge
carries.

[`ModelBuilder`]: https://docs.rs/syren/latest/syren/model/struct.ModelBuilder.html
[build]: https://docs.rs/syren/latest/syren/model/struct.ModelBuilder.html#method.build
[`Model`]: https://docs.rs/syren/latest/syren/model/struct.Model.html
[`SubScheduler`]: https://docs.rs/syren/latest/syren/model/struct.SubScheduler.html
[`NestedModel`]: https://docs.rs/syren/latest/syren/model/struct.NestedModel.html
