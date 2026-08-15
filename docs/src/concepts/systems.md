# Systems and access

A **system** is a unit of work the scheduler runs once per tick. A system
declares an **access set** — the components it reads and writes — and the
scheduler uses that declaration to decide which systems can run together.

## FnSystem and query-derived access

A system is usually an [`FnSystem`], which wraps a closure.
`FnSystem::from_queries` derives the access set from the queries the system runs:

```rust,ignore
let step_query = QueryBuilder::with_registry(registry).write::<Position>()?.build()?;
let run_query = step_query.clone();

let system = FnSystem::from_queries(
    0,                 // system id
    "random_walk",     // name
    &[&step_query],    // queries whose access this system needs
    move |ecs| {
        ecs.for_each_entity_w1::<Position>(run_query.clone(), |entity, pos| {
            // ...
        })
    },
);
```

Because the access set is derived from the queries, it matches what the system
touches, and changing the query changes the access. A system whose access is not
captured by a single query can construct an [`AccessSets`] directly.

## Access sets and conflicts

Two systems **conflict** when one writes a component the other reads or writes.
Conflicting systems do not run at the same time. Non-conflicting systems — for
example, two systems that write disjoint components — run in parallel. The
scheduler computes this from the declared access; see [scheduling](scheduling.md).

## Channels for ordering

A system may need to run after another even when the two touch different
components, when one consumes an effect the other produces. This ordering is
expressed with **channels**: a system declares that it *produces* or *consumes* a
named channel, and the scheduler orders producers before consumers. See [order
systems with channels](../how-to/channels.md).

## Backends

A system runs on the CPU by default. A system can also declare a GPU backend
through the [`GpuSystem`] trait, so the scheduler dispatches it as a compute
shader; see [CPU and GPU state](cpu-gpu.md).

[`FnSystem`]: https://docs.rs/syren/latest/syren/struct.FnSystem.html
[`AccessSets`]: https://docs.rs/syren/latest/syren/struct.AccessSets.html
[`GpuSystem`]: https://docs.rs/syren/latest/syren/struct.GpuSystem.html
