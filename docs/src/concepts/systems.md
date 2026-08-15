# Systems and access

A **system** is a unit of work the scheduler runs once per tick. A system
declares an **access set** — the components it reads and writes — and the
scheduler uses that declaration to decide which systems can run together.

## FnSystem and query-derived access

The common way to write a system is [`FnSystem`], which wraps a closure. The
recommended constructor is `FnSystem::from_queries`, which **derives the access
set from the queries the system runs**:

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

Deriving access from the queries means the declaration cannot drift from what the
system actually touches: if you change the query, the access follows. You can
still construct an [`AccessSets`] by hand where a system's access is not captured
by a single query, but the derived path is preferred.

## Access sets and conflicts

Two systems **conflict** when one writes a component the other reads or writes.
Conflicting systems must not run at the same time. Non-conflicting systems — for
example, two systems that write disjoint components — can run in parallel. The
scheduler computes this from the declared access; see
[scheduling](scheduling.md).

## Channels for ordering

Access conflicts are not the only ordering constraint. Sometimes system B must
run after system A even though they touch different components — B consumes an
effect A produces. That ordering is expressed with **channels**: a system
declares that it *produces* or *consumes* a named channel, and the scheduler
orders producers before consumers. See [order systems with
channels](../how-to/channels.md).

## Backends

A system runs on the CPU by default. A system can also declare a GPU backend
(via the [`GpuSystem`] trait) so the scheduler dispatches it as a compute
shader; see [CPU and GPU state](cpu-gpu.md).

[`FnSystem`]: https://docs.rs/syren/latest/syren/struct.FnSystem.html
[`AccessSets`]: https://docs.rs/syren/latest/syren/struct.AccessSets.html
[`GpuSystem`]: https://docs.rs/syren/latest/syren/struct.GpuSystem.html
