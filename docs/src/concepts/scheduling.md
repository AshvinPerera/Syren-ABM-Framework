# Scheduling, stages, and channels

The **scheduler** compiles a set of systems into an ordered execution plan and
runs it each tick. It packs systems that declare non-conflicting access into
stages that run in parallel.

## Stages

The scheduler packs systems into **stages**. A stage is a group of systems with
no conflicting access; the systems in a stage run in parallel, and stages run in
sequence. Packing is driven by:

- **Access conflicts** — a write to a component that another system reads or
  writes places the two in different stages.
- **Channel ordering** — a consumer of a channel runs in a later stage than the
  channel's producers.

The plan is deterministic: the same set of systems and constraints produces the
same stages every run.

## Channels

A **channel** is a named ordering edge. A system declares that it produces or
consumes a channel, and the scheduler runs producers before consumers. A channel
orders two systems that have no shared component access. Environment values and
message boundaries each own a channel, so a system that reads an environment
value runs after the systems that write it.

## Boundaries

A **boundary** is a model-owned resource written during a stage and made visible
at the stage edge; the environment and the message buffers are boundaries.
Boundary writes are staged per worker during parallel execution and merged at the
stage edge in a fixed order, so concurrent writers do not contend and the merge
is deterministic.

## Activation order and the seed

Within a stage, the **activation order** of systems is fixed and seeded, so an
order-sensitive step is reproducible. The seed set with `ModelBuilder::with_seed`
feeds both this activation order and each system's run context. See
[reproducibility](../reproducibility/guarantees.md).

## Inspecting the plan

A built [`Model`] prints its execution plan as text or as a Graphviz DOT graph,
which shows the stage each system is placed in:

```rust,ignore
println!("{}", model.execution_plan_text());
```

[`Model`]: https://docs.rs/syren/latest/syren/model/struct.Model.html
