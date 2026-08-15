# Scheduling, stages, and channels

The **scheduler** turns a set of systems into an ordered execution plan and runs
it each tick. It is the piece that turns declared access into safe parallelism.

## Stages

The scheduler packs systems into **stages**. A stage is a group of systems with
no conflicting access; the systems in a stage run in parallel, and stages run one
after another. Packing is driven by:

- **Access conflicts** — a write to a component that another system reads or
  writes forces the two into different stages.
- **Channel ordering** — a consumer of a channel runs in a later stage than the
  channel's producers.

The resulting plan is deterministic: the same set of systems and constraints
produces the same stages every run.

## Channels

A **channel** is a named ordering edge. A system declares that it produces or
consumes a channel; the scheduler guarantees producers run before consumers.
Channels express "run me after credit clears" without inventing a fake data
dependency. Environment values and message boundaries each own a channel, so a
system that reads an environment value is ordered after the systems that write
it.

## Boundaries

A **boundary** is a model-owned resource that is written during a stage and made
visible at the stage edge — the environment and the message buffers are
boundaries. Boundary writes are staged per worker during parallel execution and
finalised at the boundary between stages, which keeps concurrent writers from
contending while preserving a deterministic merge order.

## Activation order and the seed

Where a stage runs several systems, their **activation order** is fixed and
seeded, so any order-sensitive step is reproducible. The model seed set with
`ModelBuilder::with_seed` feeds both this activation order and each system's run
context. See [reproducibility](../science/reproducibility.md).

## Inspecting the plan

A built [`Model`] can print its execution plan as text or as a Graphviz DOT
graph, which is useful when checking that systems land in the stages you expect.

```rust,ignore
println!("{}", model.execution_plan_text());
```

[`Model`]: https://docs.rs/syren/latest/syren/model/struct.Model.html
