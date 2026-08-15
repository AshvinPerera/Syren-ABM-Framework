# Glossary

**Access set** — the components a system reads and writes, plus the channels it
produces and consumes. Used by the scheduler to order and parallelise systems.

**Activation order** — the fixed, seeded order in which systems within a stage
run, so any order-sensitive step is reproducible.

**Archetype** — the set of entities sharing exactly the same components. Storage
is organised per archetype.

**Attribute** — the columnar array storing one component for one archetype, split
into chunks.

**Boundary** — a model-owned resource (the environment, message buffers) written
during a stage and finalised at the stage edge.

**Channel** — a named ordering edge. A system produces or consumes a channel; the
scheduler runs producers before consumers.

**Chunk** — a fixed-size block of rows within an attribute; the unit of parallel
iteration.

**Component** — a plain data type holding one facet of an entity's state.

**DetRng** — the deterministic RNG, keyed on the run context and a salt so draws
do not depend on thread scheduling.

**Entity** — a compact handle (index and version) identifying an agent.

**Environment** — model-wide values keyed by name and type.

**Feature** — a Cargo feature gating an optional layer (`model`, `messaging`,
`gpu`, and so on).

**Migration** — moving an entity between archetypes when its component set
changes.

**Model** — a built simulation: world, environment, agents, and schedulers,
advanced with `tick`.

**Nested model** — an isolated child `Model` with its own world and seed, joined
to a parent through a bridge.

**Reduction** — a fold over a query's component column into an accumulator, with
per-chunk partials combined stably.

**Run context** — the per-system `(simulation_seed, tick, system_id)` passed to
each system; the basis for deterministic draws.

**Shard** — one partition of entity storage, typically one per worker.

**Specialisation** — how a message type is indexed and delivered (brute-force,
bucket, spatial, targeted).

**Stage** — a group of non-conflicting systems that run in parallel; stages run in
sequence.

**Sub-scheduler** — a named scope of systems sharing the model's world, run before
the root scheduler each tick.

**System** — a unit of per-tick work with a declared access set.
