# Engine architecture

Syren is layered. The **engine** is the core ECS; the optional layers (model,
agents, environment, messaging, space, GPU, profiling) build on it behind
features. Understanding the ownership — who holds what, and when it is fixed — is
the key to reading the code.

## Ownership

- **`ComponentRegistry`** maps component types to identifiers and is **frozen**
  before the world runs. Freezing is what lets storage layouts and query
  resolution be decided once.
- **`EntityShards`** owns entity allocation, partitioned into shards (typically
  one per worker) so spawns and despawns do not contend on a single structure.
- **Archetypes** own the columnar storage: one chunked attribute per component.
  An entity's components live in the archetype for its exact component set;
  changing that set migrates the entity.
- **`ECSManager`** ties these together into the world and exposes an
  `ECSReference` to systems.
- **`Scheduler`** owns the execution plan: it packs systems into stages from
  their declared access and channel constraints, and runs them.
- **Boundaries** (environment, message buffers) are model-owned resources written
  during a stage and finalised at the stage edge.

## The tick

A tick runs sub-schedulers, then nested models, then the root scheduler. Within a
scheduler, stages run in sequence and the systems in a stage run in parallel over
Rayon. Structural mutation (spawns, despawns, migrations) is deferred and applied
at the scheduler boundary, keeping it out of the parallel region.

## Determinism by construction

Determinism is designed in, not bolted on:

- The scheduler produces the same stages and activation order every run.
- `DetRng` keys draws on `(seed, tick, system_id, salt)` rather than a shared
  stream, so work stealing cannot change results.
- Parallel accumulation combines per-worker partials in a fixed order.

These are the invariants the reproducibility guarantee rests on; see
[reproducibility](../science/reproducibility.md) and [safety
invariants](../reference/safety.md).

## The model layer

The `model` layer wraps the engine: `ModelBuilder` registers components, agent
templates, environment keys, message types, sub-schedulers, and nested models,
validates them, and constructs a `Model`. It is where the seed is applied to the
root scheduler and shared sub-schedulers. The engine has no notion of "agents" or
"environment"; those are conveniences the model layer provides on top of entities,
components, and boundaries.
