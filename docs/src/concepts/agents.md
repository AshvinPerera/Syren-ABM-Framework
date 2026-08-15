# Agents and lifecycle hooks

_Requires the `agents` feature (implied by `model`)._

At the ECS level everything is an entity with components. The **agent** layer
adds a naming and templating convention on top: a way to say "a firm is an entity
with these components" and to create and destroy firms in bulk.

## Agent templates

An [`AgentTemplate`] names a kind of agent and lists its components. You build one
with the fluent builder and register it with the model:

```rust,ignore
AgentTemplate::builder("walker")
    .with_component::<Position>(position_id)?
    .with_capacity(count)
    .build()
```

The capacity reserves storage for the expected population so the columns are
allocated once. A template is keyed by its name; populations and lifecycle
operations refer to that name.

## Populations

For an initial population, prefer the bulk path: `ModelBuilder::with_agent_population`
materialises a whole component column at once rather than spawning agents one at
a time. See [initialise and mutate populations](../how-to/populations.md).

At runtime, a model spawns or despawns agents in batches keyed by template name.
Batch spawn and despawn are atomic: a batch either applies in full or, on error,
rolls back, so a failure never leaves a half-created cohort.

## Lifecycle hooks

A template can carry **lifecycle hooks** that fire when agents of that kind are
spawned or despawned. Hooks do not run in the middle of a system; they fire at
the **scheduler boundary**, after the stage that requested the structural change
completes. This keeps structural mutation out of the parallel region and its
timing deterministic.

## Agents versus raw entities

You can always drop to the raw ECS and spawn entities directly. The agent layer
is a convenience for models that think in named populations; it does not change
how storage or scheduling work underneath.

[`AgentTemplate`]: https://docs.rs/syren/latest/syren/agents/struct.AgentTemplate.html
