# Agents and lifecycle hooks

_Requires the `agents` feature (implied by `model`)._

At the ECS level, every agent is an entity with components. The **agent** layer
adds a naming and templating convention on top: a named kind of agent with a
fixed set of components, and bulk creation and destruction of that kind.

## Agent templates

An [`AgentTemplate`] names a kind of agent and lists its components. Build one
with the fluent builder and register it with the model:

```rust,ignore
AgentTemplate::builder("walker")
    .with_component::<Position>(position_id)?
    .with_capacity(count)
    .build()
```

The capacity reserves storage for the expected population, so the columns are
allocated once. A template is keyed by its name; populations and lifecycle
operations refer to that name.

## Populations

For an initial population, use the bulk path:
`ModelBuilder::with_agent_population` materialises a whole component column at
once rather than spawning agents individually. See [initialise and mutate
populations](../how-to/populations.md).

At runtime, a model spawns or despawns agents in batches keyed by template name.
Batch spawn and despawn are atomic: a batch either applies in full or, on error,
rolls back, so a failure leaves no half-created cohort.

## Lifecycle hooks

A template can carry **lifecycle hooks** that fire when agents of that kind are
spawned or despawned. Hooks do not run inside a system; they fire at the
**scheduler boundary**, after the stage that requested the structural change
completes. Structural mutation therefore stays outside the parallel region and
its timing is deterministic.

## Agents versus raw entities

A model can also drop to the raw ECS and spawn entities directly. The agent layer
is optional and does not change how storage or scheduling work.

[`AgentTemplate`]: https://docs.rs/syren/latest/syren/agents/struct.AgentTemplate.html
