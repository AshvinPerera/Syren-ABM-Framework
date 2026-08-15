# Define components and agent templates

## Define a component

A component is a plain type. Keep it `Copy` where you can — components are stored
in columnar arrays and copied during migration.

```rust,ignore
#[derive(Clone, Copy, Default)]
struct Position {
    x: i64,
}
```

Register it and freeze the registry before building the model:

```rust,ignore
let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
let position_id = registry.write().unwrap().register::<Position>()?;
registry.write().unwrap().freeze();
```

`register` returns the `ComponentID` you pass to templates and populations.
Freeze once, after all component types are registered.

## Define an agent template

_Requires the `agents` feature (implied by `model`)._

A template names a kind of agent and lists its components and capacity:

```rust,ignore
let walker = AgentTemplate::builder("walker")
    .with_component::<Position>(position_id)?
    .with_capacity(10_000)
    .build();
```

Register templates on the builder with `with_agent_template`. A template with
several components lists each `with_component` before `build`; the population is
then supplied one column per component (see [initialise and mutate
populations](populations.md)).

## Multi-component agents

Give a template every component the agent kind has:

```rust,ignore
AgentTemplate::builder("firm")
    .with_component::<Firm>(ids.firm)?
    .with_component::<FirmStocks>(ids.firm_stocks)?
    .with_capacity(firm_count)
    .build()
```

The capacity should be the expected population so the columns are allocated once
rather than grown incrementally.
