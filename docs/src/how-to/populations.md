# Initialise and mutate populations

## Build an initial population in bulk

Prefer the bulk path for the starting population: build a `Vec` per component
column and hand each to `with_agent_population`. This materialises the columns
once instead of spawning agents individually.

```rust,ignore
let population: Vec<Position> = vec![Position { x: 0 }; 10_000];

let model = ModelBuilder::new()
    .with_component_registry(Arc::clone(&registry))
    .with_shards(EntityShards::new(1)?)
    .with_agent_template(
        AgentTemplate::builder("walker")
            .with_component::<Position>(position_id)?
            .with_capacity(population.len())
            .build(),
    )?
    .with_agent_population("walker", position_id, population)?
    .build()?;
```

For a multi-component template, call `with_agent_population` once per component,
each with a `Vec` of the same length. The builder groups columns by template name
and spawns each agent once with all of its columns.

## Spawn and despawn at runtime

At runtime, add or remove agents in **batches** keyed by template name, through
the model's batch spawn and despawn methods. Batches are atomic: on error the
whole batch rolls back, so a failure never leaves a partial cohort. Structural
changes take effect at the scheduler boundary, after the current stage, which is
also where lifecycle hooks fire.

Prefer a batch over many single spawns: batched structural mutation is both
faster and easier to reason about, because the whole change lands at one
well-defined point in the tick.

## Choosing shard count

`EntityShards::new(n)` fixes the number of shards. A shard addresses a bounded
number of entities, so size the shard count for the largest population the model
will hold. The macroeconomy example derives its shard count from the population;
see its `shards_for_population` for a worked rule.
