//! Compiles a minimal model against the packaged `syren` crate to prove the
//! public `model` API is usable by an external consumer and that the published
//! package contains every source file the library needs.

use std::sync::{Arc, RwLock};

use syren::agents::AgentTemplate;
use syren::model::ModelBuilder;
use syren::{ComponentRegistry, DetRng, FnSystem, QueryBuilder, Welford};

#[derive(Clone, Copy, Default)]
struct Position {
    x: i64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let position_id = registry.write().unwrap().register::<Position>()?;
    registry.write().unwrap().freeze();

    let step_query = QueryBuilder::with_registry(Arc::clone(&registry))
        .write::<Position>()?
        .build()?;
    let walk_query = step_query.clone();

    let mut model = ModelBuilder::new()
        .with_seed(42)
        .with_component_registry(Arc::clone(&registry))
        .with_shards(syren::advanced::EntityShards::new(1)?)
        .with_agent_template(
            AgentTemplate::builder("walker")
                .with_component::<Position>(position_id)?
                .with_capacity(4)
                .build(),
        )?
        .with_agent_population("walker", position_id, vec![Position { x: 0 }; 4])?
        .with_system(FnSystem::from_queries(
            0,
            "random_walk",
            &[&step_query],
            move |ecs| {
                let context = ecs.run_context();
                ecs.for_each_entity_w1::<Position>(walk_query.clone(), move |entity, pos| {
                    let mut rng = DetRng::from_context(context, u64::from(entity.index()));
                    pos.x += if rng.next_below(2) == 0 { -1 } else { 1 };
                })
            },
        ))
        .build()?;

    model.run(3)?;

    let summary_query = QueryBuilder::with_registry(registry)
        .read::<Position>()?
        .build()?;
    let stats = model.ecs().world_ref().reduce_read::<Position, Welford>(
        summary_query,
        Welford::default,
        |acc, pos| acc.push(pos.x as f64),
        |acc, other| acc.combine(other),
    )?;

    assert_eq!(stats.n, 4, "reduction must see every walker");
    println!("consumer smoke ok: mean={:.4}", stats.mean);
    Ok(())
}
