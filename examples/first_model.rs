//! Introductory model: a population of walkers on an integer number line.
//!
//! A minimal model that exercises the framework end to end. Each tick, every
//! walker takes one deterministic step left or right; after a fixed number of
//! ticks a reduction reports the population's spread.
//!
//! Run it with the `model` feature:
//!
//! ```bash
//! cargo run --example first_model --features model
//! ```
//!
//! The guide walks through this file section by section; the `ANCHOR` comments
//! mark the ranges it includes, so the guide always shows compiled code.

// ANCHOR: component
use std::sync::{Arc, RwLock};

use syren::agents::AgentTemplate;
use syren::model::ModelBuilder;
use syren::{advanced::EntityShards, ComponentRegistry, DetRng, FnSystem, QueryBuilder, Welford};

/// Each walker holds a single integer position on a number line.
#[derive(Clone, Copy, Default)]
struct Position {
    x: i64,
}
// ANCHOR_END: component

/// Number of walkers in the population.
const WALKERS: usize = 10_000;
/// Number of ticks (steps) to simulate.
const TICKS: u64 = 50;
/// Model seed. The same seed reproduces the same walk at any thread count.
const SEED: u64 = 42;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ANCHOR: build
    // Register the component up front, then freeze the registry so archetype
    // layouts are fixed for the life of the model.
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let position_id = registry.write().unwrap().register::<Position>()?;
    registry.write().unwrap().freeze();

    // Every walker starts at the origin.
    let population: Vec<Position> = vec![Position { x: 0 }; WALKERS];

    // A query that writes `Position`. The system derives its access set from
    // this query (`from_queries`), so the declared access cannot drift from
    // what the system actually touches.
    let step_query = QueryBuilder::with_registry(Arc::clone(&registry))
        .write::<Position>()?
        .build()?;

    let walk_query = step_query.clone();
    let mut model = ModelBuilder::new()
        .with_seed(SEED)
        .with_component_registry(Arc::clone(&registry))
        .with_shards(EntityShards::new(1)?)
        .with_agent_template(
            AgentTemplate::builder("walker")
                .with_component::<Position>(position_id)?
                .with_capacity(WALKERS)
                .build(),
        )?
        .with_agent_population("walker", position_id, population)?
        .with_system(FnSystem::from_queries(
            0,
            "random_walk",
            &[&step_query],
            move |ecs| {
                // `run_context` carries the seed, tick, and system id. Salting a
                // per-entity `DetRng` with the entity's identity makes each
                // walker's step independent of the order workers visit rows in,
                // so the walk is identical whatever the thread count.
                let context = ecs.run_context();
                ecs.for_each_entity_w1::<Position>(walk_query.clone(), move |entity, pos| {
                    let mut rng = DetRng::from_context(context, u64::from(entity.index()));
                    pos.x += if rng.next_below(2) == 0 { -1 } else { 1 };
                })
            },
        ))
        .build()?;
    // ANCHOR_END: build

    // ANCHOR: run
    model.run(TICKS)?;
    // ANCHOR_END: run

    // ANCHOR: reduce
    // A single-pass reduction over the population: `Welford` accumulates count,
    // mean, and variance and combines partials from parallel chunks stably.
    let summary_query = QueryBuilder::with_registry(registry)
        .read::<Position>()?
        .build()?;
    let stats = model.ecs().world_ref().reduce_read::<Position, Welford>(
        summary_query,
        Welford::default,
        |acc, pos| acc.push(pos.x as f64),
        |acc, other| acc.combine(other),
    )?;
    println!(
        "after {TICKS} ticks: count={} mean={:.4} variance={:.4}",
        stats.n,
        stats.mean,
        stats.variance()
    );
    // ANCHOR_END: reduce

    Ok(())
}
