#![allow(dead_code)]

use std::sync::{Arc, RwLock};

use abm_framework::{
    Bundle,
    ComponentRegistry,
    ECSManager,
    EntityShards,
    Command,
    ECSResult,
    ECSError,
    QueryBuilder,
    ComponentID,
};

pub const AGENTS_SMALL: usize = 100_000;
pub const AGENTS_MED: usize = 1_000_000;
pub const AGENTS_LARGE: usize = 10_000_000;

#[derive(Clone, Copy)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy)]
pub struct Wealth {
    pub value: f32,
}

#[derive(Clone, Copy)]
pub struct Productivity {
    pub rate: f32,
}

/// Creates a shared, frozen component registry with Position, Wealth, and
/// Productivity registered.  Returns the registry handle together with the
/// three component IDs so callers never need to lock the registry just to
/// look up an ID.
pub fn make_registry() -> (Arc<RwLock<ComponentRegistry>>, ComponentID, ComponentID, ComponentID) {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let (pos_id, wealth_id, prod_id) = {
        let mut reg = registry.write().unwrap();
        let pos = reg.register::<Position>().unwrap();
        let wealth = reg.register::<Wealth>().unwrap();
        let prod = reg.register::<Productivity>().unwrap();
        reg.freeze();
        (pos, wealth, prod)
    };
    (registry, pos_id, wealth_id, prod_id)
}

/// Constructs an ECSManager backed by the given instance-owned registry.
pub fn make_world(shards: usize, registry: Arc<RwLock<ComponentRegistry>>) -> ECSManager {
    let shards = EntityShards::new(shards).unwrap();
    ECSManager::with_registry(shards, registry)
}

/// Spawns `agent_count` entities, each carrying Position, Wealth, and
/// Productivity components.
pub fn populate(
    ecs: &ECSManager,
    agent_count: usize,
    pos_id: ComponentID,
    wealth_id: ComponentID,
    prod_id: ComponentID,
) -> ECSResult<()> {
    let world = ecs.world_ref();

    world.with_exclusive(|_| {
        for _ in 0..agent_count {
            let mut bundle = Bundle::new();

            bundle.insert(pos_id, Position { x: 0.0, y: 0.0 });
            bundle.insert(wealth_id, Wealth { value: 100.0 });
            bundle.insert(prod_id, Productivity { rate: 1.0 });

            world.defer(Command::Spawn { bundle })?;
        }

        Ok::<(), ECSError>(())
    })?;

    ecs.apply_deferred_commands()?;
    Ok(())
}

/// Convenience helper: builds a QueryBuilder that resolves component IDs
/// through the given instance-owned registry (not the global registry).
pub fn query_builder(registry: &Arc<RwLock<ComponentRegistry>>) -> QueryBuilder {
    QueryBuilder::with_registry(Arc::clone(registry))
}
