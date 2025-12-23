#![allow(dead_code)]

use abm_framework::engine::component::{
    Bundle,
    register_component,
    freeze_components,
    component_id_of,
};
use abm_framework::engine::entity::EntityShards;
use abm_framework::engine::manager::{ECSManager, ECSData};
use abm_framework::engine::commands::Command;
use abm_framework::engine::error::{ECSResult, ExecutionError};

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

pub fn setup_world(agent_count: usize) -> ECSResult<ECSManager> {
    register_component::<Position>();
    register_component::<Wealth>();
    register_component::<Productivity>();
    freeze_components();

    let shards = EntityShards::new(4);
    let data = ECSData::new(shards);
    let ecs = ECSManager::new(data);

    let world = ecs.world_ref();

   world.with_exclusive(|_data| {
        for _ in 0..agent_count {
            let mut bundle = Bundle::new();
            bundle.insert(
                component_id_of::<Position>(),
                Position { x: 0.0, y: 0.0 },
            );
            bundle.insert(
                component_id_of::<Wealth>(),
                Wealth { value: 100.0 },
            );
            bundle.insert(
                component_id_of::<Productivity>(),
                Productivity { rate: 1.0 },
            );

            world.defer(Command::Spawn { bundle })?;
        }

        Ok::<(), ExecutionError>(())
    })??;

    ecs.apply_deferred_commands()?;
    Ok(ecs)
}
