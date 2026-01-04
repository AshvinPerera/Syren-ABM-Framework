#![allow(dead_code)]

use std::sync::Once;

use abm_framework::engine::component::{
    Bundle,
    register_component,
    freeze_components,
    component_id_of,
};
use abm_framework::engine::entity::EntityShards;
use abm_framework::engine::manager::{ECSManager, ECSData};
use abm_framework::engine::commands::Command;
use abm_framework::engine::error::{ECSResult, ECSError};

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

static INIT: Once = Once::new();

pub fn init_components() {
    INIT.call_once(|| {
        register_component::<Position>().unwrap();
        register_component::<Wealth>().unwrap();
        register_component::<Productivity>().unwrap();
        freeze_components().unwrap();
    });
}

pub fn make_world(shards: usize) -> ECSManager {
    let shards = EntityShards::new(shards);
    let data = ECSData::new(shards);
    ECSManager::new(data)
}

pub fn populate(ecs: &ECSManager, agent_count: usize) -> ECSResult<()> {
    let world = ecs.world_ref();

    world.with_exclusive(|_| {
        for _ in 0..agent_count {
            let mut bundle = Bundle::new();

            bundle.insert(
                component_id_of::<Position>()?,
                Position { x: 0.0, y: 0.0 },
            );
            bundle.insert(
                component_id_of::<Wealth>()?,
                Wealth { value: 100.0 },
            );
            bundle.insert(
                component_id_of::<Productivity>()?,
                Productivity { rate: 1.0 },
            );

            world.defer(Command::Spawn { bundle })?;
        }

        Ok::<(), ECSError>(())
    })?;

    ecs.apply_deferred_commands()?;
    Ok(())
}
