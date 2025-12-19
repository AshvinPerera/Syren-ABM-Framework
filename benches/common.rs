use abm_framework::engine::component::{register_component, freeze_components, component_id_of};
use abm_framework::engine::entity::EntityShards;
use abm_framework::engine::manager::ECSManager;
use abm_framework::engine::types::{Bundle};
use abm_framework::engine::commands::Command;


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

pub fn setup_world(agent_count: usize) -> ECSManager {
    register_component::<Position>();
    register_component::<Wealth>();
    register_component::<Productivity>();
    freeze_components();

    let shards = EntityShards::new(4);
    let ecs = ECSManager::new(shards);

    {
        let world = ecs.world_ref();
        let data = world.data_mut();

        for _ in 0..agent_count {
            let mut b = Bundle::new();

            b.insert(component_id_of::<Position>(), Position { x: 0.0, y: 0.0 });
            b.insert(component_id_of::<Wealth>(), Wealth { value: 100.0 });
            b.insert(component_id_of::<Productivity>(), Productivity { rate: 1.0 });

            data.defer(Command::Spawn { bundle: b });
        }
    }

    ecs.apply_deferred_commands();
    ecs
}
