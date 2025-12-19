use std::hint::black_box;

use criterion::*;
use abm_framework::engine::component::{register_component, freeze_components, component_id_of};
use abm_framework::engine::entity::EntityShards;
use abm_framework::engine::manager::ECSManager;
use abm_framework::engine::types::{Bundle};
use abm_framework::engine::commands::Command;

mod common;
use common::Position;

fn spawn_benchmark(c: &mut Criterion) {
    register_component::<Position>();
    freeze_components();

    let mut group = c.benchmark_group("spawn");

    group.bench_function("spawn_1M_agents", |b| {
        b.iter(|| {
            let shards = EntityShards::new(4);
            let ecs = ECSManager::new(shards);

            {
                let world = ecs.world_ref();
                let data = world.data_mut();

                for _ in 0..common::AGENTS_MED {
                    let mut bundle = Bundle::new();
                    bundle.insert(
                        component_id_of::<Position>(),
                        Position { x: 0.0, y: 0.0 },
                    );

                    data.defer(Command::Spawn { bundle });
                }
            }

            // this is where entities actually materialize
            ecs.apply_deferred_commands();

            black_box(ecs);
        });
    });

    group.finish();
}

criterion_group!(benches, spawn_benchmark);
criterion_main!(benches);
