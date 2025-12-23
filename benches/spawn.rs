use std::hint::black_box;
use std::sync::Once;

use criterion::*;
use abm_framework::engine::component::{
    Bundle,
    register_component,
    freeze_components,
    component_id_of,
};
use abm_framework::engine::entity::EntityShards;
use abm_framework::engine::manager::{ECSManager, ECSData};
use abm_framework::engine::commands::Command;
use abm_framework::engine::error::ExecutionError;

mod common;
use common::Position;


static INIT: Once = Once::new();

fn init_components() {
    INIT.call_once(|| {
        register_component::<Position>();
        freeze_components();
    });
}

fn spawn_benchmark(c: &mut Criterion) {
    init_components();

    let mut group = c.benchmark_group("spawn");

    group.bench_function("spawn_1M_agents", |b| {
        b.iter(|| {
            let shards = EntityShards::new(4);
            let data = ECSData::new(shards);
            let ecs = ECSManager::new(data);

            let world = ecs.world_ref();

            let _ = world
                .with_exclusive(|_data| {
                    for _ in 0..common::AGENTS_MED {
                        let mut bundle = Bundle::new();
                        bundle.insert(
                            component_id_of::<Position>(),
                            Position { x: 0.0, y: 0.0 },
                        );

                        world
                            .defer(Command::Spawn { bundle })
                            .expect("spawn defer failed in benchmark");
                    }

                    Ok::<(), ExecutionError>(())
                })
                .expect("exclusive world setup failed");

            ecs.apply_deferred_commands()
                .expect("apply_deferred_commands failed");

            black_box(ecs);
        });
    });

    group.finish();
}

criterion_group!(benches, spawn_benchmark);
criterion_main!(benches);
