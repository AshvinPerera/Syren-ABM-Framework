use std::hint::black_box;
use std::sync::{Arc, RwLock};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use syren::{
    advanced::EntityShards, Bundle, Command, ComponentRegistry, ECSManager, ECSResult, Entity,
};

const N: usize = 4096;

#[derive(Clone, Copy)]
struct CoreValue {
    _value: u32,
}

#[derive(Clone, Copy)]
struct AddedValue {
    _value: u32,
}

fn registry() -> (
    Arc<RwLock<ComponentRegistry>>,
    syren::ComponentID,
    syren::ComponentID,
) {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let ids = {
        let mut reg = registry.write().unwrap();
        let core = reg.register::<CoreValue>().unwrap();
        let added = reg.register::<AddedValue>().unwrap();
        reg.freeze();
        (core, added)
    };
    (registry, ids.0, ids.1)
}

fn world_with_core(
    registry: Arc<RwLock<ComponentRegistry>>,
    core_id: syren::ComponentID,
) -> ECSResult<(ECSManager, Vec<Entity>)> {
    let ecs = ECSManager::with_registry(EntityShards::new(4)?, registry);
    let world = ecs.world_ref();
    for i in 0..N {
        let mut bundle = Bundle::new();
        bundle.insert(core_id, CoreValue { _value: i as u32 });
        world.defer(Command::Spawn { bundle })?;
    }
    let events = ecs.apply_deferred_commands()?;
    Ok((
        ecs,
        events.spawned.iter().map(|event| event.entity).collect(),
    ))
}

fn structural_mutation_benchmarks(c: &mut Criterion) {
    let (registry, core_id, added_id) = registry();
    let mut group = c.benchmark_group("structural_mutation");

    group.bench_function("add_component_4096", |b| {
        let registry = Arc::clone(&registry);
        b.iter_batched(
            || world_with_core(Arc::clone(&registry), core_id).unwrap(),
            |(ecs, entities)| {
                let world = ecs.world_ref();
                for (i, entity) in entities.into_iter().enumerate() {
                    world
                        .defer(Command::Add {
                            entity,
                            component_id: added_id,
                            value: Box::new(AddedValue { _value: i as u32 }),
                        })
                        .unwrap();
                }
                ecs.apply_deferred_commands().unwrap();
                black_box(ecs);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("remove_component_4096", |b| {
        let registry = Arc::clone(&registry);
        b.iter_batched(
            || {
                let (ecs, entities) = world_with_core(Arc::clone(&registry), core_id).unwrap();
                let world = ecs.world_ref();
                for (i, entity) in entities.iter().copied().enumerate() {
                    world
                        .defer(Command::Add {
                            entity,
                            component_id: added_id,
                            value: Box::new(AddedValue { _value: i as u32 }),
                        })
                        .unwrap();
                }
                ecs.apply_deferred_commands().unwrap();
                (ecs, entities)
            },
            |(ecs, entities)| {
                let world = ecs.world_ref();
                for entity in entities {
                    world
                        .defer(Command::Remove {
                            entity,
                            component_id: added_id,
                        })
                        .unwrap();
                }
                ecs.apply_deferred_commands().unwrap();
                black_box(ecs);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("set_component_4096", |b| {
        let registry = Arc::clone(&registry);
        b.iter_batched(
            || world_with_core(Arc::clone(&registry), core_id).unwrap(),
            |(ecs, entities)| {
                let world = ecs.world_ref();
                for (i, entity) in entities.into_iter().enumerate() {
                    world
                        .defer(Command::Set {
                            entity,
                            component_id: core_id,
                            value: Box::new(CoreValue {
                                _value: (i as u32) + 1,
                            }),
                        })
                        .unwrap();
                }
                ecs.apply_deferred_commands().unwrap();
                black_box(ecs);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("despawn_4096", |b| {
        let registry = Arc::clone(&registry);
        b.iter_batched(
            || world_with_core(Arc::clone(&registry), core_id).unwrap(),
            |(ecs, entities)| {
                let world = ecs.world_ref();
                for entity in entities {
                    world.defer(Command::Despawn { entity }).unwrap();
                }
                ecs.apply_deferred_commands().unwrap();
                black_box(ecs);
            },
            BatchSize::SmallInput,
        );
    });

    // Columnar batch migration: one command, bitwise row moves, no boxing.
    group.bench_function("add_batch_4096", |b| {
        let registry = Arc::clone(&registry);
        b.iter_batched(
            || world_with_core(Arc::clone(&registry), core_id).unwrap(),
            |(ecs, entities)| {
                let world = ecs.world_ref();
                let n = entities.len();
                let values: Vec<AddedValue> =
                    (0..n).map(|i| AddedValue { _value: i as u32 }).collect();
                world
                    .defer(Command::AddComponentBatch {
                        entities,
                        component_id: added_id,
                        values: Box::new(values),
                        len: n,
                    })
                    .unwrap();
                ecs.apply_deferred_commands().unwrap();
                black_box(ecs);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("remove_batch_4096", |b| {
        let registry = Arc::clone(&registry);
        b.iter_batched(
            || {
                let (ecs, entities) = world_with_core(Arc::clone(&registry), core_id).unwrap();
                let world = ecs.world_ref();
                let n = entities.len();
                let values: Vec<AddedValue> =
                    (0..n).map(|i| AddedValue { _value: i as u32 }).collect();
                world
                    .defer(Command::AddComponentBatch {
                        entities: entities.clone(),
                        component_id: added_id,
                        values: Box::new(values),
                        len: n,
                    })
                    .unwrap();
                ecs.apply_deferred_commands().unwrap();
                (ecs, entities)
            },
            |(ecs, entities)| {
                let world = ecs.world_ref();
                world
                    .defer(Command::RemoveComponentBatch {
                        entities,
                        component_id: added_id,
                    })
                    .unwrap();
                ecs.apply_deferred_commands().unwrap();
                black_box(ecs);
            },
            BatchSize::SmallInput,
        );
    });

    // Batched despawn: one command, column locks taken once per archetype.
    group.bench_function("despawn_batch_4096", |b| {
        let registry = Arc::clone(&registry);
        b.iter_batched(
            || world_with_core(Arc::clone(&registry), core_id).unwrap(),
            |(ecs, entities)| {
                let world = ecs.world_ref();
                world
                    .defer(Command::DespawnBatchTagged {
                        entities,
                        template_id: syren::AgentTemplateId(0),
                    })
                    .unwrap();
                ecs.apply_deferred_commands().unwrap();
                black_box(ecs);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, structural_mutation_benchmarks);
criterion_main!(benches);
