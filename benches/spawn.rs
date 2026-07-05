use criterion::*;
use std::hint::black_box;

use abm_framework::advanced::EntityShards;
use abm_framework::{AgentTemplateId, Command, ECSManager, Signature};
use abm_framework::{BatchColumn, SpawnBatch};

mod common;
use common::*;

fn spawn_benchmark(c: &mut Criterion) {
    let (registry, pos_id, wealth_id, prod_id) = make_registry();

    let mut group = c.benchmark_group("spawn");
    group.sample_size(10);

    for &(label, n) in &[
        ("spawn_100k", AGENTS_SMALL),
        ("spawn_1M", AGENTS_MED),
        // ("spawn_10M",  AGENTS_LARGE),
    ] {
        let registry = registry.clone();
        group.bench_function(label, |b| {
            b.iter_batched(
                || make_world(4, registry.clone()),
                |ecs| {
                    populate(&ecs, n, pos_id, wealth_id, prod_id).unwrap();
                    black_box(ecs)
                },
                BatchSize::LargeInput,
            );
        });
    }

    // Columnar batch path: one Vec per component for the whole population.
    for &(label, n) in &[
        ("spawn_batch_100k", AGENTS_SMALL),
        ("spawn_batch_1M", AGENTS_MED),
    ] {
        let registry = registry.clone();
        group.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let ecs: ECSManager =
                        ECSManager::with_registry(EntityShards::new(4).unwrap(), registry.clone());
                    let positions: Vec<Position> =
                        (0..n).map(|_| Position { x: 0.0, y: 0.0 }).collect();
                    let wealths: Vec<Wealth> = (0..n).map(|_| Wealth { value: 100.0 }).collect();
                    let prods: Vec<Productivity> =
                        (0..n).map(|_| Productivity { rate: 1.0 }).collect();
                    (ecs, positions, wealths, prods)
                },
                |(ecs, positions, wealths, prods)| {
                    let mut signature = Signature::default();
                    signature.set(pos_id);
                    signature.set(wealth_id);
                    signature.set(prod_id);
                    ecs.world_ref()
                        .defer(Command::SpawnBatchTagged {
                            batch: SpawnBatch {
                                count: n,
                                signature,
                                columns: vec![
                                    BatchColumn {
                                        component_id: pos_id,
                                        values: Box::new(positions),
                                        len: n,
                                    },
                                    BatchColumn {
                                        component_id: wealth_id,
                                        values: Box::new(wealths),
                                        len: n,
                                    },
                                    BatchColumn {
                                        component_id: prod_id,
                                        values: Box::new(prods),
                                        len: n,
                                    },
                                ],
                            },
                            template_id: AgentTemplateId(0),
                        })
                        .unwrap();
                    ecs.apply_deferred_commands().unwrap();
                    black_box(ecs)
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, spawn_benchmark);
criterion_main!(benches);
