use criterion::*;
use std::hint::black_box;

use abm_framework::{Read, Write};

mod common;
use common::*;

fn iterate_benchmark(c: &mut Criterion) {
    let (registry, pos_id, wealth_id, prod_id) = make_registry();

    let mut group = c.benchmark_group("iterate");

    group.bench_function("for_each_write_wealth_1M", |b| {
        let registry = registry.clone();
        b.iter_batched(
            || {
                let ecs = make_world(4, registry.clone());
                populate(&ecs, AGENTS_MED, pos_id, wealth_id, prod_id).unwrap();

                let q = query_builder(&registry)
                    .write::<Wealth>()
                    .unwrap()
                    .build()
                    .unwrap();

                (ecs, q)
            },
            |(ecs, q)| {
                ecs.world_ref()
                    .for_each::<(Write<Wealth>,)>(q, &|w| {
                        w.0.value *= 1.0001;
                    })
                    .unwrap();

                black_box(ecs);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("for_each_read_productivity_1M", |b| {
        let registry = registry.clone();
        b.iter_batched(
            || {
                let ecs = make_world(4, registry.clone());
                populate(&ecs, AGENTS_MED, pos_id, wealth_id, prod_id).unwrap();

                let q = query_builder(&registry)
                    .read::<Productivity>()
                    .unwrap()
                    .build()
                    .unwrap();

                (ecs, q)
            },
            |(ecs, q)| {
                let total = ecs
                    .world_ref()
                    .reduce_read::<Productivity, f32>(
                        q,
                        || 0.0f32,
                        |acc, p| *acc += p.rate,
                        |acc, other| *acc += other,
                    )
                    .unwrap();

                black_box(total);
                black_box(ecs);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("for_each_read_write_prod_to_wealth_1M", |b| {
        let registry = registry.clone();
        b.iter_batched(
            || {
                let ecs = make_world(4, registry.clone());
                populate(&ecs, AGENTS_MED, pos_id, wealth_id, prod_id).unwrap();

                let q = query_builder(&registry)
                    .read::<Productivity>()
                    .unwrap()
                    .write::<Wealth>()
                    .unwrap()
                    .build()
                    .unwrap();

                (ecs, q)
            },
            |(ecs, q)| {
                ecs.world_ref()
                    .for_each::<(Read<Productivity>, Write<Wealth>)>(q, &|(p, w)| {
                        w.value += p.rate;
                    })
                    .unwrap();

                black_box(ecs);
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(benches, iterate_benchmark);
criterion_main!(benches);
