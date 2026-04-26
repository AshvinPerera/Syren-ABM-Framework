use criterion::*;
use std::hint::black_box;

mod common;
use common::*;

fn reduce_benchmark(c: &mut Criterion) {
    let (registry, pos_id, wealth_id, prod_id) = make_registry();

    let mut group = c.benchmark_group("reduce");

    group.bench_function("reduce_sum_wealth_1M", |b| {
        let registry = registry.clone();
        b.iter_batched(
            || {
                let ecs = make_world(4, registry.clone());
                populate(&ecs, AGENTS_MED, pos_id, wealth_id, prod_id).unwrap();

                let q = query_builder(&registry)
                    .read::<Wealth>()
                    .unwrap()
                    .build()
                    .unwrap();

                (ecs, q)
            },
            |(ecs, q)| {
                let total = ecs
                    .world_ref()
                    .reduce_read::<Wealth, f32>(
                        q,
                        || 0.0f32,
                        |acc, w| *acc += w.value,
                        |acc, other| *acc += other,
                    )
                    .unwrap();

                black_box(total);
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(benches, reduce_benchmark);
criterion_main!(benches);
