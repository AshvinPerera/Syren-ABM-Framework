use criterion::*;
use std::hint::black_box;

mod common;
use common::*;

fn iterate_benchmark(c: &mut Criterion) {
    init_components();

    let mut group = c.benchmark_group("iterate");

    group.bench_function("for_each_write_wealth_1M", |b| {
        b.iter_batched(
            || {
                let ecs = make_world(4);
                populate(&ecs, AGENTS_MED).unwrap();

                // IMPORTANT: your QueryBuilder is type-driven (no ComponentID args)
                let q = ecs
                    .world_ref()
                    .query().unwrap()
                    .write::<Wealth>().unwrap()
                    .build().unwrap();

                (ecs, q)
            },
            |(ecs, q)| {
                ecs.world_ref()
                    .for_each_write::<Wealth>(q, |w| {
                        w.value *= 1.0001;
                    })
                    .unwrap();

                black_box(ecs);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("for_each_read_productivity_1M", |b| {
        b.iter_batched(
            || {
                let ecs = make_world(4);
                populate(&ecs, AGENTS_MED).unwrap();

                let q = ecs
                    .world_ref()
                    .query().unwrap()
                    .read::<Productivity>().unwrap()
                    .build().unwrap();

                (ecs, q)
            },
            |(ecs, q)| {
                let total = ecs.world_ref()
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
        b.iter_batched(
            || {
                let ecs = make_world(4);
                populate(&ecs, AGENTS_MED).unwrap();

                let q = ecs
                    .world_ref()
                    .query().unwrap()
                    .read::<Productivity>().unwrap()
                    .write::<Wealth>().unwrap()
                    .build().unwrap();

                (ecs, q)
            },
            |(ecs, q)| {
                ecs.world_ref()
                    .for_each_read_write::<Productivity, Wealth>(q, |p, w| {
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
