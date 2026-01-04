use criterion::*;
use std::hint::black_box;

mod common;
use common::*;

fn reduce_benchmark(c: &mut Criterion) {
    init_components();

    let mut group = c.benchmark_group("reduce");

    group.bench_function("reduce_sum_wealth_1M", |b| {
        b.iter_batched(
            || {
                let ecs = make_world(4);
                populate(&ecs, AGENTS_MED).unwrap();

                let q = ecs
                    .world_ref()
                    .query().unwrap()
                    .read::<Wealth>().unwrap()
                    .build().unwrap();

                (ecs, q)
            },
            |(_ecs, q)| {
                let total = _ecs
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
