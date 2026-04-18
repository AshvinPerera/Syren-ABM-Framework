use criterion::*;
use std::hint::black_box;

mod common;
use common::*;

fn spawn_benchmark(c: &mut Criterion) {
    let (registry, pos_id, wealth_id, prod_id) = make_registry();

    let mut group = c.benchmark_group("spawn");

    for &(label, n) in &[
        ("spawn_100k", AGENTS_SMALL),
        ("spawn_1M",   AGENTS_MED),
        // ("spawn_10M",  AGENTS_LARGE),
    ] {
        let registry = registry.clone();
        group.bench_function(label, |b| {
            b.iter_batched(
                || make_world(4, registry.clone()),
                |ecs| {
                    populate(&ecs, n, pos_id, wealth_id, prod_id).unwrap();
                    black_box(ecs);
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, spawn_benchmark);
criterion_main!(benches);