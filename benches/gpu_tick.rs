use criterion::*;
use std::hint::black_box;

use abm_framework::engine::scheduler::Scheduler;

mod gpu_common;
use gpu_common::*;

fn gpu_tick_scaling(c: &mut Criterion) {
    init_components();

    let mut group = c.benchmark_group("gpu_tick_scaling");

    for &(label, n) in &[
        ("gpu_tick_100k", 100_000usize),
        ("gpu_tick_1M",   1_000_000usize),
    ] {
        group.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let ecs = make_world(4);
                    populate_energy(&ecs, n).unwrap();

                    let mut scheduler = Scheduler::new();
                    scheduler.add_system(EnergyDecayGpu::new(1));

                    ecs.run(&mut scheduler).unwrap();

                    (ecs, scheduler)
                },
                |(ecs, mut scheduler)| {
                    ecs.run(&mut scheduler).unwrap();
                    black_box(ecs);
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, gpu_tick_scaling);
criterion_main!(benches);
