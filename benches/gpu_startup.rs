use criterion::*;
use std::hint::black_box;

use abm_framework::engine::scheduler::Scheduler;

mod gpu_common;
use gpu_common::*;

fn gpu_startup_benchmark(c: &mut Criterion) {
    init_components();

    c.bench_function("gpu_cold_first_tick_1M", |b| {
        b.iter_batched(
            || {
                let ecs = make_world(4);
                populate_energy(&ecs, AGENTS).unwrap();

                let mut scheduler = Scheduler::new();
                scheduler.add_system(EnergyDecayGpu::new(1));

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

criterion_group!(benches, gpu_startup_benchmark);
criterion_main!(benches);
