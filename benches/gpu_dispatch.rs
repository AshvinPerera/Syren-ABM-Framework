use criterion::*;
use std::hint::black_box;

use abm_framework::engine::scheduler::Scheduler;

mod gpu_common;
use gpu_common::*;

fn gpu_dispatch_hot_benchmark(c: &mut Criterion) {
    init_components();

    let ecs = make_world(4);
    populate_energy(&ecs, AGENTS).unwrap();

    let mut scheduler = Scheduler::new();
    scheduler.add_system(EnergyDecayGpu::new(1));

    ecs.run(&mut scheduler).unwrap();

    c.bench_function("gpu_hot_tick_1M", |b| {
        b.iter(|| {
            ecs.run(&mut scheduler).unwrap();
            black_box(&ecs);
        });
    });
}

criterion_group!(benches, gpu_dispatch_hot_benchmark);
criterion_main!(benches);
