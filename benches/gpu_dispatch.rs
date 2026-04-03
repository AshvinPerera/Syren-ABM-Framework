use criterion::*;
use std::hint::black_box;

use abm_framework::Scheduler;

mod gpu_common;
use gpu_common::*;

fn gpu_dispatch_hot_benchmark(c: &mut Criterion) {
    let (registry, energy_id) = make_registry();

    let ecs = make_world(4, registry);
    populate_energy(&ecs, AGENTS, energy_id).unwrap();

    let mut scheduler = Scheduler::new();
    scheduler.add_system(EnergyDecayGpu::new(1, energy_id));

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
