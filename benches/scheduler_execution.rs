use std::hint::black_box;
use std::sync::{Arc, RwLock};

use abm_framework::{
    advanced::EntityShards, AccessSets, Bundle, Command, ComponentRegistry, ECSManager, FnSystem,
    Scheduler,
};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

const SYSTEMS: usize = 256;
const ROWS: usize = 4096;

#[derive(Clone, Copy)]
struct TickValue {
    _value: u32,
}

fn scheduler_execution_benchmark(c: &mut Criterion) {
    c.bench_function("scheduler_execution/256_noop_systems", |b| {
        b.iter_batched(
            || {
                let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
                let value_id = {
                    let mut reg = registry.write().unwrap();
                    let id = reg.register::<TickValue>().unwrap();
                    reg.freeze();
                    id
                };
                let world = ECSManager::with_registry(EntityShards::new(4).unwrap(), registry);
                for i in 0..ROWS {
                    let mut bundle = Bundle::new();
                    bundle.insert(value_id, TickValue { _value: i as u32 });
                    world.world_ref().defer(Command::Spawn { bundle }).unwrap();
                }
                world.apply_deferred_commands().unwrap();

                let mut scheduler = Scheduler::new();
                for i in 0..SYSTEMS {
                    scheduler.add_system(FnSystem::new(
                        i as u16,
                        "noop",
                        AccessSets::default(),
                        |_| Ok(()),
                    ));
                }
                (world, scheduler)
            },
            |(world, mut scheduler)| {
                world.run(&mut scheduler).unwrap();
                black_box(world);
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, scheduler_execution_benchmark);
criterion_main!(benches);
