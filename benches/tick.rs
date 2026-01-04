use criterion::*;
use std::hint::black_box;

use abm_framework::engine::component::{Signature, component_id_of};
use abm_framework::engine::scheduler::Scheduler;
use abm_framework::engine::systems::{FnSystem, AccessSets};

mod common;
use common::*;

fn tick_benchmark(c: &mut Criterion) {
    init_components();

    let mut group = c.benchmark_group("tick");

    group.bench_function("tick_2_systems_1M", |b| {
        b.iter_batched(
            || {
                let ecs = make_world(4);
                populate(&ecs, AGENTS_MED).unwrap();

                let q_prod_to_wealth = ecs
                    .world_ref()
                    .query().unwrap()
                    .read::<Productivity>().unwrap()
                    .write::<Wealth>().unwrap()
                    .build().unwrap();

                let q_decay_wealth = ecs
                    .world_ref()
                    .query().unwrap()
                    .write::<Wealth>().unwrap()
                    .build().unwrap();

                let prod_id = component_id_of::<Productivity>().unwrap();
                let wealth_id = component_id_of::<Wealth>().unwrap();

                let mut scheduler = Scheduler::new();

                // System 1: wealth += productivity
                let access_prod_to_wealth = AccessSets {
                    read: {
                        let mut s = Signature::default();
                        s.set(prod_id);
                        s
                    },
                    write: {
                        let mut s = Signature::default();
                        s.set(wealth_id);
                        s
                    },
                };

                scheduler.add_system(FnSystem::new(
                    1,
                    "production",
                    access_prod_to_wealth,
                    move |world| {
                        world.for_each_read_write::<Productivity, Wealth>(
                            q_prod_to_wealth.clone(),
                            |p, w| w.value += p.rate,
                        )?;
                        Ok(())
                    },
                ));

                // System 2: wealth decay
                let access_decay = AccessSets {
                    read: Signature::default(),
                    write: {
                        let mut s = Signature::default();
                        s.set(wealth_id);
                        s
                    },
                };

                scheduler.add_system(FnSystem::new(
                    2,
                    "decay",
                    access_decay,
                    move |world| {
                        world.for_each_write::<Wealth>(
                            q_decay_wealth.clone(),
                            |w| w.value *= 0.9999,
                        )?;
                        Ok(())
                    },
                ));

                (ecs, scheduler)
            },
            |(ecs, mut scheduler)| {
                ecs.run(&mut scheduler).unwrap();
                black_box(ecs);
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(benches, tick_benchmark);
criterion_main!(benches);
