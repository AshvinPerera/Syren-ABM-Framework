use criterion::*;
use std::hint::black_box;

use abm_framework::{
    Signature,
    Scheduler,
    FnSystem,
    AccessSets,
    Read,
    Write,
};

mod common;
use common::*;

fn tick_benchmark(c: &mut Criterion) {
    let (registry, _pos_id, wealth_id, prod_id) = make_registry();

    let mut group = c.benchmark_group("tick");

    group.bench_function("tick_2_systems_1M", |b| {
        let registry = registry.clone();
        b.iter_batched(
            || {
                let ecs = make_world(4, registry.clone());
                populate(&ecs, AGENTS_MED, _pos_id, wealth_id, prod_id).unwrap();

                let q_prod_to_wealth = query_builder(&registry)
                    .read::<Productivity>().unwrap()
                    .write::<Wealth>().unwrap()
                    .build().unwrap();

                let q_decay_wealth = query_builder(&registry)
                    .write::<Wealth>().unwrap()
                    .build().unwrap();

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
                        world.for_each::<(Read<Productivity>, Write<Wealth>)>(
                            q_prod_to_wealth.clone(),
                            &|(p, w)| w.value += p.rate,
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
                        world.for_each::<(Write<Wealth>,)>(
                            q_decay_wealth.clone(),
                            &|w| w.0.value *= 0.9999,
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
