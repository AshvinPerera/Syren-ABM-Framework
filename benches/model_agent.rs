use std::hint::black_box;
use std::sync::{Arc, RwLock};

use abm_framework::advanced::EntityShards;
use abm_framework::agents::AgentTemplate;
use abm_framework::model::ModelBuilder;
use abm_framework::Write;
use abm_framework::{AccessSets, ComponentRegistry, FnSystem};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

const AGENTS: usize = 4096;
const AGENTS_DESPAWN_LARGE: usize = 65_536;
const AGENTS_LARGE: usize = 1_000_000;

#[derive(Clone, Copy, Default)]
struct BenchAgent {
    value: u32,
}

fn make_registry() -> (Arc<RwLock<ComponentRegistry>>, abm_framework::ComponentID) {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let agent_id = {
        let mut reg = registry.write().unwrap();
        let id = reg.register::<BenchAgent>().unwrap();
        reg.freeze();
        id
    };
    (registry, agent_id)
}

fn build_model(agent_id: abm_framework::ComponentID) -> abm_framework::model::Model {
    let (registry, _) = make_registry();
    let registered = AgentTemplate::builder("bench_agent")
        .with_component::<BenchAgent>(agent_id)
        .unwrap()
        .build();
    ModelBuilder::new()
        .with_component_registry(registry)
        .with_shards(EntityShards::new(4).unwrap())
        .with_agent_template(registered)
        .unwrap()
        .build()
        .unwrap()
}

fn spawn_agents(model: &mut abm_framework::model::Model, agent_id: abm_framework::ComponentID) {
    let world = model.ecs().world_ref();
    let template = model.agents().get("bench_agent").unwrap();
    for i in 0..AGENTS {
        template
            .spawner()
            .set(agent_id, BenchAgent { value: i as u32 })
            .unwrap()
            .spawn(world)
            .unwrap();
    }
    model.ecs().apply_deferred_commands().unwrap();
}

fn make_agents(count: usize) -> Vec<BenchAgent> {
    (0..count).map(|i| BenchAgent { value: i as u32 }).collect()
}

fn model_agent_benchmarks(c: &mut Criterion) {
    let (registry, agent_id) = make_registry();
    drop(registry);

    c.bench_function("model_agent/template_spawn_4096", |b| {
        b.iter_batched(
            || build_model(agent_id),
            |mut model| {
                spawn_agents(&mut model, agent_id);
                black_box(model);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("model_agent/bulk_spawn_4096", |b| {
        b.iter_batched(
            || build_model(agent_id),
            |mut model| {
                model
                    .spawn_agent_batch("bench_agent", agent_id, make_agents(AGENTS))
                    .unwrap();
                black_box(model);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("model_agent/builder_population_1M", |b| {
        b.iter_batched(
            || {
                let (registry, agent_id) = make_registry();
                (registry, agent_id, make_agents(AGENTS_LARGE))
            },
            |(registry, agent_id, agents)| {
                let model = ModelBuilder::new()
                    .with_component_registry(registry)
                    .with_shards(EntityShards::new(4).unwrap())
                    .with_agent_template(
                        AgentTemplate::builder("bench_agent")
                            .with_component::<BenchAgent>(agent_id)
                            .unwrap()
                            .with_capacity(AGENTS_LARGE)
                            .build(),
                    )
                    .unwrap()
                    .with_agent_population("bench_agent", agent_id, agents)
                    .unwrap()
                    .build()
                    .unwrap();
                black_box(model);
            },
            BatchSize::LargeInput,
        );
    });

    c.bench_function("model_agent/bulk_despawn_4096", |b| {
        b.iter_batched(
            || {
                let mut model = build_model(agent_id);
                let entities = model
                    .spawn_agent_batch("bench_agent", agent_id, make_agents(AGENTS))
                    .unwrap();
                (model, entities)
            },
            |(mut model, entities)| {
                model.despawn_agent_batch("bench_agent", entities).unwrap();
                black_box(model);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("model_agent/bulk_despawn_65536", |b| {
        b.iter_batched(
            || {
                let mut model = build_model(agent_id);
                let entities = model
                    .spawn_agent_batch("bench_agent", agent_id, make_agents(AGENTS_DESPAWN_LARGE))
                    .unwrap();
                (model, entities)
            },
            |(mut model, entities)| {
                model.despawn_agent_batch("bench_agent", entities).unwrap();
                black_box(model);
            },
            BatchSize::LargeInput,
        );
    });

    c.bench_function("model_tick/empty_model", |b| {
        b.iter_batched(
            || ModelBuilder::new().build().unwrap(),
            |mut model| {
                model.tick().unwrap();
                black_box(model.tick_count());
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("model_tick/two_systems_4096", |b| {
        b.iter_batched(
            || {
                let (registry, agent_id) = make_registry();
                let mut access1 = AccessSets::default();
                access1.write.set(agent_id);
                let mut access2 = AccessSets::default();
                access2.read.set(agent_id);

                let mut model = ModelBuilder::new()
                    .with_component_registry(registry.clone())
                    .with_shards(EntityShards::new(4).unwrap())
                    .with_agent_template(
                        AgentTemplate::builder("bench_agent")
                            .with_component::<BenchAgent>(agent_id)
                            .unwrap()
                            .build(),
                    )
                    .unwrap()
                    .with_system(FnSystem::new(1, "increment", access1, move |ecs| {
                        let q = ecs.query()?.write::<BenchAgent>()?.build()?;
                        ecs.for_each::<(Write<BenchAgent>,)>(q, &|agent| {
                            agent.0.value = agent.0.value.wrapping_add(1);
                        })?;
                        Ok(())
                    }))
                    .with_system(FnSystem::new(2, "sum", access2, move |ecs| {
                        let q = ecs.query()?.read::<BenchAgent>()?.build()?;
                        let total = ecs.reduce_read::<BenchAgent, u64>(
                            q,
                            || 0,
                            |acc, agent| *acc += agent.value as u64,
                            |acc, other| *acc += other,
                        )?;
                        black_box(total);
                        Ok(())
                    }))
                    .build()
                    .unwrap();
                spawn_agents(&mut model, agent_id);
                model
            },
            |mut model| {
                model.tick().unwrap();
                black_box(model);
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, model_agent_benchmarks);
criterion_main!(benches);
