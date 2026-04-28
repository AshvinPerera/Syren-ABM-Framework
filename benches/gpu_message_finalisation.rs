#![cfg(all(feature = "model", feature = "messaging_gpu"))]

use std::hint::black_box;
use std::sync::{Arc, Mutex, RwLock};

use abm_framework::advanced::EntityShards;
use abm_framework::agents::AgentTemplate;
use abm_framework::messaging::{
    BruteForceMessage, Capacity, GpuMessage, Message, MessageBufferSet,
};
use abm_framework::model::ModelBuilder;
use abm_framework::{
    AccessSets, ComponentRegistry, ECSError, ECSReference, ECSResult, ExecutionError, GPUPod,
    GpuSystem, System, SystemBackend,
};
use criterion::{criterion_group, criterion_main, Criterion};

const N: usize = 4096;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuAgent {
    id: u32,
    cash: u32,
    desired: u32,
    _pad: u32,
}

unsafe impl GPUPod for GpuAgent {}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOrder {
    household_id: u32,
    quantity: u32,
    max_spend: u32,
    _pad: u32,
}

impl Message for GpuOrder {}
impl BruteForceMessage for GpuOrder {}
unsafe impl GpuMessage for GpuOrder {}

struct GpuOrderProducer {
    access: AccessSets,
    resources: [abm_framework::GPUResourceID; 1],
    writes: [abm_framework::GPUResourceID; 1],
}

impl GpuOrderProducer {
    fn new(
        agent_id: abm_framework::ComponentID,
        handle: abm_framework::messaging::GpuMessageHandle<GpuOrder>,
    ) -> Self {
        let mut access = AccessSets::default();
        access.read.set(agent_id);
        access.produces.insert(handle.channel_id());
        let resource = handle.resource_ids().resource;
        Self {
            access,
            resources: [resource],
            writes: [resource],
        }
    }
}

impl System for GpuOrderProducer {
    fn id(&self) -> abm_framework::SystemID {
        1
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn backend(&self) -> SystemBackend {
        SystemBackend::GPU
    }

    fn run(&self, _world: ECSReference<'_>) -> ECSResult<()> {
        Ok(())
    }

    fn gpu(&self) -> Option<&dyn GpuSystem> {
        Some(self)
    }
}

impl GpuSystem for GpuOrderProducer {
    fn shader(&self) -> &'static str {
        r#"
struct Agent {
    id: u32,
    cash: u32,
    desired: u32,
    pad: u32,
};

struct Order {
    household_id: u32,
    quantity: u32,
    max_spend: u32,
    pad: u32,
};

struct Params {
    entity_len: u32,
    archetype_base: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> agents: array<Agent>;
@group(0) @binding(1) var<uniform> params: Params;

@group(1) @binding(1) var<storage, read_write> raw_gpu: array<Order>;
@group(1) @binding(2) var<storage, read_write> valid_flags: array<u32>;
@group(1) @binding(5) var<storage, read_write> control: array<atomic<u32>>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.entity_len) { return; }
    let slot = params.archetype_base + i;
    if (slot >= arrayLength(&raw_gpu) || slot >= arrayLength(&valid_flags)) {
        atomicStore(&control[4], 1u);
        return;
    }
    let agent = agents[i];
    if (agent.desired == 0u) { return; }
    raw_gpu[slot] = Order(agent.id, agent.desired, agent.cash, 0u);
    valid_flags[slot] = 1u;
}
"#
    }

    fn workgroup_size(&self) -> u32 {
        128
    }

    fn uses_resources(&self) -> &[abm_framework::GPUResourceID] {
        &self.resources
    }

    fn writes_resources(&self) -> &[abm_framework::GPUResourceID] {
        &self.writes
    }
}

fn build_model() -> ECSResult<(abm_framework::model::Model, Arc<Mutex<usize>>)> {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let agent_id = {
        let mut reg = registry.write().unwrap();
        let id = reg.register_gpu::<GpuAgent>().unwrap();
        reg.freeze();
        id
    };

    let mut builder = ModelBuilder::new()
        .with_component_registry(Arc::clone(&registry))
        .with_shards(EntityShards::new(4)?);
    let message_boundary = builder.message_boundary_id();
    let orders = builder
        .register_gpu_brute_force_message::<GpuOrder>(Capacity::bounded(N, N))
        .unwrap();

    let seen = Arc::new(Mutex::new(0usize));
    let seen_for_system = Arc::clone(&seen);
    let mut consume = AccessSets::default();
    consume.consumes.insert(orders.channel_id());

    let model = builder
        .with_agent_template(
            AgentTemplate::builder("gpu_agent")
                .with_component::<GpuAgent>(agent_id)
                .unwrap()
                .build(),
        )
        .unwrap()
        .with_system(GpuOrderProducer::new(agent_id, orders))
        .with_system(abm_framework::FnSystem::new(
            2,
            "count_orders",
            consume,
            move |ecs| {
                let buffers = ecs.boundary::<MessageBufferSet>(message_boundary)?;
                let count = buffers.brute_force(orders.cpu())?.count();
                *seen_for_system.lock().unwrap() = count;
                Ok(())
            },
        ))
        .build()
        .unwrap();

    let world = model.ecs().world_ref();
    let template = model.agents().get("gpu_agent").unwrap();
    for i in 0..N {
        template
            .spawner()
            .set(
                agent_id,
                GpuAgent {
                    id: i as u32,
                    cash: 10,
                    desired: 1,
                    _pad: 0,
                },
            )
            .unwrap()
            .spawn(world)
            .unwrap();
    }
    model.ecs().apply_deferred_commands().unwrap();
    Ok((model, seen))
}

fn gpu_message_finalisation_benchmark(c: &mut Criterion) {
    let (mut model, seen) = build_model().unwrap();
    match model.tick() {
        Ok(()) => {}
        Err(ECSError::Execute(ExecutionError::GpuInitFailed { message })) => {
            eprintln!("skipping GPU message benchmark: {message}");
            return;
        }
        Err(err) => panic!("GPU message benchmark warm-up failed: {err:?}"),
    }

    c.bench_function("gpu_message_finalisation/fixed_slot_orders_4096", |b| {
        b.iter(|| {
            model.tick().unwrap();
            black_box(*seen.lock().unwrap());
        });
    });
}

criterion_group!(benches, gpu_message_finalisation_benchmark);
criterion_main!(benches);
