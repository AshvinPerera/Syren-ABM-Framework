#![allow(dead_code)]

use std::hint::black_box;
use std::sync::{Arc, RwLock};

use criterion::{BatchSize, Criterion};

use abm_framework::{
    advanced::EntityShards, AccessSets, Bundle, Command, ComponentRegistry, ECSManager, FnSystem,
    Scheduler,
};

const QUERY_ARCHETYPES: usize = 64;
const QUERY_ROWS_PER_ARCHETYPE: usize = 64;
const SCHEDULER_SYSTEMS: usize = 1_000;

#[derive(Clone, Copy)]
struct QueryValue {
    value: u32,
}

#[derive(Clone, Copy)]
struct QueryTag0 {
    _tag: u8,
}
#[derive(Clone, Copy)]
struct QueryTag1 {
    _tag: u8,
}
#[derive(Clone, Copy)]
struct QueryTag2 {
    _tag: u8,
}
#[derive(Clone, Copy)]
struct QueryTag3 {
    _tag: u8,
}
#[derive(Clone, Copy)]
struct QueryTag4 {
    _tag: u8,
}
#[derive(Clone, Copy)]
struct QueryTag5 {
    _tag: u8,
}

#[cfg(any(feature = "environment", feature = "messaging"))]
fn empty_world() -> ECSManager {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    registry.write().unwrap().freeze();
    ECSManager::with_registry(EntityShards::new(4).unwrap(), registry)
}

fn query_matching_world() -> (ECSManager, Arc<RwLock<ComponentRegistry>>) {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let (value_id, tag_ids) = {
        let mut reg = registry.write().unwrap();
        let value_id = reg.register::<QueryValue>().unwrap();
        let tag_ids = [
            reg.register::<QueryTag0>().unwrap(),
            reg.register::<QueryTag1>().unwrap(),
            reg.register::<QueryTag2>().unwrap(),
            reg.register::<QueryTag3>().unwrap(),
            reg.register::<QueryTag4>().unwrap(),
            reg.register::<QueryTag5>().unwrap(),
        ];
        reg.freeze();
        (value_id, tag_ids)
    };

    let world = ECSManager::with_registry(EntityShards::new(4).unwrap(), Arc::clone(&registry));
    let ecs = world.world_ref();
    for mask in 0..QUERY_ARCHETYPES {
        for row in 0..QUERY_ROWS_PER_ARCHETYPE {
            let mut bundle = Bundle::new();
            bundle.insert(
                value_id,
                QueryValue {
                    value: (mask * QUERY_ROWS_PER_ARCHETYPE + row) as u32,
                },
            );
            for (bit, tag_id) in tag_ids.iter().enumerate() {
                if (mask & (1 << bit)) != 0 {
                    match bit {
                        0 => bundle.insert(*tag_id, QueryTag0 { _tag: 0 }),
                        1 => bundle.insert(*tag_id, QueryTag1 { _tag: 1 }),
                        2 => bundle.insert(*tag_id, QueryTag2 { _tag: 2 }),
                        3 => bundle.insert(*tag_id, QueryTag3 { _tag: 3 }),
                        4 => bundle.insert(*tag_id, QueryTag4 { _tag: 4 }),
                        5 => bundle.insert(*tag_id, QueryTag5 { _tag: 5 }),
                        _ => unreachable!(),
                    }
                }
            }
            ecs.defer(Command::Spawn { bundle }).unwrap();
        }
    }
    world.apply_deferred_commands().unwrap();
    (world, registry)
}

pub fn query_matching_benchmark(c: &mut Criterion) {
    let (world, registry) = query_matching_world();
    let query = abm_framework::QueryBuilder::with_registry(registry)
        .read::<QueryValue>()
        .unwrap()
        .build()
        .unwrap();
    let ecs = world.world_ref();

    c.bench_function("query_matching/heavy_synthetic_tick_64_archetypes", |b| {
        b.iter(|| {
            let sum = ecs
                .reduce_read::<QueryValue, u64>(
                    query.clone(),
                    || 0,
                    |acc, value| *acc += value.value as u64,
                    |acc, other| *acc += other,
                )
                .unwrap();
            black_box(sum);
        });
    });
}

fn make_scheduler_for_packing() -> Scheduler {
    let mut scheduler = Scheduler::new();
    for i in 0..SCHEDULER_SYSTEMS {
        let mut access = AccessSets::default();
        access.write.set((i % 64) as u16);
        access.read.set(((i + 1) % 64) as u16);
        if i % 4 == 0 {
            access.produces.insert((i / 4) as u32);
        } else if i % 4 == 1 {
            access.consumes.insert((i / 4) as u32);
        }
        scheduler.add_system(FnSystem::new(i as u16, "packing_node", access, |_| Ok(())));
    }
    scheduler
}

pub fn scheduler_packing_benchmark(c: &mut Criterion) {
    c.bench_function(
        "scheduler_packing/rebuild_1000_system_realistic_graph",
        |b| {
            b.iter_batched(
                make_scheduler_for_packing,
                |mut scheduler| {
                    scheduler.try_rebuild().unwrap();
                    black_box(scheduler.plan().len());
                },
                BatchSize::SmallInput,
            );
        },
    );
}

pub fn environment_dirty_tracking_benchmark(c: &mut Criterion) {
    #[cfg(feature = "environment")]
    {
        use abm_framework::environment::{EnvironmentBoundary, EnvironmentBuilder};

        let env = EnvironmentBuilder::new()
            .register::<f32>("rate", 0.01)
            .unwrap()
            .register::<u32>("width", 128)
            .unwrap()
            .build()
            .unwrap();
        let rate_channel = env.channel_of("rate").unwrap();
        let world = empty_world();
        world
            .register_boundary(EnvironmentBoundary::new(Arc::clone(&env)))
            .unwrap();

        let mut access = AccessSets::default();
        access.produces.insert(rate_channel);
        let env_for_system = Arc::clone(&env);
        let mut scheduler = Scheduler::new();
        scheduler.add_system(FnSystem::new(1, "env_dirty_writer", access, move |_| {
            let rate = env_for_system.get::<f32>("rate")?;
            env_for_system.set("rate", rate + 0.0001)?;
            Ok(())
        }));

        c.bench_function("environment_dirty_tracking/finalise_dirty_channel", |b| {
            b.iter(|| {
                world.run(&mut scheduler).unwrap();
                black_box(env.get::<f32>("rate").unwrap());
            });
        });
    }

    #[cfg(not(feature = "environment"))]
    {
        let _ = c;
    }
}

pub fn messaging_finalisation_benchmark(c: &mut Criterion) {
    #[cfg(feature = "messaging")]
    {
        use abm_framework::advanced::ChannelAllocator;
        use abm_framework::messaging::{
            BucketMessage, Capacity, Message, MessageBufferSet, MessageRegistry,
        };

        #[derive(Clone, Copy)]
        struct BenchBucketMsg {
            bucket: u32,
            _value: u32,
        }

        impl Message for BenchBucketMsg {}
        impl BucketMessage for BenchBucketMsg {
            fn bucket_key(&self) -> u32 {
                self.bucket
            }
        }

        let mut allocator = ChannelAllocator::new();
        let mut registry = MessageRegistry::new();
        let handle = registry
            .register_bucket::<BenchBucketMsg>(&mut allocator, 64, Capacity::unbounded(4096))
            .unwrap();
        registry.freeze();
        let registry = Arc::new(registry);
        let world = empty_world();
        world
            .register_boundary(MessageBufferSet::new(Arc::clone(&registry)).unwrap())
            .unwrap();

        let mut access = AccessSets::default();
        access.produces.insert(handle.channel_id());
        let mut scheduler = Scheduler::new();
        scheduler.add_system(FnSystem::new(
            1,
            "message_bucket_producer",
            access,
            move |ecs| {
                let buffers = ecs.boundary::<MessageBufferSet>(0)?;
                for i in 0..4096u32 {
                    buffers.emit(
                        handle,
                        BenchBucketMsg {
                            bucket: i % 64,
                            _value: i,
                        },
                    )?;
                }
                Ok(())
            },
        ));

        c.bench_function("messaging_finalisation/bucket_4096_messages", |b| {
            b.iter(|| {
                world.run(&mut scheduler).unwrap();
                black_box(&world);
            });
        });
    }

    #[cfg(not(feature = "messaging"))]
    {
        let _ = c;
    }
}

#[cfg(feature = "gpu")]
mod gpu_bench {
    use super::*;
    use abm_framework::{
        ECSError, ECSReference, ECSResult, ExecutionError, GPUPod, GpuSystem, Signature, System,
        SystemBackend,
    };

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct GpuBenchValue {
        value: u32,
    }

    unsafe impl GPUPod for GpuBenchValue {}

    #[derive(Clone, Copy)]
    struct GpuTag0 {
        _tag: u8,
    }
    #[derive(Clone, Copy)]
    struct GpuTag1 {
        _tag: u8,
    }
    #[derive(Clone, Copy)]
    struct GpuTag2 {
        _tag: u8,
    }
    #[derive(Clone, Copy)]
    struct GpuTag3 {
        _tag: u8,
    }
    #[derive(Clone, Copy)]
    struct GpuTag4 {
        _tag: u8,
    }

    pub struct GpuWriteSystem {
        access: AccessSets,
    }

    impl GpuWriteSystem {
        pub fn new(value_id: abm_framework::ComponentID) -> Self {
            let mut write = Signature::default();
            write.set(value_id);
            Self {
                access: AccessSets {
                    write,
                    ..AccessSets::default()
                },
            }
        }
    }

    impl System for GpuWriteSystem {
        fn id(&self) -> abm_framework::SystemID {
            77
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

    impl GpuSystem for GpuWriteSystem {
        fn shader(&self) -> &'static str {
            r#"
struct Params {
    entity_len: u32,
    archetype_base: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0)
var<storage, read_write> values: array<u32>;

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.entity_len) {
        return;
    }
    values[i] = values[i] + params.archetype_base + i + 1u;
}
"#
        }

        fn entry_point(&self) -> &'static str {
            "main"
        }

        fn workgroup_size(&self) -> u32 {
            128
        }
    }

    fn make_gpu_world(archetypes: usize, rows_per_archetype: usize) -> (ECSManager, Scheduler) {
        let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
        let (value_id, tag_ids) = {
            let mut reg = registry.write().unwrap();
            let value_id = reg.register_gpu::<GpuBenchValue>().unwrap();
            let tag_ids = [
                reg.register::<GpuTag0>().unwrap(),
                reg.register::<GpuTag1>().unwrap(),
                reg.register::<GpuTag2>().unwrap(),
                reg.register::<GpuTag3>().unwrap(),
                reg.register::<GpuTag4>().unwrap(),
            ];
            reg.freeze();
            (value_id, tag_ids)
        };

        let world = ECSManager::with_registry(EntityShards::new(4).unwrap(), Arc::clone(&registry));
        let ecs = world.world_ref();
        for mask in 0..archetypes {
            for row in 0..rows_per_archetype {
                let mut bundle = Bundle::new();
                bundle.insert(
                    value_id,
                    GpuBenchValue {
                        value: (mask * rows_per_archetype + row) as u32,
                    },
                );
                for (bit, tag_id) in tag_ids.iter().enumerate() {
                    if (mask & (1 << bit)) != 0 {
                        match bit {
                            0 => bundle.insert(*tag_id, GpuTag0 { _tag: 0 }),
                            1 => bundle.insert(*tag_id, GpuTag1 { _tag: 1 }),
                            2 => bundle.insert(*tag_id, GpuTag2 { _tag: 2 }),
                            3 => bundle.insert(*tag_id, GpuTag3 { _tag: 3 }),
                            4 => bundle.insert(*tag_id, GpuTag4 { _tag: 4 }),
                            _ => unreachable!(),
                        }
                    }
                }
                ecs.defer(Command::Spawn { bundle }).unwrap();
            }
        }
        world.apply_deferred_commands().unwrap();

        let mut scheduler = Scheduler::new();
        scheduler.add_system(GpuWriteSystem::new(value_id));
        (world, scheduler)
    }

    pub fn maybe_warm_gpu(world: &ECSManager, scheduler: &mut Scheduler) -> bool {
        match world.run(scheduler) {
            Ok(()) => true,
            Err(ECSError::Execute(ExecutionError::GpuInitFailed { message })) => {
                eprintln!("skipping GPU review benches: {message}");
                false
            }
            Err(err) => panic!("GPU benchmark warm-up failed: {err:?}"),
        }
    }

    pub fn dispatch_poll(c: &mut Criterion) {
        let (world, mut scheduler) = make_gpu_world(1, 1024);
        if !maybe_warm_gpu(&world, &mut scheduler) {
            return;
        }

        c.bench_function("gpu_dispatch_poll/single_archetype_1024", |b| {
            b.iter(|| {
                world.run(&mut scheduler).unwrap();
                black_box(&world);
            });
        });
    }

    pub fn readback(c: &mut Criterion) {
        let (world, mut scheduler) = make_gpu_world(1, 65_536);
        if !maybe_warm_gpu(&world, &mut scheduler) {
            return;
        }

        c.bench_function("gpu_readback/single_archetype_65536", |b| {
            b.iter(|| {
                world.run(&mut scheduler).unwrap();
                black_box(&world);
            });
        });
    }

    pub fn bind_group_creation(c: &mut Criterion) {
        let (world, mut scheduler) = make_gpu_world(32, 64);
        if !maybe_warm_gpu(&world, &mut scheduler) {
            return;
        }

        c.bench_function("gpu_bind_group_creation/32_archetypes", |b| {
            b.iter(|| {
                world.run(&mut scheduler).unwrap();
                black_box(&world);
            });
        });
    }
}

pub fn gpu_dispatch_poll_benchmark(c: &mut Criterion) {
    #[cfg(feature = "gpu")]
    gpu_bench::dispatch_poll(c);

    #[cfg(not(feature = "gpu"))]
    {
        let _ = c;
    }
}

pub fn gpu_readback_benchmark(c: &mut Criterion) {
    #[cfg(feature = "gpu")]
    gpu_bench::readback(c);

    #[cfg(not(feature = "gpu"))]
    {
        let _ = c;
    }
}

pub fn gpu_bind_group_creation_benchmark(c: &mut Criterion) {
    #[cfg(feature = "gpu")]
    gpu_bench::bind_group_creation(c);

    #[cfg(not(feature = "gpu"))]
    {
        let _ = c;
    }
}
