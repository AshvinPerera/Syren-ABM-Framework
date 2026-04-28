#![cfg(feature = "messaging")]

use std::hint::black_box;
use std::sync::{Arc, RwLock};

use abm_framework::advanced::{ChannelAllocator, EntityShards};
use abm_framework::messaging::{
    BruteForceMessage, BucketMessage, Capacity, Message, MessageBufferSet, MessageRegistry,
    SpatialConfig, SpatialMessage, TargetedMessage,
};
use abm_framework::{AccessSets, ComponentRegistry, ECSManager, Entity, FnSystem, Scheduler};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

const N: u32 = 4096;

#[derive(Clone, Copy)]
struct BruteMsg {
    _value: u32,
}
impl Message for BruteMsg {}
impl BruteForceMessage for BruteMsg {}

#[derive(Clone, Copy)]
struct BucketMsg {
    bucket: u32,
    _value: u32,
}
impl Message for BucketMsg {}
impl BucketMessage for BucketMsg {
    fn bucket_key(&self) -> u32 {
        self.bucket
    }
}

#[derive(Clone, Copy)]
struct SpatialMsg {
    x: f32,
    y: f32,
    _value: u32,
}
impl Message for SpatialMsg {}
impl SpatialMessage for SpatialMsg {
    fn position(&self) -> (f32, f32) {
        (self.x, self.y)
    }
}

#[derive(Clone, Copy)]
struct TargetMsg {
    to: Entity,
    _value: u32,
}
impl Message for TargetMsg {}
impl TargetedMessage for TargetMsg {
    fn recipient(&self) -> Entity {
        self.to
    }
}

fn empty_world(registry: Arc<MessageRegistry>) -> ECSManager {
    let components = Arc::new(RwLock::new(ComponentRegistry::new()));
    components.write().unwrap().freeze();
    let world = ECSManager::with_registry(EntityShards::new(4).unwrap(), components);
    world
        .register_boundary(MessageBufferSet::new(registry).unwrap())
        .unwrap();
    world
}

fn message_specialisation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_specialisations");

    group.bench_function("brute_force_4096", |b| {
        b.iter_batched(
            || {
                let mut alloc = ChannelAllocator::new();
                let mut registry = MessageRegistry::new();
                let handle = registry
                    .register_brute_force::<BruteMsg>(&mut alloc, Capacity::unbounded(N as usize))
                    .unwrap();
                registry.freeze();
                let world = empty_world(Arc::new(registry));
                let mut access = AccessSets::default();
                access.produces.insert(handle.channel_id());
                let mut scheduler = Scheduler::new();
                scheduler.add_system(FnSystem::new(1, "produce_brute", access, move |ecs| {
                    let buffers = ecs.boundary::<MessageBufferSet>(0)?;
                    for i in 0..N {
                        buffers.emit(handle, BruteMsg { _value: i })?;
                    }
                    Ok(())
                }));
                (world, scheduler)
            },
            |(world, mut scheduler)| {
                world.run(&mut scheduler).unwrap();
                black_box(world);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("bucket_4096_64_buckets", |b| {
        b.iter_batched(
            || {
                let mut alloc = ChannelAllocator::new();
                let mut registry = MessageRegistry::new();
                let handle = registry
                    .register_bucket::<BucketMsg>(&mut alloc, 64, Capacity::unbounded(N as usize))
                    .unwrap();
                registry.freeze();
                let world = empty_world(Arc::new(registry));
                let mut access = AccessSets::default();
                access.produces.insert(handle.channel_id());
                let mut scheduler = Scheduler::new();
                scheduler.add_system(FnSystem::new(1, "produce_bucket", access, move |ecs| {
                    let buffers = ecs.boundary::<MessageBufferSet>(0)?;
                    for i in 0..N {
                        buffers.emit(
                            handle,
                            BucketMsg {
                                bucket: i % 64,
                                _value: i,
                            },
                        )?;
                    }
                    Ok(())
                }));
                (world, scheduler)
            },
            |(world, mut scheduler)| {
                world.run(&mut scheduler).unwrap();
                black_box(world);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("spatial_4096", |b| {
        b.iter_batched(
            || {
                let mut alloc = ChannelAllocator::new();
                let mut registry = MessageRegistry::new();
                let handle = registry
                    .register_spatial::<SpatialMsg>(
                        &mut alloc,
                        SpatialConfig {
                            width: 256.0,
                            height: 256.0,
                            cell_size: 8.0,
                        },
                        Capacity::unbounded(N as usize),
                    )
                    .unwrap();
                registry.freeze();
                let world = empty_world(Arc::new(registry));
                let mut access = AccessSets::default();
                access.produces.insert(handle.channel_id());
                let mut scheduler = Scheduler::new();
                scheduler.add_system(FnSystem::new(1, "produce_spatial", access, move |ecs| {
                    let buffers = ecs.boundary::<MessageBufferSet>(0)?;
                    for i in 0..N {
                        buffers.emit(
                            handle,
                            SpatialMsg {
                                x: (i % 256) as f32,
                                y: ((i / 256) % 256) as f32,
                                _value: i,
                            },
                        )?;
                    }
                    Ok(())
                }));
                (world, scheduler)
            },
            |(world, mut scheduler)| {
                world.run(&mut scheduler).unwrap();
                black_box(world);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("targeted_4096", |b| {
        b.iter_batched(
            || {
                let mut alloc = ChannelAllocator::new();
                let mut registry = MessageRegistry::new();
                let handle = registry
                    .register_targeted::<TargetMsg>(&mut alloc, Capacity::unbounded(N as usize))
                    .unwrap();
                registry.freeze();
                let world = empty_world(Arc::new(registry));
                let mut access = AccessSets::default();
                access.produces.insert(handle.channel_id());
                let mut scheduler = Scheduler::new();
                scheduler.add_system(FnSystem::new(1, "produce_targeted", access, move |ecs| {
                    let buffers = ecs.boundary::<MessageBufferSet>(0)?;
                    for i in 0..N {
                        buffers.emit(
                            handle,
                            TargetMsg {
                                to: Entity::from_raw((i % 64) as u64),
                                _value: i,
                            },
                        )?;
                    }
                    Ok(())
                }));
                (world, scheduler)
            },
            |(world, mut scheduler)| {
                world.run(&mut scheduler).unwrap();
                black_box(world);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, message_specialisation_benchmarks);
criterion_main!(benches);
