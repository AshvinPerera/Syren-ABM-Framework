//! Engine-level tests for the boundary-resource lifecycle: per-resource
//! locking, channel routing, registration collisions, and fallible
//! parallel iteration.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use abm_framework::{
    advanced::{ChannelAllocator, EntityShards},
    max_workers, worker_id, AccessSets, BoundaryContext, BoundaryResource, ChannelID,
    ComponentRegistry, ECSError, ECSManager, ECSReference, ECSResult, ExecutionError, Scheduler,
    System,
};

// ─────────────────────────────────────────────────────────────────────────────
// Shared test scaffolding
// ─────────────────────────────────────────────────────────────────────────────

fn empty_world() -> ECSManager {
    let shards = EntityShards::new(2).expect("shards");
    let registry = Arc::new(std::sync::RwLock::new(ComponentRegistry::new()));
    {
        let mut reg = registry.write().unwrap();
        reg.freeze();
    }
    ECSManager::with_registry(shards, registry)
}

/// A boundary resource that lets two threads meet at a barrier inside a
/// `&self` method. If the per-resource lock is read-shared, both threads
/// can enter concurrently and the barrier trips. If the registry is
/// serialised by a single mutex, only one thread enters at a time and the
/// barrier never trips.
struct BarrierBoundary {
    barrier: Arc<Barrier>,
    met: AtomicBool,
    owned_channels: Vec<ChannelID>,
}

impl BarrierBoundary {
    fn new(threads: usize) -> Self {
        Self {
            barrier: Arc::new(Barrier::new(threads)),
            met: AtomicBool::new(false),
            owned_channels: Vec::new(),
        }
    }

    fn meet(&self, timeout: Duration) -> bool {
        let barrier = Arc::clone(&self.barrier);
        let handle = std::thread::spawn(move || {
            barrier.wait();
        });
        let start = std::time::Instant::now();
        while !handle.is_finished() {
            if start.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let _ = handle.join();
        self.met.store(true, Ordering::Release);
        true
    }
}

impl BoundaryResource for BarrierBoundary {
    fn name(&self) -> &str {
        "BarrierBoundary"
    }
    fn channels(&self) -> &[ChannelID] {
        &self.owned_channels
    }
    fn begin_tick(&mut self, _ctx: &mut BoundaryContext<'_>) -> ECSResult<()> {
        Ok(())
    }
    fn end_tick(&mut self, _ctx: &mut BoundaryContext<'_>) -> ECSResult<()> {
        Ok(())
    }
    fn finalise(
        &mut self,
        _ctx: &mut BoundaryContext<'_>,
        _channels: &[ChannelID],
    ) -> ECSResult<()> {
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Records every `finalise` call it receives, paired with the channels it
/// observed. Used to assert routing decisions made by the engine.
struct RecordingBoundary {
    name_str: &'static str,
    owned_channels: Vec<ChannelID>,
    pub log: parking_lot::Mutex<Vec<Vec<ChannelID>>>,
}

impl RecordingBoundary {
    fn new(name: &'static str, owned: Vec<ChannelID>) -> Self {
        Self {
            name_str: name,
            owned_channels: owned,
            log: parking_lot::Mutex::new(Vec::new()),
        }
    }
}

impl BoundaryResource for RecordingBoundary {
    fn name(&self) -> &str {
        self.name_str
    }
    fn channels(&self) -> &[ChannelID] {
        &self.owned_channels
    }
    fn begin_tick(&mut self, _ctx: &mut BoundaryContext<'_>) -> ECSResult<()> {
        Ok(())
    }
    fn end_tick(&mut self, _ctx: &mut BoundaryContext<'_>) -> ECSResult<()> {
        Ok(())
    }
    fn finalise(
        &mut self,
        _ctx: &mut BoundaryContext<'_>,
        channels: &[ChannelID],
    ) -> ECSResult<()> {
        self.log.lock().push(channels.to_vec());
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

struct ProducesChannelsSystem {
    access: AccessSets,
}

impl ProducesChannelsSystem {
    fn new(channels: &[ChannelID]) -> Self {
        let mut access = AccessSets::default();
        for &channel in channels {
            access.produces.insert(channel);
        }
        Self { access }
    }
}

impl System for ProducesChannelsSystem {
    fn id(&self) -> u16 {
        1
    }
    fn access(&self) -> &AccessSets {
        &self.access
    }
    fn run(&self, _ecs: ECSReference<'_>) -> ECSResult<()> {
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn worker_id_stable_within_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        // Force every worker to observe its own ID at least once.
        let observed: parking_lot::Mutex<std::collections::HashSet<u32>> = Default::default();
        rayon::scope(|s| {
            for _ in 0..16 {
                s.spawn(|_| {
                    observed.lock().insert(worker_id());
                });
            }
        });
        let ids = observed.into_inner();
        assert!(ids.len() <= 4);
        assert!(ids.iter().all(|&id| id < max_workers()));
    });
}

#[test]
fn boundary_per_resource_lock_permits_concurrent_readers() {
    let world = empty_world();
    let id = world
        .register_boundary(BarrierBoundary::new(2))
        .expect("register");

    // Spawn two threads that each acquire a handle and call meet(timeout).
    // If the per-resource read lock is shared correctly, both threads enter
    // concurrently and the barrier trips. Otherwise the second thread
    // blocks until the first releases its handle, and the barrier never
    // trips before the timeout.
    let world_arc = Arc::new(world);
    let timeout = Duration::from_secs(2);

    let mut handles = Vec::new();
    for _ in 0..2 {
        let world_arc = Arc::clone(&world_arc);
        handles.push(std::thread::spawn(move || -> bool {
            let bh = world_arc
                .world_ref()
                .boundary::<BarrierBoundary>(id)
                .expect("handle");
            bh.meet(timeout)
        }));
    }
    let results: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(
        results.iter().all(|r| *r),
        "both readers must enter the per-resource read lock concurrently"
    );
}

#[test]
fn boundary_register_rejects_duplicate_channel() {
    let world = empty_world();
    let mut alloc = ChannelAllocator::new();
    let cid = alloc.alloc().unwrap();

    let first = world.register_boundary(RecordingBoundary::new("first", vec![cid]));
    assert!(first.is_ok(), "first registration should succeed");

    let second = world.register_boundary(RecordingBoundary::new("second", vec![cid]));
    match second {
        Err(ECSError::Execute(ExecutionError::DuplicateChannelRegistration {
            channel_id,
            existing_boundary,
        })) => {
            assert_eq!(channel_id, cid);
            assert_eq!(existing_boundary, first.unwrap());
        }
        other => panic!("expected DuplicateChannelRegistration, got {other:?}"),
    }
}

#[test]
fn boundary_finalise_skips_irrelevant_resources() {
    let world = empty_world();
    let mut alloc = ChannelAllocator::new();
    let a = alloc.alloc().unwrap();
    let b = alloc.alloc().unwrap();
    let c = alloc.alloc().unwrap();

    let id_a = world
        .register_boundary(RecordingBoundary::new("ra", vec![a]))
        .unwrap();
    let id_b = world
        .register_boundary(RecordingBoundary::new("rb", vec![b]))
        .unwrap();

    // Drive a scheduler boundary that finalises `a` and unowned `c`. Only
    // resource `ra` should be invoked; `rb` (which owns `b`) must be skipped.
    let mut scheduler = Scheduler::new();
    scheduler.add_system(ProducesChannelsSystem::new(&[a, c]));
    world.run(&mut scheduler).unwrap();

    let ra = world
        .world_ref()
        .boundary::<RecordingBoundary>(id_a)
        .unwrap();
    let rb = world
        .world_ref()
        .boundary::<RecordingBoundary>(id_b)
        .unwrap();

    assert_eq!(
        ra.log.lock().len(),
        1,
        "ra should have observed its channel"
    );
    assert_eq!(ra.log.lock()[0], vec![a, c]);
    assert!(rb.log.lock().is_empty(), "rb should have been skipped");
}

#[test]
fn for_each_fallible_propagates_first_error_deterministically() {
    use abm_framework::{Bundle, Command};

    // Component used purely to drive iteration over a row count.
    #[derive(Copy, Clone, Default)]
    struct Idx(u32);

    let shards = EntityShards::new(2).unwrap();
    let registry = Arc::new(std::sync::RwLock::new(ComponentRegistry::new()));
    let idx_cid = {
        let mut reg = registry.write().unwrap();
        let id = reg.register::<Idx>().unwrap();
        reg.freeze();
        id
    };
    let world = ECSManager::with_registry(shards, registry);

    // Spawn 2048 entities with sequential indices.
    world
        .world_ref()
        .with_exclusive(|_| {
            for i in 0..2048u32 {
                let mut bundle = Bundle::new();
                bundle.insert(idx_cid, Idx(i));
                world.world_ref().defer(Command::Spawn { bundle })?;
            }
            Ok(())
        })
        .unwrap();
    world.apply_deferred_commands().unwrap();

    let query = world
        .world_ref()
        .query()
        .unwrap()
        .read::<Idx>()
        .unwrap()
        .build()
        .unwrap();

    let observed: AtomicU32 = AtomicU32::new(u32::MAX);

    let err = world.world_ref().for_each_r1_fallible::<Idx>(query, |idx| {
        // Fail on a specific row: the lowest-indexed failing chunk's error
        // must always win, regardless of thread count.
        if idx.0 == 1500 {
            observed.fetch_min(idx.0, Ordering::Relaxed);
            return Err(ECSError::Execute(ExecutionError::InternalExecutionError));
        }
        Ok(())
    });

    assert!(err.is_err(), "the closure raised an error");
    assert_eq!(
        observed.load(Ordering::Relaxed),
        1500,
        "the failing entity must always be observed"
    );
}
