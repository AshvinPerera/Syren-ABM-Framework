use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use abm_framework::{
    advanced::{ChannelAllocator, EntityShards},
    AccessSets, ActivationOrder, BoundaryContext, BoundaryResource, Bundle, ChannelID, Command,
    ComponentRegistry, Count, ECSError, ECSManager, ECSReference, ECSResult, ExecutionError,
    Scheduler, System, SystemID, CHUNK_CAP,
};

fn empty_world() -> ECSManager {
    let shards = EntityShards::new(2).expect("shards");
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    registry.write().unwrap().freeze();
    ECSManager::with_registry(shards, registry)
}

struct LogSystem {
    id: SystemID,
    access: AccessSets,
    log: Arc<Mutex<Vec<SystemID>>>,
}

impl LogSystem {
    fn new(id: SystemID, access: AccessSets, log: Arc<Mutex<Vec<SystemID>>>) -> Self {
        Self { id, access, log }
    }
}

impl System for LogSystem {
    fn id(&self) -> SystemID {
        self.id
    }
    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, _world: ECSReference<'_>) -> ECSResult<()> {
        self.log.lock().unwrap().push(self.id);
        Ok(())
    }
}

struct RecordingBoundary {
    channels: Vec<ChannelID>,
    log: Arc<Mutex<Vec<Vec<ChannelID>>>>,
}

impl RecordingBoundary {
    fn new(channels: Vec<ChannelID>, log: Arc<Mutex<Vec<Vec<ChannelID>>>>) -> Self {
        Self { channels, log }
    }
}

impl BoundaryResource for RecordingBoundary {
    fn name(&self) -> &str {
        "RecordingBoundary"
    }
    fn channels(&self) -> &[ChannelID] {
        &self.channels
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
        self.log.lock().unwrap().push(channels.to_vec());
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

const BOUNDARY_EVENT: SystemID = 999;

struct EventBoundary {
    channels: Vec<ChannelID>,
    log: Arc<Mutex<Vec<SystemID>>>,
}

impl EventBoundary {
    fn new(channels: Vec<ChannelID>, log: Arc<Mutex<Vec<SystemID>>>) -> Self {
        Self { channels, log }
    }
}

impl BoundaryResource for EventBoundary {
    fn name(&self) -> &str {
        "EventBoundary"
    }
    fn channels(&self) -> &[ChannelID] {
        &self.channels
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
        self.log.lock().unwrap().push(BOUNDARY_EVENT);
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[test]
fn channel_producer_runs_before_lower_id_consumer() {
    let world = empty_world();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut alloc = ChannelAllocator::new();
    let channel = alloc.alloc().unwrap();

    let mut consumer_access = AccessSets::default();
    consumer_access.consumes.insert(channel);
    let mut producer_access = AccessSets::default();
    producer_access.produces.insert(channel);

    let mut scheduler = Scheduler::new();
    scheduler.add_system(LogSystem::new(1, consumer_access, Arc::clone(&log)));
    scheduler.add_system(LogSystem::new(10, producer_access, Arc::clone(&log)));

    world.run(&mut scheduler).unwrap();

    assert_eq!(*log.lock().unwrap(), vec![10, 1]);
}

#[test]
fn channel_dependency_overrides_component_conflict_id_tiebreak() {
    let world = empty_world();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut alloc = ChannelAllocator::new();
    let channel = alloc.alloc().unwrap();

    let mut consumer_access = AccessSets::default();
    consumer_access.consumes.insert(channel);
    consumer_access.read.set(0);

    let mut producer_access = AccessSets::default();
    producer_access.produces.insert(channel);
    producer_access.write.set(0);

    let mut scheduler = Scheduler::new();
    scheduler.add_system(LogSystem::new(1, consumer_access, Arc::clone(&log)));
    scheduler.add_system(LogSystem::new(10, producer_access, Arc::clone(&log)));

    world.run(&mut scheduler).unwrap();

    assert_eq!(*log.lock().unwrap(), vec![10, 1]);
}

#[test]
fn channel_dependency_cycle_is_rejected() {
    let world = empty_world();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut alloc = ChannelAllocator::new();
    let a = alloc.alloc().unwrap();
    let b = alloc.alloc().unwrap();

    let mut access_a = AccessSets::default();
    access_a.produces.insert(a);
    access_a.consumes.insert(b);
    let mut access_b = AccessSets::default();
    access_b.produces.insert(b);
    access_b.consumes.insert(a);

    let mut scheduler = Scheduler::new();
    scheduler.add_system(LogSystem::new(1, access_a, Arc::clone(&log)));
    scheduler.add_system(LogSystem::new(2, access_b, log));

    let err = world.run(&mut scheduler).unwrap_err();
    assert!(matches!(
        err,
        ECSError::Execute(ExecutionError::SchedulerCycle)
    ));
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
struct Marker(u8);

struct SpawnMarkerSystem {
    component_id: abm_framework::ComponentID,
    access: AccessSets,
}

impl System for SpawnMarkerSystem {
    fn id(&self) -> SystemID {
        10
    }
    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, world: ECSReference<'_>) -> ECSResult<()> {
        let mut bundle = Bundle::new();
        bundle.insert(self.component_id, Marker(1));
        world.defer(Command::Spawn { bundle })
    }
}

struct CountMarkerSystem {
    access: AccessSets,
    observed: Arc<AtomicU64>,
}

impl System for CountMarkerSystem {
    fn id(&self) -> SystemID {
        1
    }
    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, world: ECSReference<'_>) -> ECSResult<()> {
        let query = world.query()?.read::<Marker>()?.build()?;
        let count = world.reduce_read::<Marker, Count>(
            query,
            Count::default,
            |acc, _| acc.0 += 1,
            |a, b| a.0 += b.0,
        )?;
        self.observed.store(count.0, Ordering::Release);
        Ok(())
    }
}

#[test]
fn explicit_ordering_applies_deferred_commands_before_dependent() {
    let shards = EntityShards::new(2).unwrap();
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let marker_id = {
        let mut reg = registry.write().unwrap();
        let id = reg.register::<Marker>().unwrap();
        reg.freeze();
        id
    };
    let world = ECSManager::with_registry(shards, registry);
    let observed = Arc::new(AtomicU64::new(0));

    let mut consumer_access = AccessSets::default();
    consumer_access.read.set(marker_id);

    let mut scheduler = Scheduler::new();
    scheduler.add_system(CountMarkerSystem {
        access: consumer_access,
        observed: Arc::clone(&observed),
    });
    scheduler.add_system(SpawnMarkerSystem {
        component_id: marker_id,
        access: AccessSets::default(),
    });
    scheduler.add_ordering(1, 10);

    world.run(&mut scheduler).unwrap();

    assert_eq!(observed.load(Ordering::Acquire), 1);
}

#[derive(Clone, Copy)]
struct Visit(u32);

struct FirstVisitErrorSystem {
    access: AccessSets,
}

impl System for FirstVisitErrorSystem {
    fn id(&self) -> SystemID {
        77
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, world: ECSReference<'_>) -> ECSResult<()> {
        let query = world.query()?.read::<Visit>()?.build()?;
        world.for_each_r1_fallible::<Visit>(query, |visit| {
            Err(ECSError::Execute(ExecutionError::UnknownSystemId {
                system_id: visit.0 as SystemID,
            }))
        })
    }
}

fn world_with_visits(n: usize) -> (ECSManager, abm_framework::ComponentID) {
    let shards = EntityShards::new(2).unwrap();
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let visit_id = {
        let mut reg = registry.write().unwrap();
        let id = reg.register::<Visit>().unwrap();
        reg.freeze();
        id
    };
    let world = ECSManager::with_registry(shards, registry);
    let ecs = world.world_ref();
    for i in 0..n {
        let mut bundle = Bundle::new();
        bundle.insert(visit_id, Visit(i as u32));
        ecs.defer(Command::Spawn { bundle }).unwrap();
    }
    world.apply_deferred_commands().unwrap();
    (world, visit_id)
}

fn first_visit_error(order: ActivationOrder, seed: u64, n: usize) -> u32 {
    let (world, visit_id) = world_with_visits(n);
    let mut access = AccessSets::default();
    access.read.set(visit_id);
    let mut scheduler = Scheduler::new();
    scheduler.add_system(FirstVisitErrorSystem { access });
    scheduler.seed(seed);
    scheduler.set_activation_order(77, order);

    let err = world.run(&mut scheduler).unwrap_err();
    match err {
        ECSError::Execute(ExecutionError::UnknownSystemId { system_id }) => system_id as u32,
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn activation_order_is_deterministic_for_fixed_seed_and_changes_with_seed() {
    let n = 64;
    let a = first_visit_error(ActivationOrder::ShuffleFull, 7, n);
    let b = first_visit_error(ActivationOrder::ShuffleFull, 7, n);
    let c = first_visit_error(ActivationOrder::ShuffleFull, 8, n);

    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn sequential_and_shuffled_activation_visit_different_first_work() {
    let sequential = first_visit_error(ActivationOrder::Sequential, 0, CHUNK_CAP + 1);
    let shuffled_chunks = first_visit_error(ActivationOrder::ShuffleChunks, 2, CHUNK_CAP + 1);
    let shuffled_rows = first_visit_error(ActivationOrder::ShuffleFull, 7, 64);

    assert_eq!(sequential, 0);
    assert_ne!(shuffled_chunks, sequential);
    assert_ne!(shuffled_rows, sequential);
}

#[test]
fn component_conflicts_serialize_by_system_id_without_boundary() {
    let world = empty_world();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut access = AccessSets::default();
    access.write.set(0);

    let mut scheduler = Scheduler::new();
    scheduler.add_system(LogSystem::new(2, access.clone(), Arc::clone(&log)));
    scheduler.add_system(LogSystem::new(1, access, Arc::clone(&log)));

    world.run(&mut scheduler).unwrap();

    assert_eq!(*log.lock().unwrap(), vec![1, 2]);

    let plan = scheduler.plan();
    assert_eq!(plan.iter().filter(|stage| !stage.is_boundary()).count(), 2);
    assert!(!plan[0].is_boundary());
    assert!(!plan[1].is_boundary());
}

#[test]
fn non_conflicting_cpu_systems_share_a_stage() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut scheduler = Scheduler::new();
    scheduler.add_system(LogSystem::new(1, AccessSets::default(), Arc::clone(&log)));
    scheduler.add_system(LogSystem::new(2, AccessSets::default(), log));

    scheduler.rebuild();

    let non_boundary: Vec<_> = scheduler
        .plan()
        .iter()
        .filter(|stage| !stage.is_boundary())
        .collect();
    assert_eq!(non_boundary.len(), 1);
    assert_eq!(non_boundary[0].system_indices().len(), 2);
}

#[test]
fn channel_finalises_once_after_last_producer_before_consumer() {
    let world = empty_world();
    let log = Arc::new(Mutex::new(Vec::new()));
    let boundary_log = Arc::new(Mutex::new(Vec::new()));
    let mut alloc = ChannelAllocator::new();
    let channel = alloc.alloc().unwrap();

    world
        .register_boundary(RecordingBoundary::new(
            vec![channel],
            Arc::clone(&boundary_log),
        ))
        .unwrap();

    let mut producer_access = AccessSets::default();
    producer_access.produces.insert(channel);
    let mut consumer_access = AccessSets::default();
    consumer_access.consumes.insert(channel);

    let mut scheduler = Scheduler::new();
    scheduler.add_system(LogSystem::new(2, producer_access.clone(), Arc::clone(&log)));
    scheduler.add_system(LogSystem::new(3, producer_access, Arc::clone(&log)));
    scheduler.add_system(LogSystem::new(10, consumer_access, log));

    world.run(&mut scheduler).unwrap();

    let boundary_log = boundary_log.lock().unwrap();
    assert_eq!(boundary_log.len(), 1);
    assert_eq!(boundary_log[0], vec![channel]);
}

#[test]
fn unconsumed_channel_finalises_at_trailing_boundary() {
    let world = empty_world();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut alloc = ChannelAllocator::new();
    let channel = alloc.alloc().unwrap();

    world
        .register_boundary(EventBoundary::new(vec![channel], Arc::clone(&log)))
        .unwrap();

    let mut producer_access = AccessSets::default();
    producer_access.produces.insert(channel);

    let mut scheduler = Scheduler::new();
    scheduler.add_system(LogSystem::new(1, producer_access, Arc::clone(&log)));
    scheduler.add_system(LogSystem::new(2, AccessSets::default(), Arc::clone(&log)));
    scheduler.add_system(LogSystem::new(3, AccessSets::default(), Arc::clone(&log)));
    scheduler.add_ordering(3, 2);

    world.run(&mut scheduler).unwrap();

    let events = log.lock().unwrap();
    let boundary_pos = events
        .iter()
        .position(|id| *id == BOUNDARY_EVENT)
        .expect("channel finalise event");
    let dependent_pos = events
        .iter()
        .position(|id| *id == 3)
        .expect("dependent system event");

    assert!(
        dependent_pos < boundary_pos,
        "unconsumed channels should finalise at the trailing boundary"
    );
    assert_eq!(events.iter().filter(|id| **id == BOUNDARY_EVENT).count(), 1);
}

#[test]
fn duplicate_system_id_is_rejected() {
    let world = empty_world();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut scheduler = Scheduler::new();
    scheduler.add_system(LogSystem::new(1, AccessSets::default(), Arc::clone(&log)));
    scheduler.add_system(LogSystem::new(1, AccessSets::default(), log));

    let err = world.run(&mut scheduler).unwrap_err();
    assert!(matches!(
        err,
        ECSError::Execute(ExecutionError::DuplicateSystemId { system_id: 1 })
    ));
}

#[test]
fn ordering_unknown_system_id_is_rejected() {
    let world = empty_world();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut scheduler = Scheduler::new();
    scheduler.add_system(LogSystem::new(1, AccessSets::default(), log));
    scheduler.add_ordering(1, 99);

    let err = world.run(&mut scheduler).unwrap_err();
    assert!(matches!(
        err,
        ECSError::Execute(ExecutionError::UnknownSystemId { system_id: 99 })
    ));
}

#[test]
fn ordering_self_dependency_is_rejected() {
    let world = empty_world();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut scheduler = Scheduler::new();
    scheduler.add_system(LogSystem::new(1, AccessSets::default(), log));
    scheduler.add_ordering(1, 1);

    let err = world.run(&mut scheduler).unwrap_err();
    assert!(matches!(
        err,
        ECSError::Execute(ExecutionError::SelfSystemOrdering { system_id: 1 })
    ));
}
