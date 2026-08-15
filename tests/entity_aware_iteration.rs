use std::sync::{Arc, RwLock};

use syren::{
    advanced::EntityShards, AccessSets, Bundle, Command, ComponentID, ComponentRegistry,
    ECSManager, ECSReference, ECSResult, Entity, Read, Scheduler, System, Write,
};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Position {
    x: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Velocity {
    dx: i32,
}

fn world_with_components() -> (ECSManager, ComponentID, ComponentID) {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let (position_id, velocity_id) = {
        let mut reg = registry.write().unwrap();
        let position_id = reg.register::<Position>().unwrap();
        let velocity_id = reg.register::<Velocity>().unwrap();
        reg.freeze();
        (position_id, velocity_id)
    };
    (
        ECSManager::with_registry(EntityShards::new(2).unwrap(), registry),
        position_id,
        velocity_id,
    )
}

fn spawn_position(world: &ECSManager, position_id: ComponentID, x: i32) -> Entity {
    let mut bundle = Bundle::new();
    bundle.insert(position_id, Position { x });
    world.world_ref().defer(Command::Spawn { bundle }).unwrap();
    world.apply_deferred_commands().unwrap().spawned[0].entity
}

fn spawn_position_velocity(
    world: &ECSManager,
    position_id: ComponentID,
    velocity_id: ComponentID,
    x: i32,
    dx: i32,
) -> Entity {
    let ecs = world.world_ref();
    let mut bundle = Bundle::new();
    bundle.insert(position_id, Position { x });
    bundle.insert(velocity_id, Velocity { dx });
    ecs.defer(Command::Spawn { bundle }).unwrap();
    world.apply_deferred_commands().unwrap().spawned[0].entity
}

#[test]
fn entity_aware_write_iteration_returns_matching_entities() {
    let (world, position_id, _) = world_with_components();
    let ecs = world.world_ref();
    let first = spawn_position(&world, position_id, 10);
    let second = spawn_position(&world, position_id, 20);

    let q = ecs
        .query()
        .unwrap()
        .write::<Position>()
        .unwrap()
        .build()
        .unwrap();
    let seen = parking_lot::Mutex::new(Vec::new());
    ecs.for_each_entity_w1::<Position>(q, |entity, position| {
        seen.lock().push((entity, position.x));
        position.x += 1;
    })
    .unwrap();

    let mut seen = seen.into_inner();
    seen.sort_by_key(|(_, x)| *x);
    assert_eq!(seen, vec![(first, 10), (second, 20)]);
    assert_eq!(
        ecs.read_entity_component::<Position>(first, position_id)
            .unwrap(),
        Position { x: 11 }
    );
    assert_eq!(
        ecs.read_entity_component::<Position>(second, position_id)
            .unwrap(),
        Position { x: 21 }
    );
}

#[test]
fn entity_aware_tuple_iteration_returns_matching_entities() {
    let (world, position_id, velocity_id) = world_with_components();
    let ecs = world.world_ref();
    let first = spawn_position_velocity(&world, position_id, velocity_id, 1, 4);
    let second = spawn_position_velocity(&world, position_id, velocity_id, 2, 8);

    let q = ecs
        .query()
        .unwrap()
        .read::<Velocity>()
        .unwrap()
        .write::<Position>()
        .unwrap()
        .build()
        .unwrap();
    let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let seen_capture = Arc::clone(&seen);
    ecs.for_each_entity::<(Read<Velocity>, Write<Position>), _>(
        q,
        move |(entity, velocity, position)| {
            seen_capture.lock().push((entity, velocity.dx, position.x));
            position.x += velocity.dx;
        },
    )
    .unwrap();

    let mut seen = seen.lock().clone();
    seen.sort_by_key(|(_, _, x)| *x);
    assert_eq!(seen, vec![(first, 4, 1), (second, 8, 2)]);
    assert_eq!(
        ecs.read_entity_component::<Position>(first, position_id)
            .unwrap(),
        Position { x: 5 }
    );
    assert_eq!(
        ecs.read_entity_component::<Position>(second, position_id)
            .unwrap(),
        Position { x: 10 }
    );
}

#[test]
fn run_context_is_visible_inside_scheduled_systems() {
    struct CaptureContext {
        access: AccessSets,
        out: Arc<parking_lot::Mutex<Option<(u64, u64, u16)>>>,
    }

    impl System for CaptureContext {
        fn id(&self) -> u16 {
            7
        }

        fn access(&self) -> &AccessSets {
            &self.access
        }

        fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
            let ctx = ecs.run_context();
            *self.out.lock() = Some((ctx.simulation_seed, ctx.tick, ctx.system_id));
            Ok(())
        }
    }

    let (world, _, _) = world_with_components();
    let out = Arc::new(parking_lot::Mutex::new(None));
    let mut scheduler = Scheduler::new();
    scheduler.add_system(CaptureContext {
        access: AccessSets::default(),
        out: Arc::clone(&out),
    });

    scheduler
        .run_with_context(world.world_ref(), 1234, 99)
        .unwrap();

    assert_eq!(*out.lock(), Some((1234, 99, 7)));
}

#[test]
fn fallible_entity_iteration_propagates_error() {
    let (world, position_id, _) = world_with_components();
    let ecs = world.world_ref();
    let _ = spawn_position(&world, position_id, 10);

    let q = ecs
        .query()
        .unwrap()
        .write::<Position>()
        .unwrap()
        .build()
        .unwrap();
    let err = ecs
        .for_each_entity_w1_fallible::<Position>(q, |_entity, _position| {
            Err(syren::ECSError::Execute(
                syren::ExecutionError::InternalExecutionError,
            ))
        })
        .unwrap_err();

    assert!(matches!(
        err,
        syren::ECSError::Execute(syren::ExecutionError::InternalExecutionError)
    ));
}
