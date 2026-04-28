#![cfg(feature = "gpu")]

use std::sync::{Arc, RwLock};

use abm_framework::{
    advanced::EntityShards, AccessSets, Bundle, Command, ComponentRegistry, ECSError, ECSManager,
    ECSReference, ECSResult, ExecutionError, GPUPod, GpuSystem, Signature, System, SystemBackend,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DispatchValue {
    value: u32,
}

unsafe impl GPUPod for DispatchValue {}

#[derive(Clone, Copy)]
struct DispatchTag {
    _tag: u32,
}

struct CaptureDispatchParams {
    access: AccessSets,
}

impl CaptureDispatchParams {
    fn new(value_id: abm_framework::ComponentID) -> Self {
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

impl System for CaptureDispatchParams {
    fn id(&self) -> abm_framework::SystemID {
        31
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

impl GpuSystem for CaptureDispatchParams {
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.entity_len) {
        return;
    }
    values[i] = params.entity_len * 1000u + params.archetype_base + i;
}
"#
    }

    fn entry_point(&self) -> &'static str {
        "main"
    }

    fn workgroup_size(&self) -> u32 {
        64
    }
}

#[test]
fn gpu_dispatch_params_match_each_archetype() {
    let result = run_gpu_dispatch_params_test();
    if let Err(ECSError::Execute(ExecutionError::GpuInitFailed { message })) = result {
        eprintln!("skipping GPU dispatch params test: {message}");
        return;
    }
    result.unwrap();
}

fn run_gpu_dispatch_params_test() -> ECSResult<()> {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let (value_id, tag_id) = {
        let mut reg = registry.write().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "component registry",
            })
        })?;
        let value_id = reg.register_gpu::<DispatchValue>()?;
        let tag_id = reg.register::<DispatchTag>()?;
        reg.freeze();
        (value_id, tag_id)
    };

    let world = ECSManager::with_registry(EntityShards::new(2)?, Arc::clone(&registry));
    let ecs = world.world_ref();

    for _ in 0..3 {
        let mut bundle = Bundle::new();
        bundle.insert(value_id, DispatchValue { value: 0 });
        ecs.defer(Command::Spawn { bundle })?;
    }
    for _ in 0..2 {
        let mut bundle = Bundle::new();
        bundle.insert(value_id, DispatchValue { value: 0 });
        bundle.insert(tag_id, DispatchTag { _tag: 1 });
        ecs.defer(Command::Spawn { bundle })?;
    }

    let events = world.apply_deferred_commands()?;
    let entities: Vec<_> = events.spawned.iter().map(|event| event.entity).collect();

    let mut scheduler = abm_framework::Scheduler::new();
    scheduler.add_system(CaptureDispatchParams::new(value_id));
    world.run(&mut scheduler)?;

    let observed: Vec<u32> = entities
        .iter()
        .map(|&entity| {
            ecs.read_entity_component::<DispatchValue>(entity, value_id)
                .map(|value| value.value)
        })
        .collect::<ECSResult<_>>()?;

    assert_eq!(observed, vec![3000, 3001, 3002, 2003, 2004]);
    Ok(())
}
