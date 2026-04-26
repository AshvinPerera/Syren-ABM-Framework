#![cfg(feature = "gpu")]
#![allow(dead_code)]

use std::sync::{Arc, RwLock};

use abm_framework::{
    advanced::EntityShards, AccessSets, Bundle, Command, ComponentID, ComponentRegistry, ECSError,
    ECSManager, ECSReference, ECSResult, GpuSystem, Signature, System, SystemBackend, SystemID,
};

pub const AGENTS: usize = 1_000_000;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Energy {
    pub v: f32,
}

/// Creates a shared, frozen component registry with Energy registered.
/// Returns the registry handle and the Energy component ID.
pub fn make_registry() -> (Arc<RwLock<ComponentRegistry>>, ComponentID) {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let energy_id = {
        let mut reg = registry.write().unwrap();
        let id = reg.register::<Energy>().unwrap();
        reg.freeze();
        id
    };
    (registry, energy_id)
}

/// Constructs an ECSManager backed by the given instance-owned registry.
pub fn make_world(shards: usize, registry: Arc<RwLock<ComponentRegistry>>) -> ECSManager {
    let shards = EntityShards::new(shards).unwrap();
    ECSManager::with_registry(shards, registry)
}

/// Spawns `n` entities each carrying an Energy component.
pub fn populate_energy(ecs: &ECSManager, n: usize, energy_id: ComponentID) -> ECSResult<()> {
    let world = ecs.world_ref();

    world.with_exclusive(|_| {
        for _ in 0..n {
            let mut bundle = Bundle::new();
            bundle.insert(energy_id, Energy { v: 100.0 });
            world.defer(Command::Spawn { bundle })?;
        }
        Ok::<(), ECSError>(())
    })?;

    ecs.apply_deferred_commands()?;
    Ok(())
}

pub struct EnergyDecayGpu {
    pub id: SystemID,
    access: AccessSets,
}

impl EnergyDecayGpu {
    pub fn new(id: SystemID, energy_id: ComponentID) -> Self {
        let mut write = Signature::default();
        write.set(energy_id);
        Self {
            id,
            access: AccessSets {
                read: Signature::default(),
                write,
                ..AccessSets::default()
            },
        }
    }
}

impl System for EnergyDecayGpu {
    fn id(&self) -> SystemID {
        self.id
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    #[inline]
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

impl GpuSystem for EnergyDecayGpu {
    fn shader(&self) -> &'static str {
        r#"
            struct Energy { v: f32, };

            @group(0) @binding(0)
            var<storage, read_write> energy: array<Energy>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                if (i < arrayLength(&energy)) {
                    energy[i].v = energy[i].v * 0.999;
                }
            }
        "#
    }

    fn entry_point(&self) -> &'static str {
        "main"
    }

    fn workgroup_size(&self) -> u32 {
        256
    }
}
