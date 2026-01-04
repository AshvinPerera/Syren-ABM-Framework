#![cfg(feature = "gpu")]
#![allow(dead_code)]

use std::sync::Once;

use abm_framework::engine::commands::Command;
use abm_framework::engine::component::{Bundle, freeze_components, component_id_of};
use abm_framework::engine::entity::EntityShards;
use abm_framework::engine::error::{ECSResult, ECSError};
use abm_framework::engine::manager::{ECSData, ECSManager};
use abm_framework::engine::systems::{AccessSets, GpuSystem, System, SystemBackend};
use abm_framework::engine::types::SystemID;
use abm_framework::engine::component::Signature;

#[cfg(feature = "gpu")]
use abm_framework::engine::component::{GPUPod, register_gpu_component};

pub const AGENTS: usize = 1_000_000;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Energy {
    pub v: f32,
}

unsafe impl GPUPod for Energy {}

static INIT: Once = Once::new();

pub fn init_components() {
    INIT.call_once(|| {
        register_gpu_component::<Energy>().unwrap();
        freeze_components().unwrap();
    });
}

pub fn make_world(shards: usize) -> ECSManager {
    let shards = EntityShards::new(shards);
    let data = ECSData::new(shards);
    ECSManager::new(data)
}

pub fn populate_energy(ecs: &ECSManager, n: usize) -> ECSResult<()> {
    let world = ecs.world_ref();
    let energy_id = component_id_of::<Energy>()?;

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
}

impl EnergyDecayGpu {
    pub fn new(id: SystemID) -> Self { Self { id } }

    fn access_sets() -> AccessSets {
        let mut write = Signature::default();
        write.set(component_id_of::<Energy>().unwrap());
        AccessSets { read: Signature::default(), write }
    }
}

impl System for EnergyDecayGpu {
    fn id(&self) -> SystemID { self.id }

    fn access(&self) -> AccessSets { Self::access_sets() }

    #[inline]
    fn backend(&self) -> SystemBackend { SystemBackend::GPU }

    fn run(&self, _world: abm_framework::engine::manager::ECSReference<'_>) -> ECSResult<()> {
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

    fn entry_point(&self) -> &'static str { "main" }

    fn workgroup_size(&self) -> u32 { 256 }
}
