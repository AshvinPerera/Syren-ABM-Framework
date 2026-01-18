//! GPU systems for the Sugarscape model.

#![cfg(feature = "gpu")]

use abm_framework::engine::{
    systems::{System, SystemBackend, AccessSets, GpuSystem},
    component::component_id_of,
    manager::ECSReference,
    error::ECSResult,
    types::GPUResourceID,
};

use crate::sugarscape::components::*;

pub struct MetabolismGpuSystem;

impl System for MetabolismGpuSystem {
    fn id(&self) -> u16 { 3 }
    fn backend(&self) -> SystemBackend { SystemBackend::GPU }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Metabolism>().unwrap());
        a.read.set(component_id_of::<Alive>().unwrap());
        a.write.set(component_id_of::<Sugar>().unwrap());
        a
    }

    fn run(&self, _: ECSReference<'_>) -> ECSResult<()> { Ok(()) }
    fn gpu(&self) -> Option<&dyn GpuSystem> { Some(self) }
}

impl GpuSystem for MetabolismGpuSystem {
    fn shader(&self) -> &'static str {
        r#"
struct Params { entity_len: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read> metab : array<f32>;
@group(0) @binding(1) var<storage, read> alive : array<u32>;
@group(0) @binding(2) var<storage, read_write> sugar : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.entity_len) { return; }
    if (alive[i] == 0u) { return; }
    sugar[i] = sugar[i] - metab[i];
}
"#
    }
    fn entry_point(&self) -> &'static str { "main" }
    fn workgroup_size(&self) -> u32 { 256 }
}

pub struct DeathGpuSystem;

impl System for DeathGpuSystem {
    fn id(&self) -> u16 { 4 }
    fn backend(&self) -> SystemBackend { SystemBackend::GPU }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Sugar>().unwrap());
        a.write.set(component_id_of::<Alive>().unwrap());
        a
    }

    fn run(&self, _: ECSReference<'_>) -> ECSResult<()> { Ok(()) }
    fn gpu(&self) -> Option<&dyn GpuSystem> { Some(self) }
}

impl GpuSystem for DeathGpuSystem {
    fn shader(&self) -> &'static str {
        r#"
struct Params { entity_len: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read> sugar : array<f32>;
@group(0) @binding(1) var<storage, read_write> alive : array<u32>;
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.entity_len) { return; }
    if (alive[i] == 0u) { return; }
    if (sugar[i] <= 0.0) { alive[i] = 0u; }
}
"#
    }
    fn entry_point(&self) -> &'static str { "main" }
    fn workgroup_size(&self) -> u32 { 256 }
}

pub struct AgentIntentGpuSystem {
    resources: [GPUResourceID; 2],
    writes:    [GPUResourceID; 1],
}

impl AgentIntentGpuSystem {
    pub fn new(sugar_grid: GPUResourceID, intent: GPUResourceID) -> Self {
        Self {
            resources: [sugar_grid, intent],
            writes: [intent],
        }
    }
}

impl System for AgentIntentGpuSystem {
    fn id(&self) -> u16 { 10 }
    fn backend(&self) -> SystemBackend { SystemBackend::GPU }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Vision>().unwrap());
        a.read.set(component_id_of::<Position>().unwrap());
        a.read.set(component_id_of::<Alive>().unwrap());
        a
    }

    fn run(&self, _: ECSReference<'_>) -> ECSResult<()> { Ok(()) }
    fn gpu(&self) -> Option<&dyn GpuSystem> { Some(self) }
}

impl GpuSystem for AgentIntentGpuSystem {
    fn shader(&self) -> &'static str { include_str!("shaders/agent_intent.wgsl") }
    fn entry_point(&self) -> &'static str { "main" }
    fn workgroup_size(&self) -> u32 { 256 }
    fn uses_resources(&self) -> &[GPUResourceID] { &self.resources }
    fn writes_resources(&self) -> &[GPUResourceID] { &self.writes }
}

pub struct ResolveHarvestGpuSystem {
    resources: [GPUResourceID; 2],
    writes:    [GPUResourceID; 1],
}

impl ResolveHarvestGpuSystem {
    pub fn new(sugar_grid: GPUResourceID, intent: GPUResourceID) -> Self {
        Self {
            resources: [sugar_grid, intent],
            writes: [sugar_grid],
        }
    }
}

impl System for ResolveHarvestGpuSystem {
    fn id(&self) -> u16 { 11 }
    fn backend(&self) -> SystemBackend { SystemBackend::GPU }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Alive>().unwrap());
        a.write.set(component_id_of::<Position>().unwrap());
        a.write.set(component_id_of::<Sugar>().unwrap());
        a
    }

    fn run(&self, _: ECSReference<'_>) -> ECSResult<()> { Ok(()) }
    fn gpu(&self) -> Option<&dyn GpuSystem> { Some(self) }
}

impl GpuSystem for ResolveHarvestGpuSystem {
    fn shader(&self) -> &'static str { include_str!("shaders/resolve_harvest.wgsl") }
    fn entry_point(&self) -> &'static str { "main" }
    fn workgroup_size(&self) -> u32 { 256 }
    fn uses_resources(&self) -> &[GPUResourceID] { &self.resources }
    fn writes_resources(&self) -> &[GPUResourceID] { &self.writes }
}

pub struct SugarRegrowthGpuSystem {
    resources: [GPUResourceID; 1],
}

impl SugarRegrowthGpuSystem {
    pub fn new(sugar_grid: GPUResourceID) -> Self {
        Self { resources: [sugar_grid] }
    }
}

impl System for SugarRegrowthGpuSystem {
    fn id(&self) -> u16 { 12 }
    fn backend(&self) -> SystemBackend { SystemBackend::GPU }
    fn access(&self) -> AccessSets { AccessSets::default() }
    fn run(&self, _: ECSReference<'_>) -> ECSResult<()> { Ok(()) }
    fn gpu(&self) -> Option<&dyn GpuSystem> { Some(self) }
}

impl GpuSystem for SugarRegrowthGpuSystem {
    fn shader(&self) -> &'static str { include_str!("shaders/sugar_regrowth.wgsl") }
    fn entry_point(&self) -> &'static str { "main" }
    fn workgroup_size(&self) -> u32 { 256 }
    fn uses_resources(&self) -> &[GPUResourceID] { &self.resources }
    fn writes_resources(&self) -> &[GPUResourceID] { &self.resources }
}

pub struct ClearOccupancyGpuSystem {
    resources: [GPUResourceID; 1],
}

impl ClearOccupancyGpuSystem {
    pub fn new(sugar_grid: GPUResourceID) -> Self {
        Self { resources: [sugar_grid] }
    }
}

impl System for ClearOccupancyGpuSystem {
    fn id(&self) -> u16 { 13 }
    fn backend(&self) -> SystemBackend { SystemBackend::GPU }
    fn access(&self) -> AccessSets { AccessSets::default() }
    fn run(&self, _: ECSReference<'_>) -> ECSResult<()> { Ok(()) }
    fn gpu(&self) -> Option<&dyn GpuSystem> { Some(self) }
}

impl GpuSystem for ClearOccupancyGpuSystem {
    fn shader(&self) -> &'static str { include_str!("shaders/clear_occupancy.wgsl") }
    fn entry_point(&self) -> &'static str { "main" }
    fn workgroup_size(&self) -> u32 { 256 }
    fn uses_resources(&self) -> &[GPUResourceID] { &self.resources }
    fn writes_resources(&self) -> &[GPUResourceID] { &self.resources }
}
