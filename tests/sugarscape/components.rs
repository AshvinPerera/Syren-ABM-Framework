#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct AgentTag(pub u8);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Sugar(pub f32);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Metabolism(pub f32);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Vision(pub i32);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RNG {
    pub state: u64,
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Alive(pub u32);

#[cfg(feature = "gpu")]
use abm_framework::GPUPod;

#[cfg(feature = "gpu")]
unsafe impl GPUPod for Sugar {}

#[cfg(feature = "gpu")]
unsafe impl GPUPod for Metabolism {}

#[cfg(feature = "gpu")]
unsafe impl GPUPod for Alive {}

#[cfg(feature = "gpu")]
unsafe impl GPUPod for Position {}

#[cfg(feature = "gpu")]
unsafe impl GPUPod for Vision {}

#[cfg(feature = "gpu")]
unsafe impl GPUPod for RNG {}
