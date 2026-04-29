//! # GPU Context
//!
//! This module provides a minimal, centralized abstraction for initializing and
//! owning the GPU device state used by the ECS GPU backend.
//!
//! ## Purpose
//! `GPUContext` encapsulates the creation and lifetime of:
//! * a [`wgpu::Device`], and
//! * a corresponding [`wgpu::Queue`],
//!
//! which together form the execution environment for GPU-backed systems.
//!
//! ## Design philosophy
//!
//! * **Single point of GPU initialization**
//!   - All GPU access in the ECS flows from a single `GPUContext` instance.
//! * **Backend-agnostic**
//!   - Uses `wgpu` to remain portable across Vulkan, Metal, DX12, and WebGPU.
//! * **Explicit failure handling**
//!   - Initialization failures are surfaced as ECS execution errors.
//!
//! ## Concurrency model
//!
//! `wgpu::Device` and `wgpu::Queue` are internally thread-safe and may be shared
//! across threads. The ECS enforces higher-level synchronization around GPU usage
//! (e.g. backend stages, exclusive access) rather than relying on GPU-level locks.
//!
//! ## Feature gating
//!
//! This module is only compiled when the `gpu` feature is enabled. When the
//! feature is disabled, the ECS remains fully functional using CPU execution
//! paths only.
//!
//! ## Failure modes
//!
//! GPU initialization may fail due to:
//! * lack of a compatible adapter,
//! * driver or backend initialization errors,
//! * platform limitations.
//!
//! All such failures are reported as [`ExecutionError::GpuInitFailed`] to ensure
//! consistent error propagation through the ECS API.

#![cfg(feature = "gpu")]

use wgpu::Instance;

use crate::engine::error::{ECSError, ECSResult, ExecutionError};

/// Owned GPU execution context.
///
/// ## Role
/// `GPUContext` owns the low-level GPU objects required to execute compute
/// workloads:
/// * a logical [`wgpu::Device`], and
/// * a submission [`wgpu::Queue`].
///
/// ## Responsibilities
/// * Perform adapter selection.
/// * Create the logical device and queue.
/// * Serve as a shared handle for GPU subsystems.
///
/// ## Thread safety
/// Both fields are safe to share across threads. The ECS runtime is responsible
/// for ensuring correct usage ordering and synchronization.
#[derive(Debug)]
pub struct GPUContext {
    /// The GPU device used for GPU path execution
    pub device: wgpu::Device,

    /// The command queue for the GPU
    pub queue: wgpu::Queue,
}

impl GPUContext {
    /// Initializes a new GPU execution context.
    ///
    /// ## Behaviour
    /// This function:
    /// 1. Creates a default `wgpu::Instance`.
    /// 2. Requests a high-performance GPU adapter.
    /// 3. Creates a logical device and submission queue.
    ///
    /// The configuration intentionally requests:
    /// * no optional GPU features,
    /// * default resource limits,
    /// * no tracing or experimental features.
    ///
    /// This ensures maximum compatibility across platforms.
    ///
    /// ## Blocking behaviour
    /// GPU initialization is performed synchronously using `pollster::block_on`.
    /// This is expected to occur during application startup, not in hot paths.
    ///
    /// ## Errors
    /// Returns [`ExecutionError::GpuInitFailed`] if:
    /// * no compatible adapter is found,
    /// * device creation fails,
    /// * or the GPU backend cannot be initialized.
    pub fn new() -> ECSResult<Self> {
        let instance = Instance::default();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| {
            ECSError::from(ExecutionError::GpuInitFailed {
                message: format!("{e:?}").into(),
            })
        })?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("abm_framework_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 10,
                ..wgpu::Limits::default()
            },
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| {
            ECSError::from(ExecutionError::GpuInitFailed {
                message: format!("{e:?}").into(),
            })
        })?;

        Ok(Self { device, queue })
    }
}
