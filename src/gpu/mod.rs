//! # GPU Execution Backend
//!
//! This module implements the **GPU execution backend** for the ECS engine,
//! enabling selected systems to be executed as **compute shaders** via `wgpu`.
//!
//! The GPU backend is an **optional, feature-gated extension** (`feature = "gpu"`)
//! that integrates tightly with the ECS runtime while preserving all safety,
//! scheduling, and structural invariants enforced by the CPU execution path.
//!
//! ## Design goals
//!
//! * Execute data-parallel ECS systems efficiently on the GPU
//! * Preserve archetype-based memory layout semantics
//! * Maintain strict ECS phase discipline and structural safety
//! * Provide explicit, deterministic error propagation
//! * Avoid hidden synchronization or implicit state
//!
//! ---
//!
//! ## High-level execution model
//!
//! GPU execution proceeds in **four explicit stages**:
//!
//! 1. **Upload**
//!    * Component columns required by the system are copied from archetype storage
//!      into GPU buffers.
//!    * Only components explicitly marked as GPU-safe may be uploaded.
//!
//! 2. **Dispatch**
//!    * A compute pipeline is selected or created based on:
//!       - system identity
//!       - shader module
//!       - component binding layout
//!    * Each matching archetype is dispatched independently.
//!
//! 3. **Synchronization**
//!    * GPU execution is explicitly synchronized using `wgpu::Device::poll`.
//!    * All GPU errors are surfaced and mapped into ECS execution errors.
//!
//! 4. **Download**
//!    * Mutated component buffers are copied back into archetype storage.
//!    * chunk structure and row ordering is preserved.
//!
//! All GPU execution occurs inside an **exclusive ECS phase**, preventing
//! concurrent CPU iteration or structural mutation.
//!
//! ---
//!
//! ## Module structure
//!
//! * [`context`] — GPU device and queue initialization
//! * [`mirror`] — Host to GPU buffer mirroring for component columns
//! * [`pipeline`] — Compute pipeline creation and caching
//! * [`layout`] — Bind group layout construction for component access
//! * [`dispatch`] — System execution orchestration
//!
//! Only the high-level entry point is exposed publicly.
//!
//! ---
//!
//! ## Safety and correctness
//!
//! This module contains unsafe and low-level GPU code.
//! Correctness relies on the following invariants:
//!
//! * ECS phase discipline prevents structural mutation during GPU execution
//! * Component borrow rules are enforced before GPU dispatch
//! * GPU buffers are treated as **type-erased mirrors** of component columns
//! * Synchronization is explicit and never implicit
//! * GPU errors are never ignored or silently suppressed
//!
//! Violating these invariants may result in undefined behavior or corrupted
//! ECS state.
//!
//! ---
//!
//! ## Intended usage
//!
//! GPU execution is not automatic.
//! Systems must explicitly implement the `GpuSystem` trait and provide:
//!
//! * a compute shader
//! * a binding layout contract
//! * a workgroup configuration
//!
//! Systems that do not implement `GpuSystem` are always executed on the CPU.
//!
//! ---
//!
//! ## Public API
//!
//! * [`execute_gpu_system`] — Executes a single ECS system using the GPU backend
//!
//! This function is invoked by the scheduler when a system is marked
//! for GPU execution.
//!

#![cfg(feature = "gpu")]

mod context;
mod mirror;
mod pipeline;
mod layout;
mod dispatch;

pub use dispatch::{
    execute_gpu_system,
    sync_pending_to_cpu
};
