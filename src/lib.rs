//! # ABM Framework
//!
//! High-performance, parallel Entity-Component-System (ECS) framework
//! designed for large-scale Agent-Based Models (ABM).
//!
//! ## Design Goals
//! - Archetype-based storage for cache efficiency
//! - Deterministic scheduling
//! - Parallel CPU execution (GPU-ready architecture)
//! - Safe, explicit data access

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![allow(clippy::module_inception)]

pub(crate) mod engine;

#[cfg(feature = "gpu")]
pub mod gpu;

/// Chrome Trace (flame-style) profiler for ECS execution.
///
/// This module provides a feature-gated, zero-overhead (when disabled)
/// profiling API that emits Chrome Trace JSON compatible with:
/// - chrome://tracing
/// - https://ui.perfetto.dev
///
/// Enable with `--features profiling`.
pub(crate) mod profiling;

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports (Public API)
// ─────────────────────────────────────────────────────────────────────────────

// Core ECS manager and world access

pub use engine::manager::{
    ECSManager,
    ECSReference,
    ECSData,
    Read,
    Write,
};

// Entity types

pub use engine::entity::{
    Entity,
    EntityLocation,
    EntityShards,
};

// Component registry and registration

pub use engine::component::{
    Signature,
    ComponentRegistry,
    DynamicBundle,
    Bundle,
    ComponentDesc,
};

#[cfg(feature = "gpu")]
pub use engine::component::{
    GPUPod,
    register_gpu_component,
};

// Reduction primitives

pub use engine::reduce::{Count, Sum, MinMax, Welford};

// Query construction

pub use engine::query::QueryBuilder;

// Systems and scheduling

pub use engine::systems::System;
pub use engine::systems::{FnSystem, SystemBackend, AccessSets};
#[cfg(feature = "gpu")]
pub use engine::systems::GpuSystem;

pub use engine::scheduler::{
    Stage,
    Scheduler
};

// Deferred commands

pub use engine::commands::Command;

// Error types

pub use engine::error::{
    ECSResult,
    ECSError,
    SpawnError,
    AttributeError,
    ExecutionError,
    MoveError,
};

// Primitive type aliases and constants

pub use engine::types::{
    EntityID,
    ComponentID,
    ArchetypeID,
    SystemID,
    ChunkID,
    CHUNK_CAP,
};

#[cfg(feature = "gpu")]
pub use engine::types::{
    GPUAccessMode,
    GPUResourceID,
};

// Archetype and chunk borrowing

pub use engine::archetype::{Archetype, ChunkBorrow};

// Storage utilities

pub use engine::storage::{
    Attribute,
    TypeErasedAttribute,
    cast_slice,
    cast_slice_mut,
};

// Profiling public API
pub use profiling::profiler::{init, shutdown, span, span_fmt, thread_name, next_arg, SpanGuard, SpanName, Arg};

// ─────────────────────────────────────────────────────────────────────────────
// Prelude (Optional but recommended)
// ─────────────────────────────────────────────────────────────────────────────

/// Commonly used ECS types.
///
/// Import with:
/// ```rust
/// use abm_framework::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        ECSManager,
        ECSReference,
        Entity,
        QueryBuilder,
        System,
        FnSystem,
        SystemBackend,
        Signature,
        ComponentRegistry,
    };
}
