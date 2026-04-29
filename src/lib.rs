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
/// - <https://ui.perfetto.dev>
///
/// Enable with `--features profiling`.
pub(crate) mod profiling;

// -----------------------------------------------------------------------------
// Re-exports (Public API)
// -----------------------------------------------------------------------------

// Core ECS manager and world access

pub use engine::manager::{BoundaryHandle, ECSManager, ECSReference, Read, Write};

// Entity types

pub use engine::entity::{Entity, EntityLocation};

// Component registry and registration

pub use engine::component::{Bundle, ComponentDesc, ComponentRegistry, DynamicBundle, Signature};

#[cfg(feature = "gpu")]
pub use engine::component::{register_gpu_component, GPUPod};

// Reduction primitives

pub use engine::reduce::{Count, MinMax, Sum, Welford};

// Query construction

pub use engine::query::{BuiltQuery, QueryBuilder, QueryComponent, QuerySignature};

// Systems and scheduling

#[cfg(feature = "gpu")]
pub use engine::systems::GpuSystem;
pub use engine::systems::System;
pub use engine::systems::{AccessSets, FnSystem, SystemBackend};

// Channel-aware scheduling

pub use engine::systems::{ChannelOrder, ChannelSet};

pub use engine::scheduler::{Scheduler, Stage};

// Deferred commands

pub use engine::commands::Command;

// Error types

pub use engine::error::{
    AttributeError, ECSError, ECSResult, ExecutionError, MoveError, SpawnError,
};

// User-attributable error context for boundary access.
pub use engine::error::BoundaryAccessFailure;

// Primitive type aliases and constants

pub use engine::types::{
    AgentTemplateId, ArchetypeID, ChunkID, ComponentID, EntityID, SystemID, CHUNK_CAP,
};

// Opaque scheduling identifiers.
pub use engine::types::{BoundaryID, ChannelID};

#[cfg(feature = "gpu")]
pub use engine::types::{GPUAccessMode, GPUResourceID};

pub use engine::activation::ActivationOrder;
pub use engine::boundary::{BoundaryChannelProfile, BoundaryContext, BoundaryResource};
pub use engine::dot_export::DotExport;
pub use engine::plan_display::PlanDisplay;
pub use engine::workers::{max_workers, worker_id};

// Profiling public API
pub use profiling::profiler::{
    flush_thread, init, next_arg, shutdown, span, span_fmt, thread_name, try_init, Arg,
    ProfilingError, SpanGuard, SpanName,
};

/// Advanced extension APIs that expose storage and scheduling internals.
///
/// These types are intentionally kept out of the root API because callers must
/// preserve ECS storage invariants manually when using them.
pub mod advanced {
    pub use crate::engine::archetype::{Archetype, ChunkBorrow};
    pub use crate::engine::channel_allocator::ChannelAllocator;
    pub use crate::engine::entity::EntityShards;
    pub use crate::engine::manager::ECSData;
    pub use crate::engine::storage::{cast_slice, cast_slice_mut, Attribute, TypeErasedAttribute};
}

#[cfg(feature = "agents")]
pub mod agents;

#[cfg(feature = "environment")]
pub mod environment;

#[cfg(feature = "messaging")]
pub mod messaging;

#[cfg(feature = "model")]
pub mod model;

// -----------------------------------------------------------------------------
// Prelude (Optional but recommended)
// -----------------------------------------------------------------------------

/// Commonly used ECS types.
///
/// Import with:
/// ```rust
/// use abm_framework::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        BuiltQuery, ComponentRegistry, ECSManager, ECSReference, Entity, FnSystem, QueryBuilder,
        QueryComponent, QuerySignature, Signature, System, SystemBackend,
    };
}
