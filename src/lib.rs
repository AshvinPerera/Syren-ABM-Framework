//! # Syren
//!
//! Syren is a parallel Rust framework for agent-based models. It stores agents
//! in an archetype entity-component-system (ECS), runs systems over them through
//! a deterministic stage scheduler on top of Rayon, and adds agent, environment,
//! messaging, and optional GPU layers behind Cargo features.
//!
//! This is the API reference. For a task-oriented introduction, worked examples,
//! and the reproducibility guarantees, see the guide at
//! <https://ashvinperera.github.io/Syren-ABM-Framework/>.
//!
//! ## Installation
//!
//! Syren has no default features; enable the ones your model needs:
//!
//! ```toml
//! [dependencies]
//! syren = { version = "0.6.0-rc.1", features = ["model"] }
//! ```
//!
//! ## Features
//!
//! - `agents`, `environment` — agent templates and typed model-wide values.
//! - `model` — the [`ModelBuilder`](model::ModelBuilder) layer (implies `agents`
//!   and `environment`).
//! - `messaging` — the four message specialisations.
//! - `gpu`, `messaging_gpu` — GPU state mirroring and compute dispatch.
//! - `profiling` — tracing spans and Chrome Trace output.
//!
//! ## Getting started
//!
//! The `first_model` example is the smallest complete model: register a
//! component, build a population with [`ModelBuilder`](model::ModelBuilder), run
//! a system whose access is derived from its query, draw per-entity randomness
//! from the run context with [`DetRng`], and summarise with a [`Welford`]
//! reduction. Run it with `cargo run --example first_model --features model`.
//!
//! ## Determinism
//!
//! A model run with the same version, features, seed, and initial state produces
//! the same trajectory regardless of thread count. Draw randomness through
//! [`DetRng::from_context`], keyed on the run context and a salt, and set the
//! model seed with [`ModelBuilder::with_seed`](model::ModelBuilder::with_seed).
//! See the guide's reproducibility chapter for the model author's obligations.
//!
//! ## Stability
//!
//! Syren is pre-1.0: patch releases keep the public API; minor releases may break
//! it with migration notes. Lower-level building blocks live in the [`advanced`]
//! module and may change with less notice.

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

pub use engine::manager::{
    BoundaryHandle, ECSManager, ECSReference, EntityQueryParam, QueryParam, Read, Write,
};

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
pub use engine::commands::{BatchColumn, SpawnBatch};

// Error types

pub use engine::error::{
    AttributeError, ECSError, ECSResult, ExecutionError, MoveError, RegistryError, SpawnError,
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

pub use engine::activation::{ActivationOrder, RunContext};
pub use engine::boundary::{BoundaryChannelProfile, BoundaryContext, BoundaryResource};
pub use engine::dot_export::DotExport;
pub use engine::plan_display::PlanDisplay;
pub use engine::random::DetRng;
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
    pub use crate::engine::worker_stage::WorkerStage;
    pub use crate::engine::workers::{max_workers, worker_id};
}

#[cfg(feature = "agents")]
pub mod agents;

#[cfg(feature = "environment")]
pub mod environment;

#[cfg(feature = "messaging")]
pub mod messaging;

#[cfg(feature = "model")]
pub mod model;

#[cfg(feature = "environment")]
pub mod space;

// -----------------------------------------------------------------------------
// Prelude (Optional but recommended)
// -----------------------------------------------------------------------------

/// Commonly used ECS types.
///
/// Import with:
/// ```rust
/// use syren::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        BuiltQuery, ComponentRegistry, ECSManager, ECSReference, Entity, FnSystem, QueryBuilder,
        QueryComponent, QuerySignature, RunContext, Signature, System, SystemBackend,
    };
}
