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
//!
//! This crate builds as both:
//! - `rlib` (for Rust usage & integration tests)
//! - `cdylib` (for FFI / DLL usage)

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![allow(clippy::module_inception)]
#![deny(dead_code)]

pub mod engine;

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports (Public API)
// ─────────────────────────────────────────────────────────────────────────────

// Core ECS types

pub use engine::manager::{
    ECSManager,
    ECSReference,
};

pub use engine::entity::{
    Entity,
    EntityLocation,
    EntityShards,
};

pub use engine::component::{
    Signature,
    register_component,
    freeze_components,
    component_id_of,
};

pub use engine::query::QueryBuilder;

pub use engine::systems::System;
pub use engine::systems::{FnSystem, SystemBackend};
pub use engine::scheduler::{
    Stage,
    Scheduler
};

pub use engine::commands::Command;

pub use engine::error::{
    ECSResult,
    ECSError,
    SpawnError,
    AttributeError,
    ExecutionError,
    MoveError,
};

pub use engine::types::{
    EntityID,
    ComponentID,
    ArchetypeID
};

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
        register_component,
        freeze_components,
        component_id_of,
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature placeholders
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "rollback")]
mod rollback_support {}
