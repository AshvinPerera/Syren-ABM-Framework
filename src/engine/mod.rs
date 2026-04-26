//! # Engine Module
//!
//! Internal ECS engine implementation.
//!
//! This module contains all core ECS building blocks such as:
//! - Archetypes
//! - Entity management
//! - Component storage
//! - Query execution
//! - Scheduling and systems
//!
//! Public API exposure is controlled by `lib.rs`.

pub mod activation;
pub mod archetype;
pub mod borrow;
pub mod boundary;
pub mod channel_allocator;
pub mod commands;
pub mod component;
pub mod dot_export;
pub mod entity;
pub mod error;
pub mod manager;
pub mod plan_display;
pub mod query;
pub mod reduce;
pub mod scheduler;
pub mod storage;
pub mod systems;
pub mod types;
pub mod workers;

#[cfg(feature = "gpu")]
pub mod dirty;
