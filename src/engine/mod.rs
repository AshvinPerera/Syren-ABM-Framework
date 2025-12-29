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

pub mod types;
pub mod error;
pub mod component;
pub mod storage;
pub mod entity;
pub mod archetype;
pub mod query;
pub mod commands;
pub mod systems;
pub mod scheduler;
pub mod manager;
pub mod borrow;
pub mod reduce;
