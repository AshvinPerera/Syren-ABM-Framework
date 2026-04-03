//! # Entity Management
//!
//! This module defines the entity identity, lifecycle, and shard-based storage
//! used by the ECS.
//!
//! ## Purpose
//! Entities are lightweight, opaque identifiers that reference rows in archetype
//! storage. This module is responsible for:
//!
//! - Generating stable entity identifiers
//! - Tracking entity liveness via versioning
//! - Mapping entities to archetype locations
//! - Managing scalable entity allocation using sharded storage
//!
//! ## Entity Model
//! An `Entity` is a compact, versioned handle composed of:
//!
//! - A **shard ID**, identifying which shard owns the entity
//! - An **index**, identifying the slot within the shard
//! - A **version**, used to detect stale or recycled entities
//!
//! This layout allows fast validation and prevents use-after-free bugs when
//! entities are despawned and reused.
//!
//! ## Sharding
//! Entities are distributed across multiple shards (`EntityShards`) to reduce
//! contention during concurrent spawning, despawning, and lookup operations.
//!
//! Each shard maintains:
//! - A dense pool of entity slots
//! - Version counters for stale detection
//! - Location metadata pointing into archetype storage
//!
//! ## Invariants
//! - An entity is considered alive if and only if its version matches the
//!   version stored in its shard and its slot is marked alive.
//! - Entity locations must always reflect the actual archetype row.
//! - Despawning an entity invalidates all previous handles to that entity.
//!
//! ## Concurrency
//! - Shards synchronize entity allocation and metadata using internal locks.
//! - Global counters are maintained using atomics for low-contention signals.
//!
//! ## Safety
//! Correctness relies on:
//! - Updating entity locations atomically with archetype row moves
//! - Never mutating entity metadata while systems hold archetype borrows
//! - Applying structural changes only at synchronization points

mod entity;
mod location;
mod entities;
mod shards;

pub use entity::Entity;
pub use location::EntityLocation;
pub use shards::EntityShards;
