//! # Archetype management and entity migration
//!
//! This module defines the central orchestration layer of the ECS, responsible
//! for:
//!
//! * owning archetypes and their component storage,
//! * coordinating entity movement between archetypes,
//! * managing deferred structural mutations via commands,
//! * providing controlled parallel access to component data,
//! * executing chunk-based parallel iteration.
//!
//! ## Concurrency model
//!
//! The ECS uses **fine-grained, column-level synchronization** to support
//! safe parallel execution:
//!
//! * Each component column inside an archetype is protected by an `RwLock`.
//! * Parallel systems may:
//!   * read the same component concurrently,
//!   * write disjoint component sets concurrently,
//!   * read while others read.
//! * Writes to the same component column are mutually exclusive.
//!
//! Structural mutations (spawn, despawn, archetype migration) **must not**
//! occur during parallel iteration and must be executed at explicit
//! synchronization points.
//!
//! This constraint is enforced by execution phase discipline.
//! Violating it may result in deadlock or panic.
//!
//! ## Lock ordering
//!
//! To prevent deadlock, all code in this module and in `manager.rs` must
//! acquire locks in the following order:
//!
//! 1. **Column locks** (`LockedAttribute` read/write guards), acquired in
//!    ascending `ComponentID` order.
//! 2. **Archetype metadata lock** (`self.meta` `RwLock`), acquired *after*
//!    all column locks have been released or within a scope where no column
//!    locks are held.
//!
//! Acquiring `meta` while holding a column lock — or acquiring column locks
//! out of `ComponentID` order — may produce lock-order inversion and deadlock
//! under concurrent access.
//!
//! ## Safety model
//!
//! * Component data access is synchronized via column-level locks.
//! * Raw pointers are only exposed after acquiring the appropriate locks.
//! * Parallel iteration relies on chunk-level disjointness guaranteed by
//!   archetype layout.
//!
//! ## Unsafe code
//!
//! This module contains `unsafe` code for:
//!
//! * converting locked component storage into raw byte slices,
//! * parallel execution using Rayon,
//! * low-level performance optimizations.

mod core;
mod columns;
mod lifecycle;
mod migration;
mod access;

pub use core::{Archetype, ArchetypeMatch};
pub use access::ChunkBorrow;
