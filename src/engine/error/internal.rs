//! Typed enumeration of internal ECS framework invariant violations.
//!
//! This module defines [`InternalViolation`], a structured error type that
//! represents bugs in the ECS engine or its scheduling layer - not recoverable
//! user errors. Each variant encodes a specific invariant that was broken,
//! making these errors matchable, testable, and grep-friendly compared to
//! freeform string messages.
//!
//! ## Organization
//!
//! Variants are grouped by the module in which the violation originates:
//!
//! - **`archetype.rs`** - violations of per-archetype structural invariants
//!   (slot occupancy, component presence, metadata consistency, registry state).
//! - **`manager.rs`** - violations of query shape contracts and archetype
//!   pair access rules enforced by the world manager.
//!
//! ## Usage
//!
//! `InternalViolation` is intended to be wrapped by the crate's primary error
//! type, typically as something like `ECSError::Internal(InternalViolation)`.
//! Callers that need to distinguish between specific engine bugs can match on
//! individual variants; all other callers can treat the type opaquely and
//! propagate it upward.
//!
//! ## Stability
//!
//! This enum is marked `#[non_exhaustive]`. New variants may be added as
//! additional invariants are identified without constituting a breaking change.

use std::fmt;

/// Typed enumeration of internal invariant violations.
///
/// These replace the previous strongly-typed `ECSError::Internal(Cow<'static, str>)`
/// variant, making internal errors matchable, testable, and grep-friendly.
///
/// Each variant corresponds to a specific framework-level invariant that was
/// violated. These are **not** user-recoverable - they indicate bugs in the
/// ECS engine or its scheduling layer.
///
/// ## Design
/// Variants are grouped by originating module for clarity.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InternalViolation {
    // -- archetype.rs -------------------------------------------------------
    /// An `RwLock` protecting archetype metadata was poisoned.
    ArchetypeMetaLockPoisoned,

    /// `insert_empty_component` was called for a component that already
    /// exists in the archetype.
    ComponentAlreadyPresent,

    /// `spawn_on` found that the target entity slot was already occupied.
    SpawnSlotOccupied,

    /// `despawn_on` was called for an entity that does not belong to
    /// this archetype.
    DespawnEntityNotInArchetype,

    /// During `despawn_on`, a swap-remove produced a misaligned component
    /// position across columns.
    DespawnSwapMisalignment,

    /// During `despawn_on`, the moved (swapped) slot had no associated
    /// entity - metadata is out of sync.
    DespawnMovedSlotMissingEntity,

    /// `from_components` encountered a component type that is not
    /// registered in the global registry.
    ComponentTypeNotRegistered,

    /// `from_components` detected a mismatch between the archetype's
    /// signature and the set of provided storage columns.
    SignatureStorageMismatch,

    // -- manager.rs ---------------------------------------------------------
    /// A typed iteration helper (e.g. `for_each_read`, `for_each_write`)
    /// was invoked with a query whose read/write shape does not match the
    /// helper's requirements.
    ///
    /// ## Fields
    /// * `method` - name of the helper that was called.
    /// * `expected_reads` - number of read components the helper expects.
    /// * `expected_writes` - number of write components the helper expects.
    QueryShapeMismatch {
        /// Name of the iteration helper that was called.
        method: &'static str,

        /// Number of read components the helper expects.
        expected_reads: usize,

        /// Number of write components the helper expects.
        expected_writes: usize,
    },

    /// `reduce_abstraction` was called with a query that declares write
    /// access, which is not supported for reductions.
    ReduceWritesNotAllowed,

    /// `get_archetype_pair_mut` was called with two identical archetype IDs.
    ArchetypePairSameId,

    /// A [`std::sync::Mutex`] or [`std::sync::RwLock`] was found in a
    /// poisoned state, indicating that another thread panicked while holding
    /// the lock.
    PoisonedLock,
}

impl fmt::Display for InternalViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InternalViolation::ArchetypeMetaLockPoisoned => {
                f.write_str("archetype meta lock poisoned")
            }
            InternalViolation::ComponentAlreadyPresent => {
                f.write_str("insert_empty_component: component already present in archetype")
            }
            InternalViolation::SpawnSlotOccupied => {
                f.write_str("spawn_on: target entity slot already occupied")
            }
            InternalViolation::DespawnEntityNotInArchetype => {
                f.write_str("despawn_on: entity not in this archetype")
            }
            InternalViolation::DespawnSwapMisalignment => {
                f.write_str("despawn_on: component swap misalignment")
            }
            InternalViolation::DespawnMovedSlotMissingEntity => {
                f.write_str("despawn_on: moved slot missing entity; metadata out of sync")
            }
            InternalViolation::ComponentTypeNotRegistered => {
                f.write_str("from_components: component type not registered")
            }
            InternalViolation::SignatureStorageMismatch => {
                f.write_str("from_components: archetype signature and storage mismatch")
            }
            InternalViolation::QueryShapeMismatch {
                method,
                expected_reads,
                expected_writes,
            } => {
                write!(
                    f,
                    "{}: query must have exactly {} read(s) and {} write(s)",
                    method, expected_reads, expected_writes
                )
            }
            InternalViolation::ReduceWritesNotAllowed => {
                f.write_str("reduce_abstraction: writes not allowed")
            }
            InternalViolation::ArchetypePairSameId => f.write_str(
                "get_archetype_pair_mut: source and destination archetype IDs are equal",
            ),
            InternalViolation::PoisonedLock => {
                f.write_str("a Mutex or RwLock was found in a poisoned state")
            }
        }
    }
}

impl std::error::Error for InternalViolation {}
