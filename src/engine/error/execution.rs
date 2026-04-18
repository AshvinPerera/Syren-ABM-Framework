//! Runtime errors for ECS execution and query iteration.
//!
//! This module defines [`ExecutionError`], the single error type returned by all
//! fallible ECS operations, along with the supporting enums [`AccessKind`] and
//! [`InvalidAccessReason`] that appear in its variants.
//!
//! ## Error categories
//!
//! | Variant | Cause |
//! |---|---|
//! | [`StructuralMutationDuringIteration`] | Spawn/despawn or component add/remove called while a query iterator is live |
//! | [`BorrowConflict`] | Conflicting read/write access between parallel systems or within a single query |
//! | [`InvalidQueryAccess`] | Contradictory access declarations (e.g. read + write on the same component) |
//! | [`MissingComponent`] | Query tried to fetch a component absent from the matched archetype |
//! | [`SchedulerInvariantViolation`] | Scheduler broke its own declared access guarantees |
//! | [`LockPoisoned`] | A thread panicked while holding a synchronization primitive |
//! | `Gpu*` *(feature = "gpu")* | GPU unavailable, component not GPU-safe, or dispatch failure |
//! | [`InternalExecutionError`] | Unsafe execution path invoked incorrectly |
//!
//! ## Guarantees
//!
//! All errors in this module are:
//! - **Deterministic** — the same incorrect usage always produces the same error.
//! - **Non-destructive** — an `ExecutionError` is never returned after a partial
//!   mutation; the ECS state remains consistent.
//! - **Bug indicators** — every variant reflects incorrect API usage or a
//!   scheduling bug, never an expected runtime condition.
//!
//! [`StructuralMutationDuringIteration`]: ExecutionError::StructuralMutationDuringIteration
//! [`BorrowConflict`]: ExecutionError::BorrowConflict
//! [`InvalidQueryAccess`]: ExecutionError::InvalidQueryAccess
//! [`MissingComponent`]: ExecutionError::MissingComponent
//! [`SchedulerInvariantViolation`]: ExecutionError::SchedulerInvariantViolation
//! [`LockPoisoned`]: ExecutionError::LockPoisoned
//! [`InternalExecutionError`]: ExecutionError::InternalExecutionError

use std::fmt;

use crate::engine::types::{ComponentID};

#[cfg(feature = "gpu")]
use crate::engine::types::{ArchetypeID};

#[cfg(feature = "gpu")]
use crate::engine::types::GPUAccessMode;

/// Kind of component access requested or held during ECS execution.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    /// Shared, read-only access to a component.
    Read,

    /// Exclusive, mutable access to a component.
    Write,
}

/// Reason why a query's declared component access was invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidAccessReason {
    /// The same component was declared as both read and write.
    ReadAndWrite,

    /// The same component appeared multiple times in access set.
    DuplicateAccess,

    /// A component was declared writable while also excluded (`without`).
    WriteAndWithout,
}

/// Errors that occur during ECS execution and iteration.
///
/// These errors represent **violations of ECS runtime safety rules**
/// detected dynamically by the engine.
///
/// ## Characteristics
/// * Caused by incorrect API usage
/// * Always deterministic
/// * Never indicate partial ECS mutation

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionError {

    /// Attempted to mutate ECS structure while iteration was active.
    ///
    /// This includes:
    /// * spawning or despawning entities
    /// * adding or removing components
    /// * calling `with_exclusive` during iteration

    StructuralMutationDuringIteration,

    /// Component borrow rules were violated at runtime.
    ///
    /// This occurs when:
    /// * two systems attempt conflicting access in parallel
    /// * a query declares incompatible read/write sets
    /// * the borrow tracker's spin limit was exceeded, indicating
    ///   a probable scheduling bug

    BorrowConflict {
        /// Component whose borrow was violated.
        component_id: ComponentID,

        /// Existing access mode already held.
        held: AccessKind,

        /// Access mode that was requested.
        requested: AccessKind,
    },

    /// A query declared invalid or contradictory component access.
    ///
    /// Examples:
    /// * component appears in both read and write sets
    /// * duplicate component entries
    /// * write access combined with `without`

    InvalidQueryAccess {
        /// The component whose access declaration was invalid.
        component_id: ComponentID,

        /// The specific reason the access was rejected.
        reason: InvalidAccessReason,
    },

    /// A query attempted to access a component not present
    /// in a matched archetype.
    ///
    /// This indicates a bug in query construction or execution.

    MissingComponent {
        /// The missing component identifier.
        component_id: ComponentID,
    },

    /// Execution was aborted because the scheduler violated
    /// its declared access guarantees.

    SchedulerInvariantViolation,

    /// A synchronization primitive was poisoned (panic while held).
    LockPoisoned {
        /// message for the lock poisoned error
        what: &'static str,
    },

    /// GPU execution requested but crate was built without `--features gpu`.
    GpuNotEnabled,

    /// Component used by GPU system is not registered as GPU-safe.
    #[cfg(feature = "gpu")]
    GpuUnsupportedComponent {
        /// component ID of the unsupported component.
        component_id: ComponentID,
        /// text name of the unsupported component.
        name: &'static str,
    },

    /// GPU init failed.
    #[cfg(feature = "gpu")]
    GpuInitFailed {
        /// initialization failure reason.
        message: std::borrow::Cow<'static, str>,
    },

    /// GPU dispatch failed.
    #[cfg(feature = "gpu")]
    GpuDispatchFailed {
        /// dispatch failure reason.
        message: std::borrow::Cow<'static, str>,
    },

    /// A required GPU buffer was missing during dispatch.
    #[cfg(feature = "gpu")]
    GpuMissingBuffer {
        /// Archetype for which the GPU buffer was requested.
        archetype_id: ArchetypeID,

        /// Component whose GPU buffer was missing.
        component_id: ComponentID,

        /// Intended GPU access mode for the missing buffer.
        access: GPUAccessMode,
    },

    /// Unsafe execution path was invoked incorrectly.
    InternalExecutionError,
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutionError::StructuralMutationDuringIteration => {
                f.write_str("structural mutation attempted during ECS iteration")
            }

            ExecutionError::BorrowConflict { component_id, held, requested } => {
                write!(
                    f,
                    "borrow conflict on component {}: {:?} already held, {:?} requested",
                    component_id, held, requested
                )
            }

            ExecutionError::InvalidQueryAccess { component_id, reason } => {
                write!(
                    f,
                    "invalid query access for component {}: {:?}",
                    component_id, reason
                )
            }

            ExecutionError::MissingComponent { component_id } => {
                write!(f, "query attempted to access missing component {}", component_id)
            }

            ExecutionError::SchedulerInvariantViolation => {
                f.write_str("scheduler violated declared access invariants")
            }

            ExecutionError::LockPoisoned { what } => write!(f, "lock poisoned: {}", what),

            ExecutionError::GpuNotEnabled => {
                f.write_str("GPU execution requested but the `gpu` feature has not been enabled")
            }
            #[cfg(feature = "gpu")]
            ExecutionError::GpuUnsupportedComponent { component_id, name } => {
                write!(f, "component {} ({}) is not GPU-safe (register_gpu_component required)", component_id, name)
            }
            #[cfg(feature = "gpu")]
            ExecutionError::GpuInitFailed { message } => write!(f, "GPU initialization failed: {}", message),
            #[cfg(feature = "gpu")]
            ExecutionError::GpuDispatchFailed { message } => write!(f, "GPU dispatch failed: {}", message),
            #[cfg(feature = "gpu")]
            ExecutionError::GpuMissingBuffer {
                archetype_id,
                component_id,
                access,
            } => write!(
                f,
                "GPU dispatch failed: missing {:?} buffer for component {} in archetype {}",
                access, component_id, archetype_id
            ),

            ExecutionError::InternalExecutionError => f.write_str("internal ECS execution error"),
        }
    }
}

impl std::error::Error for ExecutionError {}
