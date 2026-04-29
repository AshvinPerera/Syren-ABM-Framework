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
//! | [`DuplicateSystemId`] | Scheduler was given multiple systems with the same ID |
//! | [`UnknownSystemId`] | Scheduler ordering referenced an unregistered system ID |
//! | [`SelfSystemOrdering`] | Scheduler ordering made a system depend on itself |
//! | [`SchedulerInvariantViolation`] | Scheduler broke its own declared access guarantees |
//! | [`SchedulerCycle`] | Channel or explicit-ordering edges form a dependency cycle |
//! | [`LockPoisoned`] | A thread panicked while holding a synchronisation primitive |
//! | `Gpu*` *(feature = "gpu")* | GPU unavailable, component not GPU-safe, or dispatch failure |
//! | [`InternalExecutionError`] | Unsafe execution path invoked incorrectly |
//!
//! [`StructuralMutationDuringIteration`]: ExecutionError::StructuralMutationDuringIteration
//! [`BorrowConflict`]: ExecutionError::BorrowConflict
//! [`InvalidQueryAccess`]: ExecutionError::InvalidQueryAccess
//! [`MissingComponent`]: ExecutionError::MissingComponent
//! [`DuplicateSystemId`]: ExecutionError::DuplicateSystemId
//! [`UnknownSystemId`]: ExecutionError::UnknownSystemId
//! [`SelfSystemOrdering`]: ExecutionError::SelfSystemOrdering
//! [`SchedulerInvariantViolation`]: ExecutionError::SchedulerInvariantViolation
//! [`SchedulerCycle`]: ExecutionError::SchedulerCycle
//! [`LockPoisoned`]: ExecutionError::LockPoisoned
//! [`InternalExecutionError`]: ExecutionError::InternalExecutionError

use std::fmt;

use crate::engine::types::ComponentID;

#[cfg(feature = "gpu")]
use crate::engine::types::ArchetypeID;

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
    /// A component was declared readable while also excluded (`without`).
    ReadAndWithout,
}

/// Reason a boundary-resource access failed.
///
/// Carried by [`ExecutionError::BoundaryAccessFailed`] to distinguish the
/// two user-attributable failure modes of
/// [`ECSReference::boundary`](crate::engine::manager::ECSReference::boundary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryAccessFailure {
    /// The supplied `BoundaryID` is outside the registered range.
    OutOfRange,
    /// The stored resource's concrete type does not match the requested `R`.
    TypeMismatch,
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
    /// This includes spawning, despawning, adding/removing components,
    /// or calling `with_exclusive` during iteration.
    StructuralMutationDuringIteration,

    /// Component borrow rules were violated at runtime.
    BorrowConflict {
        /// Component whose borrow was violated.
        component_id: ComponentID,
        /// Existing access mode already held.
        held: AccessKind,
        /// Access mode that was requested.
        requested: AccessKind,
    },

    /// A query declared invalid or contradictory component access.
    InvalidQueryAccess {
        /// The component whose access declaration was invalid.
        component_id: ComponentID,
        /// The specific reason the access was rejected.
        reason: InvalidAccessReason,
    },

    /// A system declared the same channel in both its `produces` and
    /// `consumes` sets.
    ///
    /// This is structurally unsound: the stage packer would place the system
    /// in a stage where it both produces and consumes the channel in one
    /// pass, so the `consumes` observation would read un-finalised channel
    /// data. Always a bug in the system's `AccessSets` declaration.
    ///
    /// Surfaced by [`AccessSets::validate`](crate::engine::systems::AccessSets::validate)
    /// at system-registration time.
    SelfChannelAlias {
        /// The channel ID that appeared in both `produces` and `consumes`.
        channel_id: crate::engine::types::ChannelID,
    },

    /// An attempt to access a boundary resource via
    /// [`ECSReference::boundary`](crate::engine::manager::ECSReference::boundary)
    /// failed because the [`BoundaryID`](crate::engine::types::BoundaryID)
    /// was out of range or the stored resource's concrete type did not
    /// match the requested type.
    ///
    /// Distinct from [`ExecutionError::InternalExecutionError`] because boundary-access
    /// failures are caller-attributable: passing the wrong `BoundaryID` or
    /// the wrong type parameter `R` is a user-level bug, not an engine
    /// invariant violation.
    BoundaryAccessFailed {
        /// Which form the access failure took.
        reason: BoundaryAccessFailure,
        /// The `BoundaryID` that was passed to `ECSReference::boundary`.
        id: crate::engine::types::BoundaryID,
    },

    /// Two boundary resources both declared ownership of the same
    /// [`ChannelID`](crate::engine::types::ChannelID).
    ///
    /// Channel IDs must be unique across all boundary resources so the
    /// scheduler can route `finalise` calls deterministically. This error is
    /// raised by
    /// [`ECSManager::register_boundary`](crate::engine::manager::ECSManager::register_boundary)
    /// when the resource being registered claims a channel that is already
    /// owned by an earlier-registered resource.
    DuplicateChannelRegistration {
        /// The channel ID that appeared in two resources.
        channel_id: crate::engine::types::ChannelID,
        /// The `BoundaryID` of the resource that already owned this channel.
        existing_boundary: crate::engine::types::BoundaryID,
    },

    /// Channel ID allocation exhausted the `u32` identifier space.
    ChannelIdOverflow,

    /// A query attempted to access a component not present in a matched archetype.
    MissingComponent {
        /// The missing component identifier.
        component_id: ComponentID,
    },

    /// A typed query helper was invoked with a Rust type that does not match
    /// the component type bound into the built query.
    QueryTypeMismatch {
        /// Name of the helper that detected the mismatch.
        method: &'static str,
        /// Whether the mismatched column was read-only or writable.
        access: AccessKind,
        /// Column index within the query declaration order.
        index: usize,
        /// Runtime component identifier for the mismatched column.
        component_id: ComponentID,
        /// Type name recorded when the query was built.
        expected: &'static str,
        /// Type name requested by the typed helper.
        actual: &'static str,
    },

    /// The scheduler was given more than one system with the same ID.
    ///
    /// System IDs are used as stable ordering keys in the execution graph, so
    /// duplicates would make deterministic planning ambiguous.
    DuplicateSystemId {
        /// The duplicated system identifier.
        system_id: crate::engine::types::SystemID,
    },

    /// An explicit scheduler ordering referenced a system ID that has not
    /// been registered with the scheduler.
    UnknownSystemId {
        /// The missing system identifier.
        system_id: crate::engine::types::SystemID,
    },

    /// An explicit scheduler ordering made a system depend on itself.
    SelfSystemOrdering {
        /// The self-dependent system identifier.
        system_id: crate::engine::types::SystemID,
    },

    /// Execution was aborted because the scheduler violated its declared access guarantees.
    SchedulerInvariantViolation,

    /// The scheduler detected a dependency cycle in the combined
    /// (component-conflict + channel-ordering + explicit-ordering) partial order.
    ///
    /// Common cycle shape: system A produces channel X and consumes channel Y;
    /// system B produces channel Y and consumes channel X. Break the cycle by
    /// removing one dependency, or restructure communication through ECS
    /// components / shared `Arc<Environment>` instead.
    SchedulerCycle,

    /// A synchronization primitive was poisoned (panic while held).
    LockPoisoned {
        /// Description of the poisoned lock.
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
            ExecutionError::StructuralMutationDuringIteration =>
                f.write_str("structural mutation attempted during ECS iteration"),

            ExecutionError::BorrowConflict { component_id, held, requested } =>
                write!(f,
                       "borrow conflict on component {}: {:?} already held, {:?} requested",
                       component_id, held, requested),

            ExecutionError::InvalidQueryAccess { component_id, reason } =>
                write!(f, "invalid query access for component {}: {:?}",
                       component_id, reason),

            ExecutionError::SelfChannelAlias { channel_id } => {
                write!(
                    f,
                    "system access sets alias channel {}: it appears in both produces and consumes",
                    channel_id
                )
            }

            ExecutionError::BoundaryAccessFailed { reason, id } => {
                match reason {
                    BoundaryAccessFailure::OutOfRange => write!(
                        f,
                        "boundary access failed: BoundaryID {} is out of range",
                        id
                    ),
                    BoundaryAccessFailure::TypeMismatch => write!(
                        f,
                        "boundary access failed: stored resource at BoundaryID {} has a different concrete type than requested",
                        id
                    ),
                }
            }

            ExecutionError::DuplicateChannelRegistration { channel_id, existing_boundary } => {
                write!(
                    f,
                    "boundary registration failed: channel {} is already owned by BoundaryID {}",
                    channel_id, existing_boundary
                )
            }

            ExecutionError::ChannelIdOverflow =>
                f.write_str("channel ID allocation overflowed"),

            ExecutionError::MissingComponent { component_id } =>
                write!(f, "query attempted to access missing component {}", component_id),

            ExecutionError::QueryTypeMismatch {
                method,
                access,
                index,
                component_id,
                expected,
                actual,
            } => write!(
                f,
                "{method}: {:?} column {index} for component {component_id} was built for {expected}, got {actual}",
                access
            ),

            ExecutionError::DuplicateSystemId { system_id } =>
                write!(f, "scheduler registration failed: duplicate system id {}", system_id),

            ExecutionError::UnknownSystemId { system_id } =>
                write!(f, "scheduler ordering references unknown system id {}", system_id),

            ExecutionError::SelfSystemOrdering { system_id } =>
                write!(f, "scheduler ordering cannot make system {} depend on itself", system_id),

            ExecutionError::SchedulerInvariantViolation =>
                f.write_str("scheduler violated declared access invariants"),

            ExecutionError::SchedulerCycle =>
                f.write_str(
                    "scheduler detected a dependency cycle in the system graph; \
                     check AccessSets::produces/consumes and explicit ordering edges"
                ),

            ExecutionError::LockPoisoned { what } =>
                write!(f, "lock poisoned: {}", what),

            ExecutionError::GpuNotEnabled =>
                f.write_str("GPU execution requested but the `gpu` feature has not been enabled"),

            #[cfg(feature = "gpu")]
            ExecutionError::GpuUnsupportedComponent { component_id, name } =>
                write!(f,
                       "component {} ({}) is not GPU-safe (register_gpu_component required)",
                       component_id, name),

            #[cfg(feature = "gpu")]
            ExecutionError::GpuInitFailed { message } =>
                write!(f, "GPU initialization failed: {}", message),

            #[cfg(feature = "gpu")]
            ExecutionError::GpuDispatchFailed { message } =>
                write!(f, "GPU dispatch failed: {}", message),

            #[cfg(feature = "gpu")]
            ExecutionError::GpuMissingBuffer { archetype_id, component_id, access } =>
                write!(f,
                       "GPU dispatch failed: missing {:?} buffer for component {} in archetype {}",
                       access, component_id, archetype_id),

            ExecutionError::InternalExecutionError =>
                f.write_str("internal ECS execution error"),
        }
    }
}

impl std::error::Error for ExecutionError {}
