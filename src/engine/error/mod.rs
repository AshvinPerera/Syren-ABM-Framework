//! Error types for ECS storage, spawning, migration, and execution.
//!
//! This module defines **focused, composable error types** used across the
//! entity–component system, covering both **structural failures** (such as
//! storage bounds or capacity limits) and **runtime execution violations**
//! (such as borrow conflicts or invalid query access).
//!
//! The errors in this module are designed to be:
//! * **Deterministic** - the same misuse always yields the same error,
//! * **Actionable** - errors carry structured context for diagnostics,
//! * **Composable** - low-level errors can be promoted into higher-level ones
//!   without losing information.
//!
//! ## Error Categories
//!
//! The module is broadly divided into three conceptual layers:
//!
//! ### 1. Storage and Structural Errors
//!
//! These errors arise during entity creation, component storage, or archetype
//! migration, and typically indicate violated structural assumptions.
//!
//! Examples include:
//! * insufficient entity capacity ([`CapacityError`]),
//! * invalid shard or row indices ([`ShardBoundsError`], [`PositionOutOfBoundsError`]),
//! * component type mismatches ([`TypeMismatchError`]),
//! * inconsistent archetype layouts ([`MoveError`]).
//!
//! These errors are usually surfaced during spawning or component mutation and
//! are aggregated into higher-level errors such as [`SpawnError`].
//!
//! ### 2. Runtime Execution Errors
//!
//! These errors occur during ECS **iteration and system execution**, when
//! runtime safety rules are violated.
//!
//! Examples include:
//! * attempting structural mutation during iteration,
//! * conflicting component borrows across parallel systems,
//! * invalid or contradictory query access declarations,
//! * scheduler invariant violations.
//!
//! Such failures are reported via [`ExecutionError`] and represent **incorrect
//! API usage or scheduler misconfiguration**, rather than internal corruption.
//!
//! ### 3. Invariant Violations
//!
//! Certain errors indicate serious internal inconsistencies (e.g. scheduler
//! invariants being violated, poisoned locks, or metadata drift). These are
//! reported explicitly via [`InternalViolation`] and are considered
//! framework-level bugs rather than recoverable user errors.
//!
//! ## Typical Error Flow
//!
//! Lower-level operations return precise, domain-specific errors (such as
//! [`AttributeError`] or [`MoveError`]). Higher-level orchestration code uses
//! `From<T>` conversions and the `?` operator to promote these into aggregate
//! error types like [`SpawnError`] or [`ExecutionError`].
//!
//! ## Examples
//!
//! Converting a low-level storage failure into a spawn-level error:
//! ```ignore
//! fn spawn_one(world: &mut World, components: Components) -> Result<Entity, SpawnError> {
//!     world.storage.push_components(components)?; // AttributeError → SpawnError
//!     Ok(world.entities.alloc())
//! }
//! ```
//!
//! Handling execution-time safety violations:
//! ```ignore
//! match world.run_systems() {
//!     Ok(()) => {}
//!     Err(ExecutionError::BorrowConflict { component_id, .. }) => {
//!         eprintln!("borrow conflict on component {}", component_id);
//!     }
//!     Err(e) => {
//!         eprintln!("execution failed: {e}");
//!     }
//! }
//! ```
//!
//! ## Display vs. Debug
//!
//! * [`fmt::Display`] implementations are concise and suitable for logs or
//!   user-facing diagnostics.
//! * [`fmt::Debug`] retains full structural detail for debugging and telemetry.

mod primitives;
mod attribute;
mod registry;
mod spawn;
mod move_error;
mod execution;
mod internal;

pub use primitives::{
    CapacityError,
    ShardBoundsError,
    StaleEntityError,
    PositionOutOfBoundsError,
    TypeMismatchError,
};

pub use attribute::{AttributeError, AttributeInvariantViolation};

pub use registry::{RegistryError, RegistryResult};

pub use spawn::SpawnError;

pub use move_error::MoveError;

pub use execution::{AccessKind, InvalidAccessReason, ExecutionError};

pub use internal::InternalViolation;

/// Unified error type for the public ECS API.
#[non_exhaustive]
#[derive(Debug)]
pub enum ECSError {
    /// Entity spawning error
    Spawn(SpawnError),
    /// ECS execution error
    Execute(ExecutionError),
    /// Component move error
    Move(MoveError),

    /// Component registry and component factory errors
    Registry(RegistryError),

    /// Low-level component storage access error
    Attribute(AttributeError),

    /// Internal framework invariant violation.
    ///
    /// These indicate bugs in the ECS engine itself, not user-recoverable
    /// conditions. Each variant of [`InternalViolation`] maps to a specific
    /// invariant that was broken.
    Internal(InternalViolation),

    /// Environment parameter store error.
    ///
    /// Wraps [`EnvironmentError`](crate::environment::error::EnvironmentError)
    /// so that environment failures propagate with full diagnostic context
    /// through scheduler and system boundaries.
    #[cfg(feature = "environment")]
    Environment(crate::environment::error::EnvironmentError),
}

impl std::fmt::Display for ECSError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ECSError::Spawn(e) => write!(f, "spawn error: {e}"),
            ECSError::Execute(e) => write!(f, "execution error: {e}"),
            ECSError::Move(e) => write!(f, "move error: {e}"),
            ECSError::Registry(e) => write!(f, "registry error: {e}"),
            ECSError::Attribute(e) => write!(f, "attribute error: {e}"),
            ECSError::Internal(v) => write!(f, "internal error: {v}"),
            #[cfg(feature = "environment")]
            ECSError::Environment(e) => write!(f, "environment error: {e}"),
        }
    }
}

impl std::error::Error for ECSError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ECSError::Spawn(e) => Some(e),
            ECSError::Execute(e) => Some(e),
            ECSError::Move(e) => Some(e),
            ECSError::Registry(e) => Some(e),
            ECSError::Attribute(e) => Some(e),
            ECSError::Internal(v) => Some(v),
            #[cfg(feature = "environment")]
            ECSError::Environment(e) => Some(e),
        }
    }
}

impl From<SpawnError> for ECSError {
    fn from(e: SpawnError) -> Self { ECSError::Spawn(e) }
}
impl From<ExecutionError> for ECSError {
    fn from(e: ExecutionError) -> Self { ECSError::Execute(e) }
}
impl From<MoveError> for ECSError {
    fn from(e: MoveError) -> Self { ECSError::Move(e) }
}
impl From<RegistryError> for ECSError {
    fn from(e: RegistryError) -> Self { ECSError::Registry(e) }
}
impl From<AttributeError> for ECSError {
    fn from(e: AttributeError) -> Self { ECSError::Attribute(e) }
}
impl From<InternalViolation> for ECSError {
    fn from(v: InternalViolation) -> Self { ECSError::Internal(v) }
}
#[cfg(feature = "environment")]
impl From<crate::environment::error::EnvironmentError> for ECSError {
    fn from(e: crate::environment::error::EnvironmentError) -> Self {
        ECSError::Environment(e)
    }
}

/// Result type used by the ECS engine.
pub type ECSResult<T> = Result<T, ECSError>;
