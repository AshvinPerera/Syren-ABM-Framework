//! Error types for ECS storage, spawning, migration, and execution.
//!
//! This module defines **focused, composable error types** used across the
//! entity–component system, covering both **structural failures** (such as
//! storage bounds or capacity limits) and **runtime execution violations**
//! (such as borrow conflicts or invalid query access).
//!
//! The errors in this module are designed to be:
//! * **Deterministic** — the same misuse always yields the same error,
//! * **Actionable** — errors carry structured context for diagnostics,
//! * **Composable** — low-level errors can be promoted into higher-level ones
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
//! invariants being violated). These are still reported explicitly but are
//! considered framework-level bugs rather than recoverable user errors.
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


use std::fmt;
use std::any::TypeId;

use crate::engine::types::{ShardID, ChunkID, RowID, ComponentID};


/// Returned when the system cannot satisfy a request to create or place
/// additional entities because the target container has insufficient capacity.
///
/// This typically arises during batch spawns or when attempting to grow a shard
/// beyond its configured limit.
///
/// ### Fields
/// * `entities_needed` — Total number of entities the operation attempted to
///   create or accommodate.
/// * `capacity` — The current upper bound that prevented the operation.
///
/// ### Example
/// ```ignore
/// if requested > shard.capacity() {
///     return Err(CapacityError { entities_needed: requested as u64, capacity: shard.capacity() as u64 }.into());
/// }
/// ```

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityError {
    
    /// Total entities the operation attempted to allocate.
    pub entities_needed: u64,   

    /// Current capacity limiting the operation.
    pub capacity: u64,          
}

impl fmt::Display for CapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "entity limit reached ({} needed; capacity {})",
            self.entities_needed, self.capacity
        )
    }
}

impl std::error::Error for CapacityError {}

/// Returned when a shard index is outside the valid range for the target shard
/// set or collection.
///
/// ### Fields
/// * `index` — The shard index that was requested.
/// * `max_index` — The maximum valid shard index (inclusive).
///
/// ### Example
/// ```ignore
/// let max = shards.len().saturating_sub(1) as u32;
/// if idx > max {
///     return Err(ShardBoundsError { index: idx, max_index: max }.into());
/// }
/// ```

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardBoundsError {
    
    /// Offending shard index that was requested.
    pub index: ShardID,  

    /// Maximum valid shard index (inclusive) for the collection.       
    pub max_index: u32,        
}

impl fmt::Display for ShardBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "shard index {} out of bounds (max index {})",
            self.index, self.max_index
        )
    }
}

impl std::error::Error for ShardBoundsError {}

/// Returned when an `Entity` handle is no longer valid—typically because it
/// was despawned or its generation/version no longer matches live storage.
///
/// Use this to prevent use-after-free style logic errors at the API boundary.
///
/// ### Example
/// ```ignore
/// if !entities.is_live(entity) {
///     return Err(StaleEntityError.into());
/// }
/// ```

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaleEntityError;

impl fmt::Display for StaleEntityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("stale or dead entity reference")
    }
}

impl std::error::Error for StaleEntityError {}

/// Returned when an operation expects an archetype to contain at least one
/// component, but the archetype is empty.
///
/// ## Context
/// This typically indicates a logic error during entity construction or
/// component migration, where an archetype signature was assumed to be
/// non-empty but was not.
///
/// ## Notes
/// This error represents a structural invariant violation rather than a
/// recoverable runtime condition.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyArchetypeError;

impl fmt::Display for EmptyArchetypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("archetype contains no components")
    }
}

impl std::error::Error for EmptyArchetypeError {}

/// Returned when a `(ChunkID, RowID)` pair refers to a position outside
/// valid component storage bounds.
///
/// ## Context
/// Used by attribute and archetype storage to report invalid addressing,
/// typically caused by stale metadata or incorrect index calculations.
///
/// ## Invariants
/// - `chunk < chunks`
/// - `row < capacity` for all but the last chunk

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionOutOfBoundsError {
    
    /// Chunk index that was addressed.
    pub chunk: ChunkID,

    /// Row index that was addressed.
    pub row: RowID,

    /// Total number of chunks in the storage.
    pub chunks: usize,

    /// Maximum row capacity per chunk.
    pub capacity: usize,

    /// Number of valid rows in the final chunk.
    pub last_chunk_length: usize,
}

impl fmt::Display for PositionOutOfBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "position out of bounds: chunk {} (of {}), row {} (capacity per chunk {}, last chunk length {})",
            self.chunk, self.chunks, self.row, self.capacity, self.last_chunk_length
        )
    }
}

impl std::error::Error for PositionOutOfBoundsError {}

/// Returned when an attribute/component write targets a storage slot whose
/// element type does not match the provided value's type.
///
/// This is a logic/configuration error surfaced by storage when component
/// type IDs diverge (e.g. writing `Velocity` into a `Position` column).
///
/// ### Fields
/// * `expected` — The [`TypeId`] that the destination storage declares.
/// * `actual` — The [`TypeId`] of the value provided by the caller.
///
/// ### Example
/// ```ignore
/// if actual_type != expected_type {
///     return Err(TypeMismatchError { expected: expected_type, actual: actual_type }.into());
/// }
/// ```

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeMismatchError {
    
    /// Destination storage's declared element type.
    pub expected: TypeId,
    
    /// Provided value's dynamic type.           
    pub actual: TypeId,             
}

impl fmt::Display for TypeMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type mismatch: expected {:?}, actual {:?}", self.expected, self.actual)
    }
}

impl std::error::Error for TypeMismatchError {}

/// Aggregate error for attribute (component column) operations.
///
/// This wraps precise, low-level failures that can occur when pushing or
/// writing component data into storage. Typical sources include:
///
/// * index math issues while constructing column positions (e.g. index overflow),
/// * position bounds problems when addressing `(ChunkID, RowID)` pairs,
/// * type mismatches between a column’s declared element type and a provided value.
///
/// Conversions (`From<T>`) are implemented for common low-level errors so callers
/// can write `?` and still return a single, expressive type.
///
/// ### Display
/// `Display` messages are concise and suitable for logs. For deep inspection,
/// prefer `Debug` which includes full structure.
///
/// ### Example
/// ```ignore
/// fn push_component<T: 'static>(col: &mut Column, value: T) -> Result<(), AttributeError> {
///     // may become AttributeError::TypeMismatch via From<TypeMismatchError>
///     check_type::<T>(col)?;
///
///     // may become AttributeError::IndexOverflow("row") etc.
///     let pos = compute_position(col.len())?;
///
///     // may become AttributeError via From<PositionOutOfBoundsError>
///     col.write(pos, value)?;
///     Ok(())
/// }
/// ```

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeError {
    
    /// A `(ChunkID, RowID)` addressed storage outside valid bounds.
    Position(PositionOutOfBoundsError),
    
    /// The dynamic type of a value did not match the component storage type.
    TypeMismatch(TypeMismatchError),

    /// Index arithmetic overflow occurred while constructing a storage index.
    ///
    /// The string identifies which index overflowed (e.g. `"row"` or `"chunk"`).
    IndexOverflow(&'static str),
}

impl fmt::Display for AttributeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttributeError::Position(e) => write!(f, "{e}"),
            AttributeError::TypeMismatch(e) => write!(f, "{e}"),
            AttributeError::IndexOverflow(which) => write!(f, "index overflow constructing {}", which),
        }
    }
}

impl std::error::Error for AttributeError {}

impl From<PositionOutOfBoundsError> for AttributeError {
    fn from(e: PositionOutOfBoundsError) -> Self { AttributeError::Position(e) }
}

impl From<TypeMismatchError> for AttributeError {
    fn from(e: TypeMismatchError) -> Self { AttributeError::TypeMismatch(e) }
}

/// High-level error for entity spawning.
///
/// This aggregates the most common failure modes encountered while creating
/// entities and attaching their components. It intentionally preserves the
/// underlying structured error to keep diagnostics actionable.
///
/// ### Variants (typical)
/// * `Capacity(CapacityError)` — Not enough room to allocate the requested
///   number of entities.
/// * `ShardBounds(ShardBoundsError)` — Target shard index was invalid.
/// * `StaleEntity(StaleEntityError)` — A supplied entity reference was not live.
/// * `StoragePushFailedWith(AttributeError)` — Component push/write failed.
///
/// ### Usage
/// `From<T>` conversions allow `?` from low-level operations:
/// ```ignore
/// fn spawn_batch(world: &mut World, batch: Batch) -> Result<Vec<Entity>, SpawnError> {
///     ensure_capacity(world, batch.len())?;     // -> Capacity -> SpawnError
///     let shard = world.shards.get_mut(batch.shard_index)?; // -> ShardBounds -> SpawnError
///     for comps in batch.components {
///         shard.storage.push_components(comps)?; // -> AttributeError -> SpawnError
///     }
///     Ok(/* entities */)
/// }
/// ```
///
/// ### Display
/// Human-readable, single-line messages suitable for logs.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpawnError {

    /// Entity creation failed due to insufficient capacity.
    Capacity(CapacityError),

    /// A shard index was outside the valid range.
    ShardBounds(ShardBoundsError),
    
    /// A shard-level operation failed unexpectedly.
    ShardError,
    
    /// An entity handle was stale or referred to a despawned entity.
    StaleEntity,
    
    /// An operation required a non-empty archetype.
    EmptyArchetype,
    
    /// Attempted to remove a component from an archetype that still
    /// contained entities.
    ArchetypeNotEmpty,
    
    /// Failed while swap-removing a component from storage.
    StorageSwapRemoveFailed(AttributeError),
    
    /// Failed while pushing component data into storage.
    StoragePushFailedWith(AttributeError),
    
    /// A required component was missing during entity construction.
    MissingComponent { 
        /// Runtime type identifier of the missing component.
        type_id: TypeId, 
        
        /// Human-readable component name.
        name: &'static str 
    },
    
    /// Component storages disagreed on the row position of an entity.
    ///
    /// This indicates a serious internal invariant violation.
    MisalignedStorage { 
        /// Expected `(chunk, row)` position.
        expected: (ChunkID, RowID), 

        /// Actual `(chunk, row)` encountered.
        got: (ChunkID, RowID) 
    },
    
    /// An invalid or unregistered component ID was encountered.
    InvalidComponentId,
}

impl fmt::Display for SpawnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpawnError::Capacity(e) => write!(f, "{e}"),
            SpawnError::ShardBounds(e) => write!(f, "{e}"),
            SpawnError::ShardError => write!(f, "shard operation failed"),
            SpawnError::StaleEntity => write!(f, "stale or dead entity reference"),
            SpawnError::EmptyArchetype => write!(f, "archetype contains no components"),
            SpawnError::ArchetypeNotEmpty => write!(f, "cannot remove component from non-empty archetype"), 
            SpawnError::StorageSwapRemoveFailed(e) => write!(f, "failed to swap-remove from storage: {e}"),
            SpawnError::StoragePushFailedWith(e) => write!(f, "failed to push into storage: {e}"),
            SpawnError::MissingComponent { name, .. } => write!(f, "missing component: {}", name),
            SpawnError::MisalignedStorage { expected, got } => write!(
                f,
                "component storages became misaligned; expected position {:?}, got {:?}",
                expected, got
            ),
            SpawnError::InvalidComponentId => write!(f, "invalid component id"),
        }
    }
}

impl std::error::Error for SpawnError {}

impl From<CapacityError> for SpawnError {
    fn from(e: CapacityError) -> Self { SpawnError::Capacity(e) }
}
impl From<ShardBoundsError> for SpawnError {
    fn from(e: ShardBoundsError) -> Self { SpawnError::ShardBounds(e) }
}
impl From<AttributeError> for SpawnError {
    fn from(e: AttributeError) -> Self { SpawnError::StoragePushFailedWith(e) }
}

/// Errors that can occur while moving an entity between archetypes.
///
/// ## Context
/// `MoveError` is used by archetype migration logic when transferring
/// component rows between archetypes during add/remove operations.
///
/// ## Notes
/// These errors generally indicate internal inconsistencies or violated
/// invariants rather than recoverable user-facing failures.

#[derive(Debug)]
pub enum MoveError {

    /// Component storage layouts were inconsistent between archetypes.
    InconsistentStorage,
    
    /// Failed to move component data from the source archetype.
    PushFromFailed { 
        /// Component being transferred.
        component_id: ComponentID, 
        
        /// Underlying attribute error.
        source_error: AttributeError 
    },
    
    /// Component columns disagreed on the destination row.
    RowMisalignment { 
        /// Expected `(chunk, row)` position.
        expected: (ChunkID, RowID), 
        
        /// Actual `(chunk, row)` encountered.
        got: (ChunkID, RowID), 

        /// Component whose storage was misaligned.
        component_id: ComponentID 
    },
    
    /// No components were transferred during the move.
    NoComponentsMoved,
    
    /// Failed while inserting component data into the destination archetype.    
    PushFailed { 
        /// Component being inserted.
        component_id: ComponentID,

        /// Underlying attribute error. 
        source_error: AttributeError 
    },
    
    /// Failed while removing component data from the source archetype.
    SwapRemoveError { 
        /// Component being removed.
        component_id: ComponentID, 

        /// Underlying attribute error.
        source_error: AttributeError 
    },
    
    /// Swap-remove operations yielded inconsistent metadata.
    InconsistentSwapInfo,
    
    /// Entity metadata could not be updated consistently after the move.
    MetadataFailure,

    /// Rollback failed after a partial archetype migration.
    /// This indicates a catastrophic internal error.
    RollbackFailed,
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MoveError::InconsistentStorage => {
                f.write_str("component storage layouts are inconsistent between archetypes")
            }

            MoveError::PushFromFailed { component_id, source_error } => {
                write!(
                    f,
                    "failed to move component {} from source archetype: {}",
                    component_id, source_error
                )
            }

            MoveError::RowMisalignment { expected, got, component_id } => {
                write!(
                    f,
                    "component {} storage misaligned: expected position {:?}, got {:?}",
                    component_id, expected, got
                )
            }

            MoveError::NoComponentsMoved => {
                f.write_str("no components were moved during archetype transition")
            }

            MoveError::PushFailed { component_id, source_error } => {
                write!(
                    f,
                    "failed to insert component {} into destination archetype: {}",
                    component_id, source_error
                )
            }

            MoveError::SwapRemoveError { component_id, source_error } => {
                write!(
                    f,
                    "failed to remove component {} from source archetype: {}",
                    component_id, source_error
                )
            }

            MoveError::InconsistentSwapInfo => {
                f.write_str("swap-remove produced inconsistent metadata")
            }

            MoveError::MetadataFailure => {
                f.write_str("failed to update entity metadata after archetype move")
            }

            MoveError::RollbackFailed => {
                f.write_str("rollback failed after partial archetype migration")
            }
        }
    }
}

/// Kind of component access requested or held during ECS execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    /// Shared, read-only access to a component.
    Read,

    /// Exclusive, mutable access to a component.
    Write,
}

/// Reason why a query’s declared component access was invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidAccessReason {
    /// The same component was declared as both read and write.
    ReadAndWrite,

    /// The same component appeared multiple times in an access set.
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

            ExecutionError::InternalExecutionError => {
                f.write_str("internal ECS execution error")
            }
        }
    }
}

impl std::error::Error for ExecutionError {}
