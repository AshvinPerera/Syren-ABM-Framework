//! Error types for entity spawning and structural world mutations.
//!
//! The central type is [`SpawnError`], a non-exhaustive enum that aggregates
//! the failure modes encountered when creating entities, attaching components,
//! and mutating archetype structure. It is designed to be used with `?` via
//! its `From` implementations for lower-level errors such as [`CapacityError`],
//! [`ShardBoundsError`], [`AttributeError`], [`RegistryError`], and
//! [`StaleEntityError`].
//!
//! # Error hierarchy
//!
//! ```text
//! SpawnError
//! ├── Capacity(CapacityError)                — storage full
//! ├── ShardBounds(ShardBoundsError)          — shard index out of range
//! ├── Registry(RegistryError)               — component registration failure
//! ├── StaleEntity(StaleEntityError)         — entity handle no longer valid
//! ├── StoragePushFailedWith(AttributeError)
//! └── StorageSwapRemoveFailed(AttributeError)
//! ```
//!
//! Variants without a source error (e.g. [`SpawnError::MisalignedStorage`])
//! represent invariant violations or programmer errors that do not wrap a
//! lower-level cause.
//!
//! # Usage
//!
//! Import [`SpawnError`] wherever entity construction or archetype mutation can
//! fail. The `From` conversions let all low-level errors propagate with `?`
//! without manual wrapping:
//!
//! ```ignore
//! use crate::errors::spawn::SpawnError;
//!
//! fn spawn(world: &mut World) -> Result<Entity, SpawnError> {
//!     world.ensure_capacity(1)?;          // CapacityError      -> SpawnError
//!     let shard = world.shards.get_mut(0)?; // ShardBoundsError -> SpawnError
//!     shard.push_components(bundle)?;     // AttributeError     -> SpawnError
//!     Ok(entity)
//! }
//! ```

use std::fmt;
use std::any::TypeId;

use crate::engine::types::{ChunkID, RowID};

use super::primitives::{CapacityError, ShardBoundsError, StaleEntityError};
use super::attribute::AttributeError;
use super::registry::RegistryError;

/// High-level error for entity spawning.
///
/// This aggregates the most common failure modes encountered while creating
/// entities and attaching their components. It intentionally preserves the
/// underlying structured error to keep diagnostics actionable.
///
/// ### Variants (typical)
/// * `Capacity(CapacityError)` - Not enough room to allocate the requested
///   number of entities.
/// * `ShardBounds(ShardBoundsError)` - Target shard index was invalid.
/// * `StaleEntity(StaleEntityError)` - A supplied entity reference was not live.
/// * `StoragePushFailedWith(AttributeError)` - Component push/write failed.
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

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpawnError {

    /// Entity creation failed due to insufficient capacity.
    Capacity(CapacityError),

    /// A shard index was outside the valid range.
    ShardBounds(ShardBoundsError),

    /// Returned by [`EntityShards::new`] when `n_shards` is zero or exceeds
    /// the maximum representable count given `SHARD_BITS`.
    InvalidShardCount {
        /// Number of shards that was requested.
        requested: u16,

        /// Maximum number of shards supported by the current bit layout.
        max: u16,
    },

    /// An entity handle was stale or referred to a despawned entity.
    StaleEntity(StaleEntityError),

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

    /// A shard mutex was poisoned (panic occurred while holding it).
    ShardLockPoisoned,

    /// An invalid or unregistered component ID was encountered.
    InvalidComponentId,

    /// Component registry / factory error encountered during spawn or structural mutation.
    Registry(RegistryError),
}

impl fmt::Display for SpawnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpawnError::Capacity(e) => write!(f, "{e}"),
            SpawnError::ShardBounds(e) => write!(f, "{e}"),
            SpawnError::InvalidShardCount { requested, max } => write!(
                f,
                "invalid shard count: requested {}, maximum supported {}",
                requested, max
            ),
            SpawnError::StaleEntity(e) => write!(f, "{e}"),
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
            SpawnError::ShardLockPoisoned => write!(f, "shard lock poisoned"),
            SpawnError::InvalidComponentId => write!(f, "invalid component id"),
            SpawnError::Registry(e) => write!(f, "registry error: {e}"),
        }
    }
}

impl std::error::Error for SpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SpawnError::Capacity(e) => Some(e),
            SpawnError::ShardBounds(e) => Some(e),
            SpawnError::StaleEntity(e) => Some(e),
            SpawnError::StorageSwapRemoveFailed(e) => Some(e),
            SpawnError::StoragePushFailedWith(e) => Some(e),
            SpawnError::Registry(e) => Some(e),
            _ => None,
        }
    }
}

impl From<CapacityError> for SpawnError {
    fn from(e: CapacityError) -> Self { SpawnError::Capacity(e) }
}

impl From<ShardBoundsError> for SpawnError {
    fn from(e: ShardBoundsError) -> Self { SpawnError::ShardBounds(e) }
}

impl From<StaleEntityError> for SpawnError {
    fn from(e: StaleEntityError) -> Self { SpawnError::StaleEntity(e) }
}

impl From<AttributeError> for SpawnError {
    fn from(e: AttributeError) -> Self { SpawnError::StoragePushFailedWith(e) }
}

impl From<RegistryError> for SpawnError {
    fn from(e: RegistryError) -> Self { SpawnError::Registry(e) }
}
