//! Low-level structural error types.
//!
//! These are the foundational, self-contained error structs used throughout
//! the ECS. They have no dependencies on other error types within this module
//! and are composed into higher-level aggregate errors elsewhere.
//!
//! # Error Types
//!
//! | Error | Description |
//! |-------|-------------|
//! | [`CapacityError`] | Insufficient capacity to create or place additional entities. |
//! | [`ShardBoundsError`] | Shard index is outside the valid range for a shard set. |
//! | [`StaleEntityError`] | Entity handle is no longer valid (despawned or generation mismatch). |
//! | [`EmptyArchetypeError`] | Archetype contains no components when at least one was expected. |
//! | [`PositionOutOfBoundsError`] | `(ChunkID, RowID)` pair addresses a position outside storage bounds. |
//! | [`TypeMismatchError`] | Component write targets a storage slot with a mismatched element type. |

use std::fmt;

use crate::engine::types::{ChunkID, RowID, ShardID};

/// Returned when the system cannot satisfy a request to create or place
/// additional entities because the target container has insufficient capacity.
///
/// This typically arises during batch spawns or when attempting to grow a shard
/// beyond its configured limit.
///
/// ### Fields
/// * `entities_needed` - Total number of entities the operation attempted to
///   create or accommodate.
/// * `capacity` - The current upper bound that prevented the operation.
///
/// ### Example
/// ```text
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
/// * `index` - The shard index that was requested.
/// * `max_index` - The maximum valid shard index (inclusive).
///
/// ### Example
/// ```text
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

/// Returned when an `Entity` handle is no longer valid - typically because it
/// was despawned or its generation/version no longer matches live storage.
///
/// Use this to prevent use-after-free style logic errors at the API boundary.
///
/// ### Example
/// ```text
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
/// * `expected` - The [`TypeId`] that the destination storage declares.
/// * `actual` - The [`TypeId`] of the value provided by the caller.
/// * `expected_name` - Human-readable name of the expected type, obtained via
///   [`std::any::type_name`].
/// * `actual_name` - Human-readable name of the actual type, obtained via
///   [`std::any::type_name`].
///
/// ### Example
/// ```text
/// if actual_type != expected_type {
///     return Err(TypeMismatchError {
///         expected: expected_type,
///         actual: actual_type,
///         expected_name: std::any::type_name::<Expected>(),
///         actual_name: std::any::type_name::<Actual>(),
///     }.into());
/// }
/// ```

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeMismatchError {
    /// Destination storage's declared element type.
    pub expected: std::any::TypeId,

    /// Provided value's dynamic type.
    pub actual: std::any::TypeId,

    /// Human-readable name of the expected type.
    pub expected_name: &'static str,

    /// Human-readable name of the actual type.
    pub actual_name: &'static str,
}

impl fmt::Display for TypeMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "type mismatch: expected {} ({:?}), actual {} ({:?})",
            self.expected_name, self.expected, self.actual_name, self.actual
        )
    }
}

impl std::error::Error for TypeMismatchError {}
