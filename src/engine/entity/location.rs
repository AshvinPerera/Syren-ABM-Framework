//! Physical storage location of an entity within archetype storage.
//!
//! This module defines [`EntityLocation`], a lightweight descriptor that maps
//! an entity handle to the exact position of its component data within the
//! ECS storage hierarchy: an [`ArchetypeID`] identifies which archetype owns
//! the entity, a [`ChunkID`] selects the memory chunk within that archetype,
//! and a [`RowID`] pinpoints the individual row within that chunk.
//!
//! Locations are kept in sync with the storage layer - any operation that
//! moves component rows (insertion, removal, archetype migration, or despawn)
//! must update or invalidate the corresponding `EntityLocation` atomically.

use crate::engine::types::{ArchetypeID, ChunkID, RowID};

/// Physical storage location of an entity within archetype storage.
///
/// ## Purpose
/// Maps an entity handle to its actual component data by identifying
/// the archetype, chunk, and row that contain its components.
///
/// ## Invariants
/// - Must always reflect the true location of the entity's component row.
/// - Updated atomically with archetype row moves.
/// - Invalidated immediately on despawn.

#[derive(Clone, Copy, Debug, Default)]
#[must_use]
pub struct EntityLocation {
    /// Archetype containing the entity.
    pub archetype: ArchetypeID,

    /// Chunk index within the archetype.
    pub chunk: ChunkID,

    /// Row index within the chunk.
    pub row: RowID,
}
