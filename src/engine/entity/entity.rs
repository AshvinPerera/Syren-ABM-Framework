//! Opaque, versioned entity identifier and bit-packing helpers.
//!
//! This module defines [`Entity`], the primary handle type used throughout the
//! ECS engine to refer to live entities, along with the low-level bit-packing
//! primitives that construct and decompose those handles.
//!
//! # Layout
//!
//! Every [`Entity`] wraps a single [`EntityID`] integer whose bits are divided
//! into three contiguous fields, from least-significant to most-significant:
//!
//! ```text
//! ┌─────────────────┬───────────────┬──────────────────────────────┐
//! │     version     │     shard     │            index             │
//! │  (upper bits)   │  (SHARD_BITS) │          (INDEX_BITS)        │
//! └─────────────────┴───────────────┴──────────────────────────────┘
//! ```
//!
//! - **`index`** (`INDEX_BITS` wide) — slot within the owning shard's storage array.
//! - **`shard`** (`SHARD_BITS` wide) — identifies which shard owns the entity.
//! - **`version`** (remaining upper bits) — incremented each time a slot is
//!   recycled, so stale handles from before a despawn can be detected.
//!
//! The exact widths of each field are controlled by the constants imported from
//! [`crate::engine::types`].
//!
//! # Key items
//!
//! | Item | Kind | Description |
//! |---|---|---|
//! | [`Entity`] | `struct` | The public handle type; cheap to copy, hash, and compare. |
//! | [`make_entity`] | `fn` | Constructs an [`Entity`] from `(shard, index, version)`. |
//! | [`make_id`] | `fn` | `const` variant that returns a raw [`EntityID`]. |
//! | [`split_entity`] | `fn` | Decomposes an [`Entity`] back into its three fields. |
//!
//! # Validity and liveness
//!
//! An [`Entity`] handle is *valid* if it was produced by this engine (i.e. via
//! [`make_entity`]) and has not been forged from an arbitrary integer.  A valid
//! handle is *live* if the version encoded in the handle matches the version
//! currently stored in the shard slot **and** that slot is marked alive.
//! Handles that fail either check are considered stale and must not be used to
//! access component data.
//!
//! # Crate-internal helpers
//!
//! [`make_entity`], [`make_id`], and [`split_entity`] are `pub(super)` and
//! intended only for use by the engine's shard and registry layers. External
//! code should interact exclusively with the [`Entity`] API and obtain handles
//! through the engine's spawn/query interfaces.

use crate::engine::types::{
    EntityID, ShardID, IndexID, VersionID,
    SHARD_BITS, INDEX_BITS, INDEX_MASK, SHARD_MASK,
};


/// Opaque, versioned identifier for an ECS entity.
///
/// ## Purpose
/// `Entity` is a compact handle that uniquely identifies an entity instance
/// at a point in time. It encodes enough information to:
///
/// - Detect stale or recycled entity references
/// - Route entity operations to the correct shard
/// - Index directly into shard-local storage
///
/// ## Representation
/// Internally, an `Entity` packs three values into a single integer:
///
/// - **Shard ID** - identifies which shard owns the entity
/// - **Index** - slot within the shard
/// - **Version** - incremented on despawn to invalidate stale handles
///
/// ## Invariants
/// - Two entities with the same `(shard, index)` but different versions
///   are considered distinct.
/// - An entity is alive iff its version matches the stored version and
///   its slot is marked alive.
///
/// ## Notes
/// `Entity` values are cheap to copy and compare and are safe to pass
/// across threads.

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[must_use]
pub struct Entity(EntityID);

impl Entity {
    /// Creates an `Entity` from a raw ID.
    ///
    /// Intended for deserialization or low-level interop only. The caller
    /// is responsible for ensuring the ID was produced by this ECS engine.
    #[inline]
    pub fn from_raw(id: EntityID) -> Self {
        Entity(id)
    }

    /// Returns the raw `EntityID` backing this entity handle.
    #[inline]
    pub fn to_raw(self) -> EntityID {
        self.0
    }

    /// Returns the `(shard, index, version)` parts of this entity.
    #[inline]
    pub fn parts(self) -> (ShardID, IndexID, VersionID) { split_entity(self) }

    /// Deprecated: use [`parts()`](Entity::parts) instead.
    #[deprecated(since = "0.2.0", note = "Renamed to `parts()` to avoid confusion with ECS components")]
    #[inline]
    pub fn components(self) -> (ShardID, IndexID, VersionID) { self.parts() }

    /// Returns the shard identifier encoded in this entity.
    #[inline]
    pub fn shard(self) -> ShardID { ((self.0 >> INDEX_BITS) & SHARD_MASK) as ShardID }

    /// Returns the index component of this entity.
    #[inline]
    pub fn index(self) -> IndexID { (self.0 & INDEX_MASK) as IndexID }

    /// Returns the version component of this entity.
    #[inline]
    pub fn version(self) -> VersionID { (self.0 >> (INDEX_BITS + SHARD_BITS)) as VersionID }
}

#[inline]
pub(super) const fn make_id(shard: ShardID, index: IndexID, version: VersionID) -> EntityID {
    ((version as EntityID) << (SHARD_BITS + INDEX_BITS)) |
        ((shard as EntityID) << INDEX_BITS) |
        (index as EntityID)
}

#[inline]
pub(super) fn make_entity(shard: ShardID, index: IndexID, version: VersionID) -> Entity {
    debug_assert!((index as EntityID) <= INDEX_MASK);
    debug_assert!((shard as EntityID) <= SHARD_MASK);
    Entity(make_id(shard, index, version))
}

#[inline]
pub(super) const fn split_entity(entity: Entity) -> (ShardID, IndexID, VersionID) {
    let id = entity.0;
    let shard = ((id >> INDEX_BITS) & SHARD_MASK) as ShardID;
    let index = (id & INDEX_MASK) as IndexID;
    let version = (id >> (INDEX_BITS + SHARD_BITS)) as VersionID;
    (shard, index, version)
}
