//! Shard-local entity pool managing slot allocation, versioning, and liveness tracking.
//!
//! # Overview
//!
//! This module provides [`Entities`], a shard-local data structure that tracks the full
//! lifecycle of entities within a single shard: allocation, despawn, version invalidation,
//! and archetype location metadata.
//!
//! # Entity Lifecycle
//!
//! 1. **Spawn** - a free index is claimed from the free list (or new slots are grown),
//!    marked alive, and bundled with a shard ID and version into an [`Entity`] handle.
//! 2. **Live** - the slot holds a valid [`EntityLocation`] pointing into archetype storage.
//!    Location can be updated as entities move between archetypes.
//! 3. **Despawn** - the version is incremented, the slot is marked dead, its location is
//!    cleared, and the index is returned to the free list. All previously issued handles
//!    for the entity become permanently stale.
//!
//! # Design Notes
//!
//! - **Free list allocation**: indices are recycled via `free_store`, keeping allocation O(1).
//! - **Generational versioning**: each slot carries a [`VersionID`] that is incremented on
//!   despawn, making stale [`Entity`] handles detectable without extra bookkeeping.
//! - **Dense parallel vecs**: `versions`, `alive`, and `locations` are kept in lockstep,
//!   all indexed by [`IndexID`], enabling cache-friendly access patterns.
//! - **Capacity ceiling**: total entities per shard are bounded by [`INDEX_CAP`]; attempts
//!   to exceed it surface a [`CapacityError`].
//! - **Growth strategy**: when the free list is exhausted, capacity doubles from the current
//!   size (with a minimum growth of 1024 slots), amortizing allocation cost for large
//!   entity counts.
//!
//! # Concurrency
//!
//! [`Entities`] is **not thread-safe**. Callers are responsible for external synchronization;
//! in practice this is provided by the `Mutex` wrapping [`Entities`] inside `Shard`.

use crate::engine::error::CapacityError;
use crate::engine::types::{EntityCount, EntityID, IndexID, ShardID, VersionID, INDEX_CAP};

use super::entity::Entity;
use super::entity::{make_entity, split_entity};
use super::location::EntityLocation;

/// Shard-local entity pool.
///
/// ## Purpose
/// `Entities` manages entity slot allocation, versioning, liveness tracking,
/// and archetype location metadata for a single shard.
///
/// ## Design
/// - Entities are allocated from a free list of indices.
/// - Versions are incremented on despawn to invalidate stale entities.
/// - Storage is dense and index-addressable.
/// - When the free list is exhausted, capacity grows by doubling the current
///   slot count (minimum 1024), amortizing reallocation for large simulations.
///
/// ## Invariants
/// - `versions.len() == alive.len() == locations.len()`.
/// - If `alive[i]` is `true`, then `locations[i]` is valid.
/// - Free indices always refer to dead entity slots.
///
/// ## Concurrency
/// This type is **not thread-safe** and must be externally synchronized.
/// In practice, it is protected by a `Mutex` in `Shard`.

#[derive(Default)]
pub struct Entities {
    versions: Vec<VersionID>,
    pub(super) free_store: Vec<IndexID>,
    alive: Vec<bool>,
    locations: Vec<EntityLocation>,
}

impl Entities {
    fn ensure_capacity(&mut self, additional_entities: EntityCount) -> Result<(), CapacityError> {
        if additional_entities == 0 {
            return Ok(());
        }

        let current_entity_count = self.versions.len() as EntityID;
        let entities_needed = current_entity_count + (additional_entities as EntityID);
        let capacity = INDEX_CAP as EntityID + 1;
        if entities_needed > capacity {
            return Err(CapacityError {
                entities_needed,
                capacity,
            });
        }

        self.versions.resize(entities_needed as usize, 0);
        self.alive.resize(entities_needed as usize, false);
        self.locations
            .resize(entities_needed as usize, EntityLocation::default());

        for index in current_entity_count..entities_needed {
            self.free_store.push(index as IndexID);
        }
        Ok(())
    }

    /// Allocates a new entity slot and assigns an initial location.
    ///
    /// ## Behaviour
    /// - Reuses a free slot if available, otherwise grows storage.
    /// - When growing, capacity doubles from the current slot count (minimum 1024),
    ///   amortizing allocation cost for simulations with large entity counts.
    /// - Marks the slot as alive and records its archetype location.
    /// - Does not modify archetype storage itself.
    ///
    /// ## Errors
    /// Returns `CapacityError` if the shard exceeds its maximum entity capacity.
    ///
    /// ## Invariants
    /// - The returned entity is alive upon success.
    /// - The version is unchanged from the previous occupant of the slot.

    pub(crate) fn spawn(
        &mut self,
        shard_id: ShardID,
        location: EntityLocation,
    ) -> Result<Entity, CapacityError> {
        let index = if let Some(i) = self.free_store.pop() {
            i
        } else {
            let growth = self.versions.len().max(1024);
            self.ensure_capacity(growth as EntityCount)?;
            match self.free_store.pop() {
                Some(i) => i,
                None => {
                    let entities_needed = (self.versions.len() as u64).saturating_add(1);
                    let capacity = (INDEX_CAP as u64).saturating_add(1);
                    return Err(CapacityError {
                        entities_needed,
                        capacity,
                    });
                }
            }
        };

        let version = self.versions[index as usize];
        self.alive[index as usize] = true;
        self.locations[index as usize] = location;

        Ok(make_entity(shard_id, index, version))
    }

    /// Destroys an entity and invalidates its handle.
    ///
    /// ## Behaviour
    /// - Verifies the entity version matches the current slot version.
    /// - Marks the slot dead and increments its version.
    /// - Clears stored location metadata.
    /// - Returns the slot to the free list.
    ///
    /// ## Returns
    /// Returns `true` if the entity was alive and successfully despawned.
    /// Returns `false` if the entity was stale or invalid.
    ///
    /// ## Invariants
    /// All previously issued handles for this entity become invalid.

    pub(crate) fn despawn(&mut self, entity: Entity) -> bool {
        let (_, i, v) = split_entity(entity);
        let index = i as usize;
        match self.versions.get_mut(index) {
            Some(live) if *live == v && self.alive.get(index).copied().unwrap_or(false) => {
                *live = live.wrapping_add(1);
                self.alive[index] = false;
                self.locations[index] = EntityLocation::default();
                self.free_store.push(i);
                true
            }
            _ => false,
        }
    }

    /// Returns `true` if the entity is alive and not stale.
    pub fn is_alive(&self, entity: Entity) -> bool {
        let (_, i, v) = split_entity(entity);
        let index = i as usize;
        index < self.versions.len()
            && self.alive.get(index).copied().unwrap_or(false)
            && self.versions[index] == v
    }

    /// Returns the archetype location of an entity, if alive.
    pub fn get_location(&self, entity: Entity) -> Option<EntityLocation> {
        let (_, i, _) = split_entity(entity);
        if self.is_alive(entity) {
            Some(self.locations[i as usize])
        } else {
            None
        }
    }

    /// Updates the stored location for an entity.
    ///
    /// ## Safety
    /// Caller must ensure the entity is alive.

    pub(crate) fn set_location(&mut self, entity: Entity, location: EntityLocation) {
        let (_, i, _) = split_entity(entity);
        let index = i as usize;
        debug_assert!(
            self.is_alive(entity),
            "set_location was called on a dead or stale entity. Entity: {:?}, Location: {:?}",
            entity,
            location
        );
        if index < self.locations.len() {
            self.locations[index] = location;
        }
    }
}
