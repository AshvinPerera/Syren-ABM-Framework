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

    /// Allocates `count` entity slots under a single borrow.
    ///
    /// The bulk companion to [`spawn`](Self::spawn): capacity is ensured
    /// once, then slots are claimed from the free list in a tight loop.
    /// `location_for(k)` supplies the archetype location of the `k`-th
    /// allocated entity; handles are appended to `out` in `k` order.
    ///
    /// ## Errors
    /// Returns `CapacityError` if the shard cannot hold `count` more live
    /// entities. No slots are allocated in that case.
    pub(crate) fn spawn_many(
        &mut self,
        shard_id: ShardID,
        count: usize,
        mut location_for: impl FnMut(usize) -> EntityLocation,
        out: &mut Vec<Entity>,
    ) -> Result<(), CapacityError> {
        let free = self.free_store.len();
        if free < count {
            let deficit = count - free;
            let capacity = INDEX_CAP as usize + 1;
            let available = capacity.saturating_sub(self.versions.len());
            if available < deficit {
                return Err(CapacityError {
                    entities_needed: (self.versions.len() as EntityID)
                        + (deficit as EntityID),
                    capacity: capacity as EntityID,
                });
            }
            // Grow by at least the deficit, preferring the usual doubling
            // policy, clamped to what the shard can still address.
            let growth = deficit.max(self.versions.len()).max(1024).min(available);
            self.ensure_capacity(growth as EntityCount)?;
        }

        debug_assert!(self.free_store.len() >= count);
        out.reserve(count);
        for k in 0..count {
            // The capacity check above guarantees enough free slots.
            let Some(index) = self.free_store.pop() else {
                return Err(CapacityError {
                    entities_needed: (self.versions.len() as EntityID) + 1,
                    capacity: INDEX_CAP as EntityID + 1,
                });
            };
            let version = self.versions[index as usize];
            self.alive[index as usize] = true;
            self.locations[index as usize] = location_for(k);
            out.push(make_entity(shard_id, index, version));
        }
        Ok(())
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
                // Skip VersionID::MAX so `Entity::PLACEHOLDER` (the all-ones
                // raw bit pattern) can never collide with a live handle.
                if *live == VersionID::MAX {
                    *live = 0;
                }
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

    #[cfg(test)]
    pub(crate) fn doctor_version_for_test(&mut self, index: usize, version: VersionID) {
        self.versions[index] = version;
    }

    #[cfg(test)]
    pub(crate) fn version_for_test(&self, index: usize) -> VersionID {
        self.versions[index]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn despawn_version_increment_skips_max() {
        let mut pool = Entities::default();
        let first = pool.spawn(0, EntityLocation::default()).unwrap();
        let index = first.index() as usize;

        // Doctor the slot to the last pre-sentinel version and rebuild the
        // matching live handle.
        pool.doctor_version_for_test(index, VersionID::MAX - 1);
        let handle = make_entity(0, first.index(), VersionID::MAX - 1);

        assert!(pool.despawn(handle));
        assert_eq!(
            pool.version_for_test(index),
            0,
            "version increment must skip VersionID::MAX"
        );
    }
}
