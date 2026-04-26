//! Sharded entity manager: thread-safe allocation, destruction, and lookup.
//!
//! # Overview
//!
//! This module implements [`EntityShards`], a globally shared entity manager that
//! partitions entity storage across multiple independent [`Shard`]s. Sharding
//! reduces lock contention during concurrent spawning, despawning, and lookup by
//! ensuring that operations on different shards never block one another.
//!
//! # Architecture
//!
//! Each [`Shard`] owns a [`Mutex`]-protected [`Entities`] pool along with
//! atomic counters that provide lock-free estimates of live entity count and
//! free slot availability. These counters are intentionally approximate and
//! used only for load-balancing heuristics — they are never relied upon for
//! correctness.
//!
//! # Entity Identity
//!
//! Every [`Entity`] handle encodes its shard index directly in its bits, so
//! routing any operation to the correct shard requires no additional lookup.
//! The number of addressable shards is bounded by [`SHARD_BITS`].
//!
//! # Thread Safety
//!
//! All public methods on [`EntityShards`] are thread-safe. Shards synchronize
//! independently via their internal mutexes, meaning concurrent operations on
//! different shards proceed without contention.
//!
//! # Errors
//!
//! Operations propagate [`SpawnError`] on failure. The two most common failure
//! modes are:
//!
//! - [`SpawnError::ShardBounds`] — a shard index encoded in an [`Entity`] or
//!   passed directly exceeds the initialized shard count.
//! - [`SpawnError::ShardLockPoisoned`] — a shard mutex was poisoned by a
//!   panicking thread.

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::engine::error::{ShardBoundsError, SpawnError};
use crate::engine::types::{ShardID, SHARD_BITS};

use super::entities::Entities;
use super::entity::Entity;
use super::location::EntityLocation;

/// Synchronization unit for entity storage.
///
/// ## Purpose
/// `Shard` groups a subset of entities together to reduce contention during
/// concurrent spawning, despawning, and lookup operations.
///
/// ## Design
/// - Entity data is protected by a `Mutex`.
/// - Lightweight counters are maintained using atomics for load estimation.
/// - Shards are independent and do not share entity indices.
///
/// ## Invariants
/// - All entities stored in this shard have matching shard IDs.
/// - Atomic counters are approximate and used only for heuristics.

pub struct Shard {
    /// Entity pool protected by a mutex.
    pub(crate) entities: Mutex<Entities>,

    /// Count of live entities in the shard.
    pub(crate) live_entity_count: AtomicU32,

    /// Approximate number of free entity slots.
    pub(crate) approximate_free_store_length: AtomicU32,
}

impl Shard {
    fn new() -> Self {
        Self {
            entities: Mutex::new(Entities::default()),
            live_entity_count: AtomicU32::new(0),
            approximate_free_store_length: AtomicU32::new(0),
        }
    }

    /// Returns the approximate count of live entities in this shard.
    #[allow(dead_code)]
    #[inline]
    pub fn live_count(&self) -> u32 {
        self.live_entity_count.load(Ordering::Relaxed)
    }

    /// Returns the approximate number of free entity slots in this shard.
    #[allow(dead_code)]
    #[inline]
    pub fn approximate_free_count(&self) -> u32 {
        self.approximate_free_store_length.load(Ordering::Relaxed)
    }
}

/// Global sharded entity manager.
///
/// ## Purpose
/// Provides scalable entity allocation, destruction, and lookup by distributing
/// entities across multiple independent shards.
///
/// ## Design
/// - Shards are selected explicitly or via load-balancing heuristics.
/// - Public methods route operations to the appropriate shard.
/// - Entity handles encode shard identity directly.
///
/// ## Invariants
/// - The shard encoded in an entity must exist.
/// - Entity location metadata must remain consistent with archetype storage.
///
/// ## Concurrency
/// - Shards operate independently and synchronize internally.
/// - Public methods are thread-safe.

pub struct EntityShards {
    shards: Vec<Shard>,
}

impl fmt::Debug for EntityShards {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EntityShards")
            .field("shard_count", &self.shard_count())
            .finish()
    }
}

impl EntityShards {
    /// Returns the total number of entity shards.
    #[inline]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Lock a shard's entity pool.
    #[inline]
    fn lock_entities(&self, shard_id: ShardID) -> Result<MutexGuard<'_, Entities>, SpawnError> {
        self.shards[shard_id as usize]
            .entities
            .lock()
            .map_err(|_| SpawnError::ShardLockPoisoned)
    }

    /// Creates a new sharded entity manager.
    ///
    /// ## Purpose
    /// Initializes the global entity storage by partitioning entity management
    /// across a fixed number of independent shards. Sharding reduces contention
    /// during concurrent entity creation, destruction, and lookup.
    ///
    /// ## Behaviour
    /// - Allocates `n_shards` independent `Shard` instances.
    /// - Each shard manages its own entity pool and metadata.
    /// - Initializes internal counters used for future load-balancing heuristics.
    ///
    /// ## Errors
    /// Returns `SpawnError::InvalidShardCount` if:
    /// - `n_shards` is zero.
    /// - `n_shards` exceeds the maximum representable shard count given
    ///   the configured `SHARD_BITS`.
    ///
    /// ## Invariants
    /// - Shard identifiers encoded in `Entity` values are always in
    ///   `[0, n_shards]`.
    /// - All shards are fully initialized and independent.
    ///
    /// ## Notes
    /// The shard count is fixed for the lifetime of the ECS world and
    /// cannot be changed after initialization.

    pub fn new(n_shards: usize) -> Result<Self, SpawnError> {
        let max_shards = 1usize << SHARD_BITS;
        if n_shards == 0 || n_shards > max_shards {
            return Err(SpawnError::InvalidShardCount {
                requested: n_shards as u16,
                max: max_shards as u16,
            });
        }
        let mut shards = Vec::with_capacity(n_shards);
        for _ in 0..n_shards {
            shards.push(Shard::new());
        }
        Ok(Self { shards })
    }

    /// Spawns a new entity in the specified shard.
    ///
    /// ## Behaviour
    /// - Validates shard bounds.
    /// - Allocates a new entity slot in the shard.
    /// - Updates shard-level counters for load tracking.
    ///
    /// ## Errors
    /// - `ShardBounds` if the shard ID is invalid.
    /// - `CapacityError` if entity capacity is exceeded.
    ///
    /// ## Invariants
    /// The returned entity is alive and has a valid location.

    pub fn spawn_on(
        &self,
        shard_id: ShardID,
        location: EntityLocation,
    ) -> Result<Entity, SpawnError> {
        let shard_count = self.shard_count();
        if (shard_id as usize) >= shard_count {
            return Err(SpawnError::ShardBounds(ShardBoundsError {
                index: shard_id,
                max_index: shard_count.saturating_sub(1) as u32,
            }));
        }

        // Lock shard entities.
        let mut entities = self.lock_entities(shard_id)?;

        let before_free = entities.free_store.len();
        let entity = entities.spawn(shard_id, location)?;

        // Update counters.
        let shard = &self.shards[shard_id as usize];
        shard.live_entity_count.fetch_add(1, Ordering::Relaxed);

        if before_free > 0 {
            shard
                .approximate_free_store_length
                .fetch_sub(1, Ordering::Relaxed);
        } else {
            let after_free = entities.free_store.len();
            shard
                .approximate_free_store_length
                .fetch_add(after_free as u32, Ordering::Relaxed);
        }

        Ok(entity)
    }

    /// Returns `true` if the entity is alive.
    ///
    /// ## Errors
    /// - `SpawnError::ShardLockPoisoned` if the shard mutex is poisoned.

    pub fn is_alive(&self, entity: Entity) -> Result<bool, SpawnError> {
        let shard_id = entity.shard() as usize;
        if shard_id >= self.shard_count() {
            return Ok(false);
        }

        let entities = self.shards[shard_id]
            .entities
            .lock()
            .map_err(|_| SpawnError::ShardLockPoisoned)?;

        Ok(entities.is_alive(entity))
    }

    /// Returns the location of an entity, if alive.
    ///
    /// ## Errors
    /// - `SpawnError::ShardLockPoisoned` if the shard mutex is poisoned.

    pub fn get_location(&self, entity: Entity) -> Result<Option<EntityLocation>, SpawnError> {
        let shard_id = entity.shard() as usize;
        if shard_id >= self.shard_count() {
            return Ok(None);
        }

        let entities = self.shards[shard_id]
            .entities
            .lock()
            .map_err(|_| SpawnError::ShardLockPoisoned)?;

        Ok(entities.get_location(entity))
    }

    /// Updates the archetype location metadata for an entity.
    ///
    /// ## Purpose
    /// Used during archetype row moves to keep entity metadata consistent
    /// with component storage.
    ///
    /// ## Errors
    /// Returns `SpawnError::ShardBounds` if the shard ID is out of bounds.
    /// Returns `SpawnError::ShardLockPoisoned` if the shard mutex is poisoned.
    ///
    /// ## Safety
    /// Caller must ensure:
    /// - The entity is alive.
    /// - The provided location matches actual component storage.

    pub fn set_location(&self, entity: Entity, location: EntityLocation) -> Result<(), SpawnError> {
        let shard_id = entity.shard() as usize;
        if shard_id >= self.shard_count() {
            return Err(SpawnError::ShardBounds(ShardBoundsError {
                index: entity.shard(),
                max_index: self.shard_count().saturating_sub(1) as u32,
            }));
        }

        let mut entities = self.shards[shard_id]
            .entities
            .lock()
            .map_err(|_| SpawnError::ShardLockPoisoned)?;

        entities.set_location(entity, location);
        Ok(())
    }

    /// Despawns an entity.
    ///
    /// ## Returns
    /// - `Ok(true)` if the entity was alive and successfully despawned.
    /// - `Ok(false)` if the entity was stale, invalid, or already dead.
    ///
    /// ## Errors
    /// - `SpawnError::ShardLockPoisoned` if the shard mutex is poisoned.

    pub fn despawn(&self, entity: Entity) -> Result<bool, SpawnError> {
        let shard_id = entity.shard() as usize;
        if shard_id >= self.shard_count() {
            return Ok(false);
        }

        let shard = &self.shards[shard_id];

        let mut entities = shard
            .entities
            .lock()
            .map_err(|_| SpawnError::ShardLockPoisoned)?;

        if entities.despawn(entity) {
            shard
                .approximate_free_store_length
                .fetch_add(1, Ordering::Relaxed);
            shard.live_entity_count.fetch_sub(1, Ordering::Relaxed);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
