//! # Entity Management
//!
//! This module defines the entity identity, lifecycle, and shard-based storage
//! used by the ECS.
//!
//! ## Purpose
//! Entities are lightweight, opaque identifiers that reference rows in archetype
//! storage. This module is responsible for:
//!
//! - Generating stable entity identifiers
//! - Tracking entity liveness via versioning
//! - Mapping entities to archetype locations
//! - Managing scalable entity allocation using sharded storage
//!
//! ## Entity Model
//! An `Entity` is a compact, versioned handle composed of:
//!
//! - A **shard ID**, identifying which shard owns the entity
//! - An **index**, identifying the slot within the shard
//! - A **version**, used to detect stale or recycled entities
//!
//! This layout allows fast validation and prevents use-after-free bugs when
//! entities are despawned and reused.
//!
//! ## Sharding
//! Entities are distributed across multiple shards (`EntityShards`) to reduce
//! contention during concurrent spawning, despawning, and lookup operations.
//!
//! Each shard maintains:
//! - A dense pool of entity slots
//! - Version counters for stale detection
//! - Location metadata pointing into archetype storage
//!
//! ## Invariants
//! - An entity is considered alive if and only if its version matches the
//!   version stored in its shard and its slot is marked alive.
//! - Entity locations must always reflect the actual archetype row.
//! - Despawning an entity invalidates all previous handles to that entity.
//!
//! ## Concurrency
//! - Shards synchronize entity allocation and metadata using internal locks.
//! - Global counters are maintained using atomics for low-contention signals.
//!
//! ## Safety
//! Correctness relies on:
//! - Updating entity locations atomically with archetype row moves
//! - Never mutating entity metadata while systems hold archetype borrows
//! - Applying structural changes only at synchronization points

use std::sync::{Mutex, MutexGuard};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::engine::types::{
    EntityID, ShardID, IndexID, VersionID, EntityCount, 
    SHARD_BITS, INDEX_BITS, INDEX_MASK, SHARD_MASK, INDEX_CAP,
    ArchetypeID, RowID, ChunkID
};
use crate::engine::error::{
    CapacityError, ShardBoundsError, SpawnError
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
/// - **Shard ID** — identifies which shard owns the entity
/// - **Index** — slot within the shard
/// - **Version** — incremented on despawn to invalidate stale handles
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
pub struct Entity(pub EntityID);

#[inline]
const fn make_id(shard: ShardID, index: IndexID, version: VersionID) -> EntityID {
    ((version as EntityID) << (SHARD_BITS + INDEX_BITS)) |
    ((shard as EntityID) << INDEX_BITS) |
    (index as EntityID)
}

#[inline]
fn make_entity(shard: ShardID, index: IndexID, version: VersionID) -> Entity {
    debug_assert!((index as EntityID) <= INDEX_MASK);
    debug_assert!((shard as EntityID) <= SHARD_MASK);
    Entity(make_id(shard, index, version))
}

#[inline]
const fn split_entity(entity: Entity) -> (ShardID, IndexID, VersionID) {
    let id = entity.0;
    let shard = ((id >> INDEX_BITS) & SHARD_MASK) as ShardID;    
    let index = (id & INDEX_MASK) as IndexID;
    let version = (id >> (INDEX_BITS + SHARD_BITS)) as VersionID;
    (shard, index, version)
}

impl Entity {
    /// Returns the `(shard, index, version)` components of this entity.
    #[inline] pub fn components(self) -> (ShardID, IndexID, VersionID) { split_entity(self) }

    /// Returns the shard identifier encoded in this entity.
    #[inline] pub fn shard(self) -> ShardID { ((self.0 >> INDEX_BITS) & SHARD_MASK) as ShardID }

    /// Returns the index component of this entity.
    #[inline] pub fn index(self) -> IndexID { (self.0 & INDEX_MASK) as IndexID }

    /// Returns the version component of this entity.
    #[inline] pub fn version(self) -> VersionID { (self.0 >> (INDEX_BITS + SHARD_BITS)) as VersionID }
}

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
pub struct EntityLocation {
    /// Archetype containing the entity.
    pub archetype: ArchetypeID,

    /// Chunk index within the archetype.
    pub chunk: ChunkID,

    /// Row index within the chunk.
    pub row: RowID,
}

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
    free_store: Vec<IndexID>,
    alive: Vec<bool>,
    locations: Vec<EntityLocation>
}

impl Entities {

    /// Creates an empty entity storage.
    pub fn new() -> Self { Self::default() }

    fn ensure_capacity(&mut self, additional_entities: EntityCount) -> Result<(), CapacityError> {
        if additional_entities == 0 { return Ok(()); }

        let current_entity_count = self.versions.len() as EntityID;
        let entities_needed = current_entity_count + (additional_entities as EntityID);
        let capacity = INDEX_CAP as EntityID + 1;
        if entities_needed > capacity {
            return Err(CapacityError { entities_needed, capacity });
        }

        self.versions.resize(entities_needed as usize, 0);
        self.alive.resize(entities_needed as usize, false);
        self.locations.resize(entities_needed as usize, EntityLocation::default());

        for index in current_entity_count..entities_needed {
            self.free_store.push(index as IndexID);
        }
        Ok(())
    }

    /// Allocates a new entity slot and assigns an initial location.
    ///
    /// ## Behavior
    /// - Reuses a free slot if available, otherwise grows storage.
    /// - Marks the slot as alive and records its archetype location.
    /// - Does not modify archetype storage itself.
    ///
    /// ## Errors
    /// Returns `CapacityError` if the shard exceeds its maximum entity capacity.
    ///
    /// ## Invariants
    /// - The returned entity is alive upon success.
    /// - The version is unchanged from the previous occupant of the slot.

    fn spawn(&mut self, shard_id: ShardID, location: EntityLocation) -> Result<Entity, CapacityError> {
        let index = if let Some(i) = self.free_store.pop() {
            i
        } else {
            self.ensure_capacity(1024)?;
            match self.free_store.pop() {
                Some(i) => i,
                None => {
                    let entities_needed = (self.versions.len() as u64).saturating_add(1);
                    let capacity = (INDEX_CAP as u64).saturating_add(1);
                    return Err(CapacityError { entities_needed, capacity });
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
    /// ## Behavior
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

    pub fn despawn(&mut self, entity: Entity) -> bool {
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
        if self.is_alive(entity)
        {
            Some(self.locations[i as usize])
        } else {
            None
        }
    }

    /// Updates the stored location for an entity.
    ///
    /// ## Safety
    /// Caller must ensure the entity is alive.

    pub fn set_location(&mut self, entity: Entity, location: EntityLocation) {
        let (_, i, _) = split_entity(entity);
        let index = i as usize;
        debug_assert!(
            self.is_alive(entity),
            "set_location was called on a dead or stale entity. Entity: {:?}, Location: {:?}", 
            entity, location
        );
        if index < self.locations.len() {
            self.locations[index] = location;
        }
    }
}

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
    pub entities: Mutex<Entities>,

    /// Count of live entities in the shard.
    pub live_entity_count: AtomicU32,

    /// Approximate number of free entity slots.
    pub approximate_free_store_length: AtomicU32,
}

impl Shard {
    fn new() -> Self {
        Self {
            entities: Mutex::new(Entities::default()),
            live_entity_count: AtomicU32::new(0),
            approximate_free_store_length: AtomicU32::new(0),
        }
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
    shards: Vec<Shard>
}

impl EntityShards {
    /// Returns the total number of entity shards.
    #[inline] pub fn shard_count(&self) -> usize { self.shards.len() }

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
    /// ## Behavior
    /// - Allocates `n_shards` independent `Shard` instances.
    /// - Each shard manages its own entity pool and metadata.
    /// - Initializes internal counters used for future load-balancing heuristics.
    ///
    /// ## Panics
    /// Panics if:
    /// - `n_shards` is zero.
    /// - `n_shards` exceeds the maximum representable shard count given
    ///   the configured `SHARD_BITS`.
    ///
    /// ## Invariants
    /// - Shard identifiers encoded in `Entity` values are always in
    ///   `[0, n_shards)`.
    /// - All shards are fully initialized and independent.
    ///
    /// ## Notes
    /// The shard count is fixed for the lifetime of the ECS world and
    /// cannot be changed after initialization.

    pub fn new(n_shards: usize) -> Self {
        assert!(n_shards > 0 && n_shards <= (1usize << SHARD_BITS));
        let mut shards = Vec::with_capacity(n_shards);
        for _ in 0..n_shards {
            shards.push(Shard::new());
        }
        Self { shards}
    }

    /// Spawns a new entity in the specified shard.
    ///
    /// ## Behavior
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

    pub fn spawn_on(&self, shard_id: ShardID, location: EntityLocation) -> Result<Entity, SpawnError> {
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
            shard.approximate_free_store_length.fetch_sub(1, Ordering::Relaxed);
        } else {
            let after_free = entities.free_store.len();
            shard.approximate_free_store_length
                .fetch_add(after_free as u32, Ordering::Relaxed);
        }

        Ok(entity)
    }


    /// Returns `true` if the entity is alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        let shard_id = entity.shard() as usize;
        if shard_id >= self.shard_count() {
            return false;
        }

        match self.shards[shard_id].entities.lock() {
            Ok(entities) => entities.is_alive(entity),
            Err(_) => false,
        }
    }

    /// Returns the location of an entity, if alive.
    pub fn get_location(&self, entity: Entity) -> Option<EntityLocation> {
        let shard_id = entity.shard() as usize;
        if shard_id >= self.shard_count() {
            return None;
        }

        match self.shards[shard_id].entities.lock() {
            Ok(entities) => entities.get_location(entity),
            Err(_) => None,
        }
    }

    /// Updates the archetype location metadata for an entity.
    ///
    /// ## Purpose
    /// Used during archetype row moves to keep entity metadata consistent
    /// with component storage.
    ///
    /// ## Safety
    /// Caller must ensure:
    /// - The entity is alive.
    /// - The provided location matches actual component storage.

    pub fn set_location(&self, entity: Entity, location: EntityLocation) {
        let shard_id = entity.shard() as usize;
        if shard_id >= self.shard_count() {
            return;
        }

        if let Ok(mut entities) = self.shards[shard_id].entities.lock() {
            entities.set_location(entity, location);
        }
    }

    /// Despawns an entity.
    pub fn despawn(&self, entity: Entity) -> bool {
        let shard_id = entity.shard() as usize;
        if shard_id >= self.shard_count() {
            return false;
        }

        let shard = &self.shards[shard_id];

        let mut entities = match shard.entities.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };

        if entities.despawn(entity) {
            shard.approximate_free_store_length.fetch_add(1, Ordering::Relaxed);
            shard.live_entity_count.fetch_sub(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}
