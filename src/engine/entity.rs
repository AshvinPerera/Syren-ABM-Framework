use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::collections::{HashMap, hash_map::DefaultHasher};

use rayon::prelude::*;

use random::{tl_rand_u64};
use types::{
    EntityID, ShardID, IndexID, VersionID, EntityCount, 
    SHARD_BITS, INDEX_BITS, INDEX_MASK, SHARD_MASK, INDEX_CAP,
    ArchetypeID, RowID
};
use error::{
    CapacityError, ShardBoundsError, SpawnError
};


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
    #[inline] pub fn components(self) -> (ShardID, IndexID, VersionID) { split_entity(self) }
    #[inline] pub fn shard(self) -> ShardID { ((self.0 >> INDEX_BITS) & SHARD_MASK) as ShardID }
    #[inline] pub fn index(self) -> IndexID { (self.0 & INDEX_MASK) as IndexID }
    #[inline] pub fn version(self) -> VersionID { (self.0 >> (INDEX_BITS + SHARD_BITS)) as VersionID }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EntityLocation {
    pub archetype: ArchetypeID,
    pub chunk: ChunkID,
    pub row: RowID,
}

#[derive(Default)]
pub struct Entities {
    versions: Vec<VersionID>,
    free_store: Vec<IndexID>,
    alive: Vec<bool>,
    locations: Vec<EntityLocation>
}

impl Entities {
    pub fn new() -> Self { Self::default() }

    fn ensure_capacity(&mut self, additional_entities: EntityCount) -> Result<(), CapacityError> {
        if additional_entities == 0 { return Ok(()); }

        let current_entity_count = self.versions.len() as EntityID;
        let entities_needed = current_entity_count + (additional_entities as EntityID);
        let capacity = INDEX_CAP as EntityID + 1;
        if entities_needed > capacity {
            return Err(CapacityError { entities_needed, capacity });
        }

        self.versions.resize((entities_needed as usize), 0);
        self.alive.resize((entities_needed as usize), false);
        self.locations.resize((entities_needed as usize), EntityLocation::default());

        for index in current_entity_count..entities_needed {
            self.free_store.push(index as IndexID);
        }
        Ok(())
    }

    fn spawn(&mut self, shard_id: ShardID, location: EntityLocation) -> Result<Entity, CapacityError> {
        let index = if let Some(i) = self.free_store.pop() {
            i
        } else {
            self.ensure_capacity(1024)?;
            self.free_store.pop().expect("capacity added must yield a slot.")
        };

        let version = self.versions[(index as usize)];
        self.alive[(index as usize)] = true;
        self.locations[(index as usize)] = location;

        Ok(make_entity(shard_id, index, version))
    }

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

    pub fn is_alive(&self, entity: Entity) -> bool {
        let (_, i, v) = split_entity(entity);
        let index = i as usize; 
        index < self.versions.len()
            && self.alive.get(index).copied().unwrap_or(false)
            && self.versions[index] == v
    }

    pub fn get_location(&self, entity: Entity) -> Option<EntityLocation> {
        let (_, i, v) = split_entity(entity);
        let index = i as usize;
        if self.is_alive(entity)
        {
            Some(self.locations[(i as usize)])
        } else {
            None
        }
    }

    pub fn set_location(&mut self, entity: Entity, location: EntityLocation) {
        let (_, i, v) = split_entity(entity);
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

pub struct Shard {
    pub entities: Mutex<Entities>,
    pub live_entity_count: AtomicU32,
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

    #[inline]
    fn load_signal(&self) -> (u32, u32) {
        (
            self.live_entity_count.load(Ordering::Relaxed),
            self.approximate_free_store_length.load(Ordering::Relaxed),
        )
    }
}

pub struct EntityShards {
    shards: Vec<Shard>,
    ticket: AtomicU64,
}

impl EntityShards {
    #[inline] fn shard_count(&self) -> usize { self.shards.len() }

    pub fn new(n_shards: usize) -> Self {
        assert!(n_shards > 0 && n_shards <= (1usize << SHARD_BITS));
        let mut shards = Vec::with_capacity(n_shards);
        for _ in 0..n_shards {
            shards.push(Shard::new());
        }
        Self { shards, ticket: AtomicU64::new(0) }
    }

    #[inline]
    fn pick_shard_by_thread(&self) -> ShardID {
        let thread_id = std::thread::current().id();
        let mut hasher = DefaultHasher::new();
        thread_id.hash(&mut hasher);
        (hasher.finish() as usize % self.shards.len()) as ShardID
    }

    #[inline]
    fn pick_shard_p2c(&self) -> ShardID {
        let n = self.shard_count() as u64;
        let shard_id_a = (tl_rand_u64() % n) as ShardID;
        let shard_id_b = (tl_rand_u64() % n) as ShardID;
        if shard_id_a == shard_id_b { return shard_id_a; }

        let shard_a = self.shards[shard_id_a as usize].load_signal();
        let shard_b = self.shards[shard_id_b as usize].load_signal();

        match shard_a.0.cmp(&shard_b.0) {
            std::cmp::Ordering::Less => shard_id_a,
            std::cmp::Ordering::Greater => shard_id_b,
            std::cmp::Ordering::Equal => if shard_a.1 >= shard_b.1 { shard_id_a } else { shard_id_b },
        }
    }

    pub fn spawn_on(&self, shard_id: ShardID, location: EntityLocation) -> Result<Entity, SpawnError> {
        if (shard_id as usize) >= self.shard_count() {
            return Err(
                SpawnError::ShardBounds(
                    ShardBoundsError {
                        index: shard_id,
                        max_index: (self.shard_count() - 1) as u32,
                    }
                )
            );
        }
        
        let mut entities = self.shards[shard_id as usize].entities.lock().unwrap();
        let before = entities.free_store.len();
        let entity = entities.spawn(shard_id, location)?;
        
        self.shards[shard_id as usize]
            .live_entity_count
            .fetch_add(1, Ordering::Relaxed);
        
        if before > 0 {
            self.shards[shard_id as usize]
                .approximate_free_store_length
                .fetch_sub(1, Ordering::Relaxed);
        } else {
            let after = entities.free_store.len();
            let added_slots = after;
            self.shards[shard_id as usize]
                .approximate_free_store_length
                .fetch_add(added_slots, Ordering::Relaxed);
}

        Ok(entity)
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        let shard_id = entity.shard() as usize;
        if shard_id >= self.shard_count() { return false; }
        let entities = self.shards[shard_id].entities.lock().unwrap();
        entities.is_alive(entity)
    }

    pub fn get_location(&self, entity: Entity) -> Option<EntityLocation> {
        let shard_id = entity.shard() as usize;
        if shard_id >= self.shard_count(){ return None; }
        let entities = self.shards[shard_id].entities.lock().unwrap();
        entities.get_location(entity)
    }

    pub fn set_location(&self, entity: Entity, location: EntityLocation) {
        let shard_id = entity.shard() as usize;
        if shard_id >= self.shard_count() { return; }
        let mut entities = self.shards[shard_id].entities.lock().unwrap();
        entities.set_location(entity, location);
    }

    pub fn despawn(&self, entity: Entity) -> bool {
        let shard_id = entity.shard() as usize;
        if shard_id >= self.shard_count() { return false; }
        let mut entities = self.shards[shard_id as usize].entities.lock().unwrap();
        if entities.despawn(entity) {
            self.shards[shard_id]
                .approximate_free_store_length
                .fetch_add(1, Ordering::Relaxed);

            self.shards[shard_id]
                .live_entity_count
                .fetch_sub(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}
