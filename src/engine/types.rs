use std::any::Any;
use std::collections::HashMap;


pub type Bits = u8;

pub type EntityID = u64;
pub type ShardID = u16;
pub type IndexID = u32;
pub type VersionID = u32;
pub type EntityCount = u32;

pub const ENTITY_BITS: Bits = 64;
pub const SHARD_BITS: Bits = 10;
pub const VERSION_BITS: Bits = 32;
pub const INDEX_BITS: Bits = ENTITY_BITS - SHARD_BITS - VERSION_BITS;

const _: [(); 1] = [(); (VERSION_BITS + SHARD_BITS < ENTITY_BITS) as usize];
const _: [(); 1] = [(); (INDEX_BITS > 0) as usize];
const _: [(); 1] = [(); (INDEX_BITS < ENTITY_BITS) as usize];
const _: [(); 1] = [(); (SHARD_BITS < ENTITY_BITS) as usize];

const fn mask(bits: Bits) -> EntityID {
    if bits == 0 { 0 } else { ((1 as EntityID) << bits) - 1 }
}

pub const INDEX_MASK: EntityID = mask(INDEX_BITS);
pub const SHARD_MASK: EntityID = mask(SHARD_BITS);
pub const INDEX_CAP: IndexID = (INDEX_MASK as IndexID);

pub type ArchetypeID = u16;
pub type RowID = u32;
pub type ChunkID = u16;

pub const CHUNK_CAP: usize = 16_384;

pub type ComponentID = u16;

pub const COMPONENT_CAP: usize = 4096;
pub const SIGNATURE_SIZE: usize = (COMPONENT_CAP + 63) / 64;


#[derive(Clone, Copy, Debug, Default)]
pub struct Signature {
    pub components: [u64; SIGNATURE_SIZE],
}

impl Signature {
    #[inline]
    pub fn set(&mut self, component_id: ComponentID) {
        let index = (component_id as usize) / 64;
        let bits = (component_id as usize) % 64;
        self.components[index] |= 1u64 << bits;
    }

    #[inline]
    pub fn clear(&mut self, component_id: ComponentID) {
        let index = (component_id as usize) / 64;
        let bits = (component_id as usize) % 64;
        self.components[index] &= !(1u64 << bits);
    }

    #[inline]
    pub fn has(&self, component_id: ComponentID) -> bool {
        let index = (component_id as usize) / 64;
        let bits = (component_id as usize) % 64;
        (self.components[index] >> bits) & 1 == 1
    }

    #[inline]
    pub fn contains_all(&self, signature: &Signature) -> bool {
        for (component_a, component_b) in self.components.iter().zip(signature.components.iter()) {
            if (component_a & component_b) != *component_b { return false; }
        }
        true
    }

    pub fn iterate_over_components(&self) -> impl Iterator<Item = ComponentID> + '_ {
        self.components
            .iter()
            .enumerate()
            .flat_map(|(word_index, &word)| {
                let base = word_index * 64;
                let mut bits = word;
                std::iter::from_fn(move || {
                    if bits == 0 {
                        return None;
                    }
                    let tz = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    Some((base + tz) as ComponentID)
                })
            })
    }
}

pub fn build_signature(component_ids: &[ComponentID]) -> Signature {
    let mut signature = Signature::default();
    for &component_id in component_ids { signature.set(component_id); }
    signature
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QuerySignature {
    pub read: Signature,
    pub write: Signature,
    pub without: Signature,
}

impl QuerySignature {
    pub fn requires_all(&self, archetype_signature: &Signature) -> bool {
        archetype_signature.contains_all(&self.read)
            && archetype_signature.contains_all(&self.write)
            && archetype_signature
                .components
                .iter()
                .zip(self.without.components.iter())
                .all(|(arch_word, without_word)| (arch_word & without_word) == 0)
    }
}

pub fn set_read<T: 'static + Send + Sync>(signature: &mut QuerySignature) {
    signature.read.set(component_id_of::<T>());
}
pub fn set_write<T: 'static + Send + Sync>(signature: &mut QuerySignature) {
    signature.write.set(component_id_of::<T>());
}
pub fn set_without<T: 'static + Send + Sync>(signature: &mut QuerySignature) {
    signature.without.set(component_id_of::<T>());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessMode { Read, Write }

pub trait DynamicBundle {
    fn take(&mut self, component_id: ComponentID) -> Option<Box<dyn Any>>;
}

pub struct Bundle {
    values: Vec<Option<Box<dyn Any>>>,
}

impl Bundle {
    pub fn with_len(length: usize) -> Self {
        Self { values: vec![None; length] }
    }

    #[inline]
    pub fn clear(&mut self) {
        for value in &mut self.values { *value = None; }
    }

    #[inline]
    pub fn insert<T: Any>(&mut self, component_id: ComponentID, value: T) {
        self.values[component_id as usize] = Some(Box::new(value));
    }

    #[inline]
    pub fn extend_from_iter<T: Any, I: IntoIterator<Item = (ComponentID, T)>>(&mut self, iter: I) {
        for (component_id, value) in iter {
            self.insert(component_id, value);
        }
    }

    #[inline]
    pub fn is_complete_for(&self, required: &[bool]) -> bool {
        required.iter().enumerate().all(|(i, req)| !*req || self.values[i].is_some())
    }
}

impl DynamicBundle for Bundle {
    #[inline]
    fn take(&mut self, component_id: ComponentID) -> Option<Box<dyn Any>> {
        self.values.get_mut(component_id as usize).and_then(|slot| slot.take())
    }
}

