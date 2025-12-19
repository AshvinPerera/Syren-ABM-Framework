//! Core ECS Types, Identifiers, and Bit-Level Layouts
//!
//! This module defines the **fundamental types, identifiers, bit layouts, and
//! signatures** used throughout the ECS engine. These definitions form the
//! *semantic backbone* of the system and are shared across all subsystems,
//! including entity management, archetypes, queries, scheduling, and systems.
//!
//! ## Design Philosophy
//!
//! The ECS is designed around:
//!
//! - **Dense storage**
//! - **Bitset-based signatures**
//! - **Stable numeric identifiers**
//! - **Explicit access declaration**
//!
//! To support these goals efficiently, this module:
//!
//! - Encodes entities into a single 64-bit value,
//! - Represents component sets as fixed-size bit arrays,
//! - Uses small, copyable numeric IDs for all ECS concepts,
//! - Avoids heap allocation in hot paths.
//!
//! ## Entity Representation
//!
//! Entities are encoded as a packed 64-bit integer with the following layout:
//!
//! ```text
//! | version | shard | index |
//! ```
//!
//! - **Index** identifies the slot within a shard.
//! - **Shard** allows scalable partitioning for allocation and concurrency.
//! - **Version** enables stale-entity detection after despawning.
//!
//! The exact bit widths are controlled by compile-time constants and validated
//! using static assertions.
//!
//! ## Archetypes and Components
//!
//! Components are identified by compact [`ComponentID`] values. Archetypes are
//! described by [`Signature`] bitsets indicating which components they contain.
//!
//! Component signatures:
//!
//! - are fixed-size arrays of `u64`,
//! - support fast bitwise comparison,
//! - allow efficient iteration over set bits,
//! - are used for both archetype identity and query matching.
//!
//! ## Queries and Access Control
//!
//! Query execution relies on two related concepts:
//!
//! - [`QuerySignature`] — describes *what components* a query requires,
//! - [`AccessSets`] — describes *how components are accessed* (read/write).
//!
//! These structures enable:
//!
//! - fast archetype filtering,
//! - safe parallel scheduling,
//! - deterministic conflict detection between systems.
//!
//! ## Bundles
//!
//! The [`Bundle`] and [`DynamicBundle`] abstractions provide a type-erased way
//! to group heterogeneous component values together, typically used during
//! spawning or structural changes.
//!
//! Bundles trade compile-time typing for flexibility while remaining isolated
//! from hot iteration paths.
//!
//! ## Safety and Performance
//!
//! This module contains **no unsafe code**, but many of its types are used at
//! unsafe boundaries elsewhere in the engine. Correctness here is therefore
//! critical to overall ECS soundness.
//!
//! All constants, bit widths, and capacities are chosen to:
//!
//! - fit within cache-friendly data structures,
//! - allow fast bitwise operations,
//! - minimize memory overhead,
//! - support large-scale simulations.
//!
//! ## Intended Audience
//!
//! This module is primarily intended for:
//!
//! - ECS internals,
//! - system schedulers,
//! - query planners,
//! - archetype and storage layers.

use std::any::Any;
use crate::engine::component::{component_id_of};


/// Bit-width type used for compile-time layout calculations.
pub type Bits = u8;

/// Globally unique entity identifier encoded as a packed 64-bit value.
pub type EntityID = u64;
/// Identifier for an entity allocation shard.
pub type ShardID = u16;
/// Index within a shard.
pub type IndexID = u32;
/// Generation counter used to detect stale entities.
pub type VersionID = u32;
/// Count of live entities.
pub type EntityCount = u32;

/// Unique identifier for a system.
pub type SystemID = u16;
/// Simulation tick counter.
pub type Tick = u64;

/// Total number of bits in an [`EntityID`].
pub const ENTITY_BITS: Bits = 64;
/// Number of bits reserved for shard identification.
pub const SHARD_BITS: Bits = 10;
/// Number of bits reserved for entity versioning.
pub const VERSION_BITS: Bits = 32;
/// Number of bits reserved for entity index within a shard.
pub const INDEX_BITS: Bits = ENTITY_BITS - SHARD_BITS - VERSION_BITS;

const _: [(); 1] = [(); (VERSION_BITS + SHARD_BITS < ENTITY_BITS) as usize];
const _: [(); 1] = [(); (INDEX_BITS > 0) as usize];
const _: [(); 1] = [(); (INDEX_BITS < ENTITY_BITS) as usize];
const _: [(); 1] = [(); (SHARD_BITS < ENTITY_BITS) as usize];

const fn mask(bits: Bits) -> EntityID {
    if bits == 0 { 0 } else { ((1 as EntityID) << bits) - 1 }
}

/// Mask selecting the index portion of an [`EntityID`].
pub const INDEX_MASK: EntityID = mask(INDEX_BITS);
/// Mask selecting the shard portion of an [`EntityID`].
pub const SHARD_MASK: EntityID = mask(SHARD_BITS);
/// Maximum number of indices per shard.
pub const INDEX_CAP: IndexID = INDEX_MASK as IndexID;

/// Unique identifier for an archetype.
pub type ArchetypeID = u16;
/// Row index within a chunk.
pub type RowID = u32;
/// Chunk index within an archetype.
pub type ChunkID = u16;

/// Maximum number of rows per chunk.
pub const CHUNK_CAP: usize = 16_384;

/// Unique identifier for a component type.
pub type ComponentID = u16;

/// Maximum number of registered component types.
pub const COMPONENT_CAP: usize = 4096;
/// Number of `u64` words required to represent a full component signature.
pub const SIGNATURE_SIZE: usize = (COMPONENT_CAP + 63) / 64;

/// Bitset representing a set of components.
#[derive(Clone, Copy, Debug)]
pub struct Signature {
    /// Packed component bitset.
    pub components: [u64; SIGNATURE_SIZE],
}

impl Default for Signature {
    fn default() -> Self {
        Self {
            components: [0u64; SIGNATURE_SIZE],
        }
    }
}

impl Signature {
    /// Sets the bit corresponding to `component_id`.
    #[inline]
    pub fn set(&mut self, component_id: ComponentID) {
        let index = (component_id as usize) / 64;
        let bits = (component_id as usize) % 64;
        self.components[index] |= 1u64 << bits;
    }

    /// Clears the bit corresponding to `component_id`.
    #[inline]
    pub fn clear(&mut self, component_id: ComponentID) {
        let index = (component_id as usize) / 64;
        let bits = (component_id as usize) % 64;
        self.components[index] &= !(1u64 << bits);
    }

    /// Returns `true` if `component_id` is present in this signature.
    #[inline]
    pub fn has(&self, component_id: ComponentID) -> bool {
        let index = (component_id as usize) / 64;
        let bits = (component_id as usize) % 64;
        (self.components[index] >> bits) & 1 == 1
    }

    /// Returns `true` if all components in `signature` are present.
    #[inline]
    pub fn contains_all(&self, signature: &Signature) -> bool {
        for (component_a, component_b) in self.components.iter().zip(signature.components.iter()) {
            if (component_a & component_b) != *component_b { return false; }
        }
        true
    }

    /// Iterates over all component IDs set in this signature.
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

/// Builds a component signature from a list of component IDs.
pub fn build_signature(component_ids: &[ComponentID]) -> Signature {
    let mut signature = Signature::default();
    for &component_id in component_ids { signature.set(component_id); }
    signature
}

/// Iterates over component IDs set in a raw signature word array.
#[inline]
pub fn iter_bits_from_words<'a>(
    words: &'a [u64; SIGNATURE_SIZE],
) -> impl Iterator<Item = ComponentID> + 'a {
    words
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

/// Component signature used for query matching.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuerySignature {
    /// Components read by the query.
    pub read: Signature,

    /// Components written by the query.
    pub write: Signature,

    /// Components explicitly excluded from the query.
    pub without: Signature,
}

impl QuerySignature {
    /// Returns `true` if an archetype satisfies this query.
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

/// Marks a component type as read-only in a query signature.
pub fn set_read<T: 'static + Send + Sync>(signature: &mut QuerySignature) {
    signature.read.set(component_id_of::<T>());
}

/// Marks a component type as writable in a query signature.
pub fn set_write<T: 'static + Send + Sync>(signature: &mut QuerySignature) {
    signature.write.set(component_id_of::<T>());
}

/// Excludes a component type from a query signature.
pub fn set_without<T: 'static + Send + Sync>(signature: &mut QuerySignature) {
    signature.without.set(component_id_of::<T>());
}

/// Access mode for a component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessMode { 
    /// Read-only access.
    Read, 
    /// Exclusive write access.
    Write 
}

/// Declares the component access set of a system.
#[derive(Clone, Debug, Default)]
pub struct AccessSets {
    /// Components read by the system.
    pub read: Signature,
    /// Components written by the system.
    pub write: Signature,
}

impl AccessSets {
    /// Returns `true` if this access set conflicts with another.
    #[inline]
    pub fn conflicts_with(&self, other: &AccessSets) -> bool {
        // Conflicts if: (W ∩ W) or (W ∩ R) or (R ∩ W)
        let mut w_and_w = false;
        let mut w_and_r = false;
        let mut r_and_w = false;

        for ((a_w, a_r), (b_w, b_r)) in self.write.components.iter().zip(self.read.components.iter())
            .zip(other.write.components.iter().zip(other.read.components.iter()))
        {
            if (a_w & b_w) != 0 { w_and_w = true; }
            if (a_w & b_r) != 0 { w_and_r = true; }
            if (a_r & b_w) != 0 { r_and_w = true; }
            if w_and_w || w_and_r || r_and_w { return true; }
        }
        false
    }
}

/// Type-erased container for component values.
pub trait DynamicBundle {
    /// Removes and returns the value for `component_id`, if present.
    fn take(&mut self, component_id: ComponentID) -> Option<Box<dyn Any + Send>>;
}

/// Concrete implementation of a dynamic component bundle.
pub struct Bundle {
    /// Component presence signature
    signature: Signature,
    /// Sparse storage of component values
    values: Vec<(ComponentID, Box<dyn Any + Send>)>,
}

impl Bundle {
    /// Creates an empty bundle.
    #[inline]
    pub fn new() -> Self {
        Self {
            signature: Signature::default(),
            values: Vec::new(),
        }
    }

    /// Clears all stored component values.
    #[inline]
    pub fn clear(&mut self) {
        self.signature = Signature::default();
        self.values.clear();
    }

    /// Inserts a component value into the bundle.
    #[inline]
    pub fn insert<T: Any + Send>(&mut self, component_id: ComponentID, value: T) {
        self.signature.set(component_id);
        self.values.push((component_id, Box::new(value)));
    }

    /// Inserts multiple component values from an iterator.
    #[inline]
    pub fn extend_from_iter<T: Any + Send, I: IntoIterator<Item = (ComponentID, T)>>(
        &mut self,
        iter: I,
    ) {
        for (component_id, value) in iter {
            self.insert(component_id, value);
        }
    }

    /// Returns `true` if all required components are present.
    #[inline]
    pub fn is_complete_for(&self, required: &[bool]) -> bool {
        required
            .iter()
            .enumerate()
            .all(|(i, req)| !*req || self.signature.has(i as ComponentID))
    }

    /// Builds a signature representing the components present in this bundle.
    #[inline]
    pub fn signature(&self) -> Signature {
        self.signature
    }
}

impl DynamicBundle for Bundle {
    #[inline]
    fn take(&mut self, component_id: ComponentID) -> Option<Box<dyn Any + Send>> {
        let index = self
            .values
            .iter()
            .position(|(cid, _)| *cid == component_id)?;

        let (_, value) = self.values.swap_remove(index);
        Some(value)
    }
}
