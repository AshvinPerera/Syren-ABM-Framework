//! Core ECS Types, Identifiers, and Bit-Level Layouts
//!
//! This module defines the **fundamental types, identifiers, and bit layouts
//! used throughout the ECS engine. These definitions form the
//! *semantic backbone* of the system and are shared across all subsystems,
//! including entity management, archetypes, queries, scheduling, and systems.
//!
//! ## Design Philosophy
//!
//! The ECS is designed around **Dense storage**
//!
//! To support this goal efficiently, this module:
//!
//! - Encodes entities into a single 64-bit value,
//! - Represents component sets as fixed-size bit arrays,
//! - Uses small, copyable numeric IDs for all ECS concepts.
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
//! ## Safety and Performance
//!
//! This module contains **no unsafe code**, but many of its types are used at
//! unsafe boundaries elsewhere in the engine.
//!
//! All constants, bit widths, and capacities are chosen to:
//!
//! - fit within cache-friendly data structures,
//! - allow fast bitwise operations,
//! - minimize memory overhead,
//! - support large-scale simulations.


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
