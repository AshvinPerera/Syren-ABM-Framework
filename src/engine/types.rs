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
//! - minimise memory overhead,
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
    if bits == 0 {
        0
    } else {
        ((1 as EntityID) << bits) - 1
    }
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

/// Compact identifier for an agent template registered in an agent registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AgentTemplateId(pub u32);

/// Maximum number of registered component types.
///
/// This value controls the size of [`Signature`] bitsets: each 256-component
/// capacity requires 4 `u64` words (32 bytes) per signature. If a simulation
/// requires more than 256 distinct component types, this constant can be
/// increased in multiples of 64. Each additional 64 components adds one `u64`
/// word (8 bytes) to every signature, so increases should be made deliberately.
pub const COMPONENT_CAP: usize = 256;
/// Number of `u64` words required to represent a full component signature.
pub const SIGNATURE_SIZE: usize = COMPONENT_CAP.div_ceil(64);

/// Opaque identifier for a non-component scheduling channel.
///
/// Channels are allocated by extension modules (messaging, environment) and
/// recorded in `AccessSets::produces` / `AccessSets::consumes`. The engine
/// treats them as opaque bitset indices; it does not interpret their meaning.
///
/// Assigned by [`ChannelAllocator`](crate::engine::channel_allocator::ChannelAllocator).
/// One allocator exists per `Model`, shared by messaging and environment so
/// both live in the same `u32` ID space and the scheduler can reason about
/// them uniformly.
pub type ChannelID = u32;

/// Opaque identifier for a boundary resource registered on
/// [`ECSManager`](crate::engine::manager::ECSManager).
///
/// Returned by `ECSManager::register_boundary` and used by systems to retrieve
/// a typed reference to the resource via `ECSReference::boundary::<R>(id)`.
pub type BoundaryID = u32;

/// Unique identifier for a GPU resource.
#[cfg(feature = "gpu")]
pub type GPUResourceID = u16;

/// Declares how a component buffer is accessed during GPU execution.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPUAccessMode {
    /// Read-only access to a component buffer.
    Read,
    /// Read-write (mutable) access to a component buffer.
    Write,
}
