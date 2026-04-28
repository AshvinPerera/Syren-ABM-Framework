//! Archetype storage and query-matching primitives for the ECS engine.
//!
//! An [`Archetype`] groups entities that share an identical component
//! [`Signature`]. Component data is stored column-major in chunked, densely
//! packed buffers, enabling fast iteration and cache-friendly access during
//! system execution.
//!
//! ## Organization
//!
//! - [`ArchetypeMeta`] - lightweight metadata (entity count, per-chunk entity
//!   position maps) guarded by its own `RwLock`.
//! - [`Archetype`] - owns the component columns and exposes spawn, move, and
//!   query operations.
//! Query matching returns cached archetype IDs from `ECSData`; chunk lengths
//! stay live and are read from each archetype during iteration.
//!
//! ## Storage layout
//!
//! Component columns are held in a sorted `Vec<(ComponentID, LockedAttribute)>`,
//! ordered by ascending `ComponentID`. Lookups use `binary_search_by_key` for
//! O(log n) access (n = component count per archetype, typically small).
//! Entities are densely packed with swap-remove semantics; physical chunk
//! capacity may exceed logical row count.
//!
//! ## Concurrency and lock ordering
//!
//! Each component column is independently protected by a `RwLock` inside
//! [`LockedAttribute`]. Archetype metadata is protected by a separate `RwLock`
//! on [`ArchetypeMeta`]. When both kinds of lock must be held simultaneously,
//! **column locks must be acquired before the metadata lock**, and always in
//! ascending `ComponentID` order. The sorted `components` vec naturally
//! enforces this ordering during iteration.

use std::sync::{RwLock, RwLockWriteGuard};

use crate::engine::types::{ArchetypeID, ComponentID, CHUNK_CAP};

use crate::engine::storage::{LockedAttribute, TypeErasedAttribute};

use crate::engine::component::{iter_bits_from_words, ComponentRegistry, Signature};

use crate::engine::error::{ECSError, ECSResult, InternalViolation, MoveError, SpawnError};

use crate::engine::entity::Entity;

// ---------------------------------------------------------------------------
// Internal metadata
// ---------------------------------------------------------------------------

pub(super) struct ArchetypeMeta {
    pub(super) length: usize,
    pub(super) entity_positions: Vec<Vec<Option<Entity>>>,
}

// ---------------------------------------------------------------------------
// Archetype
// ---------------------------------------------------------------------------

/// Stores entities that share an identical component signature.
///
/// ## Purpose
/// An `Archetype` owns columnar component storage for a fixed set of component
/// types and maintains dense, chunked layouts for fast iteration and mutation.
///
/// ## Design
/// - Component data is stored column-major by component type.
/// - Entities are densely packed using swap-remove semantics.
/// - Entity locations are tracked explicitly for fast lookup.
///
/// ## Storage layout
///
/// Component columns are stored in a **sparse, sorted vector** of
/// `(ComponentID, LockedAttribute)` pairs, kept sorted by `ComponentID`.
///
/// All lookups use `binary_search_by_key` for O(log n) access where n is
/// the number of components in this archetype (typically very small).
///
/// ## Invariants
/// - All component columns have identical row counts.
/// - `entity_positions` is kept consistent with component storage.
/// - Signature bits exactly reflect allocated component attributes.
/// - `components` is always sorted by `ComponentID` in ascending order.
/// - All component access during system execution must go through
///   `LockedAttribute` read/write guards.
/// - Component *data storage* is never accessed without holding the
///   corresponding lock.
/// - Archetype metadata (`entity_positions`, `length`) is synchronized
///   independently via the archetype metadata lock.
///
/// ## Lock ordering
/// See module-level documentation. Column locks must be acquired *before*
/// the metadata lock, and always in ascending `ComponentID` order.
/// The sorted `components` vec naturally enforces ascending order when
/// iterating, which simplifies lock-ordering compliance.

pub struct Archetype {
    pub(super) archetype_id: ArchetypeID,

    /// Sparse, sorted component storage.
    /// Kept sorted by `ComponentID` to enable `binary_search_by_key` lookups
    /// and naturally ascending lock order during iteration.
    pub(super) components: Vec<(ComponentID, LockedAttribute)>,

    pub(super) signature: Signature,
    pub(super) meta: RwLock<ArchetypeMeta>,
}

impl Archetype {
    /// Creates a new empty `Archetype` with the given identifier.
    ///
    /// ## Purpose
    /// Initializes component column storage, the signature bitset, entity tracking
    /// buffers, and internal counters.
    ///
    /// ## Behaviour
    /// - Allocates component slots only for components present in the signature.
    /// - Initializes an empty `Signature`.
    /// - No component columns are allocated until explicitly inserted.
    ///
    /// ## Parameters
    /// - `registry`: The component registry used to construct empty component
    ///   storage for each component in the signature. Passing the registry
    ///   directly avoids global state and supports multi-world usage.
    ///
    /// ## Invariants
    /// The archetype contains no entities upon creation.
    /// The `components` vec is sorted by `ComponentID`.

    pub fn new(
        archetype_id: ArchetypeID,
        signature: Signature,
        registry: &ComponentRegistry,
    ) -> ECSResult<Self> {
        let mut archetype = Self {
            archetype_id,
            components: Vec::new(),
            signature: Signature::default(),
            meta: RwLock::new(ArchetypeMeta {
                length: 0,
                entity_positions: Vec::new(),
            }),
        };

        for component_id in iter_bits_from_words(&signature.components) {
            let component = registry
                .make_empty_component(component_id)
                .map_err(ECSError::from)?;
            archetype
                .components
                .push((component_id, LockedAttribute::new(component)));
            archetype.signature.set(component_id);
        }

        archetype.components.sort_unstable_by_key(|(cid, _)| *cid);

        Ok(archetype)
    }

    /// Returns the number of active entities stored in the archetype.
    ///
    /// ## Notes
    /// This reflects logical count only; physical chunk storage may contain unused rows.

    pub fn length(&self) -> ECSResult<usize> {
        Ok(self
            .meta
            .read()
            .map_err(|_| ECSError::from(InternalViolation::ArchetypeMetaLockPoisoned))?
            .length)
    }

    /// Returns the `ArchetypeID` associated with this archetype.
    ///
    /// ## Notes
    /// This value is stable for the lifetime of the archetype.

    pub fn archetype_id(&self) -> ArchetypeID {
        self.archetype_id
    }

    /// Returns a reference to the archetype's signature.
    ///
    /// ## Notes
    /// Used by query and filtering logic.

    pub fn signature(&self) -> &Signature {
        &self.signature
    }

    /// Returns `true` if this archetype contains all components described in `need`.
    ///
    /// ## Notes
    /// This performs a subset check using signature bits.

    pub fn matches_all(&self, need: &Signature) -> bool {
        self.signature.contains_all(need)
    }

    /// Returns `true` if the archetype contains the specified component.
    ///
    /// ## Notes
    /// This checks the signature only; does not inspect the attribute buffer.

    #[inline]
    pub fn has(&self, component_id: ComponentID) -> bool {
        self.signature.has(component_id)
    }

    /// Ensures that `entity_positions` contains at least `chunk_count` chunks.
    ///
    /// ## Purpose
    /// Expands chunk metadata storage to match component column allocations.
    ///
    /// ## Invariants
    /// - Each added chunk contains exactly `CHUNK_CAP` rows.
    /// - Does not allocate component data; only entity metadata.

    pub(super) fn ensure_capacity(meta: &mut ArchetypeMeta, chunk_count: usize) {
        while meta.entity_positions.len() < chunk_count {
            meta.entity_positions.push(vec![None; CHUNK_CAP]);
        }
    }

    // -----------------------------------------------------------------------
    // Sparse component lookup helpers
    // -----------------------------------------------------------------------

    /// Returns a reference to the `LockedAttribute` for `component_id`,
    /// or `None` if this archetype does not contain that component.
    ///
    /// Uses `binary_search_by_key` on the sorted `components` vec.
    #[inline]
    pub(super) fn find_component(&self, component_id: ComponentID) -> Option<&LockedAttribute> {
        self.components
            .binary_search_by_key(&component_id, |(cid, _)| *cid)
            .ok()
            .map(|idx| &self.components[idx].1)
    }

    /// Returns the locked attribute wrapper for a component.
    #[inline]
    pub fn component_locked(&self, component_id: ComponentID) -> Option<&LockedAttribute> {
        self.find_component(component_id)
    }

    /// Returns component column lengths for internal regression tests.
    #[cfg(test)]
    pub(crate) fn component_lengths_for_test(&self) -> Vec<(ComponentID, usize)> {
        self.components
            .iter()
            .map(|(component_id, column)| {
                let len = column
                    .read()
                    .map(|guard| guard.length())
                    .unwrap_or(usize::MAX);
                (*component_id, len)
            })
            .collect()
    }

    #[inline]
    pub(super) fn lock_write_spawn<'a>(
        attr: &'a LockedAttribute,
    ) -> Result<RwLockWriteGuard<'a, Box<dyn TypeErasedAttribute>>, SpawnError> {
        attr.write().map_err(SpawnError::StoragePushFailedWith)
    }

    #[inline]
    pub(super) fn lock_write_move<'a>(
        attr: &'a LockedAttribute,
        component_id: ComponentID,
    ) -> Result<RwLockWriteGuard<'a, Box<dyn TypeErasedAttribute>>, MoveError> {
        attr.write().map_err(|e| MoveError::PushFromFailed {
            component_id,
            source_error: e,
        })
    }

    /// Computes how many chunks are required to store all active rows.
    ///
    /// ## Behaviour
    /// - Returns `0` if no entities exist.
    /// - Otherwise computes `(length - 1) / CHUNK_CAP + 1`.

    pub fn chunk_count(&self) -> ECSResult<usize> {
        let len = self.length()?;
        if len == 0 {
            Ok(0)
        } else {
            Ok(((len - 1) / CHUNK_CAP) + 1)
        }
    }

    /// Returns the number of valid rows in the specified chunk.
    ///
    /// ## Behaviour
    /// - Returns `0` if the chunk is unused.
    /// - Returns `CHUNK_CAP` for fully populated chunks.
    /// - Returns remaining entity count for the final partial chunk.
    ///
    /// ## Invariants
    /// Must reflect row count across all component attributes.

    pub fn chunk_valid_length(&self, chunk_index: usize) -> ECSResult<usize> {
        let max_chunk = self.chunk_count()?.saturating_sub(1);
        if chunk_index > max_chunk {
            return Ok(0);
        }

        if chunk_index < max_chunk {
            Ok(CHUNK_CAP)
        } else {
            let len = self.length()?;
            let used = len % CHUNK_CAP;
            Ok(if used == 0 { CHUNK_CAP } else { used })
        }
    }
}
