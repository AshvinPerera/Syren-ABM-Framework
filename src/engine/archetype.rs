//! # ECS world management and execution layer
//!
//! This module defines the central orchestration layer of the ECS, responsible
//! for:
//!
//! * owning archetypes and their component storage,
//! * coordinating entity movement between archetypes,
//! * managing deferred structural mutations via commands,
//! * providing controlled parallel access to component data,
//! * executing chunk-based parallel iteration.
//!
//! ## Concurrency model
//!
//! The ECS uses **fine-grained, column-level synchronization** to support
//! safe parallel execution:
//!
//! * Each component column inside an archetype is protected by an `RwLock`.
//! * Parallel systems may:
//!   * read the same component concurrently,
//!   * write disjoint component sets concurrently,
//!   * read while others read.
//! * Writes to the same component column are mutually exclusive.
//!
//! Structural mutations (spawn, despawn, archetype migration) **must not**
//! occur during parallel iteration and must be executed at explicit
//! synchronization points.
//!
//! This constraint is enforced by execution phase discipline. 
//! Violating it may result in deadlock or panic.
//!
//! ## Safety model
//!
//! * Component data access is synchronized via column-level locks.
//! * Raw pointers are only exposed after acquiring the appropriate locks.
//! * Parallel iteration relies on chunk-level disjointness guaranteed by
//!   archetype layout.
//!
//! ## Unsafe code
//!
//! This module contains `unsafe` code for:
//!
//! * converting locked component storage into raw byte slices,
//! * parallel execution using Rayon,
//! * low-level performance optimizations.


use std::any::Any;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::engine::types::{
    ArchetypeID, 
    ShardID,
    ChunkID,
    RowID, 
    CHUNK_CAP, 
    ComponentID, 
    COMPONENT_CAP,
    SIGNATURE_SIZE
};

use crate::engine::storage::{
    TypeErasedAttribute,
    LockedAttribute
};

use crate::engine::entity::{
    Entity, 
    EntityLocation, 
    EntityShards
};

use crate::engine::component::{ 
    Signature,
    DynamicBundle,
    iter_bits_from_words,
    component_id_of_type_id,
    get_component_storage_factory
};

use crate::engine::error::{
    SpawnError,
    MoveError,
    ExecutionError,
    ECSError,
    ECSResult,
    RegistryError
};


/// Represents a temporary borrow of a single archetype chunk for system execution.
///
/// ## Safety
/// This type *maintains* column-level synchronization by holding the underlying
/// `RwLock` guards for all accessed component columns for the duration of the borrow.
///
/// While a `ChunkBorrow` exists:
/// - Any component in `read_guards` is read-locked (shared).
/// - Any component in `write_guards` is write-locked (exclusive).
/// - Other systems may read the same components but may not write them.
/// - Structural archetype mutation must not occur concurrently.
///
/// This type does **not** prevent misuse such as:
/// - retaining raw pointers beyond the borrow lifetime,
/// - performing structural ECS mutations in parallel.
///
/// Violating these constraints may cause deadlock or panic.

pub struct ChunkBorrow<'a> {
    /// Number of valid rows in the borrowed chunk.
    pub length: usize,
    /// Immutable component views as `(ptr, byte_len)` pairs.
    pub reads: Vec<(*const u8, usize)>,
    /// Mutable component data pointers for write access.
    pub writes: Vec<*mut u8>,

    /// Holds the read locks alive for the lifetime of this borrow.
    _read_guards: Vec<(ComponentID, RwLockReadGuard<'a, Box<dyn TypeErasedAttribute>>)>,
    /// Holds the write locks alive for the lifetime of this borrow.
    _write_guards: Vec<(ComponentID, RwLockWriteGuard<'a, Box<dyn TypeErasedAttribute>>)>,
}

struct ArchetypeMeta {
    length: usize,
    entity_positions: Vec<Vec<Option<Entity>>>,
}

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
/// ## Invariants
/// - All component columns have identical row counts.
/// - `entity_positions` is kept consistent with component storage.
/// - Signature bits exactly reflect allocated component attributes.
/// - All component access during system execution must go through
///   `LockedAttribute` read/write guards.
/// - Component *data storage* is never accessed without holding the
///   corresponding lock.
/// - Archetype metadata (`entity_positions`, `length`) is synchronized
///   independently via the archetype metadata lock.

pub struct Archetype {
    archetype_id: ArchetypeID,
    components: Vec<Option<LockedAttribute>>,
    signature: Signature,
    meta: RwLock<ArchetypeMeta>,
}

impl Archetype {

    /// Creates a new empty `Archetype` with the given identifier.
    ///
    /// ## Purpose
    /// Initializes component column storage, the signature bitset, entity tracking
    /// buffers, and internal counters.
    ///
    /// ## Behavior
    /// - Allocates `COMPONENT_CAP` component slots, all initially empty.
    /// - Initializes an empty `Signature`.
    /// - No component columns are allocated until explicitly inserted.
    ///
    /// ## Invariants
    /// The archetype contains no entities upon creation.

    pub fn new(archetype_id: ArchetypeID, signature: Signature) -> ECSResult<Self> {
        let mut archetype = Self {
            archetype_id,
            components: (0..COMPONENT_CAP).map(|_| None).collect(),
            signature: Signature::default(),
            meta: RwLock::new(ArchetypeMeta {
                length: 0,
                entity_positions: Vec::new(),
            }),
        };

        for component_id in iter_bits_from_words(&signature.components) {
            let component = make_empty_component_for(component_id)?;
            archetype.components[component_id as usize] = Some(LockedAttribute::new(component));
            archetype.signature.set(component_id);
        }

        Ok(archetype)
    }

    /// Returns the number of active entities stored in the archetype.
    ///
    /// ## Notes
    /// This reflects logical count only; physical chunk storage may contain unused rows.

    pub fn length(&self) -> ECSResult<usize> {
        Ok(self.meta.read().map_err(|_| ECSError::Internal("archetype meta lock poisoned".into()))?.length)
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

    pub fn signature(&self) -> &Signature { &self.signature }

    /// Returns `true` if this archetype contains all components described in `need`.
    ///
    /// ## Notes
    /// This performs a subset check using signature bits.

    pub fn matches_all(&self, need: &Signature) -> bool {
        self.signature.contains_all(need)
    }

    /// Ensures that `entity_positions` contains at least `chunk_count` chunks.
    ///
    /// ## Purpose
    /// Expands chunk metadata storage to match component column allocations.
    ///
    /// ## Invariants
    /// - Each added chunk contains exactly `CHUNK_CAP` rows.
    /// - Does not allocate component data; only entity metadata.

    fn ensure_capacity(meta: &mut ArchetypeMeta, chunk_count: usize) {
        while meta.entity_positions.len() < chunk_count {
            meta.entity_positions.push(vec![None; CHUNK_CAP]);
        }
    }

    /// Guarantees that a component attribute exists for the given `component_id`.
    ///
    /// ## Behavior
    /// - Allocates a new column using the provided factory if not already present.
    /// - Marks the component bit in the signature.
    ///
    /// ## Invariants
    /// Attribute allocation and signature must remain consistent.

    #[inline]
    pub fn ensure_component(
        &mut self,
        component_id: ComponentID,
        factory: impl FnOnce() -> Result<Box<dyn TypeErasedAttribute>, RegistryError>,
    ) -> Result<(), SpawnError> {
        let index = component_id as usize;
        if index >= COMPONENT_CAP {
            return Err(SpawnError::InvalidComponentId);
        }

        if self.components[index].is_none() {
            let col = factory()?;
            self.components[index] = Some(LockedAttribute::new(col));
            self.signature.set(component_id);
        }

        Ok(())
    }

    /// Returns `true` if the archetype contains the specified component.
    ///
    /// ## Notes
    /// This checks the signature only; does not inspect the attribute buffer.

    #[inline]
    pub fn has(&self, component_id: ComponentID) -> bool {
        self.signature.has(component_id)
    }

    /// Returns the locked attribute wrapper for a component.
    #[inline]
    pub fn component_locked(&self, component_id: ComponentID) -> Option<&LockedAttribute> {
        self.components
            .get(component_id as usize)
            .and_then(|c| c.as_ref())
    }

    #[inline]
    fn lock_write_spawn<'a>(
        attr: &'a LockedAttribute,
    ) -> Result<RwLockWriteGuard<'a, Box<dyn TypeErasedAttribute>>, SpawnError> {
        attr.write().map_err(SpawnError::StoragePushFailedWith)
    }

    #[inline]
    fn lock_write_move<'a>(
        attr: &'a LockedAttribute,
        component_id: ComponentID,
    ) -> Result<RwLockWriteGuard<'a, Box<dyn TypeErasedAttribute>>, MoveError> {
        attr.write().map_err(|e| MoveError::PushFromFailed { component_id, source_error: e })
    }

    /// Computes how many chunks are required to store all active rows.
    ///
    /// ## Behavior
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
    /// ## Behavior
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

    /// Inserts an empty component attribute into the archetype.
    ///
    /// ## Purpose
    /// Used when constructing archetypes from predefined type lists.
    ///
    /// ## Behavior
    /// - Fails if the index exceeds `COMPONENT_CAP`.
    /// - Assumes the attribute did not previously exist.
    /// - Sets the signature bit for the component.
    ///
    /// ## Invariants
    /// Component attributes must be added only before entities are inserted.

    pub fn insert_empty_component(
        &mut self,
        component_id: ComponentID,
        component: Box<dyn TypeErasedAttribute>,
    ) -> ECSResult<()> {
        let index = component_id as usize;
        if index >= COMPONENT_CAP {
            return Err(SpawnError::InvalidComponentId.into());
        }

        if self.components[index].is_some() {
            return Err(ECSError::Internal("insert_empty_component: component already present".into()));
        }

        self.components[index] = Some(LockedAttribute::new(component));
        self.signature.set(component_id);
        Ok(())
    }

    /// Removes a component attribute from an empty archetype.
    ///
    /// ## Invariants
    /// Removing attributes in a populated archetype would break row alignment.

    pub fn remove_component(
        &mut self,
        component_id: ComponentID,
    ) -> ECSResult<Option<Box<dyn TypeErasedAttribute>>> {
        if self.length()? > 0 {
            return Err(SpawnError::ArchetypeNotEmpty.into());
        }

        let index = component_id as usize;
        if index >= COMPONENT_CAP {
            return Err(SpawnError::InvalidComponentId.into());
        }

        let taken = self.components[index].take();
        if taken.is_some() {
            self.signature.clear(component_id);
        }

        match taken {
            None => Ok(None),
            Some(locked) => Ok(Some(
                locked.into_inner().map_err(|e| SpawnError::StoragePushFailedWith(e))?
            )),
        }
    }

    #[cfg(feature = "rollback")]
    pub fn move_row_across_shared_components(
        &mut self,
        destination: &mut Archetype,
        source_position: (ChunkID, RowID),
        shared_components: Vec<ComponentID>
    ) -> Result<((ChunkID, RowID), Option<(ChunkID, RowID)>, Vec<(ComponentID, Box<dyn Any>)>), MoveError>
    {
        let (source_chunk, source_row) = source_position;
        let mut destination_position: Option<(ChunkID, RowID)> = None;
        let mut swap_information: Option<(ChunkID, RowID)> = None;
        let mut rollback_sequence: Vec<(ComponentID, Box<dyn Any>)> = Vec::new();

        for component_id in shared_components {
            if !self.signature.has(component_id) || !destination.signature.has(component_id) {
                continue;
            }

            let src_attr = self.components[component_id as usize]
                .as_ref()
                .ok_or(MoveError::InconsistentStorage)?;
            let dst_attr = destination.components[component_id as usize]
                .as_ref()
                .ok_or(MoveError::InconsistentStorage)?;

            let mut src_guard = Self::lock_write_move(src_attr, component_id)?;
            let mut dst_guard = Self::lock_write_move(dst_attr, component_id)?;

            let ((dst_chunk, dst_row), moved_from, rollback) =
                match dst_guard.as_mut().push_from_dyn(src_guard.as_mut(), source_chunk, source_row) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = self.rollback_into(destination, rollback_sequence);
                        return Err(MoveError::PushFromFailed { component_id, source_error: e });
                    }
                };

            match destination_position {
                Some(pos) if pos != (dst_chunk, dst_row) => {
                    let _ = self.rollback_into(destination, rollback_sequence);
                    return Err(MoveError::RowMisalignment {
                        expected: pos,
                        got: (dst_chunk, dst_row),
                        component_id,
                    });
                }
                None => destination_position = Some((dst_chunk, dst_row)),
                _ => {}
            }

            if let Some(moved_from_info) = moved_from {
                match swap_information {
                    Some(existing) if existing != moved_from_info => {
                        let _ = self.rollback_into(destination, rollback_sequence);
                        return Err(MoveError::InconsistentSwapInfo);
                    }
                    None => swap_information = Some(moved_from_info),
                    _ => {}
                }
            }

            rollback_sequence.push((component_id, rollback));
        }

        let destination_position = destination_position.ok_or(MoveError::NoComponentsMoved)?;
        Ok((destination_position, swap_information, rollback_sequence))
    }

    /// Moves component data shared between source and destination archetypes.
    ///
    /// ## Purpose
    /// Transfers component rows that exist in both archetypes during an entity
    /// migration, preserving dense storage and row alignment.
    ///
    /// ## Behavior
    /// - Shared components are moved using `push_from_dyn`.
    /// - The first successful move determines the destination `(chunk, row)`.
    /// - All subsequent moves must resolve to the same location.
    /// - Swap-remove behavior is tracked to update entity metadata correctly.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if component columns are missing.
    /// - `PushFromFailed` if backend storage transfer fails.
    /// - `RowMisalignment` if components disagree on row placement.
    /// - `InconsistentSwapInfo` if swap metadata differs between columns.
    /// - `NoComponentsMoved` if no shared components exist.

    #[cfg(not(feature = "rollback"))]
    pub fn move_row_across_shared_components(
        &mut self,
        destination: &mut Archetype,
        source_position: (ChunkID, RowID),
        shared_components: Vec<ComponentID>
    ) -> Result<((ChunkID, RowID), Option<(ChunkID, RowID)>), MoveError>
    {
        let (source_chunk, source_row) = source_position;
        let mut destination_position: Option<(ChunkID, RowID)> = None;
        let mut swap_information: Option<(ChunkID, RowID)> = None;

        for component_id in shared_components {
            if !self.signature.has(component_id) || !destination.signature.has(component_id) {
                continue;
            }

            let src_attr = self.components[component_id as usize]
                .as_ref()
                .ok_or(MoveError::InconsistentStorage)?;
            let dst_attr = destination.components[component_id as usize]
                .as_ref()
                .ok_or(MoveError::InconsistentStorage)?;

            let mut src_guard = Self::lock_write_move(src_attr, component_id)?;
            let mut dst_guard = Self::lock_write_move(dst_attr, component_id)?;

            let ((dst_chunk, dst_row), moved_from) =
                dst_guard
                    .as_mut()
                    .push_from_dyn(src_guard.as_mut(), source_chunk, source_row)
                    .map_err(|e| MoveError::PushFromFailed { component_id, source_error: e })?;

            match destination_position {
                Some(pos) if pos != (dst_chunk, dst_row) => {
                    return Err(MoveError::RowMisalignment {
                        expected: pos,
                        got: (dst_chunk, dst_row),
                        component_id,
                    });
                }
                None => destination_position = Some((dst_chunk, dst_row)),
                _ => {}
            }

            if let Some(moved_from_info) = moved_from {
                match swap_information {
                    Some(existing) if existing != moved_from_info => {
                        return Err(MoveError::InconsistentSwapInfo);
                    }
                    None => swap_information = Some(moved_from_info),
                    _ => {}
                }
            }
        }

        let destination_position = destination_position.ok_or(MoveError::NoComponentsMoved)?;
        Ok((destination_position, swap_information))
    }

    /// Inserts newly added component values into the destination archetype at a fixed row.
    ///
    /// ## Purpose
    /// Completes entity migration by inserting component values that exist only
    /// in the destination archetype.
    ///
    /// ## Behavior
    /// - Each component value is inserted using `push_dyn`.
    /// - All inserts must resolve to the exact same `(chunk, row)` location.
    /// - Rollback data is collected for failure recovery.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if a required component column is missing.
    /// - `PushFailed` if backend storage insertion fails.
    /// - `RowMisalignment` if component columns disagree on row placement.

    #[cfg(feature = "rollback")]
    pub fn add_row_in_components_at_destination(
        &mut self,
        destination: &mut Archetype,
        destination_position: (ChunkID, RowID),
        added_components: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<Vec<(ComponentID, Box<dyn Any>)>, MoveError> {
        let mut rollback_sequence: Vec<(ComponentID, Box<dyn Any>)> = Vec::new();
        let (dst_chunk, dst_row) = destination_position;

        for (component_id, value) in added_components {
            if !destination.signature.has(component_id) {
                continue;
            }

            let dst_attr = match destination.components[component_id as usize].as_ref() {
                Some(c) => c,
                None => {
                    let _ = self.rollback_into(destination, rollback_sequence);
                    return Err(MoveError::InconsistentStorage);
                }
            };

            let mut dst_guard = Self::lock_write_move(dst_attr, component_id)?;

            let ((chunk, row), rollback) =
                match dst_guard.as_mut().push_dyn(value) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = self.rollback_into(destination, rollback_sequence);
                        return Err(MoveError::PushFailed { component_id, source_error: e });
                    }
                };

            if (chunk, row) != (dst_chunk, dst_row) {
                let _ = self.rollback_into(destination, rollback_sequence);
                return Err(MoveError::RowMisalignment {
                    expected: (dst_chunk, dst_row),
                    got: (chunk, row),
                    component_id,
                });
            }

            rollback_sequence.push((component_id, rollback));
        }

        Ok(rollback_sequence)
    }

    /// Inserts newly added component values into the destination archetype at a fixed row.
    ///
    /// ## Purpose
    /// Completes entity migration by inserting component values that exist only
    /// in the destination archetype.
    ///
    /// ## Behavior
    /// - Each component value is inserted using `push_dyn`.
    /// - All inserts must resolve to the exact same `(chunk, row)` location.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if a required component column is missing.
    /// - `PushFailed` if backend storage insertion fails.
    /// - `RowMisalignment` if component columns disagree on row placement.

    #[cfg(not(feature = "rollback"))]
    pub fn add_row_in_components_at_destination(
        &mut self,
        destination: &mut Archetype,
        destination_position: (ChunkID, RowID),
        added_components: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<(), MoveError>
    {
        let (dst_chunk, dst_row) = destination_position;

        for (component_id, value) in added_components {
            if !destination.signature.has(component_id) {
                continue;
            }

            let dst_attr = destination.components[component_id as usize]
                .as_ref()
                .ok_or(MoveError::InconsistentStorage)?;

            let mut dst_guard = Self::lock_write_move(dst_attr, component_id)?;

            let (chunk, row) =
                dst_guard
                    .as_mut()
                    .push_dyn(value)
                    .map_err(|e| MoveError::PushFailed { component_id, source_error: e })?;

            if (chunk, row) != (dst_chunk, dst_row) {
                return Err(MoveError::RowMisalignment {
                    expected: (dst_chunk, dst_row),
                    got: (chunk, row),
                    component_id,
                });
            }
        }

        Ok(())
    }

    /// Removes source-only component values from the archetype during entity migration.
    ///
    /// ## Purpose
    /// Discards component data that is not present in the destination archetype
    /// while preserving storage compactness.
    ///
    /// ## Behavior
    /// - Uses `swap_remove` to maintain dense component storage.
    /// - All removed components must report identical swap positions.
    /// - Rollback data is collected for failure recovery.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if a component column is missing.
    /// - `SwapRemoveError` if backend storage removal fails.
    /// - `InconsistentSwapInfo` if component columns disagree on swap behavior.

    #[cfg(feature = "rollback")]
    pub fn remove_row_in_components_at_source(
        &mut self,
        source_position: (ChunkID, RowID),
        removed_components: &[ComponentID],
        source_swap_position: Option<(ChunkID, RowID)>
    ) -> Result<Vec<(ComponentID, Box<dyn Any>)>, MoveError>
    {
        let (src_chunk, src_row) = source_position;
        let mut rollback_sequence: Vec<(ComponentID, Box<dyn Any>)> = Vec::new();

        for &component_id in removed_components {
            if !self.signature.has(component_id) {
                continue;
            }

            let src_attr = match self.components[component_id as usize].as_ref() {
                Some(c) => c,
                None => {
                    let _ = self.rollback_self(rollback_sequence);
                    return Err(MoveError::InconsistentStorage);
                }
            };

            let mut src_guard = Self::lock_write_move(src_attr, component_id)?;

            let (moved_from, rollback) =
                match src_guard.as_mut().swap_remove_dyn(src_chunk, src_row) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = self.rollback_self(rollback_sequence);
                        return Err(MoveError::SwapRemoveError { component_id, source_error: e });
                    }
                };

            if let Some(moved_from) = moved_from {
                if let Some(expected) = source_swap_position {
                    if expected != moved_from {
                        let _ = self.rollback_self(rollback_sequence);
                        return Err(MoveError::InconsistentSwapInfo);
                    }
                }
            }

            rollback_sequence.push((component_id, rollback));
        }

        Ok(rollback_sequence)
    }

    /// Removes source-only component values from the archetype during entity migration.
    ///
    /// ## Purpose
    /// Deletes component data that does not exist in the destination archetype
    /// while keeping component columns densely packed.
    ///
    /// ## Behavior
    /// - Uses `swap_remove` for compact storage.
    /// - All components must report identical swap positions.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if a component column is missing.
    /// - `SwapRemoveError` if storage removal fails.
    /// - `InconsistentSwapInfo` if component columns disagree on swap behavior.

    #[cfg(not(feature = "rollback"))]
    pub fn remove_row_in_components_at_source(
        &mut self,
        source_position: (ChunkID, RowID),
        removed_components: &[ComponentID],
        source_swap_position: Option<(ChunkID, RowID)>,
    ) -> Result<(), MoveError>
    {
        let (src_chunk, src_row) = source_position;

        for &component_id in removed_components {
            if !self.signature.has(component_id) {
                continue;
            }

            let src_attr = self.components[component_id as usize]
                .as_ref()
                .ok_or(MoveError::InconsistentStorage)?;

            let mut src_guard = Self::lock_write_move(src_attr, component_id)?;

            let moved_from =
                src_guard
                    .as_mut()
                    .swap_remove_dyn(src_chunk, src_row)
                    .map_err(|e| MoveError::SwapRemoveError { component_id, source_error: e })?;

            if let Some(moved_from) = moved_from {
                if let Some(expected) = source_swap_position {
                    if expected != moved_from {
                        return Err(MoveError::InconsistentSwapInfo);
                    }
                }
            }
        }

        Ok(())
    }

    /// Updates entity metadata after a row is moved between archetypes.
    ///
    /// ## Purpose
    /// Synchronizes `entity_positions` and global entity location tracking
    /// after component data has been relocated.
    ///
    /// ## Behavior
    /// - Writes the entity ID into the destination archetype metadata.
    /// - Updates the entity's global location in `EntityShards`.
    /// - Fixes metadata for any entity relocated via swap-remove.
    ///
    /// ## Errors
    /// - `MetadataFailure` if internal entity tracking is inconsistent.

    pub fn update_entity_on_row_move(
        &mut self,
        destination: &mut Archetype,
        source_position: (ChunkID, RowID),
        destination_position: (ChunkID, RowID),
        source_swap_position: Option<(ChunkID, RowID)>,
        shards: &EntityShards,
        entity: Entity,
    ) -> Result<(), MoveError> {
        let (destination_chunk, destination_row) = destination_position;
        let (source_chunk, source_row) = source_position;

        {
            let mut dest_meta = destination.meta.write().map_err(|_| MoveError::MetadataFailure)?;
            Self::ensure_capacity(&mut dest_meta, destination_chunk as usize + 1);
            dest_meta.entity_positions[destination_chunk as usize][destination_row as usize] = Some(entity);
        }

        shards.set_location(
            entity,
            EntityLocation {
                archetype: destination.archetype_id,
                chunk: destination_chunk,
                row: destination_row,
            },
        );

        let mut src_meta = self.meta.write().map_err(|_| MoveError::MetadataFailure)?;

        match source_swap_position {
            Some((last_chunk, last_row)) => {
                Self::ensure_capacity(&mut src_meta, last_chunk as usize + 1);

                let swapped_entity = src_meta.entity_positions[last_chunk as usize][last_row as usize]
                    .ok_or(MoveError::MetadataFailure)?;

                src_meta.entity_positions[source_chunk as usize][source_row as usize] = Some(swapped_entity);

                shards.set_location(
                    swapped_entity,
                    EntityLocation {
                        archetype: self.archetype_id,
                        chunk: source_chunk,
                        row: source_row,
                    },
                );

                src_meta.entity_positions[last_chunk as usize][last_row as usize] = None;
            }
            None => {
                src_meta.entity_positions[source_chunk as usize][source_row as usize] = None;
            }
        }

        Ok(())
    }

    #[cfg(feature = "rollback")]
    fn rollback_into(
        &mut self,
        destination: &mut Archetype,
        rollback_sequence: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<(), MoveError> {
        for (rolled_back_id, rollback_action) in rollback_sequence.into_iter().rev() {
            let dst_attr = destination.components[rolled_back_id as usize]
                .as_ref()
                .ok_or(MoveError::InconsistentStorage)?;
            let src_attr = self.components[rolled_back_id as usize]
                .as_ref()
                .ok_or(MoveError::InconsistentStorage)?;

            let mut dst_guard = Self::lock_write_move(dst_attr, rolled_back_id)?;
            let mut src_guard = Self::lock_write_move(src_attr, rolled_back_id)?;

            dst_guard
                .as_mut()
                .rollback_dyn(rollback_action, Some(src_guard.as_mut()))
                .map_err(|_| MoveError::RollbackFailed)?;
        }
        Ok(())
    }

    #[cfg(feature = "rollback")]
    fn rollback_self(
        &mut self,
        rollback_sequence: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<(), MoveError> {
        for (rolled_back_id, rollback_action) in rollback_sequence.into_iter().rev() {
            let attr = self.components[rolled_back_id as usize]
                .as_ref()
                .ok_or(MoveError::InconsistentStorage)?;

            let mut guard = Self::lock_write_move(attr, rolled_back_id)?;

            guard
                .as_mut()
                .rollback_dyn(rollback_action, None)
                .map_err(|_| MoveError::RollbackFailed)?;
        }
        Ok(())
    }
    
    /// Moves an entity's component row from this archetype to another.
    ///
    /// # Purpose
    /// This operation is used when an entity transitions to a new archetype
    /// because its set of components has changed (added or removed).
    ///
    /// The function constructs a new row in the destination archetype containing
    /// exactly the components described by the destination's signature.
    ///
    /// # Behavior
    ///
    /// For each component type:
    ///
    /// - **If the component exists in both source and destination**  
    ///   The component value at `(source_chunk, source_row)` is moved into the
    ///   destination component column via `push_from`, preserving its internal
    ///   ordering guarantees (including swap-remove semantics).
    ///
    /// - **If the component exists in the destination but not in the source**  
    ///   A value for this component **must** be supplied in `added_components`.
    ///   That value is inserted using `push_dyn`.
    ///
    /// - **If the component exists in the source but not in the destination**  
    ///   The component value at `(source_chunk, source_row)` is discarded using
    ///   `swap_remove`, removing the row compactly from the source column.
    ///
    /// The first column to receive the moved or inserted value defines the
    /// destination `(chunk, row)` for this entity. All other component columns
    /// for the entity must place their data **at exactly the same location**.
    /// This preserves strict row alignment across all component arrays.
    ///
    /// After all component values are written:
    ///
    /// - The destination archetype's `entity_positions` entry for the final
    ///   `(chunk, row)` is updated to record the entity's ID.
    ///
    /// - The source archetype's row at `(source_chunk, source_row)` is cleared.
    ///
    /// - If any source component column performed a swap-remove, the function
    ///   updates `entity_positions` and the global shard registry so the moved
    ///   entity now references the correct new position.
    ///
    /// - Archetype `length` counters are updated in both source and destination.

    #[cfg(feature = "rollback")]    
    pub fn move_row_to_archetype(
        &mut self,
        destination: &mut Archetype,
        shards: &EntityShards,
        entity: Entity,
        source_position: (ChunkID, RowID),
        mut added_components: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<(ChunkID, RowID), MoveError> {
        let mut shared_words = [0u64; SIGNATURE_SIZE];
        let mut source_only_words = [0u64; SIGNATURE_SIZE];
        let mut destination_only_words = [0u64; SIGNATURE_SIZE];

        for i in 0..SIGNATURE_SIZE {
            let a = self.signature.components[i];
            let b = destination.signature.components[i];

            shared_words[i] = a & b;
            source_only_words[i] = a & !b;
            destination_only_words[i] = b & !a;
        }

        let shared_components: Vec<ComponentID> =
            iter_bits_from_words(&shared_words).collect();

        let source_only_components: Vec<ComponentID> =
            iter_bits_from_words(&source_only_words).collect();

        let destination_only_components: Vec<ComponentID> =
            iter_bits_from_words(&destination_only_words).collect();

        let mut destination_only_values: Vec<(ComponentID, Box<dyn Any>)> =
            Vec::with_capacity(destination_only_components.len());

        for &need_id in &destination_only_components {
            if let Some(pos) = added_components.iter().position(|(id, _)| *id == need_id) {
                let (_id, val) = added_components.swap_remove(pos);
                destination_only_values.push((need_id, val));
            } else {
                return Err(MoveError::InconsistentStorage);
            }
        }

        let (destination_position, source_swap_position, mut moved_rollbacks) =
            self.move_row_across_shared_components(destination, source_position, shared_components)?;

        let add_rollbacks = match self.add_row_in_components_at_destination(
            destination,
            destination_position,
            destination_only_values,
        ) {
            Ok(r) => r,
            Err(e) => {
                let _ = self.rollback_into(destination, moved_rollbacks);
                return Err(e);
            }
        };
        moved_rollbacks.extend(add_rollbacks);

        let remove_rollbacks = match self.remove_row_in_components_at_source(
            source_position,
            &source_only_components,
            source_swap_position
        ) {
            Ok(r) => r,
            Err(e) => {
                self.rollback_into(destination, moved_rollbacks);
                return Err(e);
            }
        };

        if let Err(e) = self.update_entity_on_row_move(
            destination,
            source_position,
            destination_position,
            source_swap_position,
            shards,
            entity
        ) {
            let _ = self.rollback_self(remove_rollbacks);
            let _ = self.rollback_into(destination, moved_rollbacks);
            return Err(e);
        }

        {
            let mut dmeta = destination.meta.write().map_err(|_| MoveError::MetadataFailure)?;
            dmeta.length += 1;
        }

        {
            let mut smeta = self.meta.write().map_err(|_| MoveError::MetadataFailure)?;
            smeta.length = smeta.length.saturating_sub(1);
            if smeta.length == 0 {
                smeta.entity_positions.clear();
            }
        }

        Ok(destination_position)
    }      

    /// Moves an entity's component row from this archetype to another.
    ///
    /// ## Purpose
    /// Transfers an entity between archetypes when its component signature changes,
    /// constructing a new row in the destination archetype that exactly matches
    /// the destination signature.
    ///
    /// This is the core operation used when components are added to or removed
    /// from an entity.
    ///
    /// ## Behavior
    ///
    /// The move is performed in four ordered phases:
    ///
    /// 1. **Signature Analysis**
    ///    - Computes the set of components shared between source and destination.
    ///    - Computes components present only in the source (to be removed).
    ///    - Computes components present only in the destination (to be added).
    ///
    /// 2. **Shared Component Transfer**
    ///    - For each shared component, the value at `source_position` is moved
    ///      into the destination archetype using `push_from_dyn`.
    ///    - The first successful transfer determines the destination `(chunk, row)`.
    ///    - All subsequent transfers must resolve to the same location.
    ///    - Any swap-remove performed during transfer is recorded.
    ///
    /// 3. **Destination-Only Component Insertion**
    ///    - Components that exist only in the destination archetype are inserted
    ///      using values supplied in `added_components`.
    ///    - All insertions must target the previously established destination row.
    ///
    /// 4. **Source-Only Component Removal**
    ///    - Components that exist only in the source archetype are removed using
    ///      `swap_remove`, preserving dense storage.
    ///    - All removals must agree on swap behavior.
    ///
    /// After component data movement:
    /// - Entity metadata is updated in both archetypes.
    /// - Any entity relocated via swap-remove has its location corrected.
    /// - Archetype entity counts are updated.
    ///
    /// ## Parameters
    /// - `destination`: Target archetype whose signature the entity will match.
    /// - `shards`: Global entity registry used to update entity locations.
    /// - `entity`: The entity being moved.
    /// - `source_position`: The `(chunk, row)` of the entity in the source archetype.
    /// - `added_components`: Component values required by the destination archetype
    ///   but not present in the source.
    ///
    /// ## Returns
    /// Returns the `(chunk, row)` of the entity in the destination archetype.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if required component columns are missing or
    ///   `added_components` does not supply all required destination-only values.
    /// - `PushFromFailed` if transferring shared component data fails.
    /// - `PushFailed` if inserting destination-only components fails.
    /// - `SwapRemoveError` if removing source-only components fails.
    /// - `RowMisalignment` if component columns disagree on row placement.
    /// - `InconsistentSwapInfo` if swap-remove metadata differs between components.
    /// - `MetadataFailure` if entity location tracking becomes inconsistent.
    ///
    /// ## Invariants
    /// - All component columns remain row-aligned.
    /// - Source and destination archetypes remain densely packed.
    /// - Entity location metadata is always consistent with component storage.

    #[cfg(not(feature = "rollback"))]   
    pub fn move_row_to_archetype(
        &mut self,
        destination: &mut Archetype,
        shards: &EntityShards,
        entity: Entity,
        source_position: (ChunkID, RowID),
        mut added_components: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> ECSResult<(ChunkID, RowID)> {
        let mut shared_words = [0u64; SIGNATURE_SIZE];
        let mut source_only_words = [0u64; SIGNATURE_SIZE];
        let mut destination_only_words = [0u64; SIGNATURE_SIZE];

        for i in 0..SIGNATURE_SIZE {
            let a = self.signature.components[i];
            let b = destination.signature.components[i];

            shared_words[i] = a & b;
            source_only_words[i] = a & !b;
            destination_only_words[i] = b & !a;
        }

        let shared_components: Vec<ComponentID> =
            iter_bits_from_words(&shared_words).collect();

        let source_only_components: Vec<ComponentID> =
            iter_bits_from_words(&source_only_words).collect();

        let destination_only_components: Vec<ComponentID> =
            iter_bits_from_words(&destination_only_words).collect();

        let mut destination_only_values: Vec<(ComponentID, Box<dyn Any>)> =
            Vec::with_capacity(destination_only_components.len());

        for &need_id in &destination_only_components {
            if let Some(pos) = added_components.iter().position(|(id, _)| *id == need_id) {
                let (_id, val) = added_components.swap_remove(pos);
                destination_only_values.push((need_id, val));
            } else {
                return Err(MoveError::InconsistentStorage.into());
            }
        }

        let (destination_position, source_swap_position) =
            self.move_row_across_shared_components(destination, source_position, shared_components)?;

        self.add_row_in_components_at_destination(
            destination,
            destination_position,
            destination_only_values,
        )?;

        self.remove_row_in_components_at_source(
            source_position,
            &source_only_components,
            source_swap_position
        )?;

        self.update_entity_on_row_move(
            destination,
            source_position,
            destination_position,
            source_swap_position,
            shards,
            entity
        )?;

        {
            let mut dmeta = destination.meta.write().map_err(|_| ECSError::Internal("archetype meta lock poisoned".into()))?;
            dmeta.length += 1;
        }

        {
            let mut smeta = self.meta.write().map_err(|_| ECSError::Internal("archetype meta lock poisoned".into()))?;
            smeta.length = smeta.length.saturating_sub(1);
            if smeta.length == 0 {
                smeta.entity_positions.clear();
            }
        }

        Ok(destination_position)
    }  

    /// Spawns a new entity into this archetype using the provided component bundle.
    ///
    /// ## Purpose
    /// Writes a full row of component values and allocates an entity handle.
    ///
    /// ## Behavior
    /// - Each component in the archetype�s signature must be supplied by the bundle.
    /// - All component attributes must write to the same `(chunk, row)` location.
    /// - On failure, all partial writes are rolled back.
    ///
    /// ## Errors
    /// - `MissingComponent` when the bundle does not contain a required value.
    /// - `StoragePushFailedWith` on backend storage errors.
    /// - `MisalignedStorage` when attributes disagree on row placement.
    /// - `EmptyArchetype` if no components exist.
    ///
    /// ## Invariants
    /// Attribute alignment and entity position mappings must remain consistent.

pub fn spawn_on(
    &mut self,
    shards: &mut EntityShards,
    shard_id: ShardID,
    mut bundle: impl DynamicBundle,
) -> ECSResult<Entity> {
        let mut written_index: Vec<usize> = Vec::new();
        let mut reference_position: Option<(ChunkID, RowID)> = None;

        for (index, component_option) in self.components.iter().enumerate() {
            let Some(attr) = component_option.as_ref() else { continue };

            let component_id = index as ComponentID;

            // Lock column mutably for the push.
            let mut guard = Self::lock_write_spawn(attr)?;

            // Identify expected type for errors.
            let type_id = guard.as_ref().element_type_id();
            let name = guard.as_ref().element_type_name();

            let Some(value) = bundle.take(component_id) else {
                // rollback already-written components
                if let Some((c, r)) = reference_position {
                    for &j in &written_index {
                        if let Some(a) = self.components[j].as_ref() {
                            if let Ok(mut g) = Self::lock_write_spawn(a) {
                                let _ = g.as_mut().swap_remove_dyn(c, r);
                            }
                        }
                    }
                }
                return Err(SpawnError::MissingComponent { type_id, name }.into());
            };

            let position = match guard.as_mut().push_dyn(value) {
                Ok(p) => p,
                Err(e) => {
                    if let Some((c, r)) = reference_position {
                        for &j in &written_index {
                            if let Some(a) = self.components[j].as_ref() {
                                if let Ok(mut g) = Self::lock_write_spawn(a) {
                                    let _ = g.as_mut().swap_remove_dyn(c, r);
                                }

                            }
                        }
                    }
                    return Err(SpawnError::StoragePushFailedWith(e).into());
                }
            };

            if let Some(rp) = reference_position {
                if position != rp {
                    for &j in &written_index {
                        if let Some(a) = self.components[j].as_ref() {
                            if let Ok(mut g) = Self::lock_write_spawn(a) {
                                let _ = g.as_mut().swap_remove_dyn(rp.0, rp.1);
                            }

                        }
                    }
                    return Err(SpawnError::MisalignedStorage { expected: rp, got: position }.into());
                }
            } else {
                reference_position = Some(position);
            }

            written_index.push(index);
        }

        let Some((chunk, row)) = reference_position else {
            return Err(SpawnError::EmptyArchetype.into());
        };

        // Reserve metadata slot & build location
        {
            let mut meta = self.meta.write().map_err(|_| ECSError::Internal("archetype meta lock poisoned".into()))?;

            Self::ensure_capacity(&mut meta, chunk as usize + 1);

            if meta.entity_positions[chunk as usize][row as usize].is_some() {
                return Err(ECSError::Internal("spawn_on: target entity slot already occupied".into()));
            }
        }

        let location = EntityLocation { archetype: self.archetype_id, chunk, row };

        // Allocate entity handle 
        let entity = shards.spawn_on(shard_id, location).map_err(|e| {
            for &j in &written_index {
                if let Some(a) = self.components[j].as_ref() {
                    if let Ok(mut g) = Self::lock_write_spawn(a) {
                        let _ = g.as_mut().swap_remove_dyn(chunk, row);
                    }
                }
            }
            ECSError::from(e)
        })?;

        // Write entity into metadata
        {
            let mut meta = self.meta.write().map_err(|_| ECSError::Internal("archetype meta lock poisoned".into()))?;
            meta.entity_positions[chunk as usize][row as usize] = Some(entity);
            meta.length += 1;
        }

        Ok(entity)
    }

    /// Removes an entity from this archetype and maintains row compactness.
    ///
    /// ## Purpose
    /// Ensures component attributes remain dense by using `swap_remove`.
    ///
    /// ## Behavior
    /// - Updates the entity tracker to reflect despawn.
    /// - All component attributes must agree on the swapped row, if any.
    /// - Updates `entity_positions` for any entity moved via swap.
    ///
    /// ## Errors
    /// - `StaleEntity` when the entity does not exist.
    ///
    /// ## Invariants
    /// Component storage and entity metadata must remain synchronized.

    pub fn despawn_on(&mut self, shards: &mut EntityShards, entity: Entity) -> ECSResult<()> {
        let Some(location) = shards.get_location(entity) else {
            return Err(SpawnError::StaleEntity.into());
        };

        if location.archetype != self.archetype_id {
            return Err(ECSError::Internal("despawn_on: entity not in this archetype".into()));
        }

        let entity_chunk = location.chunk;
        let entity_row = location.row;

        let ok = shards.despawn(entity);
        if !ok {
            return Err(SpawnError::StaleEntity.into());
        }

        // Track whether swap_remove relocated another entity.
        let mut moved_from: Option<(ChunkID, RowID)> = None;

        for attr in self.components.iter().filter_map(|c| c.as_ref()) {
            let mut guard = Self::lock_write_spawn(attr)?;
            let pos = guard
                .as_mut()
                .swap_remove_dyn(entity_chunk, entity_row)
                .map_err(SpawnError::StorageSwapRemoveFailed)?;

            if let Some(expected) = moved_from {
                if pos != Some(expected) {
                    return Err(ECSError::Internal("despawn_on: component swap misalignment".into()));
                }
            } else {
                moved_from = pos;
            }
        }

        {
            let mut meta = self.meta.write().map_err(|_| ECSError::Internal("archetype meta lock poisoned".into()))?;
            Self::ensure_capacity(&mut meta, entity_chunk as usize + 1);

            if let Some((moved_chunk, moved_row)) = moved_from {
                Self::ensure_capacity(&mut meta, moved_chunk as usize + 1);
                let moved_entity = meta.entity_positions[moved_chunk as usize][moved_row as usize]
                    .ok_or(ECSError::Internal("despawn_on: moved slot missing entity; metadata out of sync".into()))?;

                meta.entity_positions[entity_chunk as usize][entity_row as usize] = Some(moved_entity);

                shards.set_location(
                    moved_entity,
                    EntityLocation {
                        archetype: self.archetype_id,
                        chunk: entity_chunk,
                        row: entity_row,
                    },
                );

                meta.entity_positions[moved_chunk as usize][moved_row as usize] = None;
            } else {
                meta.entity_positions[entity_chunk as usize][entity_row as usize] = None;
            }

            meta.length = meta.length.saturating_sub(1);
            if meta.length == 0 {
                meta.entity_positions.clear();
            }
        }

        Ok(())
    }

    /// Borrows a chunk of component data for system execution.
    ///
    /// ## Purpose
    /// Provides systems with efficient access to contiguous component data
    /// for iteration and mutation.
    ///
    /// ## Behavior
    /// - Read components are exposed as immutable byte slices.
    /// - Write components are exposed as raw mutable pointers.
    /// - Read and write component sets must not overlap.
    ///
    /// ## Panics
    /// Panics if a component appears in both read and write lists.
    ///
    /// ## Safety
    /// `ChunkBorrow` holds column-level locks for all accessed components.
    ///
    /// While a `ChunkBorrow` exists:
    /// - No other system may mutably access the *same component columns*.
    /// - Other systems may read the same components concurrently.
    /// - Writes to disjoint component sets are permitted.
    /// - Structural archetype mutation must not occur.
    ///
    /// These guarantees are enforced by column-level locking and execution
    /// phase discipline. Violations may cause deadlock or panic.

    pub fn borrow_chunk_for<'a>(
        &'a self,
        chunk: ChunkID,
        read_ids: &[ComponentID],
        write_ids: &[ComponentID],
    ) -> ECSResult<ChunkBorrow<'a>> {
        let length = self.chunk_valid_length(chunk as usize)
            .map_err(|_| ExecutionError::InternalExecutionError)?;

        for &id in read_ids {
            if write_ids.contains(&id) {
                return Err(ECSError::Execute(
                    ExecutionError::InvalidQueryAccess {
                        component_id: id,
                        reason: crate::engine::error::InvalidAccessReason::ReadAndWrite,
                    }
                ));
            }
        }

        // Lock in a single global order across read + write
        let mut all: Vec<(ComponentID, bool)> =
            Vec::with_capacity(read_ids.len() + write_ids.len());
        for &cid in read_ids {
            all.push((cid, false));
        }
        for &cid in write_ids {
            all.push((cid, true));
        }

        all.sort_unstable_by_key(|(cid, _)| *cid);

        // Acquire locks in one pass
        let mut read_guards: Vec<(ComponentID, RwLockReadGuard<'a, Box<dyn TypeErasedAttribute>>)> =
            Vec::with_capacity(read_ids.len());
        let mut write_guards: Vec<(ComponentID, RwLockWriteGuard<'a, Box<dyn TypeErasedAttribute>>)> =
            Vec::with_capacity(write_ids.len());

        for (cid, is_write) in all {
            let attr = self.component_locked(cid)
                .ok_or(ExecutionError::MissingComponent { component_id: cid })?;

            if is_write {
                let g = attr.write().map_err(|_| ExecutionError::InternalExecutionError)?;
                write_guards.push((cid, g));
            } else {
                let g = attr.read().map_err(|_| ExecutionError::InternalExecutionError)?;
                read_guards.push((cid, g));
            }
        }

        // Build read views in the original read_ids order
        let mut reads = Vec::with_capacity(read_ids.len());
        for &cid in read_ids {
            let guard = read_guards
                .iter()
                .find(|(id, _)| *id == cid)
                .map(|(_, g)| g)
                .ok_or(ExecutionError::InternalExecutionError)?;

            let (ptr, bytes) = guard
                .as_ref()
                .chunk_bytes(chunk, length)
                .ok_or(ExecutionError::InternalExecutionError)?;

            reads.push((ptr, bytes));
        }

        // Build write views in the original write_ids order
        let mut writes = Vec::with_capacity(write_ids.len());
        for &cid in write_ids {
            let guard = write_guards
                .iter_mut()
                .find(|(id, _)| *id == cid)
                .map(|(_, g)| g)
                .ok_or(ExecutionError::InternalExecutionError)?;

            let (ptr, _bytes) = guard
                .as_mut()
                .chunk_bytes_mut(chunk, length)
                .ok_or(ExecutionError::InternalExecutionError)?;

            writes.push(ptr);
        }

        Ok(ChunkBorrow {
            length,
            reads,
            writes,
            _read_guards: read_guards,
            _write_guards: write_guards,
        })
    }


    /// Constructs a new archetype and inserts empty attributes for the provided
    /// component types.
    ///
    /// ## Behavior
    /// - Each type ID must correspond to a registered component.
    /// - Component attributes are allocated via the storage factory.
    /// - No entities are created.
    ///
    /// ## Invariants
    /// The resulting archetype is empty but has a fully defined signature.

    pub fn from_components<T: IntoIterator<Item = std::any::TypeId>>(
        archetype_id: ArchetypeID,
        types: T,
    ) -> ECSResult<Self> {
        let mut signature = Signature::default();
        let mut component_ids = Vec::new();

        for type_id in types {
            let component_id = component_id_of_type_id(type_id)?
                .ok_or(ECSError::Internal("from_components: component type not registered".into()))?;
            signature.set(component_id);
            component_ids.push(component_id);
        }

        let archetype = Self::new(archetype_id, signature)?;

        if !component_ids.iter().all(|&id| archetype.has(id)) {
            return Err(ECSError::Internal("from_components: archetype signature and storage mismatch".into()));
        }

        Ok(archetype)
    }
}

/// Represents an archetype selected during query matching.
///
/// ## Purpose
/// Used by query systems to record which archetypes satisfy a component filter
/// and how many chunks they contain.

pub struct ArchetypeMatch {
    /// Identifier of the matched archetype.
    pub archetype_id: ArchetypeID,
    /// Number of chunks currently allocated in the archetype.
    pub chunks: usize,
}

fn make_empty_component_for(component_id: ComponentID) -> ECSResult<Box<dyn TypeErasedAttribute>> {
    let factory = get_component_storage_factory(component_id)?;
    Ok(factory())
}
