//! Entity migration between archetypes.
//!
//! This module implements [`Archetype::move_row_to_archetype`], the core operation
//! that transfers an entity's component data when its signature changes — i.e. when
//! components are added to or removed from a live entity.
//!
//! # Overview
//!
//! Archetypes store component data in dense, column-oriented storage. When an
//! entity's component set changes, its data cannot remain in the same archetype;
//! it must be relocated to the archetype whose signature matches the new set.
//! This module orchestrates that relocation safely and atomically across four
//! ordered phases:
//!
//! 1. **Signature analysis** — bit-level intersection of source and destination
//!    signatures to classify each component as shared, source-only, or
//!    destination-only.
//! 2. **Shared component transfer** — component values present in both archetypes
//!    are moved into the destination row via [`Archetype::move_row_across_shared_components`].
//! 3. **Destination-only insertion** — caller-supplied values for newly added
//!    components are written into the same destination row via
//!    [`Archetype::add_row_in_components_at_destination`].
//! 4. **Source-only removal** — components dropped from the entity are
//!    swap-removed from the source archetype via
//!    [`Archetype::remove_row_in_components_at_source`].
//!
//! Entity metadata and global location tracking ([`EntityShards`]) are reconciled
//! after all component data has moved, including any entity displaced by a
//! swap-remove.
//!
//! # Invariants
//!
//! - All component columns remain row-aligned throughout and after a migration.
//! - Both source and destination archetypes remain densely packed at all times.
//! - [`EntityShards`] location data is always consistent with component storage
//!   on success; a returned error leaves the caller responsible for recovery.
//!
//! # Locking
//!
//! Each phase acquires per-column write locks as needed and releases them before
//! the next phase begins, respecting the global ascending [`ComponentID`] lock
//! ordering contract and avoiding deadlock. Archetype metadata locks are taken
//! only after all column locks have been dropped.

use std::any::Any;

use crate::engine::types::{
    ChunkID,
    RowID,
    ComponentID,
    SIGNATURE_SIZE,
};

use crate::engine::entity::{
    Entity,
    EntityLocation,
    EntityShards,
};

use crate::engine::component::iter_bits_from_words;

use crate::engine::error::{
    MoveError,
    ECSError,
    ECSResult,
    InternalViolation,
};

use super::core::Archetype;


impl Archetype {

    /// Moves component data shared between source and destination archetypes.
    ///
    /// ## Purpose
    /// Transfers component rows that exist in both archetypes during an entity
    /// migration, preserving dense storage and row alignment.
    ///
    /// ## Behaviour
    /// - Shared components are moved using `push_from_dyn`.
    /// - The first successful move determines the destination `(chunk, row)`.
    /// - All subsequent moves must resolve to the same location.
    /// - Swap-remove behaviour is tracked to update entity metadata correctly.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if component columns are missing.
    /// - `PushFromFailed` if backend storage transfer fails.
    /// - `RowMisalignment` if components disagree on row placement.
    /// - `InconsistentSwapInfo` if swap metadata differs between columns.
    /// - `NoComponentsMoved` if no shared components exist.
    ///
    /// # Safety
    ///
    /// This function acquires two column write locks per shared component
    /// (one on the source archetype, one on the destination).  Deadlock is
    /// impossible because:
    ///
    /// (a) The caller provides `&mut self` (source) and `&mut destination`,
    ///     which are guaranteed to be distinct by `get_archetype_pair_mut`.
    ///     Therefore, the two `LockedAttribute` references always point to
    ///     different `RwLock` instances — there is no self-deadlock risk.
    ///
    /// (b) Structural mutations are serialized by phase discipline: this
    ///     function is only reachable during the exclusive write phase.
    ///     No concurrent iteration or migration can be acquiring locks on
    ///     these same archetypes in the opposite direction.
    ///
    /// (c) Within each archetype the sorted `components` vec naturally
    ///     ensures ascending `ComponentID` order when iterated, matching
    ///     the global lock-ordering contract.

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

            let src_attr = self.find_component(component_id)
                .ok_or(MoveError::InconsistentStorage)?;
            let dst_attr = destination.find_component(component_id)
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
    /// ## Behaviour
    /// - Each component value is inserted using `push_dyn`.
    /// - All inserts must resolve to the exact same `(chunk, row)` location.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if a required component column is missing.
    /// - `PushFailed` if backend storage insertion fails.
    /// - `RowMisalignment` if component columns disagree on row placement.

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

            let dst_attr = destination.find_component(component_id)
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
    /// Deletes component data that does not exist in the destination archetype
    /// while keeping component columns densely packed.
    ///
    /// ## Behaviour
    /// - Uses `swap_remove` for compact storage.
    /// - All components must report identical swap positions.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if a component column is missing.
    /// - `SwapRemoveError` if storage removal fails.
    /// - `InconsistentSwapInfo` if component columns disagree on swap behavior.

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

            let src_attr = self.find_component(component_id)
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
    /// ## Behaviour
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
            let mut dest_meta = destination.meta.write().map_err(|_| MoveError::MetadataFailure {
                entity: None,
                source_archetype: None,
                destination_archetype: None,
            })?;
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
        ).map_err(|_| MoveError::MetadataFailure {
            entity: None,
            source_archetype: None,
            destination_archetype: None,
        })?;

        let mut src_meta = self.meta.write().map_err(|_| MoveError::MetadataFailure {
            entity: None,
            source_archetype: None,
            destination_archetype: None,
        })?;

        match source_swap_position {
            Some((last_chunk, last_row)) => {
                Self::ensure_capacity(&mut src_meta, last_chunk as usize + 1);

                let swapped_entity = src_meta.entity_positions[last_chunk as usize][last_row as usize]
                    .ok_or(MoveError::MetadataFailure {
                        entity: None,
                        source_archetype: None,
                        destination_archetype: None,
                    })?;

                src_meta.entity_positions[source_chunk as usize][source_row as usize] = Some(swapped_entity);

                shards.set_location(
                    swapped_entity,
                    EntityLocation {
                        archetype: self.archetype_id,
                        chunk: source_chunk,
                        row: source_row,
                    },
                ).map_err(|_| MoveError::MetadataFailure {
                    entity: None,
                    source_archetype: None,
                    destination_archetype: None,
                })?;

                src_meta.entity_positions[last_chunk as usize][last_row as usize] = None;
            }
            None => {
                src_meta.entity_positions[source_chunk as usize][source_row as usize] = None;
            }
        }

        Ok(())
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
    /// ## Behaviour
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
    ///    - All removals must agree on swap behaviour.
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
    ///
    /// # Warning: Partial Migration Failure
    ///
    /// If `add_row_in_components_at_destination` fails after
    /// `move_row_across_shared_components` has succeeded, both archetypes
    /// will be in an inconsistent state. A rollback mechanism is planned
    /// but not yet implemented. See ADR-2 in the architecture review.

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

        // NOTE: Metadata lock is acquired only after all column locks (from the
        // phases above) have been dropped, respecting the lock ordering contract.
        {
            let mut dmeta = destination.meta.write().map_err(|_| ECSError::from(InternalViolation::ArchetypeMetaLockPoisoned))?;
            dmeta.length += 1;
        }

        {
            let mut smeta = self.meta.write().map_err(|_| ECSError::from(InternalViolation::ArchetypeMetaLockPoisoned))?;
            smeta.length = smeta.length.saturating_sub(1);
            if smeta.length == 0 {
                smeta.entity_positions.clear();
            }
        }

        Ok(destination_position)
    }
}
