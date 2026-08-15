//! Spawn and despawn operations for [`Archetype`].
//!
//! This module implements [`Archetype::spawn_on`] and [`Archetype::despawn_on`],
//! the two mutating entry points for adding and removing entities from an archetype.
//!
//! # Invariants
//!
//! Both operations uphold the following invariants:
//!
//! - **Column density** - component storage is always kept dense. Removal uses
//!   `swap_remove` to fill gaps left by departed rows.
//! - **Attribute alignment** - every column for a given archetype must agree on
//!   the `(chunk, row)` position of each entity. Any disagreement is treated as
//!   an [`InternalViolation`] and the operation is aborted.
//! - **Metadata consistency** - `entity_positions` and per-entity [`EntityLocation`]
//!   records are updated atomically with respect to the metadata lock, and only
//!   after all column operations have completed.
//!
//! # Lock ordering
//!
//! To prevent deadlocks, a strict acquisition order is observed throughout:
//!
//! 1. **Column locks** - acquired and released one at a time during the component
//!    read/write loop.
//! 2. **Metadata lock** (`self.meta`) - acquired only *after* all column locks
//!    have been released and, in the case of despawn, after the entity handle
//!    has been returned to [`EntityShards`].
//!
//! # Error recovery
//!
//! Partial writes during `spawn_on` are rolled back via `swap_remove` on every
//! column written so far before the error is propagated. Partial removals during
//! `despawn_on` leave the entity handle intact so that the archetype state
//! remains recoverable.

use crate::engine::types::{ChunkID, ComponentID, RowID, ShardID, CHUNK_CAP};

use std::any::Any;

use crate::engine::commands::BatchColumn;
use crate::engine::component::DynamicBundle;

use crate::engine::entity::{Entity, EntityLocation, EntityShards};

use crate::engine::error::{
    AttributeError, ECSError, ECSResult, InternalViolation, SpawnError, StaleEntityError,
    TypeMismatchError,
};

use crate::engine::storage::LockedAttribute;

use super::core::Archetype;

impl Archetype {
    /// Spawns a new entity into this archetype using the provided component bundle.
    ///
    /// ## Purpose
    /// Writes a full row of component values and allocates an entity handle.
    ///
    /// ## Behaviour
    /// - Each component in the archetype's signature must be supplied by the bundle.
    /// - All component attributes must write to the same `(chunk, row)` location.
    /// - On failure, all partial writes are cleaned up via swap-remove.
    ///
    /// ## Errors
    /// - `MissingComponent` when the bundle does not contain a required value.
    /// - `StoragePushFailedWith` on backend storage errors.
    /// - `MisalignedStorage` when attributes disagree on row placement.
    /// - `EmptyArchetype` if no components exist.
    ///
    /// ## Invariants
    /// Attribute alignment and entity position mappings must remain consistent.
    ///
    /// ## Lock ordering
    /// Column locks are acquired and released per-component during the push
    /// loop. The metadata lock is acquired only *after* all column writes
    /// have completed, respecting the global lock ordering contract.
    pub fn spawn_on(
        &mut self,
        shards: &EntityShards,
        shard_id: ShardID,
        mut bundle: impl DynamicBundle,
    ) -> ECSResult<Entity> {
        let mut pending_values: Vec<(usize, Box<dyn Any>)> =
            Vec::with_capacity(self.components.len());

        for idx in 0..self.components.len() {
            let (component_id, ref attr) = self.components[idx];
            let guard = Self::lock_write_spawn(attr)?;
            let type_id = guard.as_ref().element_type_id();
            let name = guard.as_ref().element_type_name();

            let Some(value) = bundle.take(component_id) else {
                return Err(SpawnError::MissingComponent { type_id, name }.into());
            };

            let actual = value.as_ref().type_id();
            if actual != type_id {
                return Err(
                    SpawnError::StoragePushFailedWith(AttributeError::TypeMismatch(
                        TypeMismatchError {
                            expected: type_id,
                            actual,
                            expected_name: name,
                            actual_name: "",
                        },
                    ))
                    .into(),
                );
            }

            let value: Box<dyn Any> = value;
            pending_values.push((idx, value));
        }

        let mut written_positions: Vec<(usize, ChunkID, RowID)> = Vec::new();
        let mut reference_position: Option<(ChunkID, RowID)> = None;

        for (idx, value) in pending_values {
            let (_component_id, ref attr) = self.components[idx];

            // Lock column mutably for the push.
            let mut guard = Self::lock_write_spawn(attr)?;

            let position = match guard.as_mut().push_dyn(value) {
                Ok(p) => p,
                Err(e) => {
                    Self::rollback_written_positions(&self.components, &written_positions);
                    return Err(SpawnError::StoragePushFailedWith(e).into());
                }
            };

            if let Some(rp) = reference_position {
                if position != rp {
                    written_positions.push((idx, position.0, position.1));
                    Self::rollback_written_positions(&self.components, &written_positions);
                    return Err(SpawnError::MisalignedStorage {
                        expected: rp,
                        got: position,
                    }
                    .into());
                }
            } else {
                reference_position = Some(position);
            }

            written_positions.push((idx, position.0, position.1));
        }

        let Some((chunk, row)) = reference_position else {
            return Err(SpawnError::EmptyArchetype.into());
        };

        // Metadata lock acquired only after all column locks are released.
        {
            let mut meta = self
                .meta
                .write()
                .map_err(|_| ECSError::from(InternalViolation::ArchetypeMetaLockPoisoned))?;

            Self::ensure_capacity(&mut meta, chunk as usize + 1);

            if meta.entity_positions[chunk as usize][row as usize] != Entity::PLACEHOLDER {
                drop(meta);
                Self::rollback_written_positions(&self.components, &written_positions);
                return Err(InternalViolation::SpawnSlotOccupied.into());
            }
        }

        let location = EntityLocation {
            archetype: self.archetype_id,
            chunk,
            row,
        };

        // Allocate entity handle
        let entity = shards.spawn_on(shard_id, location).map_err(|e| {
            Self::rollback_written_positions(&self.components, &written_positions);
            ECSError::from(e)
        })?;

        // Write entity into metadata
        {
            let mut meta = self
                .meta
                .write()
                .map_err(|_| ECSError::from(InternalViolation::ArchetypeMetaLockPoisoned))?;
            meta.entity_positions[chunk as usize][row as usize] = entity;
            meta.length += 1;
        }

        Ok(entity)
    }

    /// Removes an entity from this archetype and maintains row compactness.
    ///
    /// ## Purpose
    /// Ensures component attributes remain dense by using `swap_remove`.
    ///
    /// ## Behaviour
    /// 1. Validates the entity is alive and belongs to this archetype.
    /// 2. Removes component data from all columns via `swap_remove_dyn`.
    ///    All columns must agree on which row (if any) was swapped in.
    /// 3. Despawns the entity handle from `EntityShards`.
    /// 4. Updates `entity_positions` metadata for the despawned entity
    ///    and for any entity relocated via swap.
    ///
    /// Component data is removed *before* the entity handle is despawned
    /// to ensure that a failure during column removal does not leave a
    /// dead entity handle with orphaned component rows.
    ///
    /// ## Errors
    /// - `StaleEntity` when the entity does not exist.
    ///
    /// ## Invariants
    /// Component storage and entity metadata must remain synchronized.
    ///
    /// ## Lock ordering
    /// Column locks are acquired and released per-component during the
    /// removal loop. The metadata lock is acquired only *after* all column
    /// removals and the entity despawn have completed.
    pub fn despawn_on(&mut self, shards: &EntityShards, entity: Entity) -> ECSResult<()> {
        let Some(location) = shards.get_location(entity)? else {
            return Err(SpawnError::StaleEntity(StaleEntityError).into());
        };

        if location.archetype != self.archetype_id {
            return Err(InternalViolation::DespawnEntityNotInArchetype.into());
        }

        let entity_chunk = location.chunk;
        let entity_row = location.row;

        // Remove component data from all columns first.
        // This ensures that if any column removal fails, the entity handle
        // is still alive and the archetype state is recoverable.
        let mut moved_from: Option<(ChunkID, RowID)> = None;

        for (_, attr) in self.components.iter() {
            let mut guard = Self::lock_write_spawn(attr)?;
            let pos = guard
                .as_mut()
                .swap_remove_dyn(entity_chunk, entity_row)
                .map_err(SpawnError::StorageSwapRemoveFailed)?;

            if let Some(expected) = moved_from {
                if pos != Some(expected) {
                    return Err(InternalViolation::DespawnSwapMisalignment.into());
                }
            } else {
                moved_from = pos;
            }
        }

        // Despawn the entity handle.
        // Component data has already been cleaned up, so even if this fails
        // the archetype columns are consistent.
        let ok = shards.despawn(entity)?;
        if !ok {
            return Err(SpawnError::StaleEntity(StaleEntityError).into());
        }

        // Update metadata (acquired after all column locks).
        {
            let mut meta = self
                .meta
                .write()
                .map_err(|_| ECSError::from(InternalViolation::ArchetypeMetaLockPoisoned))?;
            Self::ensure_capacity(&mut meta, entity_chunk as usize + 1);

            if let Some((moved_chunk, moved_row)) = moved_from {
                Self::ensure_capacity(&mut meta, moved_chunk as usize + 1);
                let moved_entity = meta.entity_positions[moved_chunk as usize][moved_row as usize];
                if moved_entity == Entity::PLACEHOLDER {
                    return Err(InternalViolation::DespawnMovedSlotMissingEntity.into());
                }

                meta.entity_positions[entity_chunk as usize][entity_row as usize] = moved_entity;

                shards
                    .set_location(
                        moved_entity,
                        EntityLocation {
                            archetype: self.archetype_id,
                            chunk: entity_chunk,
                            row: entity_row,
                        },
                    )
                    .map_err(ECSError::from)?;

                meta.entity_positions[moved_chunk as usize][moved_row as usize] =
                    Entity::PLACEHOLDER;
            } else {
                meta.entity_positions[entity_chunk as usize][entity_row as usize] =
                    Entity::PLACEHOLDER;
            }

            meta.length = meta.length.saturating_sub(1);
            if meta.length == 0 {
                meta.entity_positions.clear();
            }
        }

        Ok(())
    }

    fn rollback_written_positions(
        components: &[(ComponentID, LockedAttribute)],
        positions: &[(usize, ChunkID, RowID)],
    ) {
        for &(j, chunk, row) in positions.iter().rev() {
            let (_, ref a) = components[j];
            if let Ok(mut g) = Self::lock_write_spawn(a) {
                let _ = g.as_mut().swap_remove_dyn(chunk, row);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Columnar batch operations
    // -----------------------------------------------------------------------

    /// Bulk-appends one full column per component for `count` new rows.
    ///
    /// The columnar spawn fast path: each column is a single type-erased
    /// `Vec<T>` that the storage layer downcasts once and copies in
    /// chunk-sized runs, instead of one boxed value per entity per component.
    ///
    /// ## Contract
    /// - `start` must equal the archetype's current row count.
    /// - `columns` must cover the archetype signature exactly (no missing,
    ///   duplicate, or unknown components) and each column must hold `count`
    ///   elements of the registered storage type.
    ///
    /// ## Failure semantics
    /// Appends are applied column by column in ascending [`ComponentID`]
    /// order. On any failure, every column extended so far is truncated back
    /// to `start` (bulk appends never reorder existing rows, so truncation is
    /// an exact rollback) and the error is returned.
    pub(crate) fn append_batch_columns(
        &self,
        start: usize,
        count: usize,
        mut columns: Vec<BatchColumn>,
    ) -> ECSResult<()> {
        columns.sort_by_key(|column| column.component_id);

        // Validate exact signature coverage: the sorted column ids must equal
        // the sorted signature ids.
        let mut expected = self.signature.iterate_over_components();
        for column in &columns {
            match expected.next() {
                Some(required) if required == column.component_id => {}
                _ => {
                    return Err(SpawnError::BatchColumnSet {
                        component_id: column.component_id,
                    }
                    .into());
                }
            }
            if column.len != count {
                return Err(SpawnError::BatchColumnMismatch {
                    component_id: column.component_id,
                    expected: count,
                    actual: column.len,
                }
                .into());
            }
        }
        if let Some(missing) = expected.next() {
            return Err(SpawnError::BatchColumnSet {
                component_id: missing,
            }
            .into());
        }

        let mut appended: Vec<ComponentID> = Vec::with_capacity(columns.len());
        for column in columns {
            let component_id = column.component_id;
            let attr = self
                .find_component(component_id)
                .ok_or(SpawnError::BatchColumnSet { component_id })?;
            let result = Self::lock_write_spawn(attr)
                .map_err(ECSError::from)
                .and_then(|mut guard| {
                    guard
                        .extend_from_vec_any(column.values)
                        .map_err(|e| ECSError::from(SpawnError::StoragePushFailedWith(e)))
                });

            match result {
                Ok((appended_start, appended_count))
                    if appended_start == start && appended_count == count =>
                {
                    appended.push(component_id);
                }
                Ok((appended_start, appended_count)) => {
                    // The column landed at the wrong offset, or was longer or
                    // shorter than its siblings; put everything (including this
                    // column) back to `start`.
                    appended.push(component_id);
                    self.truncate_columns_to(&appended, start);
                    // Report whichever of the two actually disagreed. The guard
                    // above failed, so at least one has, and reporting the
                    // offset unconditionally would emit `expected == actual`
                    // for every pure count mismatch.
                    let (expected, actual) = if appended_start != start {
                        (start, appended_start)
                    } else {
                        (count, appended_count)
                    };
                    return Err(SpawnError::BatchColumnMismatch {
                        component_id,
                        expected,
                        actual,
                    }
                    .into());
                }
                Err(error) => {
                    self.truncate_columns_to(&appended, start);
                    return Err(error);
                }
            }
        }

        Ok(())
    }

    /// Best-effort rollback: truncates the named columns back to `length`.
    pub(crate) fn truncate_columns_to(&self, component_ids: &[ComponentID], length: usize) {
        for &component_id in component_ids {
            if let Some(attr) = self.find_component(component_id) {
                if let Ok(mut guard) = attr.write() {
                    let _ = guard.truncate_to(length);
                }
            }
        }
    }

    /// Publishes metadata for rows `start..start + entities.len()` appended by
    /// [`append_batch_columns`](Self::append_batch_columns).
    ///
    /// Fills `entity_positions` for every new row and bumps the archetype
    /// length in a single metadata-lock acquisition.
    pub(crate) fn commit_batch_rows(&self, start: usize, entities: &[Entity]) -> ECSResult<()> {
        let mut meta = self
            .meta
            .write()
            .map_err(|_| ECSError::from(InternalViolation::ArchetypeMetaLockPoisoned))?;

        let end = start + entities.len();
        let last_chunk = if end == 0 { 0 } else { (end - 1) / CHUNK_CAP };
        Self::ensure_capacity(&mut meta, last_chunk + 1);

        for (offset, &entity) in entities.iter().enumerate() {
            let index = start + offset;
            let chunk = index / CHUNK_CAP;
            let row = index % CHUNK_CAP;
            if meta.entity_positions[chunk][row] != Entity::PLACEHOLDER {
                // Clear the rows written so far so a failed commit leaves no
                // stale entries beyond the (unchanged) archetype length.
                for cleared in 0..offset {
                    let cleared_index = start + cleared;
                    meta.entity_positions[cleared_index / CHUNK_CAP][cleared_index % CHUNK_CAP] =
                        Entity::PLACEHOLDER;
                }
                return Err(InternalViolation::SpawnSlotOccupied.into());
            }
            meta.entity_positions[chunk][row] = entity;
        }
        meta.length += entities.len();
        Ok(())
    }

    /// Despawns a batch of rows belonging to this archetype.
    ///
    /// `targets` must be sorted **descending** by linear row index and must
    /// contain no duplicates; the caller ([`ECSData`]) resolves and orders
    /// them. Descending order guarantees each swap-remove's moved row (always
    /// the current last row) is never itself a pending target, so locations
    /// resolved up front stay valid throughout the batch.
    ///
    /// Column locks are acquired once for the whole batch (ascending
    /// [`ComponentID`] order, before the metadata lock), which is the point
    /// of batching: per-entity despawn pays those acquisitions per entity.
    pub(crate) fn despawn_rows_batch(
        &mut self,
        shards: &EntityShards,
        targets: &[(Entity, ChunkID, RowID)],
    ) -> ECSResult<()> {
        if targets.is_empty() {
            return Ok(());
        }

        // Acquire every column write lock up front, ascending component id.
        let mut guards = Vec::with_capacity(self.components.len());
        for (component_id, attr) in self.components.iter() {
            let guard = Self::lock_write_spawn(attr).map_err(ECSError::from)?;
            guards.push((*component_id, guard));
        }

        let mut meta = self
            .meta
            .write()
            .map_err(|_| ECSError::from(InternalViolation::ArchetypeMetaLockPoisoned))?;

        // Shard operations are deferred and applied grouped (one mutex
        // acquisition per shard) after all rows are removed. Per-shard order
        // of `pending_moves` is preserved, so an entity relocated twice by
        // chained swaps ends up at its final location.
        let mut pending_moves: Vec<(Entity, EntityLocation)> = Vec::new();

        for &(_entity, chunk, row) in targets {
            let mut moved_from: Option<(ChunkID, RowID)> = None;
            let mut first = true;
            for (_, guard) in guards.iter_mut() {
                let pos = guard
                    .as_mut()
                    .swap_remove_dyn(chunk, row)
                    .map_err(|e| ECSError::from(SpawnError::StorageSwapRemoveFailed(e)))?;
                if first {
                    moved_from = pos;
                    first = false;
                } else if pos != moved_from {
                    return Err(InternalViolation::DespawnSwapMisalignment.into());
                }
            }

            Self::ensure_capacity(&mut meta, chunk as usize + 1);
            if let Some((moved_chunk, moved_row)) = moved_from {
                let moved_entity = meta.entity_positions[moved_chunk as usize][moved_row as usize];
                if moved_entity == Entity::PLACEHOLDER {
                    return Err(InternalViolation::DespawnMovedSlotMissingEntity.into());
                }
                meta.entity_positions[chunk as usize][row as usize] = moved_entity;
                pending_moves.push((
                    moved_entity,
                    EntityLocation {
                        archetype: self.archetype_id,
                        chunk,
                        row,
                    },
                ));
                meta.entity_positions[moved_chunk as usize][moved_row as usize] =
                    Entity::PLACEHOLDER;
            } else {
                meta.entity_positions[chunk as usize][row as usize] = Entity::PLACEHOLDER;
            }

            meta.length = meta.length.saturating_sub(1);
        }

        if meta.length == 0 {
            meta.entity_positions.clear();
        }

        shards
            .set_locations_grouped(&pending_moves)
            .map_err(ECSError::from)?;
        let despawn_targets: Vec<Entity> = targets.iter().map(|&(entity, _, _)| entity).collect();
        shards
            .despawn_grouped(&despawn_targets)
            .map_err(ECSError::from)?;

        Ok(())
    }
}
