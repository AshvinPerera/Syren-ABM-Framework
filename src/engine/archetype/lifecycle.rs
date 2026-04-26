//! Spawn and despawn operations for [`Archetype`].
//!
//! This module implements [`Archetype::spawn_on`] and [`Archetype::despawn_on`],
//! the two mutating entry points for adding and removing entities from an archetype.
//!
//! # Invariants
//!
//! Both operations uphold the following invariants:
//!
//! - **Column density** — component storage is always kept dense. Removal uses
//!   `swap_remove` to fill gaps left by departed rows.
//! - **Attribute alignment** — every column for a given archetype must agree on
//!   the `(chunk, row)` position of each entity. Any disagreement is treated as
//!   an [`InternalViolation`] and the operation is aborted.
//! - **Metadata consistency** — `entity_positions` and per-entity [`EntityLocation`]
//!   records are updated atomically with respect to the metadata lock, and only
//!   after all column operations have completed.
//!
//! # Lock ordering
//!
//! To prevent deadlocks, a strict acquisition order is observed throughout:
//!
//! 1. **Column locks** — acquired and released one at a time during the component
//!    read/write loop.
//! 2. **Metadata lock** (`self.meta`) — acquired only *after* all column locks
//!    have been released and, in the case of despawn, after the entity handle
//!    has been returned to [`EntityShards`].
//!
//! # Error recovery
//!
//! Partial writes during `spawn_on` are rolled back via `swap_remove` on every
//! column written so far before the error is propagated. Partial removals during
//! `despawn_on` leave the entity handle intact so that the archetype state
//! remains recoverable.

use crate::engine::types::{ChunkID, ComponentID, RowID, ShardID};

use std::any::Any;

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

            if meta.entity_positions[chunk as usize][row as usize].is_some() {
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
                let moved_entity = meta.entity_positions[moved_chunk as usize][moved_row as usize]
                    .ok_or(ECSError::from(
                        InternalViolation::DespawnMovedSlotMissingEntity,
                    ))?;

                meta.entity_positions[entity_chunk as usize][entity_row as usize] =
                    Some(moved_entity);

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
}
