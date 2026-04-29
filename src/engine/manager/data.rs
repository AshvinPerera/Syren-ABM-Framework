//! Core ECS storage and orchestration layer.
//!
//! This module defines [`ECSData`], the central data structure of the ECS runtime.
//! It owns all archetypes and their component storage, manages entity placement and
//! migration, and drives both structural mutations and parallel query execution.
//!
//! # Architecture
//!
//! The ECS follows an **archetype-based** storage model: entities are grouped into
//! [`Archetype`]s according to their exact component signature. Each archetype stores
//! its components in contiguous, chunk-aligned columns, enabling cache-friendly
//! parallel iteration.
//!
//! # Responsibilities
//!
//! - **Archetype management** - archetypes are created lazily and looked up by
//!   component signature via `signature_map`. Each archetype is created with an
//!   explicit `&ComponentRegistry` rather than calling global registry functions,
//!   which enables multi-world support.
//! - **Entity placement** - entity locations `(archetype, chunk, row)` are tracked
//!   through [`EntityShards`] and kept consistent with archetype storage.
//! - **Structural mutations** - adding or removing a component migrates the entity's
//!   row to a new archetype whose signature reflects the change.
//! - **Deferred commands** - [`ECSData::apply_deferred_commands`] flushes a batch of
//!   [`Command`]s (spawn, despawn, add, remove) as a single synchronisation point.
//! - **Parallel iteration** - [`ECSData::for_each_abstraction_unchecked`] and
//!   [`ECSData::reduce_abstraction_unchecked`] execute chunk-oriented queries across
//!   matching archetypes using Rayon, with per-column RwLock guards enforcing
//!   read/write separation. Column locks are acquired in ascending [`ComponentID`]
//!   order to prevent deadlocks consistent with the lock-ordering contract documented
//!   in `archetype/mod.rs`.
//!
//! # GPU support (`feature = "gpu"`)
//!
//! When the `gpu` feature is enabled, `ECSData` additionally maintains:
//! - A [`DirtyChunks`] tracker that records which component chunks were written
//!   during a query, enabling selective CPU->GPU uploads.
//! - A [`GPUResourceRegistry`] for world-owned GPU buffers and bind-group resources.
//!
//! # Safety contract
//!
//! The unchecked iteration methods expose raw byte slices to user callbacks.
//! Callers must guarantee:
//! - No structural mutations occur during iteration.
//! - Component borrow rules (read/write exclusivity) are enforced at the call site.
//! - Callbacks do not allow references to escape the closure.

use std::any::TypeId;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::engine::archetype::Archetype;
use crate::engine::commands::{
    Command, CommandEvents, DespawnEvent, SpawnEvent, TemplateLifecycleBatch,
};
use crate::engine::component::{ComponentRegistry, Signature};
use crate::engine::entity::{Entity, EntityShards};
use crate::engine::error::{
    AccessKind, AttributeError, ECSError, ECSResult, ExecutionError, InternalViolation, MoveError,
    RegistryError, SpawnError, StaleEntityError,
};
use crate::engine::query::BuiltQuery;
use crate::engine::types::{ArchetypeID, ComponentID, ShardID, SIGNATURE_SIZE};

#[cfg(feature = "gpu")]
use crate::engine::dirty::DirtyChunks;

#[cfg(feature = "gpu")]
use crate::engine::types::GPUResourceID;

#[cfg(feature = "gpu")]
use crate::gpu::{GPUResource, GPUResourceRegistry, GpuWorldState};

/// Core ECS storage and orchestration structure.
pub struct ECSData {
    archetypes: Vec<Archetype>,
    signature_map: HashMap<[u64; SIGNATURE_SIZE], ArchetypeID>,
    archetype_generation: u64,
    query_match_cache: RwLock<HashMap<QueryMatchKey, QueryMatchEntry>>,
    shards: EntityShards,
    next_spawn_shard: ShardID,
    registry: Arc<RwLock<ComponentRegistry>>,

    #[cfg(feature = "gpu")]
    gpu_dirty_chunks: DirtyChunks,

    #[cfg(feature = "gpu")]
    gpu_resources: GPUResourceRegistry,

    #[cfg(feature = "gpu")]
    gpu_world_state: GpuWorldState,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct QueryMatchKey {
    read: [u64; SIGNATURE_SIZE],
    write: [u64; SIGNATURE_SIZE],
    without: [u64; SIGNATURE_SIZE],
}

impl QueryMatchKey {
    fn from_signature(signature: &crate::engine::query::QuerySignature) -> Self {
        Self {
            read: signature.read.components,
            write: signature.write.components,
            without: signature.without.components,
        }
    }
}

#[derive(Clone, Debug)]
struct QueryMatchEntry {
    generation: u64,
    archetype_ids: Vec<ArchetypeID>,
}

/// Result of a deferred command drain that stopped before applying the whole batch.
pub(crate) struct CommandDrainFailure {
    /// Error returned by the failed command.
    pub(crate) error: ECSError,
    /// Commands that were never attempted and must remain queued.
    pub(crate) unapplied: Vec<Command>,
    /// Lifecycle events produced by commands that completed before the failure.
    pub(crate) events: CommandEvents,
}

impl ECSData {
    /// Creates a new `ECSData` instance with the specified number of entity shards
    /// and component registry.
    ///
    /// # Arguments
    /// * `shards` - Configuration for entity sharding (number of shards)
    /// * `registry` - Shared component registry for type resolution
    pub fn new(shards: EntityShards, registry: Arc<RwLock<ComponentRegistry>>) -> Self {
        Self {
            archetypes: Vec::new(),
            signature_map: HashMap::new(),
            archetype_generation: 0,
            query_match_cache: RwLock::new(HashMap::new()),
            shards,
            next_spawn_shard: 0,
            registry,

            #[cfg(feature = "gpu")]
            gpu_dirty_chunks: DirtyChunks::new(),

            #[cfg(feature = "gpu")]
            gpu_resources: GPUResourceRegistry::new(),

            #[cfg(feature = "gpu")]
            gpu_world_state: GpuWorldState::new(),
        }
    }

    /// Returns a reference to the component registry.
    ///
    /// The registry is shared via `Arc<RwLock<ComponentRegistry>>`, allowing
    /// concurrent reads and exclusive writes for registration of new components.
    #[inline]
    pub fn registry(&self) -> &Arc<RwLock<ComponentRegistry>> {
        &self.registry
    }

    #[inline]
    fn pick_spawn_shard(&mut self) -> ShardID {
        let shard = self.next_spawn_shard;
        self.next_spawn_shard = (self.next_spawn_shard + 1) % (self.shards.shard_count() as u16);
        shard
    }

    #[cfg(feature = "gpu")]
    #[inline]
    /// Returns a reference to the dirty chunks tracker.
    ///
    /// The dirty chunks tracker records which component chunks were modified
    /// during query execution, enabling selective CPU->GPU uploads. This is
    /// used by the GPU synchronisation layer to minimise data transfer
    /// between host and device.
    pub fn gpu_dirty_chunks(&self) -> &DirtyChunks {
        &self.gpu_dirty_chunks
    }

    #[cfg(feature = "gpu")]
    #[inline]
    pub(crate) fn gpu_world_state(&self) -> &GpuWorldState {
        &self.gpu_world_state
    }

    #[cfg(feature = "gpu")]
    #[inline]
    pub(crate) fn gpu_world_state_mut(&mut self) -> &mut GpuWorldState {
        &mut self.gpu_world_state
    }

    #[cfg(feature = "gpu")]
    #[inline]
    pub(crate) fn gpu_execution_parts(
        &mut self,
    ) -> (
        &[Archetype],
        &DirtyChunks,
        &mut GpuWorldState,
        &GPUResourceRegistry,
    ) {
        (
            &self.archetypes,
            &self.gpu_dirty_chunks,
            &mut self.gpu_world_state,
            &self.gpu_resources,
        )
    }

    #[cfg(feature = "gpu")]
    #[inline]
    pub(crate) fn gpu_download_parts(&mut self) -> (&mut [Archetype], &mut GpuWorldState) {
        (&mut self.archetypes, &mut self.gpu_world_state)
    }

    fn get_or_create_archetype(&mut self, signature: &Signature) -> ECSResult<ArchetypeID> {
        let key = signature.components;
        if let Some(&id) = self.signature_map.get(&key) {
            return Ok(id);
        }

        let id = self.archetypes.len() as ArchetypeID;
        self.signature_map.insert(key, id);

        let registry = self
            .registry
            .read()
            .map_err(|_| ECSError::from(RegistryError::PoisonedLock))?;
        let arch = Archetype::new(id, *signature, &registry)?;
        self.archetypes.push(arch);
        drop(registry);
        self.invalidate_query_match_cache();

        Ok(id)
    }

    fn invalidate_query_match_cache(&mut self) {
        self.archetype_generation = self.archetype_generation.wrapping_add(1);
        if let Ok(cache) = self.query_match_cache.get_mut() {
            cache.clear();
        }
    }

    fn matching_archetype_ids(
        &self,
        query: &crate::engine::query::QuerySignature,
    ) -> Result<Vec<ArchetypeID>, ExecutionError> {
        let key = QueryMatchKey::from_signature(query);
        let generation = self.archetype_generation;

        {
            let cache =
                self.query_match_cache
                    .read()
                    .map_err(|_| ExecutionError::LockPoisoned {
                        what: "query match cache",
                    })?;
            if let Some(entry) = cache.get(&key) {
                if entry.generation == generation {
                    return Ok(entry.archetype_ids.clone());
                }
            }
        }

        let mut archetype_ids = Vec::new();
        for archetype in &self.archetypes {
            if query.requires_all(archetype.signature()) {
                archetype_ids.push(archetype.archetype_id());
            }
        }

        let mut cache =
            self.query_match_cache
                .write()
                .map_err(|_| ExecutionError::LockPoisoned {
                    what: "query match cache",
                })?;
        cache.insert(
            key,
            QueryMatchEntry {
                generation,
                archetype_ids: archetype_ids.clone(),
            },
        );
        Ok(archetype_ids)
    }

    #[inline]
    fn get_archetype_pair_mut(
        archetypes: &mut [Archetype],
        a: ArchetypeID,
        b: ArchetypeID,
    ) -> ECSResult<(&mut Archetype, &mut Archetype)> {
        if a == b {
            return Err(InternalViolation::ArchetypePairSameId.into());
        }

        let (low, high) = if a < b { (a, b) } else { (b, a) };
        let (head, tail) = archetypes.split_at_mut(high as usize);

        let left = &mut head[low as usize];
        let right = &mut tail[0];

        Ok(if a < b { (left, right) } else { (right, left) })
    }

    /// Adds a component to an existing entity.
    ///
    /// If the component type already exists on the entity, this operation will fail.
    /// The entity is moved to a new archetype that includes the added component.
    ///
    /// # Arguments
    /// * `entity` - The target entity
    /// * `added_component_id` - The `ComponentID` of the component to add
    /// * `added_value` - The boxed initial value for the component
    ///
    /// # Errors
    /// Returns an error if:
    /// * The entity is stale (already despawned)
    /// * The component already exists on the entity
    /// * The registry lock is poisoned
    /// * Archetype creation or component storage allocation fails
    pub fn add_component(
        &mut self,
        entity: Entity,
        added_component_id: ComponentID,
        added_value: Box<dyn std::any::Any>,
    ) -> ECSResult<()> {
        {
            let registry = self
                .registry
                .read()
                .map_err(|_| ECSError::from(RegistryError::PoisonedLock))?;
            registry.require_component_id(added_component_id)?;
        }

        let Some(location) = self.shards.get_location(entity)? else {
            return Err(SpawnError::StaleEntity(StaleEntityError).into());
        };
        let source_id = location.archetype;

        if self.archetypes[source_id as usize]
            .signature()
            .try_has(added_component_id)?
        {
            return Err(MoveError::ComponentAlreadyPresent {
                component_id: added_component_id,
            }
            .into());
        }

        let mut new_signature = *self.archetypes[source_id as usize].signature();
        new_signature.try_set(added_component_id)?;

        let destination_id = self.get_or_create_archetype(&new_signature)?;
        let source_sig = *self.archetypes[source_id as usize].signature();
        let shards = &self.shards;

        let (source, destination) =
            Self::get_archetype_pair_mut(&mut self.archetypes, source_id, destination_id)?;

        let registry = self
            .registry
            .read()
            .map_err(|_| ECSError::from(RegistryError::PoisonedLock))?;

        let factory = || registry.make_empty_component(added_component_id);

        destination
            .ensure_component(added_component_id, factory)
            .map_err(ECSError::from)?;

        Self::ensure_shared_components(&source_sig, destination, added_component_id, &registry)?;

        drop(registry);

        source.move_row_to_archetype(
            destination,
            shards,
            entity,
            (location.chunk, location.row),
            vec![(added_component_id, added_value)],
        )?;

        Ok(())
    }

    /// Removes a component from an existing entity.
    ///
    /// The entity is moved to a new archetype that excludes the removed component.
    /// If the entity has no components remaining after removal, it is despawned.
    ///
    /// # Arguments
    /// * `entity` - The target entity
    /// * `removed_component_id` - The `ComponentID` of the component to remove
    ///
    /// # Returns
    /// Returns `Ok(())` if:
    /// * The component was successfully removed
    /// * The entity didn't have the component (no-op)
    /// * The entity was despawned due to having no components left
    ///
    /// # Errors
    /// Returns an error if:
    /// * The entity is stale (already despawned)
    /// * The registry lock is poisoned
    /// * Archetype creation or entity movement fails
    pub fn remove_component(
        &mut self,
        entity: Entity,
        removed_component_id: ComponentID,
    ) -> ECSResult<()> {
        {
            let registry = self
                .registry
                .read()
                .map_err(|_| ECSError::from(RegistryError::PoisonedLock))?;
            registry.require_component_id(removed_component_id)?;
        }

        let Some(location) = self.shards.get_location(entity)? else {
            return Err(SpawnError::StaleEntity(StaleEntityError).into());
        };
        let source_id = location.archetype;

        if !self.archetypes[source_id as usize]
            .signature()
            .try_has(removed_component_id)?
        {
            return Ok(());
        }

        let mut new_signature = *self.archetypes[source_id as usize].signature();
        new_signature.try_clear(removed_component_id)?;

        if new_signature.components.iter().all(|&bits| bits == 0) {
            self.archetypes[source_id as usize].despawn_on(&self.shards, entity)?;
            return Ok(());
        }

        let destination_id = self.get_or_create_archetype(&new_signature)?;
        let source_sig = *self.archetypes[source_id as usize].signature();
        let shards = &self.shards;

        let (source_arch, dest_arch) =
            Self::get_archetype_pair_mut(&mut self.archetypes, source_id, destination_id)?;

        let registry = self
            .registry
            .read()
            .map_err(|_| ECSError::from(RegistryError::PoisonedLock))?;

        Self::ensure_shared_components(&source_sig, dest_arch, removed_component_id, &registry)?;

        drop(registry);

        source_arch.move_row_to_archetype(
            dest_arch,
            shards,
            entity,
            (location.chunk, location.row),
            Vec::new(),
        )?;

        Ok(())
    }

    /// Writes a new component value directly into the entity's current archetype
    /// row, **without** any archetype transition.
    ///
    /// The old value stored at the target slot is dropped correctly using the
    /// stored component's real drop glue, so this path is sound for any
    /// registered component type regardless of whether it implements `Copy`.
    ///
    /// ## Errors
    ///
    /// - [`ExecutionError::MissingComponent`] - the entity's archetype does not
    ///   contain `component_id`.
    /// - [`ExecutionError::InternalExecutionError`] - the dynamic `TypeId` of
    ///   `value` does not match the column's registered element type, or the
    ///   entity's row is out of range.
    /// - [`SpawnError::StaleEntity`] - the entity no longer exists.
    /// - [`ExecutionError::LockPoisoned`] - the component column lock is poisoned.
    ///
    /// ## Safety / exclusivity
    ///
    /// Must only be called while holding the exclusive phase lock
    /// (`PhaseWrite`), which is the invariant of `apply_deferred_commands`.
    pub(crate) fn apply_set_command(
        &mut self,
        entity: Entity,
        component_id: ComponentID,
        value: Box<dyn std::any::Any + Send>,
    ) -> ECSResult<()> {
        {
            let registry = self
                .registry
                .read()
                .map_err(|_| ECSError::from(RegistryError::PoisonedLock))?;
            registry.require_component_id(component_id)?;
        }

        // 1. Resolve entity location.
        let location = self
            .shards
            .get_location(entity)?
            .ok_or(ECSError::from(SpawnError::StaleEntity(StaleEntityError)))?;

        // 2. Verify archetype contains the component.
        let archetype = self
            .archetypes
            .get(location.archetype as usize)
            .ok_or(ECSError::from(ExecutionError::InternalExecutionError))?;

        if !archetype.has(component_id) {
            return Err(ECSError::from(ExecutionError::MissingComponent {
                component_id,
            }));
        }

        // 3. Acquire an exclusive write lock on the column.
        let col_lock = archetype
            .component_locked(component_id)
            .ok_or(ECSError::from(ExecutionError::MissingComponent {
                component_id,
            }))?;

        let mut col_guard = col_lock.write().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "component column (set)",
            })
        })?;

        // 4. Delegate the actual replace to the storage layer, which:
        //    - type-checks the incoming value against the column,
        //    - drops the old value in place using its real drop glue,
        //    - consumes the incoming box cleanly (no leak, no double-free).
        //
        // `Box<dyn Any + Send>` coerces freely to `Box<dyn Any>`; the `Send`
        // bound is not needed at the apply site because we hold the exclusive
        // phase lock.
        let value: Box<dyn std::any::Any> = value;
        match col_guard.replace_slot_dyn(location.chunk, location.row, value) {
            Ok(()) => Ok(()),
            Err(AttributeError::TypeMismatch(_)) => {
                Err(ECSError::from(ExecutionError::InternalExecutionError))
            }
            Err(AttributeError::Position(_)) => {
                Err(ECSError::from(ExecutionError::InternalExecutionError))
            }
            Err(other) => Err(ECSError::from(other)),
        }
    }

    /// Applies all queued deferred commands.
    /// `Spawn`, `Despawn`, `Add`, and `Remove` variants.
    pub fn apply_deferred_commands(&mut self, commands: Vec<Command>) -> ECSResult<CommandEvents> {
        self.apply_deferred_commands_partial(commands)
            .map_err(|failure| failure.error)
    }

    /// Applies deferred commands until completion or the first error.
    ///
    /// On error, commands that were never attempted are returned so the manager
    /// can preserve their relative order in the deferred queue.
    #[allow(clippy::result_large_err)]
    pub(crate) fn apply_deferred_commands_partial(
        &mut self,
        commands: Vec<Command>,
    ) -> Result<CommandEvents, CommandDrainFailure> {
        let mut events = CommandEvents::default();
        #[cfg(feature = "gpu")]
        let mut applied_any = false;
        let mut iter = commands.into_iter();

        while let Some(command) = iter.next() {
            let result = match command {
                Command::Spawn { bundle } => {
                    let signature = bundle.signature();
                    let archetype_id = self.get_or_create_archetype(&signature);
                    let archetype_id = match archetype_id {
                        Ok(id) => id,
                        Err(error) => {
                            return Err(CommandDrainFailure {
                                error,
                                unapplied: iter.collect(),
                                events,
                            });
                        }
                    };
                    let shard_id = self.pick_spawn_shard();
                    let archetype = &mut self.archetypes[archetype_id as usize];
                    archetype
                        .spawn_on(&self.shards, shard_id, bundle)
                        .map(|entity| {
                            events.spawned.push(SpawnEvent::untagged(entity));
                        })
                }

                Command::SpawnTagged { bundle, tag } => {
                    let signature = bundle.signature();
                    let archetype_id = self.get_or_create_archetype(&signature);
                    let archetype_id = match archetype_id {
                        Ok(id) => id,
                        Err(error) => {
                            return Err(CommandDrainFailure {
                                error,
                                unapplied: iter.collect(),
                                events,
                            });
                        }
                    };
                    let shard_id = self.pick_spawn_shard();
                    let archetype = &mut self.archetypes[archetype_id as usize];
                    archetype
                        .spawn_on(&self.shards, shard_id, bundle)
                        .map(|entity| {
                            events.spawned.push(SpawnEvent::tagged(entity, tag));
                        })
                }

                Command::SpawnBatchTagged { batch, template_id } => {
                    let archetype_id = self.get_or_create_archetype(&batch.signature);
                    let archetype_id = match archetype_id {
                        Ok(id) => id,
                        Err(error) => {
                            return Err(CommandDrainFailure {
                                error,
                                unapplied: iter.collect(),
                                events,
                            });
                        }
                    };

                    let mut spawned = Vec::with_capacity(batch.count);
                    let mut columns: Vec<_> = batch
                        .columns
                        .into_iter()
                        .map(|column| (column.component_id, column.values.into_iter()))
                        .collect();
                    if let Err(error) =
                        self.archetypes[archetype_id as usize].reserve_additional_rows(batch.count)
                    {
                        return Err(CommandDrainFailure {
                            error,
                            unapplied: iter.collect(),
                            events,
                        });
                    }

                    for _ in 0..batch.count {
                        let mut bundle = crate::engine::component::Bundle::new();
                        for (component_id, values) in &mut columns {
                            let Some(value) = values.next() else {
                                return Err(CommandDrainFailure {
                                    error: ECSError::from(SpawnError::MissingComponent {
                                        type_id: std::any::TypeId::of::<()>(),
                                        name: "batch column value",
                                    }),
                                    unapplied: iter.collect(),
                                    events,
                                });
                            };
                            bundle.insert_boxed(*component_id, value);
                        }

                        let shard_id = self.pick_spawn_shard();
                        let archetype = &mut self.archetypes[archetype_id as usize];
                        match archetype.spawn_on(&self.shards, shard_id, bundle) {
                            Ok(entity) => {
                                events
                                    .spawned
                                    .push(SpawnEvent::template_tagged(entity, template_id));
                                spawned.push(entity);
                            }
                            Err(error) => {
                                return Err(CommandDrainFailure {
                                    error,
                                    unapplied: iter.collect(),
                                    events,
                                });
                            }
                        }
                    }
                    events.spawned_batches.push(TemplateLifecycleBatch {
                        template_id,
                        entities: spawned,
                    });
                    Ok(())
                }

                Command::Despawn { entity } => {
                    let loc = self
                        .shards
                        .get_location(entity)
                        .map_err(ECSError::from)
                        .and_then(|loc| {
                            loc.ok_or(ECSError::from(SpawnError::StaleEntity(StaleEntityError)))
                        });
                    match loc {
                        Ok(loc) => {
                            let archetype = &mut self.archetypes[loc.archetype as usize];
                            archetype.despawn_on(&self.shards, entity).map(|()| {
                                events.despawned.push(DespawnEvent::untagged(entity));
                            })
                        }
                        Err(error) => Err(error),
                    }
                }

                Command::DespawnTagged { entity, tag } => {
                    let loc = self
                        .shards
                        .get_location(entity)
                        .map_err(ECSError::from)
                        .and_then(|loc| {
                            loc.ok_or(ECSError::from(SpawnError::StaleEntity(StaleEntityError)))
                        });
                    match loc {
                        Ok(loc) => {
                            let archetype = &mut self.archetypes[loc.archetype as usize];
                            archetype.despawn_on(&self.shards, entity).map(|()| {
                                events.despawned.push(DespawnEvent::tagged(entity, tag));
                            })
                        }
                        Err(error) => Err(error),
                    }
                }

                Command::DespawnBatchTagged {
                    entities,
                    template_id,
                } => {
                    let mut despawned = Vec::with_capacity(entities.len());
                    for entity in entities {
                        let loc = self
                            .shards
                            .get_location(entity)
                            .map_err(ECSError::from)
                            .and_then(|loc| {
                                loc.ok_or(ECSError::from(SpawnError::StaleEntity(StaleEntityError)))
                            });
                        match loc {
                            Ok(loc) => {
                                let archetype = &mut self.archetypes[loc.archetype as usize];
                                if let Err(error) = archetype.despawn_on(&self.shards, entity) {
                                    return Err(CommandDrainFailure {
                                        error,
                                        unapplied: iter.collect(),
                                        events,
                                    });
                                }
                                events
                                    .despawned
                                    .push(DespawnEvent::template_tagged(entity, template_id));
                                despawned.push(entity);
                            }
                            Err(error) => {
                                return Err(CommandDrainFailure {
                                    error,
                                    unapplied: iter.collect(),
                                    events,
                                });
                            }
                        }
                    }
                    events.despawned_batches.push(TemplateLifecycleBatch {
                        template_id,
                        entities: despawned,
                    });
                    Ok(())
                }

                Command::Add {
                    entity,
                    component_id,
                    value,
                } => self.add_component(entity, component_id, value),

                Command::Remove {
                    entity,
                    component_id,
                } => self.remove_component(entity, component_id),

                Command::Set {
                    entity,
                    component_id,
                    value,
                } => self.apply_set_command(entity, component_id, value),
            };

            match result {
                Ok(()) => {
                    #[cfg(feature = "gpu")]
                    {
                        applied_any = true;
                    }
                }
                Err(error) => {
                    #[cfg(feature = "gpu")]
                    if applied_any {
                        self.notify_all_archetypes_dirty();
                    }
                    return Err(CommandDrainFailure {
                        error,
                        unapplied: iter.collect(),
                        events,
                    });
                }
            }
        }

        #[cfg(feature = "gpu")]
        if applied_any {
            self.notify_all_archetypes_dirty();
        }

        Ok(events)
    }

    #[cfg(feature = "gpu")]
    fn notify_all_archetypes_dirty(&self) {
        for archetype in &self.archetypes {
            self.gpu_dirty_chunks
                .notify_archetype_changed(archetype.archetype_id());
        }
    }

    fn ensure_shared_components(
        source_sig: &Signature,
        destination: &mut Archetype,
        excluded: ComponentID,
        registry: &ComponentRegistry,
    ) -> ECSResult<()> {
        for cid in source_sig.iterate_over_components() {
            if cid == excluded {
                continue;
            }
            let factory = || registry.make_empty_component(cid);
            destination
                .ensure_component(cid, factory)
                .map_err(ECSError::from)?;
        }
        Ok(())
    }

    /// Reads a component value from a specific entity, cloning it.
    ///
    /// This method provides random-access read of a single entity's component.
    /// For bulk operations, prefer using queries with `for_each` or `reduce`.
    ///
    /// # Type Parameters
    /// * `T` - The expected component type. Must match the type registered for
    ///   `component_id`.
    ///
    /// # Arguments
    /// * `entity` - The target entity
    /// * `component_id` - The `ComponentID` of the component to read
    ///
    /// # Returns
    /// Returns a cloned copy of the component value.
    ///
    /// # Errors
    /// Returns an error if:
    /// * The entity is stale (already despawned)
    /// * The archetype or component column is missing
    /// * The component type `T` doesn't match the actual stored type
    /// * The component is currently borrowed for writing
    /// * The column lock is poisoned
    pub fn read_component<T: 'static + Clone>(
        &self,
        entity: Entity,
        component_id: ComponentID,
    ) -> ECSResult<T> {
        let loc = self
            .shards
            .get_location(entity)?
            .ok_or(ECSError::from(SpawnError::StaleEntity(StaleEntityError)))?;

        let arch = self
            .archetypes
            .get(loc.archetype as usize)
            .ok_or(ECSError::from(ExecutionError::InternalExecutionError))?;

        let col_lock = arch.component_locked(component_id).ok_or(ECSError::from(
            ExecutionError::MissingComponent { component_id },
        ))?;

        let col = col_lock.try_read().map_err(|e| match e {
            std::sync::TryLockError::WouldBlock => ECSError::from(ExecutionError::BorrowConflict {
                component_id,
                held: AccessKind::Write,
                requested: AccessKind::Read,
            }),
            std::sync::TryLockError::Poisoned(_) => ECSError::from(ExecutionError::LockPoisoned {
                what: "component column",
            }),
        })?;

        if col.element_type_id() != TypeId::of::<T>() {
            return Err(ECSError::from(ExecutionError::InternalExecutionError));
        }

        let chunk_len = arch.chunk_valid_length(loc.chunk as usize)?;
        let (ptr, bytes) = col
            .chunk_bytes(loc.chunk, chunk_len)
            .ok_or(ECSError::from(ExecutionError::InternalExecutionError))?;

        let slice: &[T] = unsafe { crate::engine::storage::cast_slice::<T>(ptr, bytes) };

        let row = loc.row as usize;
        if row >= slice.len() {
            return Err(ECSError::from(ExecutionError::InternalExecutionError));
        }

        Ok(slice[row].clone())
    }

    pub(crate) fn for_each_abstraction_unchecked(
        &self,
        query: BuiltQuery,
        f: impl Fn(&[&[u8]], &mut [&mut [u8]]) + Send + Sync,
    ) -> Result<(), ExecutionError> {
        let matches = self.matching_archetype_ids(query.signature())?;
        #[cfg(feature = "gpu")]
        {
            super::query_executor::for_each_unchecked(
                &self.archetypes,
                matches,
                query,
                &self.gpu_dirty_chunks,
                crate::engine::activation::current_activation_context(),
                f,
            )
        }

        #[cfg(not(feature = "gpu"))]
        {
            super::query_executor::for_each_unchecked(
                &self.archetypes,
                matches,
                query,
                crate::engine::activation::current_activation_context(),
                f,
            )
        }
    }

    /// Parallel chunk iteration whose closure may return an error.
    ///
    /// Identical contract to [`for_each_abstraction_unchecked`] for borrow,
    /// phase, and aliasing safety, but the user closure returns
    /// [`ECSResult<()>`]. If any chunk's closure returns `Err`, every other
    /// chunk in the same archetype short-circuits and the lowest-chunk-index
    /// error is returned. The selected error is therefore deterministic in
    /// the chunk identifier rather than wall-clock first-to-error, so
    /// repeated runs over identical data produce identical error reports.
    pub(crate) fn for_each_abstraction_fallible_unchecked(
        &self,
        query: BuiltQuery,
        f: impl Fn(&[&[u8]], &mut [&mut [u8]]) -> crate::engine::error::ECSResult<()> + Send + Sync,
    ) -> crate::engine::error::ECSResult<()> {
        let matches = self
            .matching_archetype_ids(query.signature())
            .map_err(ECSError::from)?;
        #[cfg(feature = "gpu")]
        {
            super::query_executor::for_each_fallible_unchecked(
                &self.archetypes,
                matches,
                query,
                &self.gpu_dirty_chunks,
                crate::engine::activation::current_activation_context(),
                f,
            )
        }

        #[cfg(not(feature = "gpu"))]
        {
            super::query_executor::for_each_fallible_unchecked(
                &self.archetypes,
                matches,
                query,
                crate::engine::activation::current_activation_context(),
                f,
            )
        }
    }

    pub(crate) fn reduce_abstraction_unchecked<R>(
        &self,
        query: BuiltQuery,
        init: impl Fn() -> R + Send + Sync,
        fold_chunk: impl Fn(&mut R, &[&[u8]], usize) + Send + Sync,
        combine: impl Fn(&mut R, R) + Send + Sync,
    ) -> Result<R, ExecutionError>
    where
        R: Send + 'static,
    {
        let matches = self.matching_archetype_ids(query.signature())?;
        super::query_executor::reduce_unchecked(
            &self.archetypes,
            matches,
            query,
            init,
            fold_chunk,
            combine,
        )
    }

    #[cfg(feature = "gpu")]
    /// Registers a GPU resource with the ECS world.
    ///
    /// GPU resources are world-owned buffers, textures, or bind groups that
    /// persist across frames. Once registered, the resource is managed by the
    /// ECS and can be accessed from GPU systems.
    ///
    /// # Type Parameters
    /// * `R` - The GPU resource type, which must implement [`GPUResource`].
    ///
    /// # Arguments
    /// * `r` - The resource instance to register.
    ///
    /// # Returns
    /// A [`GPUResourceID`] that can be used to reference this resource in
    /// GPU systems and queries.
    ///
    /// # Example
    /// ```text
    /// let buffer = world.create_buffer(&device, &descriptor);
    /// let resource_id = ecs_data.register_gpu_resource(buffer);
    /// ```
    pub fn register_gpu_resource<R: GPUResource + 'static>(
        &mut self,
        r: R,
    ) -> ECSResult<GPUResourceID> {
        self.gpu_resources.register(r)
    }

    #[cfg(feature = "gpu")]
    #[inline]
    /// Returns an immutable reference to the GPU resource registry.
    ///
    /// The registry holds all world-owned GPU resources (buffers, textures,
    /// bind groups) and provides lookup by [`GPUResourceID`]. This is used
    /// by GPU systems to access resources during execution.
    pub fn gpu_resources(&self) -> &GPUResourceRegistry {
        &self.gpu_resources
    }

    #[cfg(feature = "gpu")]
    #[inline]
    /// Returns a mutable reference to the GPU resource registry.
    ///
    /// Allows adding, removing, or modifying GPU resources owned by the world.
    /// Used during resource creation, clean-up, or when updating resource contents.
    ///
    /// # Safety
    /// Callers must ensure that no GPU systems are currently executing when
    /// mutating the registry, as this may invalidate active resource bindings.
    pub fn gpu_resources_mut(&mut self) -> &mut GPUResourceRegistry {
        &mut self.gpu_resources
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::component::Bundle;
    use crate::engine::entity::EntityShards;
    use std::sync::{Arc, RwLock};

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Marker(u32);

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Extra(u32);

    fn test_manager() -> (crate::engine::manager::ECSManager, ComponentID, ComponentID) {
        let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
        let (marker_id, extra_id) = {
            let mut registry = registry.write().unwrap();
            let marker_id = registry.register::<Marker>().unwrap();
            let extra_id = registry.register::<Extra>().unwrap();
            registry.freeze();
            (marker_id, extra_id)
        };
        (
            crate::engine::manager::ECSManager::with_registry(
                EntityShards::new(1).unwrap(),
                registry,
            ),
            marker_id,
            extra_id,
        )
    }

    fn marker_bundle(marker_id: ComponentID, value: u32) -> Bundle {
        let mut bundle = Bundle::new();
        bundle.insert(marker_id, Marker(value));
        bundle
    }

    fn spawn_marker(
        ecs: &crate::engine::manager::ECSManager,
        marker_id: ComponentID,
        value: u32,
    ) -> Entity {
        let world = ecs.world_ref();
        world
            .defer(Command::Spawn {
                bundle: marker_bundle(marker_id, value),
            })
            .unwrap();
        ecs.apply_deferred_commands().unwrap().spawned[0].entity
    }

    fn assert_all_columns_empty(data: &ECSData) {
        for archetype in &data.archetypes {
            assert_eq!(archetype.length().unwrap(), 0);
            for (_, len) in archetype.component_lengths_for_test() {
                assert_eq!(len, 0);
            }
        }
    }

    #[test]
    fn wrong_type_spawn_rolls_back_all_written_columns() {
        let (ecs, marker_id, extra_id) = test_manager();
        let world = ecs.world_ref();
        let mut bundle = Bundle::new();
        bundle.insert(marker_id, Marker(1));
        bundle.insert(extra_id, Marker(2));

        world.defer(Command::Spawn { bundle }).unwrap();
        assert!(ecs.apply_deferred_commands().is_err());

        world
            .with_exclusive(|data| {
                assert_all_columns_empty(data);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn failed_add_component_leaves_source_destination_and_location_unchanged() {
        let (ecs, marker_id, extra_id) = test_manager();
        let entity = spawn_marker(&ecs, marker_id, 7);
        let world = ecs.world_ref();

        let before = world
            .with_exclusive(|data| {
                let loc = data.shards.get_location(entity)?.unwrap();
                let source_len = data.archetypes[loc.archetype as usize].length()?;
                let marker_len = data.archetypes[loc.archetype as usize]
                    .component_locked(marker_id)
                    .unwrap()
                    .read()
                    .unwrap()
                    .length();
                Ok((loc, source_len, marker_len))
            })
            .unwrap();

        world
            .defer(Command::Add {
                entity,
                component_id: extra_id,
                value: Box::new(Marker(99)),
            })
            .unwrap();
        assert!(ecs.apply_deferred_commands().is_err());

        world
            .with_exclusive(|data| {
                let loc = data.shards.get_location(entity)?.unwrap();
                assert_eq!(loc.archetype, before.0.archetype);
                assert_eq!(loc.chunk, before.0.chunk);
                assert_eq!(loc.row, before.0.row);
                assert_eq!(
                    data.archetypes[loc.archetype as usize].length().unwrap(),
                    before.1
                );
                assert_eq!(
                    data.archetypes[loc.archetype as usize]
                        .component_locked(marker_id)
                        .unwrap()
                        .read()
                        .unwrap()
                        .length(),
                    before.2
                );
                for archetype in &data.archetypes {
                    if archetype.archetype_id() != loc.archetype {
                        assert_eq!(archetype.length().unwrap(), 0);
                        for (_, len) in archetype.component_lengths_for_test() {
                            assert_eq!(len, 0);
                        }
                    }
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn duplicate_add_component_returns_error_without_migration() {
        let (ecs, marker_id, _extra_id) = test_manager();
        let entity = spawn_marker(&ecs, marker_id, 7);
        let world = ecs.world_ref();

        world
            .defer(Command::Add {
                entity,
                component_id: marker_id,
                value: Box::new(Marker(8)),
            })
            .unwrap();

        assert!(matches!(
            ecs.apply_deferred_commands(),
            Err(ECSError::Move(MoveError::ComponentAlreadyPresent { .. }))
        ));
    }

    #[test]
    fn disjoint_archetype_move_replaces_source_only_with_destination_only_components() {
        let (ecs, marker_id, extra_id) = test_manager();
        let entity = spawn_marker(&ecs, marker_id, 7);
        let world = ecs.world_ref();

        let (source_id, destination_id) = world
            .with_exclusive(|data| {
                let location = data.shards.get_location(entity)?.unwrap();
                let source_id = location.archetype;

                let mut destination_signature = Signature::default();
                destination_signature.set(extra_id);
                let destination_id = data.get_or_create_archetype(&destination_signature)?;

                let shards = &data.shards;
                let (source, destination) = ECSData::get_archetype_pair_mut(
                    &mut data.archetypes,
                    source_id,
                    destination_id,
                )?;

                source.move_row_to_archetype(
                    destination,
                    shards,
                    entity,
                    (location.chunk, location.row),
                    vec![(extra_id, Box::new(Extra(9)) as Box<dyn std::any::Any>)],
                )?;

                Ok((source_id, destination_id))
            })
            .unwrap();

        world
            .with_exclusive(|data| {
                let location = data.shards.get_location(entity)?.unwrap();
                assert_eq!(location.archetype, destination_id);
                assert_eq!(data.archetypes[source_id as usize].length().unwrap(), 0);
                assert_eq!(
                    data.archetypes[source_id as usize]
                        .component_locked(marker_id)
                        .unwrap()
                        .read()
                        .unwrap()
                        .length(),
                    0
                );
                assert_eq!(
                    data.archetypes[destination_id as usize].length().unwrap(),
                    1
                );
                assert_eq!(
                    data.archetypes[destination_id as usize]
                        .component_locked(extra_id)
                        .unwrap()
                        .read()
                        .unwrap()
                        .length(),
                    1
                );
                Ok(())
            })
            .unwrap();
    }

    #[cfg(feature = "gpu")]
    fn clear_dirty_for_entity(
        data: &mut ECSData,
        entity: Entity,
        component_id: ComponentID,
    ) -> (ArchetypeID, usize) {
        let loc = data.shards.get_location(entity).unwrap().unwrap();
        let chunk_count = data.archetypes[loc.archetype as usize]
            .chunk_count()
            .unwrap();
        let _ = data
            .gpu_dirty_chunks
            .take_dirty_chunks(loc.archetype, component_id, chunk_count);
        (loc.archetype, chunk_count)
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn cpu_write_query_marks_gpu_dirty_chunks() {
        let (ecs, marker_id, _extra_id) = test_manager();
        let entity = spawn_marker(&ecs, marker_id, 1);
        let world = ecs.world_ref();
        let (archetype_id, chunk_count) = world
            .with_exclusive(|data| Ok(clear_dirty_for_entity(data, entity, marker_id)))
            .unwrap();

        let query = world
            .query()
            .unwrap()
            .write::<Marker>()
            .unwrap()
            .build()
            .unwrap();
        world
            .for_each_w1(query, |marker: &mut Marker| marker.0 += 1)
            .unwrap();

        world
            .with_exclusive(|data| {
                let dirty =
                    data.gpu_dirty_chunks
                        .take_dirty_chunks(archetype_id, marker_id, chunk_count);
                assert_eq!(dirty, vec![0]);
                Ok(())
            })
            .unwrap();
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn cpu_read_query_does_not_mark_gpu_dirty_chunks() {
        let (ecs, marker_id, _extra_id) = test_manager();
        let entity = spawn_marker(&ecs, marker_id, 1);
        let world = ecs.world_ref();
        let (archetype_id, chunk_count) = world
            .with_exclusive(|data| Ok(clear_dirty_for_entity(data, entity, marker_id)))
            .unwrap();

        let query = world
            .query()
            .unwrap()
            .read::<Marker>()
            .unwrap()
            .build()
            .unwrap();
        world.for_each_r1(query, |_marker: &Marker| {}).unwrap();

        world
            .with_exclusive(|data| {
                let dirty =
                    data.gpu_dirty_chunks
                        .take_dirty_chunks(archetype_id, marker_id, chunk_count);
                assert!(dirty.is_empty());
                Ok(())
            })
            .unwrap();
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn fallible_cpu_write_query_marks_gpu_dirty_before_error() {
        let (ecs, marker_id, _extra_id) = test_manager();
        let entity = spawn_marker(&ecs, marker_id, 1);
        let world = ecs.world_ref();
        let (archetype_id, chunk_count) = world
            .with_exclusive(|data| Ok(clear_dirty_for_entity(data, entity, marker_id)))
            .unwrap();

        let query = world
            .query()
            .unwrap()
            .write::<Marker>()
            .unwrap()
            .build()
            .unwrap();
        let result = world.for_each_w1_fallible(query, |marker: &mut Marker| {
            marker.0 += 1;
            Err(ECSError::from(ExecutionError::InternalExecutionError))
        });
        assert!(result.is_err());

        world
            .with_exclusive(|data| {
                let dirty =
                    data.gpu_dirty_chunks
                        .take_dirty_chunks(archetype_id, marker_id, chunk_count);
                assert_eq!(dirty, vec![0]);
                Ok(())
            })
            .unwrap();
    }
}
