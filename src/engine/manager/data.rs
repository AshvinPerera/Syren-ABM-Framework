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
//! - **Archetype management** — archetypes are created lazily and looked up by
//!   component signature via `signature_map`. Each archetype is created with an
//!   explicit `&ComponentRegistry` rather than calling global registry functions,
//!   which enables multi-world support.
//! - **Entity placement** — entity locations `(archetype, chunk, row)` are tracked
//!   through [`EntityShards`] and kept consistent with archetype storage.
//! - **Structural mutations** — adding or removing a component migrates the entity's
//!   row to a new archetype whose signature reflects the change.
//! - **Deferred commands** — [`ECSData::apply_deferred_commands`] flushes a batch of
//!   [`Command`]s (spawn, despawn, add, remove) as a single synchronisation point.
//! - **Parallel iteration** — [`ECSData::for_each_abstraction_unchecked`] and
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
//!   during a query, enabling selective CPU→GPU uploads.
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

use std::sync::{
    Arc,
    RwLock,
    RwLockReadGuard,
    RwLockWriteGuard,
    Mutex,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::HashMap;

use smallvec::SmallVec;

use crate::engine::commands::Command;
use crate::engine::types::{
    ComponentID,
    ArchetypeID,
    ShardID,
    SIGNATURE_SIZE,
};
use crate::engine::query::{QuerySignature, BuiltQuery};
use crate::engine::archetype::{Archetype, ArchetypeMatch};
use crate::engine::entity::{Entity, EntityShards};
use crate::engine::storage::TypeErasedAttribute;
use crate::engine::component::{Signature, ComponentRegistry};
use crate::engine::error::{
    ECSResult, ECSError, ExecutionError, InternalViolation,
    RegistryError, SpawnError, StaleEntityError, AccessKind,
    AttributeError,
};

#[cfg(feature = "gpu")]
use crate::engine::dirty::DirtyChunks;

#[cfg(feature = "gpu")]
use crate::engine::types::GPUResourceID;

#[cfg(feature = "gpu")]
use crate::gpu::{GPUResource, GPUResourceRegistry};

use super::iteration::ChunkView;

/// Core ECS storage and orchestration structure.
pub struct ECSData {
    archetypes: Vec<Archetype>,
    signature_map: HashMap<[u64; SIGNATURE_SIZE], ArchetypeID>,
    shards: EntityShards,
    next_spawn_shard: ShardID,
    registry: Arc<RwLock<ComponentRegistry>>,

    #[cfg(feature = "gpu")]
    gpu_dirty_chunks: DirtyChunks,

    #[cfg(feature = "gpu")]
    gpu_resources: GPUResourceRegistry,
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
            shards,
            next_spawn_shard: 0,
            registry,

            #[cfg(feature = "gpu")]
            gpu_dirty_chunks: DirtyChunks::new(),

            #[cfg(feature = "gpu")]
            gpu_resources: GPUResourceRegistry::new(),
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
        self.next_spawn_shard =
            (self.next_spawn_shard + 1) % (self.shards.shard_count() as u16);
        shard
    }

    #[cfg(feature = "gpu")]
    #[inline]
    /// Returns a reference to the dirty chunks tracker.
    ///
    /// The dirty chunks tracker records which component chunks were modified
    /// during query execution, enabling selective CPU→GPU uploads. This is
    /// used by the GPU synchronisation layer to minimise data transfer
    /// between host and device.
    pub fn gpu_dirty_chunks(&self) -> &DirtyChunks {
        &self.gpu_dirty_chunks
    }

    fn get_or_create_archetype(&mut self, signature: &Signature) -> ECSResult<ArchetypeID> {
        let key = signature.components;
        if let Some(&id) = self.signature_map.get(&key) {
            return Ok(id);
        }

        let id = self.archetypes.len() as ArchetypeID;
        self.signature_map.insert(key, id);

        let registry = self.registry.read()
            .map_err(|_| ECSError::from(RegistryError::PoisonedLock))?;
        let arch = Archetype::new(id, *signature, &registry)?;
        self.archetypes.push(arch);

        Ok(id)
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
        let Some(location) = self.shards.get_location(entity)? else {
            return Err(SpawnError::StaleEntity(StaleEntityError).into());
        };
        let source_id = location.archetype;

        let mut new_signature = self.archetypes[source_id as usize].signature().clone();
        new_signature.set(added_component_id);

        let destination_id = self.get_or_create_archetype(&new_signature)?;
        let source_sig = self.archetypes[source_id as usize].signature().clone();
        let shards = &self.shards;

        let (source, destination) =
            Self::get_archetype_pair_mut(&mut self.archetypes, source_id, destination_id)?;

        let registry = self.registry.read()
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
        let Some(location) = self.shards.get_location(entity)? else {
            return Err(SpawnError::StaleEntity(StaleEntityError).into());
        };
        let source_id = location.archetype;

        if !self.archetypes[source_id as usize].has(removed_component_id) {
            return Ok(());
        }

        let mut new_signature = self.archetypes[source_id as usize].signature().clone();
        new_signature.clear(removed_component_id);

        if new_signature.components.iter().all(|&bits| bits == 0) {
            self.archetypes[source_id as usize]
                .despawn_on(&mut self.shards, entity)?;
            return Ok(());
        }

        let destination_id = self.get_or_create_archetype(&new_signature)?;
        let source_sig = self.archetypes[source_id as usize].signature().clone();
        let shards = &self.shards;

        let (source_arch, dest_arch) =
            Self::get_archetype_pair_mut(&mut self.archetypes, source_id, destination_id)?;

        let registry = self.registry.read()
            .map_err(|_| ECSError::from(RegistryError::PoisonedLock))?;

        Self::ensure_shared_components(
            &source_sig, dest_arch, removed_component_id, &registry,
        )?;

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
    /// - [`ExecutionError::MissingComponent`] — the entity's archetype does not
    ///   contain `component_id`.
    /// - [`ExecutionError::InternalExecutionError`] — the dynamic `TypeId` of
    ///   `value` does not match the column's registered element type, or the
    ///   entity's row is out of range.
    /// - [`SpawnError::StaleEntity`] — the entity no longer exists.
    /// - [`ExecutionError::LockPoisoned`] — the component column lock is poisoned.
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
        // 1. Resolve entity location.
        let location = self.shards.get_location(entity)?
            .ok_or(ECSError::from(SpawnError::StaleEntity(StaleEntityError)))?;

        // 2. Verify archetype contains the component.
        let archetype = self.archetypes
            .get(location.archetype as usize)
            .ok_or(ECSError::from(ExecutionError::InternalExecutionError))?;

        if !archetype.has(component_id) {
            return Err(ECSError::from(ExecutionError::MissingComponent { component_id }));
        }

        // 3. Acquire an exclusive write lock on the column.
        let col_lock = archetype
            .component_locked(component_id)
            .ok_or(ECSError::from(ExecutionError::MissingComponent { component_id }))?;

        let mut col_guard = col_lock.write().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned { what: "component column (set)" })
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
    pub fn apply_deferred_commands(
        &mut self,
        commands: Vec<Command>,
    ) -> ECSResult<Vec<Entity>> {
        let mut spawned = Vec::new();

        for command in commands {
            match command {
                Command::Spawn { bundle } => {
                    let signature = bundle.signature();
                    let archetype_id = self.get_or_create_archetype(&signature)?;
                    let shard_id = self.pick_spawn_shard();
                    let archetype = &mut self.archetypes[archetype_id as usize];
                    let entity = archetype.spawn_on(&mut self.shards, shard_id, bundle)?;
                    spawned.push(entity);
                }

                Command::Despawn { entity } => {
                    let loc = self.shards.get_location(entity)
                        .map_err(ECSError::from)?
                        .ok_or(ECSError::from(SpawnError::StaleEntity(StaleEntityError)))?;
                    let archetype = &mut self.archetypes[loc.archetype as usize];
                    archetype.despawn_on(&mut self.shards, entity)?;
                }

                Command::Add { entity, component_id, value } => {
                    self.add_component(entity, component_id, value)?;
                }

                Command::Remove { entity, component_id } => {
                    self.remove_component(entity, component_id)?;
                }

                Command::Set { entity, component_id, value } => {
                    self.apply_set_command(entity, component_id, value)?;
                }
            }
        }

        #[cfg(feature = "gpu")]
        {
            for archetype in &self.archetypes {
                self.gpu_dirty_chunks
                    .notify_archetype_changed(archetype.archetype_id());
            }
        }

        Ok(spawned)
    }

    fn ensure_shared_components(
        source_sig: &Signature,
        destination: &mut Archetype,
        excluded: ComponentID,
        registry: &ComponentRegistry,
    ) -> ECSResult<()> {
        for cid in source_sig.iterate_over_components() {
            if cid == excluded { continue; }
            let factory = || registry.make_empty_component(cid);
            destination.ensure_component(cid, factory).map_err(ECSError::from)?;
        }
        Ok(())
    }

    fn collect_read_ptrs_by_id(
        guards: &[(ComponentID, RwLockReadGuard<'_, Box<dyn TypeErasedAttribute>>)],
        declaration_order: &[ComponentID],
        chunk_id: crate::engine::types::ChunkID,
        len: usize,
        out: &mut Vec<(*const u8, usize)>,
    ) -> Result<(), ExecutionError> {
        for &cid in declaration_order {
            let (_, g) = guards.iter()
                .find(|(id, _)| *id == cid)
                .ok_or(ExecutionError::InternalExecutionError)?;
            let (ptr, bytes) = g
                .chunk_bytes(chunk_id, len)
                .ok_or(ExecutionError::InternalExecutionError)?;
            out.push((ptr, bytes));
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
        let loc = self.shards.get_location(entity)?
            .ok_or(ECSError::from(SpawnError::StaleEntity(StaleEntityError)))?;

        let arch = self.archetypes.get(loc.archetype as usize)
            .ok_or(ECSError::from(ExecutionError::InternalExecutionError))?;

        let col_lock = arch.component_locked(component_id)
            .ok_or(ECSError::from(ExecutionError::MissingComponent { component_id }))?;

        let col = col_lock.try_read().map_err(|e| match e {
            std::sync::TryLockError::WouldBlock => ECSError::from(
                ExecutionError::BorrowConflict {
                    component_id,
                    held: AccessKind::Write,
                    requested: AccessKind::Read,
                }
            ),
            std::sync::TryLockError::Poisoned(_) => ECSError::from(
                ExecutionError::LockPoisoned { what: "component column" }
            ),
        })?;

        if col.element_type_id() != TypeId::of::<T>() {
            return Err(ECSError::from(ExecutionError::InternalExecutionError));
        }

        let chunk_len = arch.chunk_valid_length(loc.chunk as usize)?;
        let (ptr, bytes) = col.chunk_bytes(loc.chunk, chunk_len)
            .ok_or(ECSError::from(ExecutionError::InternalExecutionError))?;

        let slice: &[T] = unsafe {
            crate::engine::storage::cast_slice::<T>(ptr, bytes)
        };

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
        let matches = self.matching_archetypes(&query.signature)?;
        let f = Arc::new(f);
        let abort = Arc::new(AtomicBool::new(false));
        let err: Arc<Mutex<Option<ExecutionError>>> = Arc::new(Mutex::new(None));

        for matched_archetype in matches {
            let archetype = &self.archetypes[matched_archetype.archetype_id as usize];

            let mut lock_order: Vec<(ComponentID, bool)> =
                Vec::with_capacity(query.reads.len() + query.writes.len());
            for &cid in &query.reads  { lock_order.push((cid, false)); }
            for &cid in &query.writes { lock_order.push((cid, true));  }
            lock_order.sort_unstable_by_key(|(cid, _)| *cid);
            lock_order.dedup_by_key(|(cid, _)| *cid);

            let mut read_guards:  Vec<(ComponentID, RwLockReadGuard<'_, Box<dyn TypeErasedAttribute>>)>  = Vec::new();
            let mut write_guards: Vec<(ComponentID, RwLockWriteGuard<'_, Box<dyn TypeErasedAttribute>>)> = Vec::new();

            for (cid, is_write) in &lock_order {
                let locked = archetype.component_locked(*cid)
                    .ok_or(ExecutionError::MissingComponent { component_id: *cid })?;
                if *is_write {
                    let g = locked.write().map_err(|_| ExecutionError::LockPoisoned {
                        what: "component column (write)",
                    })?;
                    write_guards.push((*cid, g));
                } else {
                    let g = locked.read().map_err(|_| ExecutionError::LockPoisoned {
                        what: "component column (read)",
                    })?;
                    read_guards.push((*cid, g));
                }
            }

            let chunk_count = archetype
                .chunk_count()
                .map_err(|_| ExecutionError::InternalExecutionError)?;
            if chunk_count == 0 { continue; }

            let mut chunk_lens = Vec::with_capacity(chunk_count);
            for c in 0..chunk_count {
                let len = archetype
                    .chunk_valid_length(c)
                    .map_err(|_| ExecutionError::InternalExecutionError)?;
                chunk_lens.push(len);
            }

            let n_reads  = query.reads.len();
            let n_writes = query.writes.len();

            let mut read_ptrs:  Vec<(*const u8, usize)> = Vec::with_capacity(chunk_count * n_reads);
            let mut write_ptrs: Vec<(*mut   u8, usize)> = Vec::with_capacity(chunk_count * n_writes);

            for chunk in 0..chunk_count {
                let len = chunk_lens[chunk];
                let chunk_id = chunk as crate::engine::types::ChunkID;

                if len == 0 {
                    for _ in 0..n_reads  { read_ptrs .push((std::ptr::null(), 0)); }
                    for _ in 0..n_writes { write_ptrs.push((std::ptr::null_mut(), 0)); }
                    continue;
                }

                Self::collect_read_ptrs_by_id(&read_guards, &query.reads, chunk_id, len, &mut read_ptrs)?;

                for &cid in &query.writes {
                    let (_, g) = write_guards.iter_mut()
                        .find(|(id, _)| *id == cid)
                        .ok_or(ExecutionError::InternalExecutionError)?;
                    let (ptr, bytes) = g.chunk_bytes_mut(chunk_id, len)
                        .ok_or(ExecutionError::InternalExecutionError)?;
                    write_ptrs.push((ptr, bytes));
                }
            }

            let views = ChunkView {
                chunk_count,
                chunk_lens,
                n_reads,
                n_writes,
                read_ptrs,
                write_ptrs,
            };

            let threads = rayon::current_num_threads().max(1);
            let grainsize = (views.chunk_count / threads).max(8);
            let views_ref = &views;
            let f_ref = &*f;
            let abort_ref = &abort;
            let err_ref = &err;

            rayon::scope(|s| {
                let mut start = 0usize;
                while start < views_ref.chunk_count {
                    let end = (start + grainsize).min(views_ref.chunk_count);
                    let abort = abort_ref.clone();
                    let _err   = err_ref.clone();
                    let views = views_ref;

                    s.spawn(move |_| {
                        if abort.load(Ordering::Acquire) { return; }

                        let mut read_views:  SmallVec<[&[u8]; 8]>      = SmallVec::new();
                        let mut write_views: SmallVec<[&mut [u8]; 8]>  = SmallVec::new();

                        for chunk in start..end {
                            let len = views.chunk_lens[chunk];
                            if len == 0 { continue; }

                            read_views.clear();
                            write_views.clear();

                            let rbase = chunk * views.n_reads;
                            for i in 0..views.n_reads {
                                let (ptr, bytes) = views.read_ptrs[rbase + i];
                                unsafe { read_views.push(std::slice::from_raw_parts(ptr, bytes)); }
                            }

                            let wbase = chunk * views.n_writes;
                            for i in 0..views.n_writes {
                                let (ptr, bytes) = views.write_ptrs[wbase + i];
                                unsafe { write_views.push(std::slice::from_raw_parts_mut(ptr, bytes)); }
                            }

                            // SAFETY: The caller (ECSReference::for_each_abstraction) holds
                            // the shared phase lock and has validated borrows; the byte slices
                            // correspond to correctly typed and aligned component storage.
                            f_ref(&read_views, &mut write_views);
                        }
                    });

                    start = end;
                }
            });

            if abort.load(Ordering::Acquire) {
                let guard = err.lock().map_err(|_| ExecutionError::LockPoisoned {
                    what: "job error latch",
                })?;
                return Err(guard.clone().unwrap_or(ExecutionError::InternalExecutionError));
            }

            drop(read_guards);
            drop(write_guards);
        }

        Ok(())
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
        let matches = self.matching_archetypes(&query.signature)?;
        let init = Arc::new(init);
        let fold_chunk = Arc::new(fold_chunk);
        let combine = Arc::new(combine);
        let partials: Arc<Mutex<Vec<(usize, R)>>> = Arc::new(Mutex::new(Vec::new()));

        for matched in matches {
            let archetype = &self.archetypes[matched.archetype_id as usize];

            let mut sorted_reads: Vec<ComponentID> = query.reads.clone();
            sorted_reads.sort_unstable();

            let mut read_guards: Vec<(ComponentID, RwLockReadGuard<'_, Box<dyn TypeErasedAttribute>>)> =
                Vec::with_capacity(query.reads.len());

            for &cid in &sorted_reads {
                let locked = archetype
                    .component_locked(cid)
                    .ok_or(ExecutionError::MissingComponent { component_id: cid })?;
                let guard = locked.read().map_err(|_| ExecutionError::LockPoisoned {
                    what: "component column (read)",
                })?;
                read_guards.push((cid, guard));
            }

            let chunk_count = archetype
                .chunk_count()
                .map_err(|_| ExecutionError::InternalExecutionError)?;
            if chunk_count == 0 { continue; }

            let mut chunk_lens = Vec::with_capacity(chunk_count);
            for c in 0..chunk_count {
                let len = archetype
                    .chunk_valid_length(c)
                    .map_err(|_| ExecutionError::InternalExecutionError)?;
                chunk_lens.push(len);
            }

            let n_reads = query.reads.len();
            let mut read_ptrs: Vec<(*const u8, usize)> =
                Vec::with_capacity(chunk_count * n_reads);

            for chunk in 0..chunk_count {
                let len = chunk_lens[chunk];
                if len == 0 {
                    for _ in 0..n_reads { read_ptrs.push((std::ptr::null(), 0)); }
                    continue;
                }
                let chunk_id = chunk as crate::engine::types::ChunkID;
                Self::collect_read_ptrs_by_id(&read_guards, &query.reads, chunk_id, len, &mut read_ptrs)?;
            }

            let views = ChunkView {
                chunk_count,
                chunk_lens,
                n_reads,
                n_writes: 0,
                read_ptrs,
                write_ptrs: Vec::new(),
            };

            let threads = rayon::current_num_threads().max(1);
            let grainsize = (views.chunk_count / threads).max(8);
            let views_ref = &views;

            rayon::scope(|s| {
                let mut start = 0usize;
                while start < views_ref.chunk_count {
                    let end = (start + grainsize).min(views_ref.chunk_count);
                    let init = init.clone();
                    let fold_chunk = fold_chunk.clone();
                    let partials = partials.clone();
                    let views = views_ref;

                    s.spawn(move |_| {
                        let mut local = init();
                        let mut read_views: SmallVec<[&[u8]; 8]> =
                            SmallVec::with_capacity(views.n_reads);

                        for chunk in start..end {
                            let len = views.chunk_lens[chunk];
                            if len == 0 { continue; }
                            read_views.clear();
                            let base = chunk * views.n_reads;
                            for i in 0..views.n_reads {
                                let (ptr, bytes) = views.read_ptrs[base + i];
                                unsafe { read_views.push(std::slice::from_raw_parts(ptr, bytes)); }
                            }
                            fold_chunk(&mut local, &read_views, len);
                        }

                        partials.lock().unwrap().push((start, local));
                    });

                    start = end;
                }
            });

            drop(read_guards);
        }

        let mut parts = partials.lock().unwrap();
        parts.sort_by_key(|(start, _)| *start);
        let mut out = init();
        for (_, p) in parts.drain(..) { combine(&mut out, p); }
        Ok(out)
    }

    fn matching_archetypes(
        &self,
        query: &QuerySignature,
    ) -> Result<Vec<ArchetypeMatch>, ExecutionError> {
        let mut out = Vec::new();
        for a in &self.archetypes {
            if !query.requires_all(a.signature()) { continue; }
            let chunks = a.chunk_count()
                .map_err(|_| ExecutionError::InternalExecutionError)?;
            out.push(ArchetypeMatch { archetype_id: a.archetype_id(), chunks });
        }
        Ok(out)
    }

    #[cfg(feature = "gpu")]
    #[inline]
    pub(crate) fn archetypes(&self) -> &[Archetype] { &self.archetypes }

    #[cfg(feature = "gpu")]
    #[inline]
    pub(crate) fn archetypes_mut(&mut self) -> &mut [Archetype] { &mut self.archetypes }

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
    /// ```ignore
    /// let buffer = world.create_buffer(&device, &descriptor);
    /// let resource_id = ecs_data.register_gpu_resource(buffer);
    /// ```
    pub fn register_gpu_resource<R: GPUResource + 'static>(&mut self, r: R) -> GPUResourceID {
        self.gpu_resources.register(r)
    }

    #[cfg(feature = "gpu")]
    #[inline]
    /// Returns an immutable reference to the GPU resource registry.
    ///
    /// The registry holds all world-owned GPU resources (buffers, textures,
    /// bind groups) and provides lookup by [`GPUResourceID`]. This is used
    /// by GPU systems to access resources during execution.
    pub fn gpu_resources(&self) -> &GPUResourceRegistry { &self.gpu_resources }

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
    pub fn gpu_resources_mut(&mut self) -> &mut GPUResourceRegistry { &mut self.gpu_resources }
}
