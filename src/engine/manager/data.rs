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
//!   [`Command`]s (spawn, despawn, add, remove) as a single synchronization point.
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
use crate::engine::error::{ECSResult, ECSError, ExecutionError, InternalViolation, RegistryError, SpawnError, StaleEntityError, AccessKind};

#[cfg(feature = "gpu")]
use crate::engine::dirty::DirtyChunks;

#[cfg(feature = "gpu")]
use crate::engine::types::GPUResourceID;

#[cfg(feature = "gpu")]
use crate::gpu::{GPUResource, GPUResourceRegistry};

use super::iteration::ChunkView;

/// Core ECS storage and orchestration structure.
///
/// ## Responsibilities
/// * Owns all archetypes and their component storage
/// * Maps signatures to archetype IDs
/// * Manages entity placement across archetypes
/// * Executes structural changes and parallel iteration
/// * Holds the component registry used for archetype creation, enabling multi-world support
///
/// ## Invariants
/// * `signature_map` and `archetypes` must remain consistent
/// * Entity locations must always point to valid archetypes
/// * Column locks are always acquired in ascending [`ComponentID`] order

pub struct ECSData {
    /// All registered archetypes.
    archetypes: Vec<Archetype>,

    /// Maps component signatures to archetype IDs.
    signature_map: HashMap<[u64; SIGNATURE_SIZE], ArchetypeID>,

    /// Entity shard allocator and location tracker.
    shards: EntityShards,

    next_spawn_shard: ShardID,

    /// Component registry used for archetype and column creation.
    registry: Arc<RwLock<ComponentRegistry>>,

    /// Chunk-level dirty tracking for CPU writes.
    #[cfg(feature = "gpu")]
    gpu_dirty_chunks: DirtyChunks,

    #[cfg(feature = "gpu")]
    gpu_resources: GPUResourceRegistry,
}

impl ECSData {

    /// Creates a new, empty `ECSData` with the given entity shard allocator and component registry.
    ///
    /// ## Semantics
    /// Archetypes are created lazily and assigned monotonically increasing IDs.
    /// The registry is used for all archetype and component column creation, enabling
    /// multi-world support without relying on global registry state.
    ///
    /// ## Complexity
    /// Amortized O(1).

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

    /// Returns a reference to the component registry held by this world.
    ///
    /// ## Purpose
    /// Provides access to the registry for callers that need to inspect or
    /// register component types without going through global state.
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
    /// Returns GPU dirty chunks
    #[inline]
    pub fn gpu_dirty_chunks(&self) -> &DirtyChunks {
        &self.gpu_dirty_chunks
    }

    /// Returns mutable references to two distinct archetypes.
    ///
    /// ## Purpose
    /// Enables safe mutation of source and destination archetypes during entity
    /// migration without violating Rust aliasing rules.
    ///
    /// ## Errors
    /// Returns `InternalViolation::ArchetypePairSameId` if `a == b`.
    ///
    /// ## Safety
    /// Relies on slice splitting to ensure disjoint mutable borrows.

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

    /// Adds a component to an existing entity, migrating it to a new archetype if necessary.
    ///
    /// This operation **changes the structural signature** of an entity. In an archetype-based ECS,
    /// entities are grouped by the exact set of components they possess; therefore adding a component
    /// requires **moving the entity's row** from its current archetype to another archetype whose
    /// signature includes the new component.
    ///
    /// ### High-level behaviour
    /// 1. The entity's current location `(archetype, chunk, row)` is retrieved.
    /// 2. A new signature is derived by setting `added_component_id`.
    /// 3. The destination archetype corresponding to that signature is located or created.
    /// 4. The destination archetype is prepared to store:
    ///    * the newly added component, and
    ///    * all components shared with the source archetype.
    /// 5. The entity's row is migrated from the source archetype to the destination archetype:
    ///    * shared components are copied/moved,
    ///    * the new component value is inserted,
    ///    * source-only components are removed.
    /// 6. Entity location metadata is updated as part of the row move.
    ///
    /// If the entity does not exist or is stale, the function returns early without performing
    /// any operation.
    ///
    /// ### Parameters
    /// * `entity` - The entity to which the component should be added.
    /// * `added_component_id` - The component type identifier to add.
    /// * `added_value` - The concrete component value, type-erased as `Box<dyn Any>`.
    ///   The dynamic type **must match** the registered type of `added_component_id`.
    ///
    /// ### Safety and correctness notes
    /// * This function assumes **exclusive access** to `ECSData`.
    /// * Archetype mutation is safe because `get_archetype_pair_mut` guarantees distinct mutable
    ///   references.
    /// * The dynamic type of `added_value` must correspond exactly to the component's registered
    ///   storage type; mismatches will cause the row move to fail internally.
    /// * Errors from `move_row_to_archetype` are intentionally ignored here; structural failures
    ///   are treated as non-recoverable or handled elsewhere.
    ///
    /// ### Complexity
    /// * **Time:** O(n) where n is the number of components in the source archetype.
    /// * **Memory:** May allocate a new archetype and component storage on first use of a signature.
    ///
    /// ### Archetype invariants
    /// * After completion, the entity will reside in an archetype whose signature includes
    ///   `added_component_id`.
    /// * Component column alignment across chunks is preserved.
    /// * Source archetype row density is maintained via swap-remove semantics.
    ///
    /// ### Example
    /// ```ignore
    /// ecs.add_component(entity, position_id, Box::new(Position { x: 1.0, y: 2.0 }));
    /// ```
    ///
    /// ### See also
    /// * [`remove_component`](Self::remove_component)
    /// * [`Archetype::move_row_to_archetype`]
    /// * [`Signature`]

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

        // Ensure all shared components exist in destination storage
        Self::ensure_shared_components(&source_sig, destination, added_component_id, &registry)?;

        drop(registry);

        // Move row + insert new value
        source.move_row_to_archetype(
            destination,
            shards,
            entity,
            (location.chunk, location.row),
            vec![(added_component_id, added_value)],
        )?;

        Ok(())
    }

    /// Removes a component from an entity, migrating it to a new archetype if needed.
    ///
    /// ## Semantics
    /// This operation performs a *structural mutation* of the ECS:
    ///
    /// 1. The entity's current archetype is determined from its stored location.
    /// 2. If the entity does not currently own `removed_component_id`, the operation
    ///    is a no-op.
    /// 3. A new component signature is constructed by clearing the specified
    ///    component bit.
    /// 4. If the resulting signature is empty, the entity is **despawned** and all
    ///    associated storage is released.
    /// 5. Otherwise, the entity's row is migrated to an archetype matching the new
    ///    signature, preserving all remaining components.
    ///
    /// ## Archetype migration
    /// * The destination archetype is created lazily if it does not already exist.
    /// * All remaining components from the source archetype are ensured to exist
    ///   in the destination before the row move.
    /// * Component storage order and chunk alignment are preserved by
    ///   `move_row_to_archetype`.
    ///
    /// ## Safety and invariants
    /// * This method assumes the entity location stored in `EntityShards` is valid.
    /// * Entity movement relies on consistent archetype metadata and component
    ///   storage alignment.
    /// * This function must **not** be called concurrently with parallel iteration
    ///   over archetypes.
    ///
    /// ## Failure behaviour
    /// * Invalid or stale entities are silently ignored.
    /// * Removing the last component always results in despawning the entity.
    /// * Storage-level errors during migration are handled internally by
    ///   `move_row_to_archetype`.
    ///
    /// ## Performance
    /// * O(number of components in the source archetype)
    /// * No heap allocations unless a new archetype must be created
    ///
    /// ## Example
    /// ```ignore
    /// // Removes Velocity from an entity; may move it to a different archetype
    /// world.remove_component(entity, Velocity::ID);
    /// ```

    pub fn remove_component(&mut self, entity: Entity, removed_component_id: ComponentID) -> ECSResult<()> {
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
            self.archetypes[source_id as usize].despawn_on(&mut self.shards, entity)?;
            return Ok(());
        }

        let destination_id = self.get_or_create_archetype(&new_signature)?;
        let source_sig = self.archetypes[source_id as usize].signature().clone();
        let shards = &self.shards;

        let (source_arch, dest_arch) =
            Self::get_archetype_pair_mut(&mut self.archetypes, source_id, destination_id)?;

        let registry = self.registry.read()
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

    /// Applies all queued deferred commands.
    ///
    /// ## Notes
    /// This method is expected to evolve as command execution is implemented.
    /// Currently acts as a structural synchronization point.

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
                    let entity = archetype.spawn_on(
                        &mut self.shards, shard_id, bundle
                    )?;
                    spawned.push(entity);
                }

                Command::Despawn { entity } => {
                    let loc = self.shards.get_location(entity)
                        .map_err(ECSError::from)?
                        .ok_or(ECSError::from(
                            SpawnError::StaleEntity(StaleEntityError)
                        ))?;
                    let archetype = &mut self.archetypes[loc.archetype as usize];
                    archetype.despawn_on(&mut self.shards, entity)?;
                }

                Command::Add { entity, component_id, value } => {
                    self.add_component(entity, component_id, value)?;
                }

                Command::Remove { entity, component_id } => {
                    self.remove_component(entity, component_id)?;
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

    /// Executes a generic, parallel, chunk-oriented ECS query **without safety checks**.
    ///
    /// ## Important
    /// This function is **unsafe by contract** and must only be called from
    /// [`ECSReference::for_each_abstraction`].
    ///
    /// ## Required invariants
    /// The caller must guarantee:
    ///
    /// * no structural mutation occurs for the duration of the call,
    /// * component borrow rules are already enforced,
    /// * each chunk represents a disjoint memory region,
    /// * the callback does not escape references.
    ///
    /// ## Lock ordering
    /// Column locks are acquired in ascending [`ComponentID`] order (reads and writes merged
    /// and sorted together) to prevent deadlocks. Guards are then reordered back to declaration
    /// order before pointer arrays are built, so callback index semantics are preserved.

    pub(crate) fn for_each_abstraction_unchecked(
        &self,
        query: BuiltQuery,
        f: impl Fn(&[&[u8]], &mut [&mut [u8]]) + Send + Sync,
    ) -> Result<(), ExecutionError> {
        let matches = self.matching_archetypes(&query.signature)?;

        // Share callback across spawned tasks
        let f = Arc::new(f);

        // Hoist the error latch outside the archetype loop to avoid
        // allocating an Arc<Mutex<...>> per archetype.
        let abort = Arc::new(AtomicBool::new(false));
        let err: Arc<Mutex<Option<ExecutionError>>> = Arc::new(Mutex::new(None));

        for matched_archetype in matches {
            let archetype = &self.archetypes[matched_archetype.archetype_id as usize];

            // Build a combined lock order sorted by ascending ComponentID.
            // This matches the contract documented in archetype/mod.rs and prevents
            // deadlocks when multiple queries run concurrently on overlapping archetypes.
            let mut lock_order: Vec<(ComponentID, bool)> =
                Vec::with_capacity(query.reads.len() + query.writes.len());
            for &cid in &query.reads {
                lock_order.push((cid, false)); // false = read
            }
            for &cid in &query.writes {
                lock_order.push((cid, true)); // true = write
            }
            lock_order.sort_unstable_by_key(|(cid, _)| *cid);

            // Acquire locks in sorted order.
            let mut read_guards: Vec<(ComponentID, RwLockReadGuard<'_, Box<dyn TypeErasedAttribute>>)> =
                Vec::with_capacity(query.reads.len());
            let mut write_guards: Vec<(ComponentID, RwLockWriteGuard<'_, Box<dyn TypeErasedAttribute>>)> =
                Vec::with_capacity(query.writes.len());

            for (component_id, is_write) in &lock_order {
                let locked = archetype
                    .component_locked(*component_id)
                    .ok_or(ExecutionError::MissingComponent { component_id: *component_id })?;

                if *is_write {
                    let guard = locked.write().map_err(|_| ExecutionError::LockPoisoned {
                        what: "component column (write)",
                    })?;
                    write_guards.push((*component_id, guard));
                } else {
                    let guard = locked.read().map_err(|_| ExecutionError::LockPoisoned {
                        what: "component column (read)",
                    })?;
                    read_guards.push((*component_id, guard));
                }
            }

            let chunk_count = archetype
                .chunk_count()
                .map_err(|_| ExecutionError::InternalExecutionError)?;

            if chunk_count == 0 {
                continue;
            }

            let mut chunk_lens = Vec::with_capacity(chunk_count);
            for chunk in 0..chunk_count {
                let len = archetype
                    .chunk_valid_length(chunk)
                    .map_err(|_| ExecutionError::InternalExecutionError)?;
                chunk_lens.push(len);
            }

            // Precompute pointers.
            // Reorder guards back to declaration order for the pointer arrays so that
            // callback index semantics (col[0] = first declared read, etc.) are preserved.
            let n_reads = query.reads.len();
            let n_writes = query.writes.len();

            let mut read_ptrs: Vec<(*const u8, usize)> = Vec::with_capacity(chunk_count * n_reads);
            let mut write_ptrs: Vec<(*mut u8, usize)> = Vec::with_capacity(chunk_count * n_writes);

            for chunk in 0..chunk_count {
                let len = chunk_lens[chunk];

                if len == 0 {
                    for _ in 0..n_reads {
                        read_ptrs.push((std::ptr::null(), 0));
                    }
                    for _ in 0..n_writes {
                        write_ptrs.push((std::ptr::null_mut(), 0));
                    }
                    continue;
                }

                let chunk_id: crate::engine::types::ChunkID = chunk
                    .try_into()
                    .map_err(|_| ExecutionError::InternalExecutionError)?;

                Self::collect_read_ptrs_by_id(&read_guards, &query.reads, chunk_id, len, &mut read_ptrs)?;

                // Collect write pointers in declaration order.
                for &cid in &query.writes {
                    let (_, g) = write_guards.iter_mut()
                        .find(|(id, _)| *id == cid)
                        .ok_or(ExecutionError::InternalExecutionError)?;
                    let (ptr, bytes) = g
                        .chunk_bytes_mut(chunk_id, len)
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

            #[cfg(feature = "gpu")]
            let archetype_id: ArchetypeID = matched_archetype.archetype_id;

            #[cfg(feature = "gpu")]
            let resolved_dirty_entries: Vec<Arc<crate::engine::dirty::Entry>> = if n_writes > 0 {
                let dirty = self.gpu_dirty_chunks();
                query.writes.iter().map(|&cid| {
                    dirty.resolve_entry(archetype_id, cid, views.chunk_count)
                }).collect()
            } else {
                Vec::new()
            };

            #[cfg(feature = "gpu")]
            let resolved_dirty_entries = Arc::new(resolved_dirty_entries);

            rayon::scope(|s| {
                let mut start = 0usize;
                while start < views_ref.chunk_count {
                    let end = (start + grainsize).min(views_ref.chunk_count);

                    let abort = abort.clone();
                    let err = err.clone();
                    let f = f.clone();
                    let views = views_ref;

                    #[cfg(feature = "gpu")]
                    let resolved_dirty_entries = resolved_dirty_entries.clone();

                    s.spawn(move |_| {
                        let mut read_views: SmallVec<[&[u8]; 8]> = SmallVec::with_capacity(views.n_reads);
                        let mut write_views: SmallVec<[&mut [u8]; 8]> = SmallVec::with_capacity(views.n_writes);

                        for chunk in start..end {
                            if abort.load(Ordering::Acquire) {
                                return;
                            }

                            let fail = |e: ExecutionError| {
                                abort.store(true, Ordering::Release);
                                if let Ok(mut slot) = err.lock() {
                                    if slot.is_none() {
                                        *slot = Some(e);
                                    }
                                }
                            };

                            let len = views.chunk_lens[chunk];
                            if len == 0 {
                                continue;
                            }

                            read_views.clear();
                            write_views.clear();

                            let read_base = chunk * views.n_reads;
                            for i in 0..views.n_reads {
                                let (ptr, bytes) = views.read_ptrs[read_base + i];
                                if ptr.is_null() || bytes == 0 {
                                    fail(ExecutionError::InternalExecutionError);
                                    return;
                                }
                                unsafe { read_views.push(std::slice::from_raw_parts(ptr, bytes)); }
                            }

                            let write_base = chunk * views.n_writes;
                            for i in 0..views.n_writes {
                                let (ptr, bytes) = views.write_ptrs[write_base + i];
                                if ptr.is_null() || bytes == 0 {
                                    fail(ExecutionError::InternalExecutionError);
                                    return;
                                }
                                unsafe { write_views.push(std::slice::from_raw_parts_mut(ptr, bytes)); }
                            }

                            #[cfg(feature = "gpu")]
                            {
                                for entry in resolved_dirty_entries.as_slice() {
                                    entry.mark_dirty(chunk);
                                }
                            }

                            // Call user callback.
                            // SAFETY: SmallVec<[&[u8]; 8]> derefs to &[&[u8]], which
                            // matches the callback signature.
                            f(&read_views, &mut write_views);
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

            // Guards drop
            drop(read_guards);
            drop(write_guards);
        }

        Ok(())
    }

    /// Executes a parallel reduction over all archetypes matching a query.
    ///
    /// ## Lock ordering
    /// Read column locks are acquired in ascending [`ComponentID`] order to prevent
    /// deadlocks, then reordered back to declaration order for pointer building.

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

        // (start_chunk, partial)
        let partials: Arc<Mutex<Vec<(usize, R)>>> =
            Arc::new(Mutex::new(Vec::new()));

        for matched in matches {
            let archetype = &self.archetypes[matched.archetype_id as usize];

            // Acquire read guards in ascending ComponentID order to match the
            // documented lock-ordering contract. Guards are stored with their
            // ComponentID so they can be retrieved in declaration order when
            // building pointer arrays.
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
            if chunk_count == 0 {
                continue;
            }

            // Precompute chunk lengths
            let mut chunk_lens = Vec::with_capacity(chunk_count);
            for c in 0..chunk_count {
                let len = archetype
                    .chunk_valid_length(c)
                    .map_err(|_| ExecutionError::InternalExecutionError)?;
                chunk_lens.push(len);
            }

            // Precompute read pointers in declaration order.
            let n_reads = query.reads.len();
            let mut read_ptrs: Vec<(*const u8, usize)> =
                Vec::with_capacity(chunk_count * n_reads);

            for chunk in 0..chunk_count {
                let len = chunk_lens[chunk];

                if len == 0 {
                    for _ in 0..n_reads {
                        read_ptrs.push((std::ptr::null(), 0));
                    }
                    continue;
                }

                let chunk_id: crate::engine::types::ChunkID = chunk
                    .try_into()
                    .map_err(|_| ExecutionError::InternalExecutionError)?;

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
                            if len == 0 {
                                continue;
                            }

                            read_views.clear();
                            let base = chunk * views.n_reads;

                            for i in 0..views.n_reads {
                                let (ptr, bytes) = views.read_ptrs[base + i];
                                unsafe {
                                    read_views.push(std::slice::from_raw_parts(ptr, bytes));
                                }
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

        // Deterministic combine
        let mut parts = partials.lock().unwrap();
        parts.sort_by_key(|(start, _)| *start);

        let mut out = init();
        for (_, p) in parts.drain(..) {
            combine(&mut out, p);
        }

        Ok(out)
    }

    /// Returns archetypes matching a query signature.
    ///
    /// ## Returns
    /// A list of archetype IDs and their chunk counts.

    fn matching_archetypes(
        &self,
        query: &QuerySignature,
    ) -> Result<Vec<ArchetypeMatch>, ExecutionError> {
        let mut out = Vec::new();

        for a in &self.archetypes {
            if !query.requires_all(a.signature()) {
                continue;
            }

            let chunks = a
                .chunk_count()
                .map_err(|_| ExecutionError::InternalExecutionError)?;

            out.push(ArchetypeMatch {
                archetype_id: a.archetype_id(),
                chunks,
            });
        }

        Ok(out)
    }

    #[cfg(feature = "gpu")]
    #[inline]
    pub(crate) fn archetypes(&self) -> &[Archetype] {
        &self.archetypes
    }

    #[cfg(feature = "gpu")]
    #[inline]
    pub(crate) fn archetypes_mut(&mut self) -> &mut [Archetype] {
        &mut self.archetypes
    }

    /// Registers a world-owned GPU resource with the ECS world.
    ///
    /// ## Semantics
    /// * The resource becomes owned by the ECS world for its entire lifetime.
    /// * GPU buffers are created lazily when the GPU runtime is initialized.
    ///
    /// ## Returns
    /// A stable `GPUResourceID` that can be referenced by GPU systems.

    #[cfg(feature = "gpu")]
    pub fn register_gpu_resource<R: GPUResource + 'static>(&mut self, r: R) -> GPUResourceID {
        self.gpu_resources.register(r)
    }

    /// Returns an immutable view of the GPU resource registry.
    ///
    /// ## Purpose
    /// Intended for GPU dispatch and scheduling logic that needs to:
    /// * resolve resource bindings,
    /// * inspect resource layouts,
    /// * or build GPU bind groups.

    #[cfg(feature = "gpu")]
    #[inline]
    pub fn gpu_resources(&self) -> &GPUResourceRegistry {
        &self.gpu_resources
    }

    /// Returns a mutable view of the GPU resource registry.
    ///
    /// ## Purpose
    /// Allows controlled mutation of GPU resource state, including:
    /// * marking CPU-side data dirty,
    /// * marking pending GPU to CPU downloads,
    /// * performing explicit upload/download synchronization.

    #[cfg(feature = "gpu")]
    #[inline]
    pub fn gpu_resources_mut(&mut self) -> &mut GPUResourceRegistry {
        &mut self.gpu_resources
    }

    /// Ensures all components from `source_sig` (excluding one) exist in `destination`.
    ///
    /// Uses the provided `registry` to create empty component storage, avoiding
    /// global registry calls and enabling multi-world support.
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
            destination.ensure_component(cid, factory).map_err(ECSError::from)?;
        }
        Ok(())
    }

    /// Collects read pointers in declaration order from guards keyed by ComponentID.
    ///
    /// Guards are stored in lock-acquisition order (ascending ComponentID), but
    /// callers need pointers in declaration order. This helper looks up each
    /// declared ComponentID in the guards list and appends the chunk pointer to `out`.
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

    /// Reads a single component value for one entity.
    ///
    /// Resolves the entity's archetype location through `EntityShards`,
    /// acquires a read lock on the target component column, and clones the
    /// value at the entity's `(chunk, row)`.
    ///
    /// # Concurrency
    ///
    /// This method requires only `&self` — no exclusive access. It is safe
    /// to call under a shared phase lock (the same lock held by `for_each`)
    /// because:
    ///
    /// * `EntityShards::get_location` acquires only the per-shard mutex,
    ///   which does not conflict with the phase lock.
    /// * The component column `RwLock` allows concurrent readers. If the
    ///   column is write-locked by the calling system's own query, the read
    ///   will block — which is correct, because reading a component that is
    ///   concurrently being written is a data race.
    /// * No structural mutation occurs.
    ///
    /// # Errors
    ///
    /// * `SpawnError::StaleEntity` — entity is dead or recycled.
    /// * `ExecutionError::MissingComponent` — archetype does not contain
    ///   this component.
    /// * `ExecutionError::LockPoisoned` — column lock is poisoned.
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
            .ok_or(ECSError::from(
                ExecutionError::MissingComponent { component_id }
            ))?;

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
}
