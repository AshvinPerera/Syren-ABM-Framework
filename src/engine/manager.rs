//! ECS world management and execution layer.
//!
//! This module defines the central orchestration layer of the ECS, responsible
//! for:
//!
//! * owning archetypes and their component storage,
//! * coordinating entity movement between archetypes,
//! * managing deferred structural mutations via commands,
//! * providing safe shared and exclusive access to world state,
//! * executing parallel component iteration with explicit read/write sets.
//!
//! ## Concurrency model
//!
//! The ECS is internally mutable and uses `UnsafeCell` to allow aliasing between
//! shared (`&`) and exclusive (`&mut`) access paths. Safety is enforced by
//! *API discipline*, not the Rust borrow checker:
//!
//! * Structural mutations must go through exclusive access (`&mut ECSData`)
//! * Parallel iteration is limited to non-overlapping component access sets
//! * Deferred commands are applied only at explicit synchronization points
//!
//! ## Safety
//!
//! This module contains unsafe code for:
//! * interior mutability (`UnsafeCell`),
//! * raw pointer component access,
//! * parallel execution using Rayon.
//!
//! All unsafe blocks rely on strict invariants documented at each boundary.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use rayon::prelude::*;

use crate::engine::commands::Command;
use crate::engine::types::{
    Signature, 
    ComponentID, 
    ArchetypeID,
    SIGNATURE_SIZE,
    QuerySignature
};
use crate::engine::query::QueryBuilder;
use crate::engine::archetype::{
    Archetype,
    ArchetypeMatch
};
use crate::engine::entity::{Entity, EntityShards};
use crate::engine::storage::{cast_slice, cast_slice_mut};
use crate::engine::component::make_empty_component;


/// A unit of parallel work operating on a single archetype chunk.
///
/// ## Purpose
/// `Job` packages raw pointers and byte lengths for three component columns
/// (two read-only, one mutable) so they can be processed in parallel by Rayon
/// without holding references into archetype storage.
///
/// ## Safety
/// All pointers must:
/// * originate from valid component storage,
/// * remain valid for the duration of the job,
/// * refer to non-overlapping mutable regions across jobs.
///
/// These invariants are established by archetype chunk partitioning.
///
/// ## Fields
/// * `len` — Number of elements logically processed by the job.
/// * `*_ptr` — Raw pointer to component data.
/// * `*_bytes` — Size in bytes of the pointed-to slice.

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct Job {
    /// Number of elements in the chunk slice.
    len: usize,

    /// Pointer to first read-only component column.
    a_ptr: *const u8,

    /// Byte length of the first read-only slice.
    a_bytes: usize,

    /// Pointer to second read-only component column.
    b_ptr: *const u8,

    /// Byte length of the second read-only slice.
    b_bytes: usize,

    /// Pointer to mutable component column.
    c_ptr: *mut u8,

    /// Byte length of the mutable slice.
    c_bytes: usize,
}

unsafe impl Send for Job {}
unsafe impl Sync for Job {}

/// Thread-safe entry point to the ECS world.
///
/// ## Role
/// `ECSManager` owns the entire ECS state and provides controlled access
/// through lightweight references (`ECSReference`). It is designed to be
/// shared across threads while enforcing safety via interior mutability.
///
/// ## Concurrency
/// * `ECSManager` is `Sync`
/// * All mutation occurs through `UnsafeCell<ECSData>`
/// * Users must respect API-level exclusivity guarantees
pub struct ECSManager {
    /// Interior-mutable ECS state.
    inner: UnsafeCell<ECSData>,
}

unsafe impl Sync for ECSManager {}

impl ECSManager {
    /// Creates a new ECS manager with the given entity shard configuration.
    ///
    /// ## Parameters
    /// * `shards` — Entity shard allocator used for entity lifetime tracking.
    ///
    /// ## Notes
    /// Archetypes are initially empty and created lazily as signatures are
    /// encountered.    

    pub fn new(shards: EntityShards) -> Self {
        Self {
            inner: UnsafeCell::new(ECSData::new(shards)),
        }
    }

    /// Returns a lightweight reference handle to the ECS world.
    ///
    /// ## Purpose
    /// Allows shared access to ECS data without transferring ownership or
    /// requiring exclusive access.
    ///
    /// ## Safety
    /// The returned reference permits both shared and mutable access via
    /// `ECSReference`, relying on caller discipline to avoid data races.

    #[inline]
    pub fn world_ref(&self) -> ECSReference<'_> {
        ECSReference { inner: &self.inner }
    }

    /// Applies all queued deferred commands.
    ///
    /// ## Semantics
    /// This is a synchronization point where structural changes requested
    /// during parallel or shared access phases are applied.
    ///
    /// ## Notes
    /// Commands are drained and executed in FIFO order.

    pub fn apply_deferred_commands(&self) {
        unsafe { &mut *self.inner.get() }.apply_deferred_commands();
    }
}

/// A non-owning handle granting access to ECS data.
///
/// ## Role
/// `ECSReference` allows systems to read or mutate ECS state while the
/// `ECSManager` remains shared.
///
/// ## Safety
/// This type exposes raw access to `ECSData` via `UnsafeCell` and relies
/// on higher-level scheduling to avoid conflicting mutable accesses.

pub struct ECSReference<'a> {
    /// Pointer to interior ECS data.
    inner: &'a UnsafeCell<ECSData>,
}

impl<'a> ECSReference<'a> {
    /// Returns an immutable reference to ECS data.
    ///
    /// ## Safety
    /// No aliasing guarantees are enforced at compile time.    
    
    #[inline]
    pub fn data(&self) -> &ECSData {
        unsafe { &*self.inner.get() }
    }

    /// Returns a mutable reference to ECS data.
    ///
    /// ## Safety
    /// Caller must ensure no other references (mutable or immutable) are
    /// active while this reference is used.

    #[inline]
    pub fn data_mut(&self) -> &mut ECSData {
        unsafe { &mut *self.inner.get() }
    }
}

/// Core ECS storage and orchestration structure.
///
/// ## Responsibilities
/// * Owns all archetypes and their component storage
/// * Maps signatures to archetype IDs
/// * Manages entity placement across archetypes
/// * Executes structural changes and parallel iteration
///
/// ## Invariants
/// * `signature_map` and `archetypes` must remain consistent
/// * Entity locations must always point to valid archetypes

pub struct ECSData {
    /// All registered archetypes.
    pub archetypes: Vec<Archetype>,

    /// Maps component signatures to archetype IDs.
    signature_map: HashMap<[u64; SIGNATURE_SIZE], ArchetypeID>,

    /// Entity shard allocator and location tracker.
    shards: EntityShards,

    /// Deferred structural commands.
    deferred: Vec<Command>,
}

impl ECSData {

    /// Retrieves the archetype matching `signature`, creating it if necessary.
    ///
    /// ## Semantics
    /// Archetypes are created lazily and assigned monotonically increasing IDs.
    ///
    /// ## Complexity
    /// Amortized O(1).

    pub fn new(shards: EntityShards) -> Self {
        Self {
            archetypes: Vec::new(),
            signature_map: HashMap::new(),
            shards,
            deferred: Vec::new(),
        }
    }

    /// Returns mutable references to two distinct archetypes.
    ///
    /// ## Purpose
    /// Enables safe mutation of source and destination archetypes during entity
    /// migration without violating Rust aliasing rules.
    ///
    /// ## Panics
    /// Panics if `a == b`.
    ///
    /// ## Safety
    /// Relies on slice splitting to ensure disjoint mutable borrows.

    fn get_or_create_archetype(&mut self, signature: &Signature) -> ArchetypeID {
        let key = signature.components;
        if let Some(&id) = self.signature_map.get(&key) {
            return id;
        }

        let id = self.archetypes.len() as ArchetypeID;
        self.signature_map.insert(key, id);
        self.archetypes.push(Archetype::new(id));
        id
    }

    /// Adds a component to an entity, migrating it to a new archetype if required.
    ///
    /// ## Semantics
    /// * Computes the destination signature
    /// * Creates destination archetype if needed
    /// * Moves the entity row between archetypes
    ///
    /// ## Notes
    /// This is a structural operation and should not be called during parallel
    /// iteration.

    #[inline]
    fn get_archetype_pair_mut(
        archetypes: &mut [Archetype],
        a: ArchetypeID,
        b: ArchetypeID,
    ) -> (&mut Archetype, &mut Archetype) {
        assert!(a != b);

        let (low, high) = if a < b { (a, b) } else { (b, a) };
        let (head, tail) = archetypes.split_at_mut(high as usize);

        let left = &mut head[low as usize];
        let right = &mut tail[0];

        if a < b { (left, right) } else { (right, left) }
    }

    /// Adds a component to an existing entity, migrating it to a new archetype if necessary.
    ///
    /// This operation **changes the structural signature** of an entity. In an archetype-based ECS,
    /// entities are grouped by the exact set of components they possess; therefore adding a component
    /// requires **moving the entity’s row** from its current archetype to another archetype whose
    /// signature includes the new component.
    ///
    /// ### High-level behavior
    /// 1. The entity’s current location `(archetype, chunk, row)` is retrieved.
    /// 2. A new signature is derived by setting `added_component_id`.
    /// 3. The destination archetype corresponding to that signature is located or created.
    /// 4. The destination archetype is prepared to store:
    ///    * the newly added component, and
    ///    * all components shared with the source archetype.
    /// 5. The entity’s row is migrated from the source archetype to the destination archetype:
    ///    * shared components are copied/moved,
    ///    * the new component value is inserted,
    ///    * source-only components are removed.
    /// 6. Entity location metadata is updated as part of the row move.
    ///
    /// If the entity does not exist or is stale, the function returns early without performing
    /// any operation.
    ///
    /// ### Parameters
    /// * `entity` — The entity to which the component should be added.
    /// * `added_component_id` — The component type identifier to add.
    /// * `added_value` — The concrete component value, type-erased as `Box<dyn Any>`.
    ///   The dynamic type **must match** the registered type of `added_component_id`.
    ///
    /// ### Safety and correctness notes
    /// * This function assumes **exclusive access** to `ECSData`.
    /// * Archetype mutation is safe because `get_archetype_pair_mut` guarantees distinct mutable
    ///   references.
    /// * The dynamic type of `added_value` must correspond exactly to the component’s registered
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
    ) {
        let Some(location) = self.shards.get_location(entity) else { return; };
        let source_id = location.archetype;

        let mut new_signature = self.archetypes[source_id as usize].signature().clone();
        new_signature.set(added_component_id);

        let destination_id = self.get_or_create_archetype(&new_signature);
        let source_sig = self.archetypes[source_id as usize].signature().clone();

        let shards = &self.shards;
        let (source, destination) =
            Self::get_archetype_pair_mut(&mut self.archetypes, source_id, destination_id);

        let result = destination.ensure_component(added_component_id, || make_empty_component(added_component_id));
        debug_assert!(
            result.is_ok(),
            "ECS invariant violation during add_component: {:?}",
            result
        );

        if cfg!(not(debug_assertions)) {
            if let Err(e) = result {
                panic!("ECS corruption detected: {e}");
            }
        }
        
        for cid in source_sig.iterate_over_components() {
            if cid == added_component_id { continue; }
            
            let result = destination.ensure_component(cid, || make_empty_component(cid));
            debug_assert!(
                result.is_ok(),
                "ECS invariant violation during add_component: {:?}",
                result
            );   

            if cfg!(not(debug_assertions)) {
                if let Err(e) = result {
                    panic!("ECS corruption detected: {e}");
                }
            }         
        }

        let result = source.move_row_to_archetype(
            destination,
            shards,
            entity,
            (location.chunk, location.row),
            vec![(added_component_id, added_value)],
        );

        debug_assert!(
            result.is_ok(),
            "ECS invariant violation during add_component: {:?}",
            result
        );

        if cfg!(not(debug_assertions)) {
            if let Err(e) = result {
                panic!("ECS corruption detected: {e}");
            }
        }
    }

    /// Removes a component from an entity, migrating it to a new archetype if needed.
    ///
    /// ## Semantics
    /// This operation performs a *structural mutation* of the ECS:
    ///
    /// 1. The entity’s current archetype is determined from its stored location.
    /// 2. If the entity does not currently own `removed_component_id`, the operation
    ///    is a no-op.
    /// 3. A new component signature is constructed by clearing the specified
    ///    component bit.
    /// 4. If the resulting signature is empty, the entity is **despawned** and all
    ///    associated storage is released.
    /// 5. Otherwise, the entity’s row is migrated to an archetype matching the new
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
    /// ## Failure behavior
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

    pub fn remove_component(&mut self, entity: Entity, removed_component_id: ComponentID) {
        let Some(location) = self.shards.get_location(entity) else { return; };
        let source_id = location.archetype;

        if !self.archetypes[source_id as usize].has(removed_component_id) {
            return;
        }

        let mut new_signature = self.archetypes[source_id as usize].signature().clone();
        new_signature.clear(removed_component_id);

        if new_signature.components.iter().all(|&bits| bits == 0) {
            let _ = self.archetypes[source_id as usize].despawn_on(&mut self.shards, entity);
            return;
        }

        let destination_id = self.get_or_create_archetype(&new_signature);
        let source_sig = self.archetypes[source_id as usize].signature().clone();

        let shards = &self.shards;
        let (source_arch, dest_arch) =
            Self::get_archetype_pair_mut(&mut self.archetypes, source_id, destination_id);

        for cid in source_sig.iterate_over_components() {
            if cid == removed_component_id { continue; }

            let result = dest_arch.ensure_component(cid, || make_empty_component(cid));
            debug_assert!(
                result.is_ok(),
                "ECS invariant violation during add_component: {:?}",
                result
            );

            if cfg!(not(debug_assertions)) {
                if let Err(e) = result {
                    panic!("ECS corruption detected: {e}");
                }
            }
        }

        let result = source_arch.move_row_to_archetype(
            dest_arch,
            shards,
            entity,
            (location.chunk, location.row),
            Vec::new(),
        );

        debug_assert!(
            result.is_ok(),
            "ECS invariant violation during add_component: {:?}",
            result
        );

        if cfg!(not(debug_assertions)) {
            if let Err(e) = result {
                panic!("ECS corruption detected: {e}");
            }
        }        
    }

    /// Executes a closure with exclusive access to ECS data.
    ///
    /// ## Purpose
    /// Provides an explicit escape hatch for complex mutations requiring full
    /// control over ECS state.

    pub fn with_exclusive<R>(&mut self, f: impl FnOnce(&mut ECSData) -> R) -> R {
        f(self)
    } 

    /// Queues a structural command for deferred execution.
    ///
    /// ## Use case
    /// Used by systems that cannot mutate ECS structure immediately.

    pub fn defer(&mut self, command: Command) {
        self.deferred.push(command);
    }

    /// Applies all queued deferred commands.
    ///
    /// ## Notes
    /// This method is expected to evolve as command execution is implemented.
    /// Currently acts as a structural synchronization point.

    pub fn apply_deferred_commands(&mut self) {
        for command in self.deferred.drain(..) {
            match command {
                Command::Spawn { .. } => {}
                Command::Despawn { .. } => {}
                Command::Add { .. } => {}
                Command::Remove { .. } => {}
            }
        }
    }

    /// Executes a parallel iteration over three components.
    ///
    /// ## Access pattern
    /// * Two read-only components
    /// * One mutable component
    ///
    /// ## Safety
    /// This function relies on:
    /// * chunk-level disjointness
    /// * correct component type registration
    /// * non-overlapping mutable slices
    ///
    /// Violating these invariants results in undefined behavior.

    pub fn par_for_each3<
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
        C: 'static + Send + Sync,
    >(
        &mut self,
        reads: [ComponentID; 2],
        writes: [ComponentID; 1],
        f: impl Fn(&A, &B, &mut C) + Send + Sync,
    ) {
        for archetype in &mut self.archetypes {
            if !(archetype.has(reads[0]) && archetype.has(reads[1]) && archetype.has(writes[0])) {
                continue;
            }

            let chunks = archetype.chunk_count();
            let mut jobs = Vec::with_capacity(chunks);

            for chunk in 0..chunks {
                let len = archetype.chunk_valid_length(chunk);
                if len == 0 { continue; }

                let (a_ptr, a_bytes) = {
                    let a_col = archetype.component(reads[0]).unwrap();
                    a_col.chunk_bytes(chunk as _, len).unwrap()
                };

                let (b_ptr, b_bytes) = {
                    let b_col = archetype.component(reads[1]).unwrap();
                    b_col.chunk_bytes(chunk as _, len).unwrap()
                };

                let (c_ptr, c_bytes) = {
                    let c_col = archetype.component_mut(writes[0]).unwrap();
                    c_col.chunk_bytes_mut(chunk as _, len).unwrap()
                };

                jobs.push(Job {
                    len,
                    a_ptr,
                    a_bytes,
                    b_ptr,
                    b_bytes,
                    c_ptr,
                    c_bytes,
                });
            }

            jobs.into_par_iter().for_each(|job| {
                unsafe {
                    let a_slice = cast_slice::<A>(job.a_ptr, job.a_bytes);
                    let b_slice = cast_slice::<B>(job.b_ptr, job.b_bytes);
                    let c_slice = cast_slice_mut::<C>(job.c_ptr, job.c_bytes);

                    let n = a_slice.len()
                        .min(b_slice.len())
                        .min(c_slice.len());

                    for i in 0..n {
                        f(&a_slice[i], &b_slice[i], &mut c_slice[i]);
                    }
                }
            });
        }
    }
}

impl ECSData {

    /// Begins construction of a component query.
    pub fn query(&mut self) -> QueryBuilder {
        QueryBuilder::new()
    }

    /// Returns archetypes matching a query signature.
    ///
    /// ## Returns
    /// A list of archetype IDs and their chunk counts.

    pub fn matching_archetypes(
        &self,
        query: &QuerySignature,
    ) -> Vec<ArchetypeMatch> {
        self.archetypes
            .iter()
            .filter(|a| query.requires_all(a.signature()))
            .map(|a| ArchetypeMatch {
                archetype_id: a.archetype_id(),
                chunks: a.chunk_count(),
            })
            .collect()
    }
}
