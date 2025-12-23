//! ECS world management and execution layer.
//!
//! This module defines the central orchestration layer of the ECS. It is
//! responsible for:
//!
//! * owning archetypes and component storage,
//! * coordinating entity movement between archetypes,
//! * managing deferred structural mutations,
//! * enforcing safe parallel iteration at runtime,
//! * providing controlled access to ECS state.
//!
//! ## Concurrency model
//!
//! The ECS uses **interior mutability** (`UnsafeCell`) to allow highly parallel
//! iteration while avoiding fine-grained locks in hot paths.
//!
//! Safety is enforced by **runtime mechanisms**:
//!
//! * **Phase discipline**
//!   - A global read/write phase lock prevents structural mutation during iteration.
//! * **IterationScope**
//!   - A global iteration counter forbids structural changes while iteration is active.
//! * **BorrowTracker**
//!   - Per-component runtime borrow tracking enforces Rust-like read/write rules.
//!
//! ## Structural mutation
//!
//! Structural changes (spawn, despawn, add/remove component) must be:
//!
//! * deferred via [`ECSReference::defer`], or
//! * performed inside an exclusive phase using [`ECSReference::with_exclusive`].
//!
//! Applying deferred commands is a global synchronization point and is
//! guaranteed not to overlap with iteration.
//!
//! ## Safety
//!
//! This module contains unsafe code for:
//!
//! * interior mutability (`UnsafeCell`),
//! * raw pointer component access,
//! * parallel execution using Rayon.
//!
//! Each unsafe block documents the invariants it relies on.
//! Violating these invariants results in undefined behavior.

use std::cell::UnsafeCell;
use std::sync::{
    Arc, 
    RwLock, 
    RwLockReadGuard, 
    RwLockWriteGuard, 
    Mutex
};
use std::sync::atomic::{
    AtomicUsize,
    AtomicBool, 
    Ordering
};
use std::collections::HashMap;
use rayon::prelude::*;

use crate::engine::commands::Command;
use crate::engine::types::{ 
    ComponentID, 
    ArchetypeID,
    ShardID,
    SIGNATURE_SIZE,
};
use crate::engine::query::{
    QuerySignature, 
    QueryBuilder, 
    BuiltQuery
};
use crate::engine::archetype::{
    Archetype,
    ArchetypeMatch
};
use crate::engine::entity::{
    Entity, 
    EntityShards
};
use crate::engine::storage::cast_slice;
use crate::engine::component::{
    Signature, 
    make_empty_component
};
use crate::engine::scheduler::Scheduler;
use crate::engine::borrow::{
    BorrowTracker, 
    BorrowGuard
};
use crate::engine::error::{
    ECSResult,
    ECSError,
    ExecutionError,
    SpawnError
};



/// A unit of parallel work operating on a single archetype chunk.
///
/// ## Purpose
/// `Job` packages raw pointers and byte lengths for component columns
/// (read-only and writable) so they can be processed in parallel by Rayon
/// without holding Rust references into archetype storage.
///
/// ## Safety
/// The pointers contained in a `Job` are valid only under the following
/// invariants:
///
/// * All pointers originate from valid component storage.
/// * Component storage is not structurally modified for the lifetime of the job.
/// * No two jobs contain overlapping mutable regions.
/// * All jobs execute within the ECS **iteration phase**, during which
///   structural mutation is prohibited.
///
/// These invariants are enforced by:
/// * execution phase discipline (read/write phase locking),
/// * `IterationScope` (prevents structural mutation),
/// * `BorrowTracker` (runtime read/write conflict detection),
/// * archetype chunk partitioning.


#[derive(Clone)]
struct Job {
    chunk: usize,
    len: usize,

    read_locks: Vec<Arc<RwLock<Box<dyn crate::engine::storage::TypeErasedAttribute>>>>,
    write_locks: Vec<Arc<RwLock<Box<dyn crate::engine::storage::TypeErasedAttribute>>>>,
}

struct IterationScope<'a>(&'a AtomicUsize);

impl<'a> IterationScope<'a> {
    #[inline]
    fn new(counter: &'a AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self(counter)
    }
}

impl Drop for IterationScope<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

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
    inner: UnsafeCell<ECSData>,
    phase: RwLock<()>,
    borrows: BorrowTracker,
    active_iters: AtomicUsize,
    deferred: Mutex<Vec<Command>>,
}

unsafe impl Sync for ECSManager {}

impl ECSManager {
    /// Creates a new ECS manager.
    ///
    /// ## Notes
    /// Archetypes are initially empty and created lazily as signatures are
    /// encountered.    

    pub fn new(data: ECSData) -> Self {
        Self {
            inner: UnsafeCell::new(data),
            phase: RwLock::new(()),
            borrows: BorrowTracker::new(),
            active_iters: AtomicUsize::new(0),
            deferred: Mutex::new(Vec::new()),
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

    /// Shared handle. This does *not* grant structural mutation by itself.
    #[inline]
    pub fn world_ref(&self) -> ECSReference<'_> {
        ECSReference { manager: self }
    }

    /// Runs the given scheduler for one full tick.
    pub fn run(&self, scheduler: &mut Scheduler) -> ECSResult<()> {
        scheduler.run(self.world_ref()).map_err(ECSError::from)?;
        self.apply_deferred_commands()?;
        Ok(())
    } 

    /// Applies all queued deferred commands.
    ///
    /// ## Semantics
    /// This is a synchronization point where structural changes requested
    /// during parallel or shared access phases are applied.
    ///
    /// ## Notes
    /// Commands are drained and executed in FIFO order.

    pub fn apply_deferred_commands(&self) -> ECSResult<()> {
        if self.active_iters.load(Ordering::Acquire) != 0 {
            return Err(ECSError::from(ExecutionError::StructuralMutationDuringIteration));
        }

        let _phase = self.phase_write()?;

        // Drain queue outside ECSData
        let commands = {
            let mut queue = self
                .deferred
                .lock()
                .map_err(|_| ECSError::from(ExecutionError::LockPoisoned { what: "deferred command queue" }))?;
            std::mem::take(&mut *queue)
        };

        // Apply to ECSData under exclusive phase
        let data = unsafe { self.data_mut_unchecked() };
        data.apply_deferred_commands(commands)?;

        Ok(())
    }

    #[inline]
    pub(crate) fn phase_read(&self) -> ECSResult<PhaseRead<'_>> {
        let g = self
            .phase
            .read()
            .map_err(|_| ECSError::from(ExecutionError::LockPoisoned { what: "ECS phase (read)" }))?;
        Ok(PhaseRead(g))
    }

    #[inline]
    pub(crate) fn phase_write(&self) -> ECSResult<PhaseWrite<'_>> {
        let g = self.phase.write().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned { what: "ECS phase (write)" })
        })?;
        Ok(PhaseWrite(g))
    }

    #[inline]
    unsafe fn data_ref_unchecked(&self) -> &ECSData {
        unsafe { &*self.inner.get() }
    }

    #[inline]
    unsafe fn data_mut_unchecked(&self) -> &mut ECSData {
        unsafe{ &mut *self.inner.get() }
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

#[derive(Copy, Clone)]
pub struct ECSReference<'a> {
    manager: &'a ECSManager,
}

impl<'a> ECSReference<'a> {

    /// Executes a closure with **exclusive access** to the ECS world.
    ///
    /// ## Purpose
    /// This function provides a **low-level escape hatch** for advanced use cases
    /// that require direct, immediate mutation of ECS state.
    ///
    /// ## Safety contract
    /// * This function **must not** be called during parallel iteration.
    /// * Calling it while iteration is active will panic.
    /// * The caller is responsible for maintaining all ECS invariants.
    ///
    /// ## Preferred alternative
    /// Most systems should use [`ECSReference::defer`] to request structural
    /// mutations instead of calling this function directly.
    ///
    /// ## Warning
    /// This API bypasses the scheduler, borrow tracker, and command buffering.
    /// Incorrect use can easily result in undefined behavior.
    
    #[inline]
    pub fn with_exclusive<R>(
        &self,
        f: impl FnOnce(&mut ECSData) -> ECSResult<R>,
    ) -> ECSResult<R> {
        if self.manager.active_iters.load(Ordering::Acquire) != 0 {
            return Err(ECSError::from(ExecutionError::StructuralMutationDuringIteration));
        }

        let _phase = self.manager.phase_write()?;
        let data = unsafe { self.manager.data_mut_unchecked() };
        f(data)
    }

    /// Queue a structural command.
    #[inline]
    pub fn defer(&self, command: Command) -> ECSResult<()> {
        let mut queue = self.manager.deferred.lock().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned { what: "deferred command queue" })
        })?;
        queue.push(command);
        Ok(())
    }

    /// Begins construction of a component query.
    #[inline]
    pub fn query(&self) -> ECSResult<QueryBuilder> {
        Ok(QueryBuilder::new())
    }

    /// Executes a generic, parallel, chunk-oriented ECS query.
    ///
    /// ## Execution model
    /// This function enforces all runtime safety guarantees before invoking
    /// the underlying unchecked iteration primitive:
    ///
    /// * acquires a global read phase,
    /// * increments the iteration counter,
    /// * acquires per-component borrows via the borrow tracker.
    ///
    /// The provided closure is invoked once per non-empty chunk.
    ///
    /// ## Safety
    /// This function is safe to call from systems. All aliasing, mutation,
    /// and structural constraints are enforced at runtime.
    ///
    /// The closure must not:
    /// * retain references beyond the call,
    /// * perform structural ECS mutations.

    pub fn for_each_abstraction(
        &self,
        query: BuiltQuery,
        f: impl Fn(&[&[u8]], &mut [&mut [u8]]) + Send + Sync,
    ) -> ECSResult<()> {
        let _phase = self.manager.phase_read()?;
        let _iter_scope = IterationScope::new(&self.manager.active_iters);

        // BorrowGuard returns ExecutionError, map into ECSError
        let _borrows = BorrowGuard::new(
            &self.manager.borrows,
            &query.reads,
            &query.writes,
        ).map_err(ECSError::from)?;

        let data = unsafe { self.manager.data_ref_unchecked() };
        data.for_each_abstraction_unchecked(query, f)
            .map_err(ECSError::from)?;

        Ok(())
    }
}

impl ECSReference<'_> {
    /// Executes a parallel iteration over a single read-only component.
    ///
    /// Invokes `f` once per entity for each matching archetype chunk.
    /// Structural ECS mutations must be deferred.
    ///
    /// ## Safety model
    /// Runtime safety is enforced by:
    /// - phase discipline (read phase)
    /// - IterationScope (no structural mutation)
    /// - BorrowGuard (no aliasing / RW conflicts)
    ///
    /// This function is safe to call from systems.

    pub fn for_each_read<A>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
    {
        if query.reads.len() != 1 || !query.writes.is_empty() {
            return Err(ECSError::Internal("for_each_read: query must have exactly 1 read and 0 writes".into()));
        }

        self.for_each_abstraction(query, move |reads, _| unsafe {
            let a = cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            for v in a {
                f(v);
            }
        })
    }

    /// Executes a parallel iteration over a single mutable component.
    pub fn for_each_write<A>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&mut A) + Send + Sync,
    ) -> ECSResult<()> {
        if !query.reads.is_empty() || query.writes.len() != 1 {
            return Err(ECSError::Internal(
                "for_each_write: query must have exactly 0 reads and 1 write".into(),
            ));
        }

        self.for_each_abstraction(query, |_, writes| {
            let a = unsafe { &mut *(writes[0].as_mut_ptr() as *mut A) };
            f(a);
        })
    }


    /// Executes a parallel iteration over two read-only components.
    pub fn for_each_read2<A, B>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &B) + Send + Sync,
    ) -> ECSResult<()> {
        if query.reads.len() != 2 || !query.writes.is_empty() {
            return Err(ECSError::Internal(
                "for_each_read2: query must have exactly 2 reads and 0 writes".into(),
            ));
        }

        self.for_each_abstraction(query, |reads, _| {
            let a = unsafe { &*(reads[0].as_ptr() as *const A) };
            let b = unsafe { &*(reads[1].as_ptr() as *const B) };
            f(a, b);
        })
    }

    /// Executes a parallel iteration over one read-only and one mutable component.
    pub fn for_each_read_write<A, B>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &mut B) + Send + Sync,
    ) -> ECSResult<()> {
        if query.reads.len() != 1 || query.writes.len() != 1 {
            return Err(ECSError::Internal(
                "for_each_read_write: query must have exactly 1 read and 1 write".into(),
            ));
        }

        self.for_each_abstraction(query, |reads, writes| {
            let a = unsafe { &*(reads[0].as_ptr() as *const A) };
            let b = unsafe { &mut *(writes[0].as_mut_ptr() as *mut B) };
            f(a, b);
        })
    }

    /// Executes a parallel iteration over two read-only components and one mutable component.
    pub fn for_each_read2_write_1<A, B, C>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &B, &mut C) + Send + Sync,
    ) -> ECSResult<()> {
        if query.reads.len() != 2 || query.writes.len() != 1 {
            return Err(ECSError::Internal(
                "for_each_read2_write_1: query must have exactly 2 reads and 1 write".into(),
            ));
        }

        self.for_each_abstraction(query, |reads, writes| {
            let a = unsafe { &*(reads[0].as_ptr() as *const A) };
            let b = unsafe { &*(reads[1].as_ptr() as *const B) };
            let c = unsafe { &mut *(writes[0].as_mut_ptr() as *mut C) };
            f(a, b, c);
        })
    }


    /// Executes a parallel iteration over two read-only and two mutable components.
    pub fn for_each_read2_write2<A, B, C, D>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &B, &mut C, &mut D) + Send + Sync,
    ) -> ECSResult<()> {
        if query.reads.len() != 2 || query.writes.len() != 2 {
            return Err(ECSError::Internal(
                "for_each_read2_write2: query must have exactly 2 reads and 2 writes".into(),
            ));
        }

        self.for_each_abstraction(query, |reads, writes| {
            let a = unsafe { &*(reads[0].as_ptr() as *const A) };
            let b = unsafe { &*(reads[1].as_ptr() as *const B) };
            let c = unsafe { &mut *(writes[0].as_mut_ptr() as *mut C) };
            let d = unsafe { &mut *(writes[1].as_mut_ptr() as *mut D) };
            f(a, b, c, d);
        })
    }

}

/// RAII guard representing the ECS read phase.
/// Holding this guard guarantees that no exclusive (structural) access exists.
#[allow(dead_code)] pub struct PhaseRead<'a>(RwLockReadGuard<'a, ()>);
/// phase write guard
#[allow(dead_code)] pub struct PhaseWrite<'a>(RwLockWriteGuard<'a, ()>);

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
    archetypes: Vec<Archetype>,

    /// Maps component signatures to archetype IDs.
    signature_map: HashMap<[u64; SIGNATURE_SIZE], ArchetypeID>,

    /// Entity shard allocator and location tracker.
    shards: EntityShards,

    next_spawn_shard: ShardID,
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
            next_spawn_shard: 0,
        }
    }

    #[inline]
    fn pick_spawn_shard(&mut self) -> ShardID {
        let shard = self.next_spawn_shard;
        self.next_spawn_shard = (self.next_spawn_shard + 1) % (self.shards.shard_count() as u16);
        shard
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

    fn get_or_create_archetype(&mut self, signature: &Signature) -> ECSResult<ArchetypeID> {
        let key = signature.components;
        if let Some(&id) = self.signature_map.get(&key) {
            return Ok(id);
        }

        let id = self.archetypes.len() as ArchetypeID;
        self.signature_map.insert(key, id);

        let arch = Archetype::new(id, *signature)?;
        self.archetypes.push(arch);

        Ok(id)
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
    ) -> ECSResult<(&mut Archetype, &mut Archetype)> {
        if a == b {
            return Err(ECSError::Internal("get_archetype_pair_mut: a == b".into()));
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
    ) -> ECSResult<()> {
        let Some(location) = self.shards.get_location(entity) else {
            return Err(SpawnError::StaleEntity.into());
        };
        let source_id = location.archetype;

        let mut new_signature = self.archetypes[source_id as usize].signature().clone();
        new_signature.set(added_component_id);

        let destination_id = self.get_or_create_archetype(&new_signature)?;
        let source_sig = self.archetypes[source_id as usize].signature().clone();
        let shards = &self.shards;

        let (source, destination) =
            Self::get_archetype_pair_mut(&mut self.archetypes, source_id, destination_id)?;

        let factory = || make_empty_component(added_component_id);

        destination
            .ensure_component(added_component_id, factory)
            .map_err(ECSError::from)?;

        // Ensure all shared components exist in destination storage
        for cid in source_sig.iterate_over_components() {
            if cid == added_component_id {
                continue;
            }

            let factory = || make_empty_component(cid);

            destination
                .ensure_component(cid, factory)
                .map_err(ECSError::from)?;

        }

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

    pub fn remove_component(&mut self, entity: Entity, removed_component_id: ComponentID) -> ECSResult<()> {
        let Some(location) = self.shards.get_location(entity) else {
            return Err(SpawnError::StaleEntity.into());
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

        for cid in source_sig.iterate_over_components() {
            if cid == removed_component_id {
                continue;
            }

            let factory = || make_empty_component(cid);

            dest_arch
                .ensure_component(cid, factory)
                .map_err(ECSError::from)?;
        }

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

    pub fn apply_deferred_commands(&mut self, commands: Vec<Command>) -> ECSResult<()> {
        for command in commands {
            match command {
                Command::Spawn { bundle } => {
                    let signature = bundle.signature();
                    let archetype_id = self.get_or_create_archetype(&signature)?;
                    let shard_id = self.pick_spawn_shard();

                    let archetype = &mut self.archetypes[archetype_id as usize];
                    let _entity = archetype.spawn_on(&mut self.shards, shard_id, bundle)?;
                }

                Command::Despawn { entity } => {
                    let loc = self.shards.get_location(entity).ok_or(SpawnError::StaleEntity)?;
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

        Ok(())
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


    pub(crate) fn for_each_abstraction_unchecked(
        &self,
        query: BuiltQuery,
        f: impl Fn(&[&[u8]], &mut [&mut [u8]]) + Send + Sync,
    ) -> Result<(), ExecutionError> {
        let matches = self.matching_archetypes(&query.signature)?;

        for m in matches {
            // Collect jobs for this archetype
            let jobs: Vec<Job> = {
                let archetype = &self.archetypes[m.archetype_id as usize];
                let mut jobs = Vec::new();

                let chunks = archetype
                    .chunk_count()
                    .map_err(|_| ExecutionError::InternalExecutionError)?;

                for chunk in 0..chunks {
                    let len = archetype
                        .chunk_valid_length(chunk)
                        .map_err(|_| ExecutionError::InternalExecutionError)?;

                    if len == 0 {
                        continue;
                    }

                    let mut reads = Vec::with_capacity(query.reads.len());
                    let mut writes = Vec::with_capacity(query.writes.len());

                    // Collect read-only component locks
                    for &component_id in &query.reads {
                        let locked = archetype
                            .component_locked(component_id)
                            .ok_or(ExecutionError::MissingComponent { component_id })?;
                        reads.push(locked.arc());
                    }

                    // Collect writable component locks
                    for &component_id in &query.writes {
                        let locked = archetype
                            .component_locked(component_id)
                            .ok_or(ExecutionError::MissingComponent { component_id })?;
                        writes.push(locked.arc());
                    }

                    jobs.push(Job {
                        chunk,
                        len,
                        read_locks: reads,
                        write_locks: writes,
                    });
                }

                jobs
            };

            let abort = Arc::new(AtomicBool::new(false));
            let err: Arc<Mutex<Option<ExecutionError>>> = Arc::new(Mutex::new(None));

            jobs.into_par_iter().for_each(|job| {
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

                let mut read_guards = Vec::with_capacity(job.read_locks.len());
                let mut write_guards = Vec::with_capacity(job.write_locks.len());
                let mut read_views = Vec::with_capacity(job.read_locks.len());
                let mut write_views = Vec::with_capacity(job.write_locks.len());

                // Read locks
                for lock in &job.read_locks {
                    let guard = match lock.read() {
                        Ok(g) => g,
                        Err(_) => {
                            fail(ExecutionError::LockPoisoned {
                                what: "component column (read)",
                            });
                            return;
                        }
                    };

                    let (ptr, bytes) = match guard.chunk_bytes(job.chunk as _, job.len) {
                        Some(v) => v,
                        None => {
                            fail(ExecutionError::InternalExecutionError);
                            return;
                        }
                    };

                    unsafe {
                        read_views.push(std::slice::from_raw_parts(ptr, bytes));
                    }
                    read_guards.push(guard);
                }

                // Write locks
                for lock in &job.write_locks {
                    let mut guard = match lock.write() {
                        Ok(g) => g,
                        Err(_) => {
                            fail(ExecutionError::LockPoisoned {
                                what: "component column (write)",
                            });
                            return;
                        }
                    };

                    let (ptr, bytes) = match guard.chunk_bytes_mut(job.chunk as _, job.len) {
                        Some(v) => v,
                        None => {
                            fail(ExecutionError::InternalExecutionError);
                            return;
                        }
                    };

                    unsafe {
                        write_views.push(std::slice::from_raw_parts_mut(ptr, bytes));
                    }
                    write_guards.push(guard);
                }

                // Execute callback
                f(&read_views, &mut write_views);
            });

            if abort.load(Ordering::Acquire) {
                let guard = err.lock().map_err(|_| {
                    ExecutionError::LockPoisoned {
                        what: "job error latch",
                    }
                })?;

                if let Some(e) = guard.clone() {
                    return Err(e);
                } else {
                    return Err(ExecutionError::InternalExecutionError);
                }
            }

        }

        Ok(())
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
}
