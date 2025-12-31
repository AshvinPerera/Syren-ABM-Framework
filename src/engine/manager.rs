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
use crate::engine::storage::{
    cast_slice,
    TypeErasedAttribute};
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

#[cfg(feature = "gpu")]
use crate::engine::dirty::DirtyChunks;


/// Per-archetype precomputed view into chunk memory.
///
/// Layout:
/// - read_ptrs[(chunk * n_reads) + i]  => (ptr, bytes) for read component i in that chunk
/// - write_ptrs[(chunk * n_writes) + i] => (ptr, bytes) for write component i in that chunk

struct ChunkView {
    chunk_count: usize,
    chunk_lens: Vec<usize>,
    n_reads: usize,
    n_writes: usize,
    read_ptrs: Vec<(*const u8, usize)>,
    write_ptrs: Vec<(*mut u8, usize)>,
}

// ChunkView contains raw pointers into archetype-owned component storage.
// These pointers are:
// - valid for the duration of for_each_abstraction_unchecked
// - protected by phase discipline (no structural mutation)
// - protected by BorrowGuard (no aliasing violations)
// - chunk-disjoint across parallel tasks

unsafe impl Send for ChunkView {}
unsafe impl Sync for ChunkView {}

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
        let world = self.world_ref();

        scheduler.run(world).map_err(ECSError::from)?;
        world.clear_borrows();
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

    /// Clears all component borrows.
    ///
    /// ## Semantics
    /// This marks the end of a scheduler execution stage.
    /// Must only be called when no systems are running.
    #[inline]
    pub(crate) fn clear_borrows(&self) {
        self.manager.borrows.clear();
    }

    #[inline]
    pub(crate) fn apply_deferred_commands(&self) -> ECSResult<()> {
        self.manager.apply_deferred_commands()
    }

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

    /// Executes a parallel, read-only reduction over ECS component data.
    ///
    /// ## Purpose
    /// This function computes a **single aggregated result** from all entities
    /// matching the given query, without mutating ECS state. It is intended for
    /// summary statistics, diagnostics, and global observations (e.g. counts,
    /// sums, means, variances).
    ///
    /// ## Execution model
    /// The reduction proceeds in two phases:
    /// 1. **Parallel accumulation** — each worker thread processes a disjoint
    ///    subset of archetype chunks and accumulates results into a thread-local
    ///    accumulator of type `R`.
    /// 2. **Deterministic combination** — all partial results are combined
    ///    serially using the provided `combine` function to produce the final
    ///    result.
    ///
    /// ## Concurrency and safety
    /// Runtime safety is enforced by:
    /// * phase discipline (read phase only),
    /// * [`IterationScope`] (prevents structural mutation),
    /// * [`BorrowGuard`] (enforces read-only component access).
    ///
    /// Structural ECS mutations and component writes are **not permitted** during
    /// a reduction. Queries containing write components will be rejected.
    ///
    /// ## Parameters
    /// * `query` — A built ECS query specifying which components to read.
    /// * `init` — Constructs a fresh accumulator value for each worker thread.
    /// * `fold_chunk` — Updates an accumulator using the raw component slices
    ///   for a single chunk.
    /// * `combine` — Merges two accumulator values; must be associative.
    ///
    /// ## Determinism
    /// Partial results are combined in a deterministic order independent of
    /// thread scheduling.
    ///
    /// ## Errors
    /// Returns an error if:
    /// * the query requests write access,
    /// * required components are missing,
    /// * a runtime safety invariant is violated.

    pub fn reduce_abstraction<R>(
        &self,
        query: BuiltQuery,
        init: impl Fn() -> R + Send + Sync,
        fold_chunk: impl Fn(&mut R, &[&[u8]], usize) + Send + Sync,
        combine: impl Fn(&mut R, R) + Send + Sync,
    ) -> ECSResult<R>
    where
        R: Send + 'static,
    {
        if !query.writes.is_empty() {
            return Err(ECSError::Internal(
                "reduce_abstraction: writes not allowed".into(),
            ));
        }

        let _phase = self.manager.phase_read()?;
        let _iter = IterationScope::new(&self.manager.active_iters);

        let _borrows = BorrowGuard::new(
            &self.manager.borrows,
            &query.reads,
            &query.writes,
        )
        .map_err(ECSError::from)?;

        let data = unsafe { self.manager.data_ref_unchecked() };
        data.reduce_abstraction_unchecked(query, init, fold_chunk, combine)
            .map_err(ECSError::from)
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
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
    {
        if !query.reads.is_empty() || query.writes.len() != 1 {
            return Err(ECSError::Internal(
                "for_each_write: expected exactly 0 reads and 1 write".into(),
            ));
        }

        self.for_each_abstraction(query, move |_, writes| unsafe {
            let bytes = writes[0].len();
            let slice =
                crate::engine::storage::cast_slice_mut::<A>(writes[0].as_mut_ptr(), bytes);

            for item in slice {
                f(item);
            }
        })
    }


    /// Executes a parallel iteration over two read-only components.
    pub fn for_each_read2<A, B>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &B) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
    {
        if query.reads.len() != 2 || !query.writes.is_empty() {
            return Err(ECSError::Internal(
                "for_each_read2: expected exactly 2 reads and 0 writes".into(),
            ));
        }

        self.for_each_abstraction(query, move |reads, _| unsafe {
            let a =
                crate::engine::storage::cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            let b =
                crate::engine::storage::cast_slice::<B>(reads[1].as_ptr(), reads[1].len());

            debug_assert_eq!(a.len(), b.len());

            for i in 0..a.len() {
                f(&a[i], &b[i]);
            }
        })
    }

    /// Executes a parallel iteration over one read-only and one mutable component.
    pub fn for_each_read_write<A, B>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &mut B) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
    {
        if query.reads.len() != 1 || query.writes.len() != 1 {
            return Err(ECSError::Internal(
                "for_each_read_write: expected exactly 1 read and 1 write".into(),
            ));
        }

        self.for_each_abstraction(query, move |reads, writes| unsafe {
            let a =
                crate::engine::storage::cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            let b = crate::engine::storage::cast_slice_mut::<B>(
                writes[0].as_mut_ptr(),
                writes[0].len(),
            );

            debug_assert_eq!(a.len(), b.len());

            for i in 0..a.len() {
                f(&a[i], &mut b[i]);
            }
        })
    }

    /// Executes a parallel iteration over two read-only components and one mutable component.
    pub fn for_each_read2_write1<A, B, C>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &B, &mut C) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
        C: 'static + Send + Sync,
    {
        if query.reads.len() != 2 || query.writes.len() != 1 {
            return Err(ECSError::Internal(
                "for_each_read2_write1: expected exactly 2 reads and 1 write".into(),
            ));
        }

        self.for_each_abstraction(query, move |reads, writes| unsafe {
            let a =
                crate::engine::storage::cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            let b =
                crate::engine::storage::cast_slice::<B>(reads[1].as_ptr(), reads[1].len());
            let c = crate::engine::storage::cast_slice_mut::<C>(
                writes[0].as_mut_ptr(),
                writes[0].len(),
            );

            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), c.len());

            for i in 0..a.len() {
                f(&a[i], &b[i], &mut c[i]);
            }
        })
    }


    /// Executes a parallel iteration over two read-only and two mutable components.
    pub fn for_each_read2_write2<A, B, C, D>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &B, &mut C, &mut D) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
        C: 'static + Send + Sync,
        D: 'static + Send + Sync,
    {
        if query.reads.len() != 2 || query.writes.len() != 2 {
            return Err(ECSError::Internal(
                "for_each_read2_write2: expected exactly 2 reads and 2 writes".into(),
            ));
        }

        self.for_each_abstraction(query, move |reads, writes| unsafe {
            let a =
                crate::engine::storage::cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            let b =
                crate::engine::storage::cast_slice::<B>(reads[1].as_ptr(), reads[1].len());
            let c = crate::engine::storage::cast_slice_mut::<C>(
                writes[0].as_mut_ptr(),
                writes[0].len(),
            );
            let d = crate::engine::storage::cast_slice_mut::<D>(
                writes[1].as_mut_ptr(),
                writes[1].len(),
            );

            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), c.len());
            debug_assert_eq!(a.len(), d.len());

            for i in 0..a.len() {
                f(&a[i], &b[i], &mut c[i], &mut d[i]);
            }
        })
    }

    /// Executes a typed, parallel reduction over a single read-only component.
    ///
    /// ## Purpose
    /// This is a convenience wrapper around [`reduce_abstraction`] that provides
    /// typed access to a single component column. The supplied `fold` function
    /// is invoked once per entity to update a thread-local accumulator.
    ///
    /// ## Execution model
    /// * Each worker thread processes disjoint chunks of entities.
    /// * The accumulator is updated locally using `fold`.
    /// * All partial accumulators are combined deterministically using `combine`.
    ///
    /// ## Safety and concurrency
    /// This function is safe to call from systems. Runtime safety is enforced via:
    /// * read-only phase discipline,
    /// * iteration scope tracking,
    /// * component borrow checking.
    ///
    /// The query must specify **exactly one read component** and no writes.
    ///
    /// ## Parameters
    /// * `query` — A built query reading exactly one component of type `A`.
    /// * `init` — Constructs a fresh accumulator for each worker thread.
    /// * `fold` — Updates the accumulator for each entity.
    /// * `combine` — Merges two accumulator values; must be associative.
    ///
    /// ## Typical use cases
    /// * Counting entities
    /// * Summing numeric component fields
    /// * Computing means or variances
    /// * Global diagnostics and convergence checks

    pub fn reduce_read<A, R>(
        &self,
        query: BuiltQuery,
        init: impl Fn() -> R + Send + Sync,
        fold: impl Fn(&mut R, &A) + Send + Sync,
        combine: impl Fn(&mut R, R) + Send + Sync,
    ) -> ECSResult<R>
    where
        A: 'static + Send + Sync,
        R: Send + 'static,
    {
        if query.reads.len() != 1 || !query.writes.is_empty() {
            return Err(ECSError::Internal(
                "reduce_read: requires exactly 1 read and 0 writes".into(),
            ));
        }

        self.reduce_abstraction(
            query,
            init,
            move |acc, cols, _| unsafe {
                let slice =
                    crate::engine::storage::cast_slice::<A>(cols[0].as_ptr(), cols[0].len());
                for v in slice {
                    fold(acc, v);
                }
            },
            combine,
        )
    }

    /// Executes a typed, parallel reduction over two read-only components.
    pub fn reduce_read2<A, B, R>(
        &self,
        query: BuiltQuery,
        init: impl Fn() -> R + Send + Sync,
        fold: impl Fn(&mut R, &A, &B) + Send + Sync,
        combine: impl Fn(&mut R, R) + Send + Sync,
    ) -> ECSResult<R>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
        R: Send + 'static,
    {
        if query.reads.len() != 2 || !query.writes.is_empty() {
            return Err(ECSError::Internal(
                "reduce_read2: requires exactly 2 reads and 0 writes".into(),
            ));
        }

        self.reduce_abstraction(
            query,
            init,
            move |acc, cols, _| unsafe {
                let slice_a =
                    crate::engine::storage::cast_slice::<A>(cols[0].as_ptr(), cols[0].len());
                let slice_b =
                    crate::engine::storage::cast_slice::<B>(cols[1].as_ptr(), cols[1].len());

                debug_assert_eq!(slice_a.len(), slice_b.len());

                for i in 0..slice_a.len() {
                    fold(acc, &slice_a[i], &slice_b[i]);
                }
            },
            combine,
        )
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

    /// Chunk-level dirty tracking for CPU writes.
    #[cfg(feature = "gpu")]
    gpu_dirty_chunks: DirtyChunks,    
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

            #[cfg(feature = "gpu")]
            gpu_dirty_chunks: DirtyChunks::new(),
        }
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

        #[cfg(feature = "gpu")]
        {
            // Structural changes invalidate all prior dirty tracking.
            self.gpu_dirty_chunks.notify_world_changed();
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

        // Share callback across spawned tasks
        let f = Arc::new(f);

        for matched_archetype in matches {
            let archetype = &self.archetypes[matched_archetype.archetype_id as usize];

            // Acquire guards
            let mut read_guards: Vec<RwLockReadGuard<'_, Box<dyn TypeErasedAttribute>>> =
                Vec::with_capacity(query.reads.len());
            let mut write_guards: Vec<RwLockWriteGuard<'_, Box<dyn TypeErasedAttribute>>> =
                Vec::with_capacity(query.writes.len());

            for &component_id in &query.reads {
                let locked = archetype
                    .component_locked(component_id)
                    .ok_or(ExecutionError::MissingComponent { component_id })?;

                let guard = locked.read().map_err(|_| ExecutionError::LockPoisoned {
                    what: "component column (read)",
                })?;

                read_guards.push(guard);
            }

            for &component_id in &query.writes {
                let locked = archetype
                    .component_locked(component_id)
                    .ok_or(ExecutionError::MissingComponent { component_id })?;

                let guard = locked.write().map_err(|_| ExecutionError::LockPoisoned {
                    what: "component column (write)",
                })?;

                write_guards.push(guard);
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

            // Precompute pointers
            let n_reads = read_guards.len();
            let n_writes = write_guards.len();

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

                for g in &read_guards {
                    let (ptr, bytes) = g
                        .chunk_bytes(chunk_id, len)
                        .ok_or(ExecutionError::InternalExecutionError)?;
                    read_ptrs.push((ptr as *const u8, bytes));
                }

                for g in &mut write_guards {
                    let (ptr, bytes) = g
                        .chunk_bytes_mut(chunk_id, len)
                        .ok_or(ExecutionError::InternalExecutionError)?;
                    write_ptrs.push((ptr as *mut u8, bytes));
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

            let abort = Arc::new(AtomicBool::new(false));
            let err: Arc<Mutex<Option<ExecutionError>>> = Arc::new(Mutex::new(None));
            let threads = rayon::current_num_threads().max(1);
            let grainsize = (views.chunk_count / threads).max(8);
            let views_ref = &views;

            #[cfg(feature = "gpu")]
            let dirty = self.gpu_dirty_chunks();

            #[cfg(feature = "gpu")]
            let archetype_id: ArchetypeID = matched_archetype.archetype_id;

            #[cfg(feature = "gpu")]
            let write_component_ids: std::sync::Arc<Vec<ComponentID>> =
                std::sync::Arc::new(query.writes.clone());

            rayon::scope(|s| {
                let mut start = 0usize;
                while start < views_ref.chunk_count {
                    let end = (start + grainsize).min(views_ref.chunk_count);

                    let abort = abort.clone();
                    let err = err.clone();
                    let f = f.clone();
                    let views = views_ref;

                    #[cfg(feature = "gpu")]
                    let write_component_ids = write_component_ids.clone();

                    s.spawn(move |_| {
                        let mut read_views: Vec<&[u8]> = Vec::with_capacity(views.n_reads);
                        let mut write_views: Vec<&mut [u8]> = Vec::with_capacity(views.n_writes);

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

                            // Mark dirty chunks for all write components touched by this query.
                            #[cfg(feature = "gpu")]
                            {
                                if views.n_writes != 0 {
                                    for &component_id in write_component_ids.as_slice() {
                                        dirty.mark_chunk_dirty(archetype_id, component_id, chunk, views.chunk_count);
                                    }
                                }
                            }

                            // Call user callback
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

                if let Some(e) = guard.clone() {
                    return Err(e);
                } else {
                    return Err(ExecutionError::InternalExecutionError);
                }
            }

            // Guards drop
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

        // (start_chunk, partial)
        let partials: Arc<Mutex<Vec<(usize, R)>>> =
            Arc::new(Mutex::new(Vec::new()));

        for matched in matches {
            let archetype = &self.archetypes[matched.archetype_id as usize];

            // Acquire read guards
            let mut read_guards: Vec<RwLockReadGuard<'_, Box<dyn TypeErasedAttribute>>> =
                Vec::with_capacity(query.reads.len());

            for &cid in &query.reads {
                let locked = archetype
                    .component_locked(cid)
                    .ok_or(ExecutionError::MissingComponent { component_id: cid })?;

                let guard = locked.read().map_err(|_| ExecutionError::LockPoisoned {
                    what: "component column (read)",
                })?;

                read_guards.push(guard);
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

            // Precompute read pointers
            let n_reads = read_guards.len();
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

                for g in &read_guards {
                    let (ptr, bytes) = g
                        .chunk_bytes(chunk_id, len)
                        .ok_or(ExecutionError::InternalExecutionError)?;
                    read_ptrs.push((ptr as *const u8, bytes));
                }
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
                        let mut read_views: Vec<&[u8]> =
                            Vec::with_capacity(views.n_reads);

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
}
