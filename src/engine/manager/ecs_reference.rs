//! Non-owning handle granting shared or exclusive access to ECS state.
//!
//! # Overview
//!
//! [`ECSReference`] is a lightweight, copyable handle into a live [`ECSManager`].
//! It is the primary interface through which systems read and mutate ECS data
//! without taking ownership of the world. All access is gated by runtime safety
//! checks: a phase lock, an iteration scope counter, and a per-component borrow
//! tracker.
//!
//! # Access modes
//!
//! | Method family             | Phase   | Borrows        | Structural mutation |
//! |---------------------------|---------|----------------|---------------------|
//! | `for_each_*` / `for_each` | Read    | Read + Write   | Forbidden           |
//! | `reduce_*`                | Read    | Read only      | Forbidden           |
//! | `defer`                   | None    | None           | Queued              |
//! | `with_exclusive`          | Write   | Bypassed       | Immediate           |
//!
//! # Iteration API
//!
//! The preferred entry point for typed, parallel iteration is the generic
//! [`ECSReference::for_each`] method.  It accepts a single type parameter
//! composed of [`Read<T>`] and [`Write<T>`] markers that encodes the full
//! read/write signature of the iteration:
//!
//! ```ignore
//! // Single read:
//! ecs.for_each::<Read<Position>>(query, |pos| { ... })?;
//!
//! // Two reads, one write (closure receives a tuple):
//! ecs.for_each::<(Read<Position>, Read<Velocity>, Write<Acceleration>)>(
//!     query,
//!     |(pos, vel, acc)| { ... },
//! )?;
//! ```
//!
//! The named `for_each_rNwM` helpers remain available for callers that prefer
//! explicit method names.
//!
//! The older combinatorial methods (`for_each_read`, `for_each_read2`,
//! `for_each_read_write`, etc.) are deprecated; prefer `for_each` or the
//! `rNwM` equivalents.
//!
//! # Reduction API
//!
//! [`ECSReference::reduce_abstraction`] and its typed wrappers (`reduce_read`,
//! `reduce_read2`) perform parallel, read-only reductions over component data.
//! Each worker thread accumulates into a local value; results are combined
//! deterministically after all threads complete.
//!
//! # Structural mutation
//!
//! Direct structural mutation (spawning, despawning, adding or removing
//! components) must go through one of two paths:
//!
//! * **Deferred** – call [`ECSReference::defer`] to enqueue a [`Command`] for
//!   execution between scheduler stages. This is the preferred path for systems.
//! * **Immediate** – call [`ECSReference::with_exclusive`] to obtain a mutable
//!   reference to [`ECSData`] directly. This bypasses the scheduler and borrow
//!   tracker; the caller must guarantee no iteration is in progress.
//!
//! # GPU resources
//!
//! When the `gpu` feature is enabled, [`ECSReference::register_gpu_resource`]
//! allows world-owned GPU resources to be registered via `with_exclusive`.

use std::sync::atomic::Ordering;

use crate::engine::commands::Command;
use crate::engine::query::{QueryBuilder, BuiltQuery};
use crate::engine::storage::{cast_slice, cast_slice_mut};
use crate::engine::borrow::BorrowGuard;
use crate::engine::error::{
    ECSResult,
    ECSError,
    ExecutionError,
    InternalViolation,
};

#[cfg(feature = "gpu")]
use crate::engine::types::GPUResourceID;

#[cfg(feature = "gpu")]
use crate::gpu::GPUResource;

use super::iteration::IterationScope;
use super::query_param::{Read, Write, QueryParam};
use super::ecs_manager::ECSManager;
use super::data::ECSData;

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
    pub(super) manager: &'a ECSManager,
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
    /// Incorrect use can easily result in undefined behaviour.

    #[inline]
    pub fn with_exclusive<R>(
        &self,
        f: impl FnOnce(&mut ECSData) -> ECSResult<R>,
    ) -> ECSResult<R> {
        if self.manager.active_iters.load(Ordering::Acquire) != 0 {
            return Err(ECSError::from(ExecutionError::StructuralMutationDuringIteration));
        }

        let _phase = self.manager.phase_write()?;
        // SAFETY: We hold the exclusive phase lock.
        let data = unsafe { self.manager.data_mut_unchecked(&_phase) };
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
    ///
    /// ## Registry resolution
    /// The returned [`QueryBuilder`] resolves component IDs through this
    /// world's instance-owned registry. This ensures correctness in
    /// multi-world setups where each world has its own independent registry
    /// and components registered in one world are not visible in another.
    #[inline]
    pub fn query(&self) -> ECSResult<QueryBuilder> {
        Ok(QueryBuilder::with_registry(self.manager.registry()))
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

        let _borrows = BorrowGuard::new(
            &self.manager.borrows,
            &query.reads,
            &query.writes,
        ).map_err(ECSError::from)?;

        // SAFETY: We hold the shared phase lock (`_phase`).
        let data = unsafe { self.manager.data_ref_unchecked(&_phase) };
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
    /// 1. **Parallel accumulation** - each worker thread processes a disjoint
    ///    subset of archetype chunks and accumulates results into a thread-local
    ///    accumulator of type `R`.
    /// 2. **Deterministic combination** - all partial results are combined
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
    /// * `query` - A built ECS query specifying which components to read.
    /// * `init` - Constructs a fresh accumulator value for each worker thread.
    /// * `fold_chunk` - Updates an accumulator using the raw component slices
    ///   for a single chunk.
    /// * `combine` - Merges two accumulator values; must be associative.
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
            return Err(InternalViolation::ReduceWritesNotAllowed.into());
        }

        let _phase = self.manager.phase_read()?;
        let _iter = IterationScope::new(&self.manager.active_iters);

        let _borrows = BorrowGuard::new(
            &self.manager.borrows,
            &query.reads,
            &query.writes,
        )
            .map_err(ECSError::from)?;

        // SAFETY: We hold the shared phase lock (`_phase`).
        let data = unsafe { self.manager.data_ref_unchecked(&_phase) };
        data.reduce_abstraction_unchecked(query, init, fold_chunk, combine)
            .map_err(ECSError::from)
    }

    /// Registers a world-owned GPU resource.
    #[cfg(feature = "gpu")]
    pub fn register_gpu_resource<R: GPUResource + 'static>(
        &self,
        r: R,
    ) -> ECSResult<GPUResourceID> {
        self.with_exclusive(|data| Ok(data.register_gpu_resource(r)))
    }
}

// ---------------------------------------------------------------------------
// Generic tuple-based iteration — `for_each`
// ---------------------------------------------------------------------------

impl ECSReference<'_> {
    /// Generic, tuple-based parallel iteration over ECS components.
    ///
    /// `P` is a [`QueryParam`] implementor — either a bare `Read<T>` /
    /// `Write<T>` or a tuple thereof — that encodes the read/write signature.
    /// The closure `f` must match `P::Closure` (e.g. `dyn Fn((&A, &mut B))`
    /// for `(Read<A>, Write<B>)`).
    ///
    /// ## Usage
    ///
    /// ```ignore
    /// ecs.for_each::<Read<Position>>(query, &|pos| { ... })?;
    ///
    /// ecs.for_each::<(Read<Position>, Read<Velocity>, Write<Acceleration>)>(
    ///     query,
    ///     &|(pos, vel, acc)| { ... },
    /// )?;
    /// ```
    ///
    /// ## Safety
    /// Identical to `for_each_abstraction`: phase discipline, iteration scope,
    /// and borrow tracking are all enforced at runtime.
    pub fn for_each<P>(
        &self,
        query: BuiltQuery,
        f: &P::Closure,
    ) -> ECSResult<()>
    where
        P: QueryParam,
        P::Closure: Send + Sync,
    {
        P::validate(&query).map_err(ECSError::from)?;

        self.for_each_abstraction(query, move |reads, writes| {
            // SAFETY: for_each_abstraction guarantees that the byte slices
            // are correctly typed and aligned for the components declared in
            // the query, and that read/write aliasing rules are upheld by the
            // borrow tracker.
            unsafe { P::for_each_chunk(reads, writes, f); }
        })
    }
}

// ---------------------------------------------------------------------------
// Typed reduction helpers
// ---------------------------------------------------------------------------

impl ECSReference<'_> {
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
    /// * `query` - A built query reading exactly one component of type `A`.
    /// * `init` - Constructs a fresh accumulator for each worker thread.
    /// * `fold` - Updates the accumulator for each entity.
    /// * `combine` - Merges two accumulator values; must be associative.
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
            return Err(InternalViolation::QueryShapeMismatch {
                method: "reduce_read",
                expected_reads: 1,
                expected_writes: 0,
            }.into());
        }

        self.reduce_abstraction(
            query,
            init,
            move |acc, cols, _| unsafe {
                let slice = cast_slice::<A>(cols[0].as_ptr(), cols[0].len());
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
            return Err(InternalViolation::QueryShapeMismatch {
                method: "reduce_read2",
                expected_reads: 2,
                expected_writes: 0,
            }.into());
        }

        self.reduce_abstraction(
            query,
            init,
            move |acc, cols, _| unsafe {
                let slice_a = cast_slice::<A>(cols[0].as_ptr(), cols[0].len());
                let slice_b = cast_slice::<B>(cols[1].as_ptr(), cols[1].len());

                debug_assert_eq!(slice_a.len(), slice_b.len());

                for i in 0..slice_a.len() {
                    fold(acc, &slice_a[i], &slice_b[i]);
                }
            },
            combine,
        )
    }

}

// ---------------------------------------------------------------------------
// Named rNwM helpers
// ---------------------------------------------------------------------------

impl ECSReference<'_> {
    /// Typed parallel iteration: single read component.
    pub fn for_each_r1<A>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
    {
        <(Read<A>,) as QueryParam>::validate(&query)
            .map_err(ECSError::from)?;

        self.for_each_abstraction(query, move |reads, _| unsafe {
            let a = cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            for v in a {
                f(v);
            }
        })
    }

    /// Typed parallel iteration: single write component.
    pub fn for_each_w1<A>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&mut A) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
    {
        <(Write<A>,) as QueryParam>::validate(&query)
            .map_err(ECSError::from)?;

        self.for_each_abstraction(query, move |_, writes| unsafe {
            let bytes = writes[0].len();
            let slice = cast_slice_mut::<A>(writes[0].as_mut_ptr(), bytes);
            for item in slice {
                f(item);
            }
        })
    }

    /// Typed parallel iteration: two read components.
    pub fn for_each_r2<A, B>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &B) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
    {
        <(Read<A>, Read<B>) as QueryParam>::validate(&query)
            .map_err(ECSError::from)?;

        self.for_each_abstraction(query, move |reads, _| unsafe {
            let a = cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            let b = cast_slice::<B>(reads[1].as_ptr(), reads[1].len());
            debug_assert_eq!(a.len(), b.len());
            for i in 0..a.len() {
                f(&a[i], &b[i]);
            }
        })
    }

    /// Typed parallel iteration: three read components.
    pub fn for_each_r3<A, B, C>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &B, &C) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
        C: 'static + Send + Sync,
    {
        <(Read<A>, Read<B>, Read<C>) as QueryParam>::validate(&query)
            .map_err(ECSError::from)?;

        self.for_each_abstraction(query, move |reads, _| unsafe {
            let a = cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            let b = cast_slice::<B>(reads[1].as_ptr(), reads[1].len());
            let c = cast_slice::<C>(reads[2].as_ptr(), reads[2].len());
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), c.len());
            for i in 0..a.len() {
                f(&a[i], &b[i], &c[i]);
            }
        })
    }

    /// Typed parallel iteration: one read, one write.
    pub fn for_each_r1w1<A, B>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &mut B) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
    {
        <(Read<A>, Write<B>) as QueryParam>::validate(&query)
            .map_err(ECSError::from)?;

        self.for_each_abstraction(query, move |reads, writes| unsafe {
            let a = cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            let b = cast_slice_mut::<B>(writes[0].as_mut_ptr(), writes[0].len());
            debug_assert_eq!(a.len(), b.len());
            for i in 0..a.len() {
                f(&a[i], &mut b[i]);
            }
        })
    }

    /// Typed parallel iteration: two reads, one write.
    pub fn for_each_r2w1<A, B, C>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &B, &mut C) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
        C: 'static + Send + Sync,
    {
        <(Read<A>, Read<B>, Write<C>) as QueryParam>::validate(&query)
            .map_err(ECSError::from)?;

        self.for_each_abstraction(query, move |reads, writes| unsafe {
            let a = cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            let b = cast_slice::<B>(reads[1].as_ptr(), reads[1].len());
            let c = cast_slice_mut::<C>(writes[0].as_mut_ptr(), writes[0].len());
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), c.len());
            for i in 0..a.len() {
                f(&a[i], &b[i], &mut c[i]);
            }
        })
    }

    /// Typed parallel iteration: two reads, two writes.
    pub fn for_each_r2w2<A, B, C, D>(
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
        <(Read<A>, Read<B>, Write<C>, Write<D>) as QueryParam>::validate(&query)
            .map_err(ECSError::from)?;

        self.for_each_abstraction(query, move |reads, writes| unsafe {
            let a = cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            let b = cast_slice::<B>(reads[1].as_ptr(), reads[1].len());
            let c = cast_slice_mut::<C>(writes[0].as_mut_ptr(), writes[0].len());
            let d = cast_slice_mut::<D>(writes[1].as_mut_ptr(), writes[1].len());
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), c.len());
            debug_assert_eq!(a.len(), d.len());
            for i in 0..a.len() {
                f(&a[i], &b[i], &mut c[i], &mut d[i]);
            }
        })
    }
}
