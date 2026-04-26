//! Non-owning handle granting shared or exclusive access to ECS state.
//!
//! [`ECSReference`] is a lightweight, copyable handle into a live [`ECSManager`].
//! It is the primary interface through which systems read and mutate ECS data.
//!
//! # Boundary Resource Access
//!
//! [`ECSReference::boundary`] provides typed access to boundary resources
//! registered on the `ECSManager`. The returned [`BoundaryHandle`] holds the
//! boundary registry mutex for its lifetime — keep it short. Systems that
//! acquire a `BoundaryHandle` must not be co-scheduled in the same parallel
//! stage (they would deadlock on the mutex).
//!
//! # Access modes
//!
//! | Method family             | Phase   | Borrows        | Structural mutation |
//! |---------------------------|---------|----------------|---------------------|
//! | `for_each_*` / `for_each` | Read    | Read + Write   | Forbidden           |
//! | `reduce_*`                | Read    | Read only      | Forbidden           |
//! | `defer`                   | None    | None           | Queued              |
//! | `with_exclusive`          | Write   | Bypassed       | Immediate           |
//! | `boundary`                | None    | None           | None (interior mut) |

use std::sync::atomic::Ordering;

use parking_lot::ArcRwLockReadGuard;
use parking_lot::RawRwLock;

use crate::engine::borrow::BorrowGuard;
use crate::engine::boundary::BoundaryResource;
use crate::engine::commands::Command;
use crate::engine::error::{ECSError, ECSResult, ExecutionError, InternalViolation};
use crate::engine::query::{BuiltQuery, QueryBuilder};
use crate::engine::storage::{cast_slice, cast_slice_mut};
use crate::engine::types::{BoundaryID, ChannelID};

#[cfg(feature = "gpu")]
use crate::engine::types::GPUResourceID;
#[cfg(feature = "gpu")]
use crate::gpu::GPUResource;
use crate::{ComponentID, Entity};

use super::data::ECSData;
use super::ecs_manager::ECSManager;
use super::iteration::IterationScope;
use super::query_param::{QueryParam, Read, Write};

/// A non-owning handle granting access to ECS data.
///
/// `ECSReference` allows systems to read or mutate ECS state while the
/// `ECSManager` remains shared across threads.
#[derive(Copy, Clone)]
pub struct ECSReference<'a> {
    pub(super) manager: &'a ECSManager,
}

impl<'a> ECSReference<'a> {
    /// Clears all component borrows. Must only be called when no systems are running.
    #[inline]
    pub(crate) fn clear_borrows(&self) {
        self.manager.borrows.clear();
    }

    #[inline]
    pub(crate) fn apply_deferred_commands(&self) -> ECSResult<Vec<Entity>> {
        self.manager.apply_deferred_commands()
    }

    /// Executes a closure with **exclusive access** to the ECS world.
    ///
    /// Must not be called during parallel iteration.
    #[inline]
    pub fn with_exclusive<R>(&self, f: impl FnOnce(&mut ECSData) -> ECSResult<R>) -> ECSResult<R> {
        if self.manager.active_iters.load(Ordering::Acquire) != 0 {
            return Err(ECSError::from(
                ExecutionError::StructuralMutationDuringIteration,
            ));
        }
        let _phase = self.manager.phase_write()?;
        // SAFETY: We hold the exclusive phase lock.
        let data = unsafe { self.manager.data_mut_unchecked(&_phase) };
        f(data)
    }

    /// Enqueues a structural command for deferred execution at the next boundary.
    #[inline]
    pub fn defer(&self, command: Command) -> ECSResult<()> {
        let mut queue = self.manager.deferred.lock().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "deferred command queue",
            })
        })?;
        queue.push(command);
        Ok(())
    }

    /// Begins construction of a component query against this world's registry.
    #[inline]
    pub fn query(&self) -> ECSResult<QueryBuilder> {
        Ok(QueryBuilder::with_registry(self.manager.registry()))
    }

    /// Returns a typed handle to a registered boundary resource.
    ///
    /// The returned [`BoundaryHandle`] owns a clone of the resource's `Arc`
    /// and a per-resource read guard. The outer registry mutex is dropped
    /// before the handle is returned, so two systems holding handles to
    /// different resources never contend, and two systems holding handles
    /// to the same resource take read locks in parallel.
    ///
    /// Drop the handle as soon as the interaction is complete; lifecycle
    /// hooks at the next boundary stage need to acquire the per-resource
    /// write lock and will block until the last reader releases.
    ///
    /// # Errors
    /// - [`ExecutionError::BoundaryAccessFailed`] with
    ///   `reason = OutOfRange` if `id` exceeds the number of registered
    ///   resources.
    /// - [`ExecutionError::BoundaryAccessFailed`] with
    ///   `reason = TypeMismatch` if the stored resource's concrete type
    ///   does not match `R`.
    /// - [`ExecutionError::LockPoisoned`] if the boundary registry mutex
    ///   is poisoned (another thread panicked while holding it).
    pub fn boundary<R: BoundaryResource + 'static>(
        &self,
        id: BoundaryID,
    ) -> ECSResult<BoundaryHandle<R>> {
        let slot = {
            let guard = self.manager.boundary_resources.lock().map_err(|_| {
                ECSError::from(ExecutionError::LockPoisoned {
                    what: "boundary_resources (ECSReference::boundary)",
                })
            })?;
            if id as usize >= guard.slots.len() {
                return Err(ECSError::from(ExecutionError::BoundaryAccessFailed {
                    reason: crate::engine::error::BoundaryAccessFailure::OutOfRange,
                    id,
                }));
            }
            guard.slots[id as usize].clone()
        };

        // Type-check before acquiring the read guard so the type-mismatch
        // path doesn't pay for a lock acquisition.
        {
            let probe = slot.read();
            if !probe.as_any().is::<R>() {
                return Err(ECSError::from(ExecutionError::BoundaryAccessFailed {
                    reason: crate::engine::error::BoundaryAccessFailure::TypeMismatch,
                    id,
                }));
            }
        }

        let read_guard = slot.read_arc();
        Ok(BoundaryHandle {
            guard: read_guard,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Generic parallel chunk-oriented ECS query.
    pub fn for_each_abstraction(
        &self,
        query: BuiltQuery,
        f: impl Fn(&[&[u8]], &mut [&mut [u8]]) + Send + Sync,
    ) -> ECSResult<()> {
        let _phase = self.manager.phase_read()?;
        let _iter_scope = IterationScope::new(&self.manager.active_iters);
        let _borrows = BorrowGuard::new(&self.manager.borrows, &query.reads, &query.writes)
            .map_err(ECSError::from)?;
        // SAFETY: shared phase lock held; iteration scope active.
        let data = unsafe { self.manager.data_ref_unchecked(&_phase) };
        data.for_each_abstraction_unchecked(query, f)
            .map_err(ECSError::from)?;
        Ok(())
    }

    /// Parallel chunk iteration whose closure may return an error.
    ///
    /// Same semantics as [`for_each_abstraction`] for borrow checking, phase
    /// discipline, and chunk-disjoint parallel execution. The closure
    /// returns [`ECSResult<()>`]; on the first error the iteration short-
    /// circuits and that error is returned. When two chunks fail
    /// concurrently the lower-indexed chunk's error wins, so the surfaced
    /// error is deterministic across thread counts.
    pub fn for_each_abstraction_fallible(
        &self,
        query: BuiltQuery,
        f: impl Fn(&[&[u8]], &mut [&mut [u8]]) -> ECSResult<()> + Send + Sync,
    ) -> ECSResult<()> {
        let _phase = self.manager.phase_read()?;
        let _iter_scope = IterationScope::new(&self.manager.active_iters);
        let _borrows = BorrowGuard::new(&self.manager.borrows, &query.reads, &query.writes)
            .map_err(ECSError::from)?;
        // SAFETY: shared phase lock held; iteration scope active.
        let data = unsafe { self.manager.data_ref_unchecked(&_phase) };
        data.for_each_abstraction_fallible_unchecked(query, f)
    }

    /// Generic parallel read-only reduction over ECS data.
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
        let _borrows = BorrowGuard::new(&self.manager.borrows, &query.reads, &query.writes)
            .map_err(ECSError::from)?;
        // SAFETY: shared phase lock held.
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

// Generic tuple for_each
impl ECSReference<'_> {
    /// Generic tuple-based parallel iteration. `P` is `Read<T>`/`Write<T>` or a tuple thereof.
    pub fn for_each<P>(&self, query: BuiltQuery, f: &P::Closure) -> ECSResult<()>
    where
        P: QueryParam,
        P::Closure: Send + Sync,
    {
        P::validate(&query).map_err(ECSError::from)?;
        self.for_each_abstraction(query, move |reads, writes| {
            // SAFETY: for_each_abstraction guarantees correct types and borrow rules.
            unsafe {
                P::for_each_chunk(reads, writes, f);
            }
        })
    }
}

// Typed reductions
impl ECSReference<'_> {
    /// Typed parallel reduction over one read component.
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
            }
            .into());
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

    /// Typed parallel reduction over two read components.
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
            }
            .into());
        }
        self.reduce_abstraction(
            query,
            init,
            move |acc, cols, _| unsafe {
                let a = cast_slice::<A>(cols[0].as_ptr(), cols[0].len());
                let b = cast_slice::<B>(cols[1].as_ptr(), cols[1].len());
                debug_assert_eq!(a.len(), b.len());
                for i in 0..a.len() {
                    fold(acc, &a[i], &b[i]);
                }
            },
            combine,
        )
    }
}

// Named rNwM helpers
impl ECSReference<'_> {
    /// Single read.
    pub fn for_each_r1<A>(&self, query: BuiltQuery, f: impl Fn(&A) + Send + Sync) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
    {
        <(Read<A>,) as QueryParam>::validate(&query).map_err(ECSError::from)?;
        self.for_each_abstraction(query, move |reads, _| unsafe {
            let a = cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            for v in a {
                f(v);
            }
        })
    }

    /// Single write.
    pub fn for_each_w1<A>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&mut A) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
    {
        <(Write<A>,) as QueryParam>::validate(&query).map_err(ECSError::from)?;
        self.for_each_abstraction(query, move |_, writes| unsafe {
            let slice = cast_slice_mut::<A>(writes[0].as_mut_ptr(), writes[0].len());
            for item in slice {
                f(item);
            }
        })
    }

    /// Two reads.
    pub fn for_each_r2<A, B>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &B) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
    {
        <(Read<A>, Read<B>) as QueryParam>::validate(&query).map_err(ECSError::from)?;
        self.for_each_abstraction(query, move |reads, _| unsafe {
            let a = cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            let b = cast_slice::<B>(reads[1].as_ptr(), reads[1].len());
            debug_assert_eq!(a.len(), b.len());
            for i in 0..a.len() {
                f(&a[i], &b[i]);
            }
        })
    }

    /// Three reads.
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
        <(Read<A>, Read<B>, Read<C>) as QueryParam>::validate(&query).map_err(ECSError::from)?;
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

    /// One read, one write.
    pub fn for_each_r1w1<A, B>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &mut B) + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
    {
        <(Read<A>, Write<B>) as QueryParam>::validate(&query).map_err(ECSError::from)?;
        self.for_each_abstraction(query, move |reads, writes| unsafe {
            let a = cast_slice::<A>(reads[0].as_ptr(), reads[0].len());
            let b = cast_slice_mut::<B>(writes[0].as_mut_ptr(), writes[0].len());
            debug_assert_eq!(a.len(), b.len());
            for i in 0..a.len() {
                f(&a[i], &mut b[i]);
            }
        })
    }

    /// Two reads, one write.
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
        <(Read<A>, Read<B>, Write<C>) as QueryParam>::validate(&query).map_err(ECSError::from)?;
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

    /// Two reads, two writes.
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

// Named rNwM helpers whose closures return ECSResult.
impl ECSReference<'_> {
    /// Single read, fallible closure.
    pub fn for_each_r1_fallible<A>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A) -> ECSResult<()> + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
    {
        <(Read<A>,) as QueryParam>::validate(&query).map_err(ECSError::from)?;
        self.for_each_abstraction_fallible(query, move |reads, _| {
            let a = unsafe { cast_slice::<A>(reads[0].as_ptr(), reads[0].len()) };
            for v in a {
                f(v)?;
            }
            Ok(())
        })
    }

    /// Single write, fallible closure.
    pub fn for_each_w1_fallible<A>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&mut A) -> ECSResult<()> + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
    {
        <(Write<A>,) as QueryParam>::validate(&query).map_err(ECSError::from)?;
        self.for_each_abstraction_fallible(query, move |_, writes| {
            let slice = unsafe { cast_slice_mut::<A>(writes[0].as_mut_ptr(), writes[0].len()) };
            for item in slice {
                f(item)?;
            }
            Ok(())
        })
    }

    /// Two reads, fallible closure.
    pub fn for_each_r2_fallible<A, B>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &B) -> ECSResult<()> + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
    {
        <(Read<A>, Read<B>) as QueryParam>::validate(&query).map_err(ECSError::from)?;
        self.for_each_abstraction_fallible(query, move |reads, _| {
            let a = unsafe { cast_slice::<A>(reads[0].as_ptr(), reads[0].len()) };
            let b = unsafe { cast_slice::<B>(reads[1].as_ptr(), reads[1].len()) };
            debug_assert_eq!(a.len(), b.len());
            for i in 0..a.len() {
                f(&a[i], &b[i])?;
            }
            Ok(())
        })
    }

    /// One read, one write, fallible closure.
    pub fn for_each_r1w1_fallible<A, B>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &mut B) -> ECSResult<()> + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
    {
        <(Read<A>, Write<B>) as QueryParam>::validate(&query).map_err(ECSError::from)?;
        self.for_each_abstraction_fallible(query, move |reads, writes| {
            let a = unsafe { cast_slice::<A>(reads[0].as_ptr(), reads[0].len()) };
            let b = unsafe { cast_slice_mut::<B>(writes[0].as_mut_ptr(), writes[0].len()) };
            debug_assert_eq!(a.len(), b.len());
            for i in 0..a.len() {
                f(&a[i], &mut b[i])?;
            }
            Ok(())
        })
    }

    /// Two reads, one write, fallible closure.
    pub fn for_each_r2w1_fallible<A, B, C>(
        &self,
        query: BuiltQuery,
        f: impl Fn(&A, &B, &mut C) -> ECSResult<()> + Send + Sync,
    ) -> ECSResult<()>
    where
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
        C: 'static + Send + Sync,
    {
        <(Read<A>, Read<B>, Write<C>) as QueryParam>::validate(&query).map_err(ECSError::from)?;
        self.for_each_abstraction_fallible(query, move |reads, writes| {
            let a = unsafe { cast_slice::<A>(reads[0].as_ptr(), reads[0].len()) };
            let b = unsafe { cast_slice::<B>(reads[1].as_ptr(), reads[1].len()) };
            let c = unsafe { cast_slice_mut::<C>(writes[0].as_mut_ptr(), writes[0].len()) };
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), c.len());
            for i in 0..a.len() {
                f(&a[i], &b[i], &mut c[i])?;
            }
            Ok(())
        })
    }

    /// Random-access single-entity component read.
    pub fn read_entity_component<T: 'static + Clone>(
        &self,
        entity: Entity,
        component_id: ComponentID,
    ) -> ECSResult<T> {
        let _phase = self.manager.phase_read()?;
        let data = unsafe { self.manager.data_ref_unchecked(&_phase) };
        data.read_component::<T>(entity, component_id)
    }

    /// Finalises boundary channels by merging thread-local buffers and building
    /// acceleration indices for the specified channels.
    ///
    /// This is called by the scheduler at boundary stages.
    #[inline]
    pub(crate) fn finalise_boundaries(&self, channels: &[ChannelID]) -> ECSResult<()> {
        self.manager.finalise_boundaries(channels)
    }
}

/// Short-lived typed reference to a boundary resource.
///
/// Acquired via [`ECSReference::boundary`]. Owns a per-resource read guard
/// (no other reader is blocked, write locks block until all handles are
/// dropped) and dereferences to `R`. Drop promptly.
pub struct BoundaryHandle<R: BoundaryResource + 'static> {
    guard: ArcRwLockReadGuard<RawRwLock, dyn BoundaryResource>,
    _phantom: std::marker::PhantomData<R>,
}

impl<R: BoundaryResource + 'static> std::ops::Deref for BoundaryHandle<R> {
    type Target = R;

    fn deref(&self) -> &R {
        // SAFETY: `ECSReference::boundary` verified that the inner resource
        // is of type `R` before constructing this handle. The Arc keeps the
        // resource pinned in memory for the duration of the guard.
        self.guard
            .as_any()
            .downcast_ref::<R>()
            .expect("BoundaryHandle downcast failed.")
    }
}
