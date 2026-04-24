//! Thread-safe entry point to the ECS world.
//!
//! This module provides [`ECSManager`], the central owner of all ECS state,
//! and the lifecycle methods used by `Model::tick` and the scheduler.
//!
//! # Architecture
//!
//! ```text
//!                        ┌─────────────────────┐
//!                        │     ECSManager      │
//!                        │                     │
//!                        │  UnsafeCell<ECSData>│  ← owns all ECS state
//!                        │  RwLock<()> phase   │  ← read/write phase gate
//!                        │  BorrowTracker      │  ← per-component borrows
//!                        │  AtomicUsize iters  │  ← active iteration count
//!                        │  Mutex<Vec<Command>>│  ← deferred command queue
//!                        │  Mutex<Vec<Box<dyn  │
//!                        │    BoundaryResource>│  ← tick-lifecycle hooks
//!                        │  >>                 │    (messaging, env, ...)
//!                        └────────┬────────────┘
//!                                 │ .world_ref()
//!                                 ▼
//!                        ┌─────────────────────┐
//!                        │    ECSReference     │  ← shared, lightweight handle
//!                        └─────────────────────┘
//! ```
//!
//! # Boundary Resources
//!
//! messaging, environment, and logging register objects
//! that implement [`BoundaryResource`] before the simulation starts. The
//! manager calls lifecycle hooks on all of them at the right points:
//!
//! - [`begin_tick`](ECSManager::begin_tick) — before any stage.
//! - [`finalise_boundaries`](ECSManager::finalise_boundaries) — at each
//!   scheduler boundary stage, after `apply_deferred_commands`.
//! - [`end_tick`](ECSManager::end_tick) — after the final stage.
//!
//! Systems access individual resources via `ECSReference::boundary::<R>(id)`.
//!
//! # Concurrency Model
//!
//! The `boundary_resources` mutex is acquired in three distinct situations:
//!
//! 1. **Boundary stages** — exclusively, no systems running.
//! 2. **`begin_tick` / `end_tick`** — exclusively, no systems running.
//! 3. **`ECSReference::boundary` called from within a running system** —
//!    the returned [`BoundaryHandle`] holds the mutex guard for its entire
//!    lifetime. Two systems in the same parallel CPU stage that both hold a
//!    `BoundaryHandle` will serialise on this mutex for the duration of
//!    their handles. The scheduler does **not** currently detect this
//!    co-scheduling hazard; modules that access boundary resources from
//!    systems must either:
//!    - declare a channel dependency via `AccessSets::produces` /
//!      `AccessSets::consumes` so the scheduler places them in different
//!      stages, or
//!    - use [`Scheduler::add_ordering`](Scheduler::add_ordering)
//!      to force an explicit ordering.

use std::cell::UnsafeCell;
use std::sync::{Arc, RwLock, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::engine::boundary::BoundaryResource;
use crate::engine::commands::Command;
use crate::engine::scheduler::Scheduler;
use crate::engine::borrow::BorrowTracker;
use crate::engine::component::ComponentRegistry;
use crate::engine::types::{BoundaryID, ChannelID};
use crate::engine::error::{
    ECSResult,
    ECSError,
    ExecutionError,
};
use crate::Entity;
use super::phase::{PhaseRead, PhaseWrite};
use super::data::ECSData;
use super::ecs_reference::ECSReference;

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
///
/// ## Safety invariant
/// `data_mut_unchecked` requires a `&PhaseWrite<'_>` token, ensuring at
/// compile time that the caller holds the exclusive phase lock before
/// obtaining a mutable reference to `ECSData`.
///
/// `data_ref_unchecked` requires a `&PhaseRead<'_>` token,
/// ensuring the caller holds the shared phase lock before obtaining a
/// shared reference to `ECSData`.
pub struct ECSManager {
    pub(super) inner: UnsafeCell<ECSData>,
    pub(super) phase: RwLock<()>,
    pub(super) borrows: BorrowTracker,
    pub(super) active_iters: AtomicUsize,
    pub(super) deferred: Mutex<Vec<Command>>,

    /// Direct handle to the instance-owned component registry.
    ///
    /// Hoisted out of [`ECSData`] so that callers which only need to clone
    /// the `Arc` (e.g., building a [`QueryBuilder`](crate::engine::query::QueryBuilder))
    /// can do so without reaching through `UnsafeCell<ECSData>` — avoiding
    /// the data race that would occur if another thread were concurrently
    /// inside [`ECSManager::data_mut_unchecked`].
    ///
    /// The underlying `Arc<RwLock<ComponentRegistry>>` is `Sync`, so
    /// cloning it requires no additional synchronisation. The stored
    /// handle is kept in lockstep with `inner.registry()` by construction
    /// and never reassigned.
    pub(super) registry: Arc<RwLock<ComponentRegistry>>,

    /// Extension-owned boundary resources (message buffers, env boundary,
    /// future per-tick accumulators, ...).
    ///
    /// Keyed by [`BoundaryID`] assigned at registration time: the ID is the
    /// index into this `Vec`. The `Vec` grows monotonically; IDs are stable.
    ///
    /// **Lock discipline**: this mutex is acquired at:
    /// - boundary stages (no systems running),
    /// - `begin_tick` / `end_tick` (no systems running),
    /// - `ECSReference::boundary` from within a running system — the
    ///   returned `BoundaryHandle` holds the guard for its lifetime.
    ///
    /// The third case means co-scheduled systems that both acquire
    /// boundary handles will contend on this mutex. The scheduler does
    /// **not** detect this automatically; see the module-level
    /// `Concurrency Model` section for the constraints extension authors
    /// must manage themselves.
    pub(super) boundary_resources: Mutex<Vec<Box<dyn BoundaryResource>>>,
}

// SAFETY: `ECSManager` is `Sync` because:
// 1. The `phase` RwLock serializes read vs. write phases.
// 2. `active_iters` prevents structural mutation during iteration.
// 3. `borrows` (BorrowTracker) enforces per-component read/write rules.
// 4. `data_mut_unchecked` requires a `PhaseWrite` token.
// 5. `data_ref_unchecked` requires a `PhaseRead` token.
// 6. `boundary_resources` is behind a `Mutex`.
// 7. `registry` is `Arc<RwLock<ComponentRegistry>>` — the `Arc` is `Sync`
//    and the `RwLock` serialises its own contents; the field is never
//    reassigned after construction.
unsafe impl Sync for ECSManager {}

impl ECSManager {
    /// Creates a new ECS manager with the given raw `ECSData`.
    pub fn new(data: ECSData) -> Self {
        // Clone the registry Arc out of `data` before moving `data` into
        // the UnsafeCell, so the top-level `registry` field mirrors the
        // one inside `ECSData` without subsequent unsafe access.
        let registry = data.registry().clone();
        Self {
            inner: UnsafeCell::new(data),
            phase: RwLock::new(()),
            borrows: BorrowTracker::new(),
            active_iters: AtomicUsize::new(0),
            deferred: Mutex::new(Vec::new()),
            registry,
            boundary_resources: Mutex::new(Vec::new()),
        }
    }

    /// Creates a new ECS manager from an entity shard configuration, using a
    /// provided component registry.
    ///
    /// Prefer this constructor when multiple ECS worlds coexist in the same
    /// process. Each world receives its own registry, avoiding contention on
    /// the global registry and enabling independent component namespaces.
    pub fn with_registry(
        shards: crate::engine::entity::EntityShards,
        registry: Arc<RwLock<ComponentRegistry>>,
    ) -> Self {
        let data = ECSData::new(shards, registry.clone());
        Self {
            inner: UnsafeCell::new(data),
            phase: RwLock::new(()),
            borrows: BorrowTracker::new(),
            active_iters: AtomicUsize::new(0),
            deferred: Mutex::new(Vec::new()),
            registry,
            boundary_resources: Mutex::new(Vec::new()),
        }
    }

    /// Returns a clone of the instance-owned component registry.
    ///
    /// Provides access to the registry so that callers such as
    /// [`ECSReference::query`] can build queries that resolve component IDs
    /// through the correct instance registry. Reads the top-level
    /// `registry` field directly — no unsafe, no phase lock required.
    #[inline]
    pub(super) fn registry(&self) -> Arc<RwLock<ComponentRegistry>> {
        self.registry.clone()
    }

    /// Returns a lightweight reference handle to the ECS world.
    ///
    /// The returned handle does not transfer ownership and relies on the
    /// caller to respect API-level exclusivity guarantees.
    #[inline]
    pub fn world_ref(&self) -> ECSReference<'_> {
        ECSReference { manager: self }
    }

    /// Runs the given scheduler for one full tick.
    ///
    /// Tick order:
    ///
    /// 1. `begin_tick` on every registered boundary resource (no systems
    ///    running).
    /// 2. Scheduler executes all stages; boundary stages internally
    ///    invoke `clear_borrows`, GPU sync (if enabled),
    ///    `apply_deferred_commands`, and `finalise_boundaries`.
    /// 3. A final `clear_borrows` and `apply_deferred_commands` drain any
    ///    tail state produced in the last stage.
    /// 4. `end_tick` on every registered boundary resource.
    ///
    /// If any step fails, the error is returned immediately and subsequent
    /// steps are skipped. `end_tick` is **not** run when an earlier step
    /// fails — boundary resources should tolerate observing `begin_tick`
    /// without a matching `end_tick` during error recovery.
    pub fn run(&self, scheduler: &mut Scheduler) -> ECSResult<()> {
        self.begin_tick()?;

        let world = self.world_ref();
        scheduler.run(world).map_err(ECSError::from)?;
        world.clear_borrows();
        let _spawned = self.apply_deferred_commands()?;

        self.end_tick()?;
        Ok(())
    }

    /// Applies all queued deferred commands under the exclusive phase lock.
    ///
    /// This is a synchronisation point where structural changes requested
    /// during parallel or shared access phases are applied. Commands are
    /// drained and executed in FIFO order.
    pub fn apply_deferred_commands(&self) -> ECSResult<Vec<Entity>> {
        if self.active_iters.load(Ordering::Acquire) != 0 {
            return Err(ECSError::from(
                ExecutionError::StructuralMutationDuringIteration
            ));
        }

        let _phase = self.phase_write()?;

        let commands = {
            let mut queue = self.deferred.lock().map_err(|_| {
                ECSError::from(ExecutionError::LockPoisoned {
                    what: "deferred command queue",
                })
            })?;
            std::mem::take(&mut *queue)
        };

        // SAFETY: We hold the exclusive phase lock (`_phase`).
        let data = unsafe { self.data_mut_unchecked(&_phase) };
        data.apply_deferred_commands(commands)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Boundary resource registry
    // ─────────────────────────────────────────────────────────────────────────

    /// Registers a boundary resource and returns its stable [`BoundaryID`].
    ///
    /// Must be called before the simulation starts (typically from
    /// `ModelBuilder::build`). The returned ID is stored by the caller and
    /// passed to `ECSReference::boundary::<R>(id)` at system runtime.
    ///
    /// # Panics
    ///
    /// Panics if the boundary-resource mutex is poisoned (indicates a prior
    /// panic inside a boundary lifecycle method, which is a hard error).
    pub fn register_boundary<R: BoundaryResource + 'static>(&self, r: R) -> BoundaryID {
        let mut guard = self.boundary_resources.lock()
            .expect("boundary_resources mutex poisoned");
        let id = guard.len() as BoundaryID;
        guard.push(Box::new(r));
        id
    }

    /// Called by `Model::tick` before running any stage.
    ///
    /// Iterates all registered boundary resources and calls their
    /// [`begin_tick`](BoundaryResource::begin_tick) method in registration order.
    pub fn begin_tick(&self) -> ECSResult<()> {
        let mut guard = self.boundary_resources.lock().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "boundary_resources (begin_tick)",
            })
        })?;
        for r in guard.iter_mut() {
            r.begin_tick()?;
        }
        Ok(())
    }

    /// Called by `Model::tick` after the last stage and final
    /// `apply_deferred_commands`.
    ///
    /// Iterates all registered boundary resources and calls their
    /// [`end_tick`](BoundaryResource::end_tick) method in registration order.
    pub fn end_tick(&self) -> ECSResult<()> {
        let mut guard = self.boundary_resources.lock().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "boundary_resources (end_tick)",
            })
        })?;
        for r in guard.iter_mut() {
            r.end_tick()?;
        }
        Ok(())
    }

    /// Called by the scheduler at each boundary stage, after
    /// `clear_borrows`, GPU sync, and `apply_deferred_commands`.
    ///
    /// `channels` is the set of channel IDs whose last producer just
    /// completed — consumers in subsequent stages will read them.
    /// Each boundary resource receives the full `channels` slice and
    /// decides internally which IDs are relevant to it.
    pub(crate) fn finalise_boundaries(&self, channels: &[ChannelID]) -> ECSResult<()> {
        let mut guard = self.boundary_resources.lock().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "boundary_resources (finalise)",
            })
        })?;
        for r in guard.iter_mut() {
            r.finalise(channels)?;
        }
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase lock helpers (unchanged from original)
    // ─────────────────────────────────────────────────────────────────────────

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

    /// Returns a shared reference to the inner `ECSData`.
    ///
    /// # Safety
    ///
    /// The caller must hold the shared phase lock. This is enforced at
    /// compile time by requiring a `&PhaseRead<'_>` token.
    #[inline]
    pub(super) unsafe fn data_ref_unchecked(&self, _phase: &PhaseRead<'_>) -> &ECSData {
        unsafe { &*self.inner.get() }
    }

    /// Returns a mutable reference to the inner `ECSData`.
    ///
    /// # Safety
    ///
    /// The caller must hold the exclusive phase lock. This is enforced at
    /// compile time by requiring a `&PhaseWrite<'_>` token.
    #[inline]
    pub(super) unsafe fn data_mut_unchecked(&self, _phase: &PhaseWrite<'_>) -> &mut ECSData {
        unsafe { &mut *self.inner.get() }
    }
}
