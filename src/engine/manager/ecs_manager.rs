//! Thread-safe entry point to the ECS world.
//!
//! This module provides [`ECSManager`], the central owner of all ECS state,
//! and [`ECSReference`], a lightweight borrowed handle used to access that
//! state without transferring ownership.
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
//!                        └────────┬────────────┘
//!                                 │ .world_ref()
//!                                 ▼
//!                        ┌─────────────────────┐
//!                        │    ECSReference     │  ← shared, lightweight handle
//!                        └─────────────────────┘
//! ```
//!
//! # Concurrency Model
//!
//! Access to [`ECSData`] is gated by a two-phase locking scheme:
//!
//! | Phase token    | Lock mode | Grants                          |
//! |----------------|-----------|---------------------------------|
//! | [`PhaseRead`]  | Shared    | `&ECSData` via `data_ref_unchecked`  |
//! | [`PhaseWrite`] | Exclusive | `&mut ECSData` via `data_mut_unchecked` |
//!
//! Phase tokens are compile-time witnesses: neither accessor can be called
//! without first acquiring the appropriate lock, closing the soundness gap
//! that would exist if they accepted only `&self`.
//!
//! A secondary [`BorrowTracker`] enforces per-component aliasing rules
//! (analogous to `RefCell` but across component types), and an
//! [`AtomicUsize`] counter prevents structural mutation while any iterator
//! is live.
//!
//! # Deferred Commands
//!
//! Structural changes (entity spawning, despawning, component insertion, etc.)
//! requested during a tick are enqueued as [`Command`]s rather than applied
//! immediately. [`ECSManager::apply_deferred_commands`] drains the queue at a
//! safe synchronization point — after all systems have finished and no
//! iterators are active — and replays them under an exclusive phase lock.

use std::cell::UnsafeCell;
use std::sync::{Arc, RwLock, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::engine::commands::Command;
use crate::engine::scheduler::Scheduler;
use crate::engine::borrow::BorrowTracker;
use crate::engine::component::ComponentRegistry;
use crate::engine::error::{
    ECSResult,
    ECSError,
    ExecutionError,
};

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
}

// SAFETY: `ECSManager` is `Sync` because:
// 1. The `phase` RwLock serializes read vs. write phases.
// 2. `active_iters` prevents structural mutation during iteration.
// 3. `borrows` (BorrowTracker) enforces per-component read/write rules.
// 4. `data_mut_unchecked` requires a `PhaseWrite` token, making the phase
//    lock requirement compile-time enforced.
// 5. `data_ref_unchecked` requires a `PhaseRead` token.
unsafe impl Sync for ECSManager {}

impl ECSManager {
    /// Creates a new ECS manager with a fresh, empty component registry.
    ///
    /// ## Notes
    /// Archetypes are initially empty and created lazily as signatures are
    /// encountered. The component registry is wrapped in `Arc<RwLock<_>>` so
    /// that it can be shared with `ECSData` and accessed by archetype creation
    /// paths without relying on the global registry.

    pub fn new(data: ECSData) -> Self {
        Self {
            inner: UnsafeCell::new(data),
            phase: RwLock::new(()),
            borrows: BorrowTracker::new(),
            active_iters: AtomicUsize::new(0),
            deferred: Mutex::new(Vec::new()),
        }
    }

    /// Creates a new ECS manager from an entity shard configuration, using a
    /// provided component registry.
    ///
    /// ## Notes
    /// Prefer this constructor when multiple ECS worlds coexist in the same
    /// process. Each world receives its own registry, avoiding contention on
    /// the global registry and enabling independent component namespaces.

    pub fn with_registry(
        shards: crate::engine::entity::EntityShards,
        registry: Arc<RwLock<ComponentRegistry>>,
    ) -> Self {
        let data = ECSData::new(shards, registry);
        Self {
            inner: UnsafeCell::new(data),
            phase: RwLock::new(()),
            borrows: BorrowTracker::new(),
            active_iters: AtomicUsize::new(0),
            deferred: Mutex::new(Vec::new()),
        }
    }

    /// Returns a clone of the instance-owned component registry.
    ///
    /// ## Purpose
    /// Provides access to the registry so that callers such as
    /// [`ECSReference::query`] can build queries that resolve component IDs
    /// through the correct instance registry rather than the global one.
    /// This is essential for multi-world correctness where each world has its
    /// own independent registry.
    ///
    /// ## Safety
    /// This method only reads the `Arc` pointer stored inside `ECSData`.
    /// It does not access any component storage or archetype state, so no
    /// phase lock is required.
    #[inline]
    pub(super) fn registry(&self) -> Arc<RwLock<ComponentRegistry>> {
        // SAFETY: We are only cloning the Arc pointer held by ECSData.
        // No component data is read or written; the Arc clone is atomic.
        unsafe { (*self.inner.get()).registry().clone() }
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

        // SAFETY: We hold the exclusive phase lock (`_phase`).
        let data = unsafe { self.data_mut_unchecked(&_phase) };
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
        unsafe{ &mut *self.inner.get() }
    }
}
