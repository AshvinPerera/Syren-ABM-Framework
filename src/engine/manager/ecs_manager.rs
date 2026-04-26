//! Thread-safe entry point to the ECS world.
//!
//! This module provides [`ECSManager`], the central owner of all ECS state,
//! and the lifecycle methods used by `Model::tick` and the scheduler.
//!
//! # Architecture
//!
//! ```text
//!                        ┌─────────────────────────┐
//!                        │      ECSManager         │
//!                        │                         │
//!                        │  UnsafeCell<ECSData>    │  ← owns all ECS state
//!                        │  RwLock<()> phase       │  ← read/write phase gate
//!                        │  BorrowTracker          │  ← per-component borrows
//!                        │  AtomicUsize iters      │  ← active iteration count
//!                        │  Mutex<Vec<Command>>    │  ← deferred command queue
//!                        │  Mutex<BoundaryRegistry>│  ← tick-lifecycle hooks
//!                        │                         │    (per-resource locks)
//!                        └─────────┬───────────────┘
//!                                  │ .world_ref()
//!                                  ▼
//!                        ┌─────────────────────────┐
//!                        │      ECSReference       │  ← shared, lightweight handle
//!                        └─────────────────────────┘
//! ```
//!
//! # Boundary Resources
//!
//! Messaging, environment, and other extension modules register objects
//! that implement [`BoundaryResource`] before the simulation starts. Each
//! registration returns a stable [`BoundaryID`]; the resource is wrapped in
//! an [`Arc<RwLock<dyn BoundaryResource>>`] so consumers can hold a typed
//! handle that does not block other resources' access. The manager calls
//! lifecycle hooks on each resource at the appropriate points:
//!
//! - [`begin_tick`](ECSManager::begin_tick) — before any stage.
//! - [`finalise_boundaries`](ECSManager::finalise_boundaries) — at each
//!   scheduler boundary stage, after `apply_deferred_commands`. Routed only
//!   to resources that own at least one of the channels in the boundary's
//!   `finalised_channels` set.
//! - [`end_tick`](ECSManager::end_tick) — after the final stage.
//!
//! Systems access individual resources via `ECSReference::boundary::<R>(id)`.
//!
//! # Concurrency Model
//!
//! Each registered resource has its own [`parking_lot::RwLock`]. The outer
//! [`Mutex`] over the resource vector is taken only at registration time.
//! From a running system, [`ECSReference::boundary`] clones the resource's
//! `Arc` and acquires the per-resource read lock; two systems in the same
//! stage that touch *different* resources never contend, and two systems
//! that touch the *same* resource take read locks in parallel. Lifecycle
//! hooks acquire the per-resource write lock; phase discipline prevents
//! these from racing with system access.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use parking_lot::RwLock as ResourceLock;

use super::data::ECSData;
use super::ecs_reference::ECSReference;
use super::phase::{PhaseRead, PhaseWrite};
use crate::engine::borrow::BorrowTracker;
use crate::engine::boundary::{BoundaryContext, BoundaryResource};
use crate::engine::commands::Command;
use crate::engine::component::ComponentRegistry;
use crate::engine::error::{ECSError, ECSResult, ExecutionError};
use crate::engine::scheduler::Scheduler;
use crate::engine::types::{BoundaryID, ChannelID};
use crate::Entity;

/// Per-resource lock used to serialise lifecycle writes against system
/// reads while keeping inter-resource access independent.
pub(super) type BoundarySlot = Arc<ResourceLock<dyn BoundaryResource>>;

/// Internal owner of all registered boundary resources, plus the
/// `ChannelID → BoundaryID` index used to route `finalise` calls.
pub(super) struct BoundaryRegistry {
    pub(super) slots: Vec<BoundarySlot>,
    pub(super) channel_owner: HashMap<ChannelID, BoundaryID>,
}

impl BoundaryRegistry {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            channel_owner: HashMap::new(),
        }
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
    /// The outer mutex protects only the slot vector itself — pushed at
    /// registration time. Individual resources are wrapped in their own
    /// [`parking_lot::RwLock`] inside an `Arc`, so handles obtained via
    /// [`ECSReference::boundary`](super::ecs_reference::ECSReference::boundary)
    /// hold a per-resource read guard rather than blocking the whole
    /// registry.
    pub(super) boundary_resources: Mutex<BoundaryRegistry>,
}

// SAFETY: `ECSManager` is `Sync` because:
// 1. The `phase` RwLock serializes read vs. write phases.
// 2. `active_iters` prevents structural mutation during iteration.
// 3. `borrows` (BorrowTracker) enforces per-component read/write rules.
// 4. `data_mut_unchecked` requires a `PhaseWrite` token.
// 5. `data_ref_unchecked` requires a `PhaseRead` token.
// 6. `boundary_resources` outer mutex serialises registration; each slot is
//    a separately-locked `Arc<RwLock<dyn BoundaryResource>>`.
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
            boundary_resources: Mutex::new(BoundaryRegistry::new()),
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
            boundary_resources: Mutex::new(BoundaryRegistry::new()),
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
                ExecutionError::StructuralMutationDuringIteration,
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
    /// The resource's [`channels`](BoundaryResource::channels) declaration
    /// is read once and indexed for routing. Each channel ID may be owned by
    /// at most one resource across the whole registry.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`ExecutionError::DuplicateChannelRegistration`](crate::engine::error::ExecutionError::DuplicateChannelRegistration)
    /// if any channel claimed by the resource is already owned by an
    /// earlier-registered resource.
    pub fn register_boundary<R: BoundaryResource + 'static>(&self, r: R) -> ECSResult<BoundaryID> {
        let mut guard = self.boundary_resources.lock().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "boundary_resources (register)",
            })
        })?;

        for &cid in r.channels() {
            if let Some(&existing) = guard.channel_owner.get(&cid) {
                return Err(ECSError::from(
                    ExecutionError::DuplicateChannelRegistration {
                        channel_id: cid,
                        existing_boundary: existing,
                    },
                ));
            }
        }

        let id = guard.slots.len() as BoundaryID;
        for &cid in r.channels() {
            guard.channel_owner.insert(cid, id);
        }
        let slot: BoundarySlot = Arc::new(ResourceLock::new(r));
        guard.slots.push(slot);
        Ok(id)
    }

    /// Called by `Model::tick` before running any stage.
    ///
    /// Iterates all registered boundary resources and calls their
    /// [`begin_tick`](BoundaryResource::begin_tick) method in registration
    /// order. Each resource is locked for write individually so concurrent
    /// access via [`ECSReference::boundary`](super::ecs_reference::ECSReference::boundary)
    /// to other resources is unaffected (callers must not invoke this from
    /// inside a parallel stage; phase discipline guarantees no system holds
    /// any handle when the manager calls into a lifecycle method).
    pub fn begin_tick(&self) -> ECSResult<()> {
        let slots = self.snapshot_slots("boundary_resources (begin_tick)")?;
        let mut ctx = self.make_boundary_context();
        for slot in &slots {
            slot.write().begin_tick(&mut ctx)?;
        }
        Ok(())
    }

    /// Called by `Model::tick` after the last stage and final
    /// `apply_deferred_commands`.
    pub fn end_tick(&self) -> ECSResult<()> {
        let slots = self.snapshot_slots("boundary_resources (end_tick)")?;
        let mut ctx = self.make_boundary_context();
        for slot in &slots {
            slot.write().end_tick(&mut ctx)?;
        }
        Ok(())
    }

    /// Called by the scheduler at each boundary stage, after
    /// `clear_borrows`, GPU sync, and `apply_deferred_commands`.
    ///
    /// Routes the call to each resource that owns at least one of
    /// `channels`. A resource that owns no listed channel is skipped.
    pub(crate) fn finalise_boundaries(&self, channels: &[ChannelID]) -> ECSResult<()> {
        if channels.is_empty() {
            return Ok(());
        }

        // Collect the unique set of resources whose owned channels
        // intersect `channels`, preserving registration order.
        let (slots, channel_owner) = {
            let guard = self.boundary_resources.lock().map_err(|_| {
                ECSError::from(ExecutionError::LockPoisoned {
                    what: "boundary_resources (finalise)",
                })
            })?;
            (guard.slots.clone(), guard.channel_owner.clone())
        };

        let mut targets: Vec<BoundaryID> = Vec::new();
        for &cid in channels {
            if let Some(&id) = channel_owner.get(&cid) {
                if !targets.contains(&id) {
                    targets.push(id);
                }
            }
        }
        targets.sort_unstable();

        let mut ctx = self.make_boundary_context();
        for id in targets {
            slots[id as usize].write().finalise(&mut ctx, channels)?;
        }
        Ok(())
    }

    /// Internal helper: clones the slot vector under the registration mutex,
    /// dropping the mutex immediately so per-resource locks can be acquired
    /// without contention.
    fn snapshot_slots(&self, what: &'static str) -> ECSResult<Vec<BoundarySlot>> {
        let guard = self
            .boundary_resources
            .lock()
            .map_err(|_| ECSError::from(ExecutionError::LockPoisoned { what }))?;
        Ok(guard.slots.clone())
    }

    /// Builds the [`BoundaryContext`] passed to lifecycle hooks.
    ///
    /// On builds with the `gpu` feature the context exposes the world's
    /// GPU resource registry through a phase-write borrow; without `gpu`
    /// it is empty. Lifecycle hooks run while no system is iterating, so
    /// taking the phase write here is uncontended.
    #[cfg(feature = "gpu")]
    fn make_boundary_context(&self) -> BoundaryContext<'_> {
        // GPU resource access during a lifecycle hook would require the
        // phase-write lock, which conflicts with the layered locking the
        // scheduler already does at boundary stages. The current shape
        // exposes `None` here so resources that need GPU access during
        // finalise should drive uploads through their own
        // [`crate::gpu::GPUResource`] registration; future work may thread
        // a properly-acquired registry handle through the boundary
        // dispatch path.
        BoundaryContext::empty()
    }

    /// Builds an empty [`BoundaryContext`] for non-GPU builds.
    #[cfg(not(feature = "gpu"))]
    fn make_boundary_context(&self) -> BoundaryContext<'_> {
        BoundaryContext::empty()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase lock helpers (unchanged from original)
    // ─────────────────────────────────────────────────────────────────────────

    #[inline]
    pub(crate) fn phase_read(&self) -> ECSResult<PhaseRead<'_>> {
        let g = self.phase.read().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "ECS phase (read)",
            })
        })?;
        Ok(PhaseRead(g))
    }

    #[inline]
    pub(crate) fn phase_write(&self) -> ECSResult<PhaseWrite<'_>> {
        let g = self.phase.write().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "ECS phase (write)",
            })
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
