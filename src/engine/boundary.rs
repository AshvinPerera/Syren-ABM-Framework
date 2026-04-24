//! Tick-lifecycle resources attached to [`ECSManager`](crate::engine::manager::ECSManager).
//!
//! A [`BoundaryResource`] is an object that participates
//! in the per-tick execution lifecycle managed by the engine. The engine knows
//! only the trait — it has no knowledge of what the resource contains or how it
//! works internally.
//!
//! ## Concurrency contract
//!
//! Methods on `BoundaryResource` come in two kinds:
//!
//! - **`&mut self`** (`begin_tick`, `end_tick`, `finalise`): called exclusively
//!   by the scheduler at boundary stages or by `ECSManager::{begin,end}_tick`.
//!   No systems are running when these are called, so exclusive access is safe
//!   without interior mutability.
//!
//! - Any `&self` methods a resource adds for systems (e.g., an `emit` method
//!   on `MessageBufferSet`) must use interior mutability (thread-local storage,
//!   atomics, `Mutex`) because they are called from inside `par_iter` across
//!   multiple Rayon threads.
//!
//! ## Access from systems
//!
//! Systems retrieve a short-lived [`BoundaryHandle`](crate::engine::manager::ECSReference)
//! via `ECSReference::boundary::<R>(id)`. The handle holds the boundary
//! registry mutex for its lifetime, serialising concurrent accesses. Systems
//! that access boundary resources should not share a parallel stage, because
//! two systems in the same Rayon `par_iter` would contend on the mutex.

use std::any::Any;

use crate::engine::error::ECSResult;
use crate::engine::types::ChannelID;


/// Trait for tick-lifecycle resources owned by [`ECSManager`](crate::engine::manager::ECSManager).
///
/// Implementors hook into three lifecycle events per tick:
///
/// 1. [`begin_tick`](BoundaryResource::begin_tick) — before any system runs.
/// 2. [`finalise`](BoundaryResource::finalise) — at each scheduler boundary
///    stage, after `clear_borrows`, GPU sync, and `apply_deferred_commands`.
/// 3. [`end_tick`](BoundaryResource::end_tick) — after the final stage and
///    final `apply_deferred_commands`.
///
/// All three methods receive `&mut self`, so no interior mutability is required
/// for the lifecycle methods themselves. Resources that need `&self` access
/// from within parallel system stages (e.g., emit buffers) implement that via
/// interior mutability on their own fields.
pub trait BoundaryResource: Any + Send + Sync {
    /// Human-readable name for diagnostics and the plan display.
    fn name(&self) -> &'static str;

    /// Called at the start of each tick, before any system runs.
    ///
    /// Typical work: clear per-tick buffers while retaining capacity, reset
    /// dirty flags, drain thread-local leftovers from the previous tick.
    ///
    /// Called under exclusive access (no systems running).
    ///
    /// # Reentrancy
    ///
    /// Implementations MUST NOT call
    /// [`ECSReference::boundary`](crate::engine::manager::ECSReference::boundary)
    /// or otherwise re-enter the boundary-resource registry. The `ECSManager`
    /// holds the registry mutex for the duration of this call; re-entering
    /// would deadlock. Cross-resource coordination must go through ECS
    /// components or channel dependencies, not through direct boundary
    /// access.
    fn begin_tick(&mut self) -> ECSResult<()>;

    /// Called at the end of each tick, after all stages and all deferred
    /// commands have completed.
    ///
    /// Typical work: flush per-tick logs, upload final GPU state, release
    /// transient allocations.
    ///
    /// Called under exclusive access (no systems running).
    ///
    /// # Reentrancy
    ///
    /// Same constraint as [`begin_tick`](BoundaryResource::begin_tick):
    /// implementations MUST NOT re-enter the boundary-resource registry
    /// via `ECSReference::boundary`. Doing so deadlocks.
    fn end_tick(&mut self) -> ECSResult<()>;

    /// Called at each scheduler boundary stage, after `clear_borrows`, GPU
    /// sync, and `apply_deferred_commands` have run.
    ///
    /// `channels` is the list of [`ChannelID`]s whose last producer just
    /// finished — consumers in subsequent stages will read them. Typically, the
    /// resource filters `channels` against its own registered channel IDs and
    /// only does work when a relevant channel appears.
    ///
    /// Typical work: merge thread-local emit slots into flat world buffers,
    /// build acceleration indices, upload GPU uniforms.
    ///
    /// Called under exclusive access (no systems running).
    ///
    /// # Reentrancy
    ///
    /// Same constraint as [`begin_tick`](BoundaryResource::begin_tick):
    /// implementations MUST NOT re-enter the boundary-resource registry
    /// via `ECSReference::boundary`. All lifecycle methods (`begin_tick`,
    /// `finalise`, `end_tick`) run while the `ECSManager` holds the
    /// registry mutex — any path that re-acquires it will deadlock.
    fn finalise(&mut self, channels: &[ChannelID]) -> ECSResult<()>;

    /// Returns a shared reference to `self` as `dyn Any` for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns a mutable reference to `self` as `dyn Any` for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
