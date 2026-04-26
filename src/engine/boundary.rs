//! Tick-lifecycle resources attached to [`ECSManager`](crate::engine::manager::ECSManager).
//!
//! A [`BoundaryResource`] is an object that participates in the per-tick
//! execution lifecycle managed by the engine. The engine knows only the
//! trait — it has no knowledge of what the resource contains or how it
//! works internally.
//!
//! # Lifecycle methods and access
//!
//! Every resource implements three lifecycle hooks called by the engine:
//!
//! - [`begin_tick`](BoundaryResource::begin_tick) — before any system runs
//!   in a tick.
//! - [`finalise`](BoundaryResource::finalise) — at each scheduler boundary
//!   stage, after `clear_borrows`, GPU sync, and `apply_deferred_commands`,
//!   for the channels whose last producer just completed.
//! - [`end_tick`](BoundaryResource::end_tick) — after the final stage and
//!   the final `apply_deferred_commands`.
//!
//! All three take `&mut self` and a [`BoundaryContext`]. The context carries
//! the engine surfaces a resource may need without granting access back to
//! the boundary registry itself; in particular it exposes
//! `gpu_resources_mut()` (with the `gpu` feature) so that a resource can
//! mark world-owned GPU buffers dirty as part of its own finalise step.
//!
//! Lifecycle hooks run while no systems are executing. The engine acquires
//! each resource's per-slot lock for write before calling them, so
//! concurrent system access via
//! [`ECSReference::boundary`](crate::engine::manager::ECSReference::boundary)
//! is impossible during a hook. Implementations therefore do not need
//! interior mutability for lifecycle state; they only need it for `&self`
//! methods that systems call concurrently from inside a parallel stage.
//!
//! # Channel ownership
//!
//! A resource declares which channel IDs it owns by overriding
//! [`channels`](BoundaryResource::channels). The engine builds an inverted
//! `ChannelID → BoundaryID` index at registration time and routes
//! `finalise` only to resources that own at least one of the channels in
//! the boundary stage. Channel IDs must be unique across all registered
//! resources; a collision causes
//! [`ECSManager::register_boundary`](crate::engine::manager::ECSManager::register_boundary)
//! to return
//! [`ExecutionError::DuplicateChannelRegistration`](crate::engine::error::ExecutionError::DuplicateChannelRegistration).
//!
//! # Reentrancy
//!
//! Lifecycle hooks must not call
//! [`ECSReference::boundary`](crate::engine::manager::ECSReference::boundary)
//! to acquire a handle on any other boundary resource — including
//! themselves. The engine holds the boundary registry mutex while
//! dispatching lifecycle calls; re-entering would deadlock. Cross-resource
//! coordination must go through ECS components or channel dependencies.

use std::any::Any;

use crate::engine::error::ECSResult;
use crate::engine::types::ChannelID;

/// Engine-side surfaces a [`BoundaryResource`] may interact with during a
/// lifecycle hook.
///
/// The context exists so resources can perform work that touches other
/// world-owned state — most commonly marking GPU buffers dirty after a
/// CPU-side acceleration index is rebuilt — without re-entering the
/// boundary registry.
pub struct BoundaryContext<'a> {
    #[cfg(feature = "gpu")]
    pub(crate) gpu_resources: Option<&'a mut crate::gpu::GPUResourceRegistry>,

    /// Marker so the lifetime parameter is always carried, even on builds
    /// that omit GPU support.
    pub(crate) _marker: std::marker::PhantomData<&'a mut ()>,
}

impl<'a> BoundaryContext<'a> {
    /// Empty context, used by callers that have no engine surfaces to
    /// expose to a lifecycle hook.
    pub(crate) fn empty() -> Self {
        Self {
            #[cfg(feature = "gpu")]
            gpu_resources: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Mutable reference to the world-owned GPU resource registry, if one is
    /// available for this lifecycle call.
    ///
    /// Returns `None` when the engine could not acquire exclusive access
    /// (caller did not set it) or when the `gpu` feature is disabled.
    #[cfg(feature = "gpu")]
    pub fn gpu_resources_mut(&mut self) -> Option<&mut crate::gpu::GPUResourceRegistry> {
        self.gpu_resources.as_deref_mut()
    }
}

/// Trait for tick-lifecycle resources owned by
/// [`ECSManager`](crate::engine::manager::ECSManager).
///
/// Implementors hook into three lifecycle events per tick — `begin_tick`,
/// `finalise`, and `end_tick` — and declare the set of channel IDs they
/// own so the scheduler can route `finalise` calls to the relevant
/// resources only.
pub trait BoundaryResource: Any + Send + Sync {
    /// Human-readable name for diagnostics and the plan display.
    fn name(&self) -> &str;

    /// Channel IDs whose finalise events this resource handles.
    ///
    /// The default returns an empty slice, meaning the resource never
    /// receives `finalise` calls. Resources with channel-driven lifecycle
    /// (message buffers, environment uniform mirrors) override this to
    /// return their owned IDs.
    ///
    /// The slice is read once during `register_boundary`; later changes are
    /// not picked up.
    fn channels(&self) -> &[ChannelID] {
        &[]
    }

    /// Called at the start of each tick, before any system runs.
    ///
    /// Typical work: clear per-tick buffers while retaining capacity, reset
    /// dirty flags, drain thread-local leftovers from the previous tick.
    fn begin_tick(&mut self, ctx: &mut BoundaryContext<'_>) -> ECSResult<()>;

    /// Called at the end of each tick, after all stages and deferred
    /// commands have completed.
    fn end_tick(&mut self, ctx: &mut BoundaryContext<'_>) -> ECSResult<()>;

    /// Called at each scheduler boundary stage that finalises a channel
    /// owned by this resource.
    ///
    /// `channels` is the full slice of channel IDs being finalised at the
    /// boundary; the resource may further filter against its own owned
    /// IDs. The engine guarantees that at least one of the listed IDs is
    /// owned by this resource — resources never receive an unrelated
    /// finalise call.
    fn finalise(&mut self, ctx: &mut BoundaryContext<'_>, channels: &[ChannelID]) -> ECSResult<()>;

    /// Returns a shared reference to `self` as `dyn Any` for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns a mutable reference to `self` as `dyn Any` for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
