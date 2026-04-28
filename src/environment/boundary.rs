//! Tick-lifecycle integration for [`Environment`].
//!
//! [`EnvironmentBoundary`] implements [`BoundaryResource`] so that the
//! scheduler can drive environment dirty-channel housekeeping - and optionally
//! GPU uniform buffer uploads - at the correct points in each tick.
//!
//! ## Lifecycle
//!
//! | Stage | Work done |
//! |---|---|
//! | `begin_tick` | No-op. Environment values persist across ticks by design. |
//! | `finalise(channels)` | If any channel in `channels` is dirty, then (GPU only) mark the uniform buffer dirty when at least one channel is **both** owned by the uniform **and** present in the environment's dirty set. Finally, clear those channels from the environment's dirty set. |
//! | `end_tick` | Drain any remaining dirty channels. If a GPU uniform buffer is present and any of its channels are still dirty, mark it dirty before clearing. Clears the entire dirty set. |
//!
//! ## GPU integration
//!
//! When a GPU uniform buffer is attached via
//! [`with_uniform`](EnvironmentBoundary::with_uniform), `finalise` bridges the
//! environment dirty-channel set to the uniform buffer's internal `cpu_dirty`
//! flag with a precise three-way intersection:
//!
//! 1. Compute the set of channels in `channels` that are **both** owned by the
//!    uniform buffer **and** currently dirty in the environment.
//! 2. If that set is non-empty, call
//!    [`mark_cpu_dirty`](super::uniform::EnvUniformBuffer::mark_cpu_dirty).
//! 3. Then clear `channels` from the environment's dirty set.
//!
//! This intersection is what prevents spurious GPU uploads when a system
//! over-declares its `produces` set: if a uniform-owned channel appears in
//! `channels` but no `Environment::set` actually wrote it, the uniform stays
//! clean and no upload is scheduled.
//!
//! ## Registration
//!
//! ```text
//! let boundary = EnvironmentBoundary::new(Arc::clone(&env));
//! let boundary_id = ecs_manager.register_boundary(Box::new(boundary));
//! ```

use std::sync::Arc;

use crate::engine::boundary::{BoundaryContext, BoundaryResource};
use crate::engine::error::ECSResult;
use crate::engine::types::ChannelID;

use super::store::Environment;

#[cfg(feature = "gpu")]
use super::uniform::EnvUniformBuffer;

// -----------------------------------------------------------------------------
// EnvironmentBoundary
// -----------------------------------------------------------------------------

/// [`BoundaryResource`] that drives dirty-channel housekeeping for an
/// [`Environment`].
///
/// Attach to an [`ECSManager`](crate::engine::manager::ECSManager) via
/// [`register_boundary`](crate::engine::manager::ECSManager::register_boundary).
/// The returned [`BoundaryID`](crate::engine::types::BoundaryID) can be used to
/// retrieve the boundary from within systems if direct access is needed.
///
/// # GPU uniform buffer
///
/// Optionally carries an [`EnvUniformBuffer`] that is marked dirty by
/// `finalise` whenever the environment's dirty channel set overlaps with the
/// channels tracked by the buffer. The actual GPU upload is deferred to the
/// registry's upload pass and is driven by the buffer's
/// [`is_cpu_dirty`](EnvUniformBuffer::is_cpu_dirty) flag.
pub struct EnvironmentBoundary {
    env: Arc<Environment>,

    /// Cached snapshot of the channel IDs owned by `env`, returned from
    /// [`channels`](BoundaryResource::channels). Populated at construction
    /// time so the engine can index them once at boundary registration.
    owned_channels: Vec<ChannelID>,

    #[cfg(feature = "gpu")]
    uniform: Option<EnvUniformBuffer>,
}

impl EnvironmentBoundary {
    /// Creates a new boundary wrapping the given environment.
    ///
    /// No GPU uniform buffer is attached. Call
    /// [`with_uniform`](Self::with_uniform) (requires `gpu` feature) to attach
    /// one.
    pub fn new(env: Arc<Environment>) -> Self {
        let owned_channels = env.all_channel_ids();
        Self {
            env,
            owned_channels,
            #[cfg(feature = "gpu")]
            uniform: None,
        }
    }

    /// Returns the environment owned by this boundary.
    pub fn environment(&self) -> &Arc<Environment> {
        &self.env
    }

    /// Attaches a GPU uniform buffer to this boundary.
    ///
    /// The buffer will be marked dirty by `finalise` and `end_tick` whenever
    /// the environment's dirty channel set contains channels owned by the
    /// buffer.
    #[cfg(feature = "gpu")]
    pub fn with_uniform(mut self, uniform: EnvUniformBuffer) -> Self {
        self.uniform = Some(uniform);
        self
    }

    /// Returns a shared reference to the attached [`EnvUniformBuffer`], or
    /// `None` if none was attached.
    #[cfg(feature = "gpu")]
    pub fn uniform(&self) -> Option<&EnvUniformBuffer> {
        self.uniform.as_ref()
    }

    /// Returns a mutable reference to the attached [`EnvUniformBuffer`], or
    /// `None` if none was attached.
    #[cfg(feature = "gpu")]
    pub fn uniform_mut(&mut self) -> Option<&mut EnvUniformBuffer> {
        self.uniform.as_mut()
    }
}

impl BoundaryResource for EnvironmentBoundary {
    fn name(&self) -> &str {
        "EnvironmentBoundary"
    }

    fn channels(&self) -> &[ChannelID] {
        &self.owned_channels
    }

    /// No per-tick reset is needed: environment values are persistent and the
    /// dirty channel set is managed by `finalise` and `end_tick`.
    fn begin_tick(&mut self, _ctx: &mut BoundaryContext<'_>) -> ECSResult<()> {
        Ok(())
    }

    /// Processes a subset of dirty channels at a scheduler boundary stage.
    ///
    /// Behaviour:
    ///
    /// 1. Returns early if no channel in `channels` is currently dirty in the
    ///    environment (`channels` may include IDs from other subsystems that
    ///    share the channel namespace).
    /// 2. (GPU only) If a uniform buffer is attached and at least one channel
    ///    in `channels` is **both** owned by the uniform **and** currently
    ///    dirty in the environment, calls
    ///    [`mark_cpu_dirty`](EnvUniformBuffer::mark_cpu_dirty). The
    ///    three-way intersection prevents spurious uploads when a system
    ///    over-declares its `produces` set.
    /// 3. Removes `channels` from the environment's dirty set via
    ///    [`Environment::clear_dirty_for_channels`]. IDs that are not in the
    ///    dirty set are silently ignored.
    fn finalise(
        &mut self,
        _ctx: &mut BoundaryContext<'_>,
        channels: &[ChannelID],
    ) -> ECSResult<()> {
        // Restrict to channels that are actually dirty in the environment.
        // `channels` may include IDs from other subsystems that share the same
        // channel namespace.
        if !self.env.has_any_dirty_channels(channels.iter().copied())? {
            return Ok(());
        }

        #[cfg(feature = "gpu")]
        if let Some(ref mut uniform) = self.uniform {
            // Mark dirty only if a uniform-owned channel is *actually* dirty
            // in the environment. Owning a channel that appears in `channels`
            // but was not written this tick must not trigger an upload.
            let touched_owned = channels.iter().try_fold(false, |touched, &id| {
                Ok::<_, crate::environment::EnvironmentError>(
                    touched || (uniform.owns_channel(id) && self.env.is_channel_dirty(id)?),
                )
            })?;
            if touched_owned {
                uniform.mark_cpu_dirty();
            }
        }

        self.env.clear_dirty_for_channels(channels)?;
        Ok(())
    }

    /// Drains all remaining dirty channels at the end of the tick.
    ///
    /// Collects any channels that were not consumed by a `finalise` call
    /// during the tick (e.g. writes that happened after the last boundary
    /// stage). If a GPU uniform buffer is attached and any of those remaining
    /// channels are owned by it, the buffer is marked dirty before the set is
    /// cleared.
    fn end_tick(&mut self, _ctx: &mut BoundaryContext<'_>) -> ECSResult<()> {
        let remaining = self.env.dirty_channel_ids()?;
        if remaining.is_empty() {
            return Ok(());
        }

        #[cfg(feature = "gpu")]
        if let Some(ref mut uniform) = self.uniform {
            if remaining.iter().any(|&id| uniform.owns_channel(id)) {
                uniform.mark_cpu_dirty();
            }
        }

        self.env.clear_dirty()?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::error::ECSError;
    use crate::environment::builder::EnvironmentBuilder;
    use crate::environment::EnvironmentError;
    use std::sync::Arc;

    fn make_boundary() -> (Arc<Environment>, EnvironmentBoundary) {
        let env = EnvironmentBuilder::new()
            .register::<f32>("interest_rate", 0.05)
            .unwrap()
            .register::<u32>("world_width", 100)
            .unwrap()
            .build()
            .unwrap();
        let boundary = EnvironmentBoundary::new(Arc::clone(&env));
        (env, boundary)
    }

    #[test]
    fn begin_tick_is_noop() {
        let (_env, mut boundary) = make_boundary();
        // begin_tick must not fail and must not alter environment state.
        let mut ctx = BoundaryContext::empty();
        boundary.begin_tick(&mut ctx).unwrap();
    }

    #[test]
    fn finalise_clears_specified_dirty_channels() {
        let (env, mut boundary) = make_boundary();
        let id_rate = env.channel_of("interest_rate").unwrap();
        let id_width = env.channel_of("world_width").unwrap();

        env.set::<f32>("interest_rate", 0.10).unwrap();
        env.set::<u32>("world_width", 200).unwrap();

        // Finalise only the interest_rate channel.
        let mut ctx = BoundaryContext::empty();
        boundary.finalise(&mut ctx, &[id_rate]).unwrap();

        let dirty = env.dirty_channel_ids().unwrap();
        assert!(
            !dirty.contains(&id_rate),
            "interest_rate should have been cleared"
        );
        assert!(
            dirty.contains(&id_width),
            "world_width should still be dirty"
        );
    }

    #[test]
    fn finalise_ignores_unrelated_channels() {
        let (env, mut boundary) = make_boundary();
        let id_rate = env.channel_of("interest_rate").unwrap();

        env.set::<f32>("interest_rate", 0.10).unwrap();

        // Pass a completely unrelated channel ID to finalise.
        let unrelated_id: ChannelID = 9999;
        let mut ctx = BoundaryContext::empty();
        boundary.finalise(&mut ctx, &[unrelated_id]).unwrap();

        // interest_rate must still be dirty - finalise saw only an unrelated ID.
        let dirty = env.dirty_channel_ids().unwrap();
        assert!(dirty.contains(&id_rate));
    }

    #[test]
    fn finalise_noop_when_env_is_clean() {
        let (env, mut boundary) = make_boundary();
        let id_rate = env.channel_of("interest_rate").unwrap();

        // Do not set any values - env is clean.
        let mut ctx = BoundaryContext::empty();
        boundary.finalise(&mut ctx, &[id_rate]).unwrap();

        assert!(env.dirty_channel_ids().unwrap().is_empty());
    }

    #[test]
    fn end_tick_clears_all_remaining_dirty_channels() {
        let (env, mut boundary) = make_boundary();

        env.set::<f32>("interest_rate", 0.10).unwrap();
        env.set::<u32>("world_width", 200).unwrap();

        let mut ctx = BoundaryContext::empty();
        boundary.end_tick(&mut ctx).unwrap();

        assert!(env.dirty_channel_ids().unwrap().is_empty());
    }

    #[test]
    fn end_tick_noop_when_already_clean() {
        let (env, mut boundary) = make_boundary();
        // No sets - nothing to drain.
        let mut ctx = BoundaryContext::empty();
        boundary.end_tick(&mut ctx).unwrap();
        assert!(env.dirty_channel_ids().unwrap().is_empty());
    }

    #[test]
    fn finalise_then_end_tick_leaves_env_clean() {
        let (env, mut boundary) = make_boundary();
        let id_rate = env.channel_of("interest_rate").unwrap();
        let id_width = env.channel_of("world_width").unwrap();

        env.set::<f32>("interest_rate", 0.10).unwrap();
        env.set::<u32>("world_width", 200).unwrap();

        let mut ctx = BoundaryContext::empty();
        boundary.finalise(&mut ctx, &[id_rate]).unwrap();
        boundary.end_tick(&mut ctx).unwrap();

        // After end_tick, the remaining world_width channel must also be gone.
        assert!(env.dirty_channel_ids().unwrap().is_empty());
        let _ = id_width; // used only implicitly via end_tick
    }

    #[test]
    fn finalise_propagates_poisoned_dirty_channel_error() {
        use std::thread;

        let (env, mut boundary) = make_boundary();
        let id_rate = env.channel_of("interest_rate").unwrap();
        let env_for_thread = Arc::clone(&env);
        let _ = thread::spawn(move || env_for_thread.poison_dirty_channels_for_test()).join();

        let err = boundary
            .finalise(&mut BoundaryContext::empty(), &[id_rate])
            .unwrap_err();
        assert!(matches!(
            err,
            ECSError::Environment(EnvironmentError::LockPoisoned {
                what: "environment dirty channels"
            })
        ));
    }

    #[test]
    fn end_tick_propagates_poisoned_dirty_channel_error() {
        use std::thread;

        let (env, mut boundary) = make_boundary();
        let env_for_thread = Arc::clone(&env);
        let _ = thread::spawn(move || env_for_thread.poison_dirty_channels_for_test()).join();

        let err = boundary.end_tick(&mut BoundaryContext::empty()).unwrap_err();
        assert!(matches!(
            err,
            ECSError::Environment(EnvironmentError::LockPoisoned {
                what: "environment dirty channels"
            })
        ));
    }
}

// -----------------------------------------------------------------------------
// GPU-integration tests: finalise / end_tick interaction with EnvUniformBuffer
// -----------------------------------------------------------------------------

#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;
    use crate::environment::builder::EnvironmentBuilder;
    use crate::environment::uniform::EnvUniformBuffer;
    use std::sync::Arc;

    /// Build an env with two keys, attach a uniform that tracks only `rate`.
    /// Returns `(env, boundary, rate_id, width_id)`.
    fn make_gpu_boundary() -> (Arc<Environment>, EnvironmentBoundary, ChannelID, ChannelID) {
        let env = EnvironmentBuilder::new()
            .register::<f32>("rate", 0.05_f32)
            .unwrap()
            .register::<u32>("width", 100_u32)
            .unwrap()
            .build()
            .unwrap();
        let id_rate = env.channel_of("rate").unwrap();
        let id_width = env.channel_of("width").unwrap();

        let uniform = EnvUniformBuffer::builder(Arc::clone(&env))
            .include::<f32>("rate")
            .build();
        let boundary = EnvironmentBoundary::new(Arc::clone(&env)).with_uniform(uniform);

        (env, boundary, id_rate, id_width)
    }

    /// Happy path: a system writes the uniform-tracked key, finalise sees
    /// that channel, and the uniform is marked dirty.
    #[test]
    fn finalise_marks_uniform_when_owned_channel_written() {
        let (env, mut boundary, id_rate, _id_width) = make_gpu_boundary();
        assert!(!boundary.uniform().unwrap().is_cpu_dirty());

        env.set::<f32>("rate", 0.10).unwrap();
        let mut ctx = BoundaryContext::empty();
        boundary.finalise(&mut ctx, &[id_rate]).unwrap();

        assert!(
            boundary.uniform().unwrap().is_cpu_dirty(),
            "uniform should be marked dirty when its owned channel was written"
        );
    }

    /// A non-tracked key is written, finalise is called for that key only:
    /// the uniform must remain clean.
    #[test]
    fn finalise_does_not_mark_uniform_when_only_untracked_channel_written() {
        let (env, mut boundary, _id_rate, id_width) = make_gpu_boundary();

        env.set::<u32>("width", 200).unwrap();
        let mut ctx = BoundaryContext::empty();
        boundary.finalise(&mut ctx, &[id_width]).unwrap();

        assert!(
            !boundary.uniform().unwrap().is_cpu_dirty(),
            "uniform should NOT be marked dirty when only an untracked channel was written"
        );
    }

    /// Regression test for the intersection fix.
    ///
    /// Setup: a system over-declares `produces: {rate, width}` but only
    /// actually writes `width` (rate is uniform-owned, width is not).
    /// Pre-fix code marked the uniform dirty because *some* channel in
    /// `channels` was uniform-owned. The intersection fix should leave the
    /// uniform clean because the uniform-owned channel was not written.
    #[test]
    fn finalise_does_not_mark_uniform_when_owned_channel_in_slice_but_not_dirty() {
        let (env, mut boundary, id_rate, id_width) = make_gpu_boundary();

        // Only `width` is actually written; `rate` is in the produces set
        // (passed to finalise) but was not written this tick.
        env.set::<u32>("width", 200).unwrap();
        let mut ctx = BoundaryContext::empty();
        boundary.finalise(&mut ctx, &[id_rate, id_width]).unwrap();

        assert!(
            !boundary.uniform().unwrap().is_cpu_dirty(),
            "uniform should stay clean: rate is in `channels` but was not written, \
             and width was written but is not uniform-owned"
        );
    }

    /// Mixed case: both an owned and a non-owned channel are written, all
    /// passed to finalise. The uniform must be marked dirty (the owned one
    /// was written).
    #[test]
    fn finalise_marks_uniform_when_both_owned_and_untracked_written() {
        let (env, mut boundary, id_rate, id_width) = make_gpu_boundary();

        env.set::<f32>("rate", 0.10).unwrap();
        env.set::<u32>("width", 200).unwrap();
        let mut ctx = BoundaryContext::empty();
        boundary.finalise(&mut ctx, &[id_rate, id_width]).unwrap();

        assert!(
            boundary.uniform().unwrap().is_cpu_dirty(),
            "uniform should be marked dirty because owned channel `rate` was written"
        );
    }

    /// `end_tick` is the safety net: if an owned channel's dirty mark was
    /// never consumed by a finalise call, end_tick must still mark the
    /// uniform dirty before draining.
    #[test]
    fn end_tick_marks_uniform_on_uncleared_owned_dirty_channel() {
        let (env, mut boundary, _id_rate, _id_width) = make_gpu_boundary();
        assert!(!boundary.uniform().unwrap().is_cpu_dirty());

        // Write an owned key but never call finalise for it.
        env.set::<f32>("rate", 0.10).unwrap();
        let mut ctx = BoundaryContext::empty();
        boundary.end_tick(&mut ctx).unwrap();

        assert!(
            boundary.uniform().unwrap().is_cpu_dirty(),
            "end_tick should mark the uniform dirty for any still-dirty owned channel"
        );
        assert!(
            env.dirty_channel_ids().unwrap().is_empty(),
            "end_tick should drain the dirty set"
        );
    }
}
