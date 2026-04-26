//! Agent activation order for per-system iteration.
//!
//! In Agent-Based Models, the order in which agents (entities) are processed
//! within a system tick can significantly affect emergent behaviour. This module
//! provides [`ActivationOrder`], an enum that controls how a system visits
//! entities during [`ECSReference::for_each`] execution.
//!
//! ## Variants
//!
//! | Variant | Behaviour | Overhead |
//! |---------|-----------|----------|
//! | [`Sequential`](ActivationOrder::Sequential) | Natural archetype/chunk order | Zero |
//! | [`ShuffleChunks`](ActivationOrder::ShuffleChunks) | Chunks shuffled; rows within each chunk remain sequential | One Fisher-Yates pass over chunk list |
//! | [`ShuffleFull`](ActivationOrder::ShuffleFull) | Individual rows shuffled within each chunk | One Fisher-Yates pass per chunk |
//!
//! `Sequential` is the default and incurs no cost. The shuffle variants use
//! [`tl_rand_u64`](crate::engine::random::tl_rand_u64) so that each worker
//! thread has an independent, deterministically seeded RNG — results are
//! reproducible given a fixed global seed set via
//! [`Scheduler::seed`](crate::engine::scheduler::Scheduler::seed).
//!
//! ## Integration with the scheduler
//!
//! Activation orders are stored in the [`Scheduler`] and applied at iteration
//! time inside the `for_each` dispatch path. They are per-system, not global:
//! different systems in the same tick may use different activation orders.
//!
//! The scheduler exposes:
//! - [`Scheduler::set_activation_order`](crate::engine::scheduler::Scheduler::set_activation_order)
//! - [`Scheduler::activation_order`](crate::engine::scheduler::Scheduler::activation_order)

/// Controls the order in which entities are visited by a system's iteration.
///
/// See the [module-level documentation](self) for a comparison of variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ActivationOrder {
    /// Entities are processed in natural archetype/chunk storage order.
    ///
    /// This is the default. It incurs zero overhead and produces the same
    /// ordering across all ticks for a given world state.
    #[default]
    Sequential,

    /// Chunks are shuffled, but rows within each chunk remain sequential.
    ///
    /// Provides coarse-grained randomisation at low cost. Useful when the
    /// aggregate behaviour is sensitive to which chunk of agents acts first,
    /// but not to the within-chunk ordering.
    ShuffleChunks,

    /// Individual rows within every chunk are shuffled.
    ///
    /// Provides fine-grained randomisation at the cost of one Fisher-Yates
    /// pass per chunk per system invocation. Appropriate when within-chunk
    /// ordering would introduce systematic bias.
    ShuffleFull,
}
