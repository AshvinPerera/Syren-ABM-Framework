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
//! a Fisher-Yates pass driven by a `splitmix64` stream keyed on the global
//! seed, system id, archetype id, and chunk index - deliberately *not* a
//! thread-local RNG, so the visit order is reproducible for a fixed seed set
//! via [`Scheduler::seed`](crate::engine::scheduler::Scheduler::seed)
//! regardless of how Rayon assigns chunks to worker threads. Model code that
//! needs per-agent randomness should use [`DetRng`](crate::DetRng) for the
//! same reason.
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
/// See the module-level documentation for a comparison of variants.
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

/// Deterministic execution context for the system currently running on this
/// thread.
///
/// The scheduler installs this context before calling `System::run`. Code
/// running outside a scheduled system observes the zero-valued default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunContext {
    /// Global simulation seed selected by the model or generated crate.
    pub simulation_seed: u64,
    /// Current model tick.
    pub tick: u64,
    /// Stable identifier of the system currently executing.
    pub system_id: crate::engine::types::SystemID,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ActivationContext {
    pub(crate) order: ActivationOrder,
    pub(crate) seed: u64,
    pub(crate) system_id: crate::engine::types::SystemID,
}

impl Default for ActivationContext {
    fn default() -> Self {
        Self {
            order: ActivationOrder::Sequential,
            seed: 0,
            system_id: 0,
        }
    }
}

thread_local! {
    static CURRENT_ACTIVATION: std::cell::Cell<ActivationContext> =
        const { std::cell::Cell::new(ActivationContext {
            order: ActivationOrder::Sequential,
            seed: 0,
            system_id: 0,
        }) };
    static CURRENT_RUN_CONTEXT: std::cell::Cell<RunContext> =
        const { std::cell::Cell::new(RunContext {
            simulation_seed: 0,
            tick: 0,
            system_id: 0,
        }) };
}

pub(crate) fn current_activation_context() -> ActivationContext {
    CURRENT_ACTIVATION.with(std::cell::Cell::get)
}

/// Puts a thread-local `Cell` back to its previous value on drop.
///
/// Restoring after the closure returns is not enough. Rayon catches a panicking
/// task, propagates it to the caller, and returns the *worker* to the pool, so
/// a system that panics would otherwise leave its context installed for
/// whatever that worker runs next - and `RunContext`'s contract is that code
/// outside a scheduled system observes the zero-valued default. Unwinding
/// through the guard restores it.
struct RestoreOnDrop<'a, T: Copy + 'static> {
    cell: &'a std::cell::Cell<T>,
    previous: T,
}

impl<'a, T: Copy + 'static> RestoreOnDrop<'a, T> {
    /// Installs `context` and captures what it replaced.
    fn install(cell: &'a std::cell::Cell<T>, context: T) -> Self {
        let previous = cell.replace(context);
        Self { cell, previous }
    }
}

impl<T: Copy + 'static> Drop for RestoreOnDrop<'_, T> {
    fn drop(&mut self) {
        self.cell.set(self.previous);
    }
}

pub(crate) fn with_activation_context<R>(context: ActivationContext, f: impl FnOnce() -> R) -> R {
    CURRENT_ACTIVATION.with(|cell| {
        let _restore = RestoreOnDrop::install(cell, context);
        f()
    })
}

pub(crate) fn current_run_context() -> RunContext {
    CURRENT_RUN_CONTEXT.with(std::cell::Cell::get)
}

pub(crate) fn with_run_context<R>(context: RunContext, f: impl FnOnce() -> R) -> R {
    CURRENT_RUN_CONTEXT.with(|cell| {
        let _restore = RestoreOnDrop::install(cell, context);
        f()
    })
}

#[cfg(test)]
mod context_tests {
    use super::*;

    /// A panicking system must not leave its context behind on the worker.
    #[test]
    fn unwinding_restores_the_previous_run_context() {
        let outer = RunContext {
            simulation_seed: 7,
            tick: 3,
            system_id: 11,
        };
        with_run_context(outer, || {
            let inner = RunContext {
                simulation_seed: 99,
                tick: 99,
                system_id: 99,
            };
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_run_context(inner, || panic!("system blew up"));
            }));
            assert!(panicked.is_err());
            assert_eq!(current_run_context(), outer);
        });
        assert_eq!(current_run_context(), RunContext::default());
    }

    #[test]
    fn unwinding_restores_the_previous_activation_context() {
        let outer = ActivationContext {
            order: ActivationOrder::ShuffleFull,
            seed: 5,
            system_id: 2,
        };
        with_activation_context(outer, || {
            let inner = ActivationContext {
                order: ActivationOrder::ShuffleChunks,
                seed: 42,
                system_id: 8,
            };
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_activation_context(inner, || panic!("system blew up"));
            }));
            assert!(panicked.is_err());
            assert_eq!(current_activation_context(), outer);
        });
        assert_eq!(current_activation_context(), ActivationContext::default());
    }
}
