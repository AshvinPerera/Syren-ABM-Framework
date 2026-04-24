//! Graphviz DOT export of a compiled scheduler plan.
//!
//! [`DotExport`] is a thin wrapper around [`Scheduler`] that implements
//! [`std::fmt::Display`], producing a Graphviz DOT graph that can be piped
//! directly to `dot`, `neato`, or any DOT-aware renderer:
//!
//! ```ignore
//! use abm_framework::engine::dot_export::DotExport;
//! std::fs::write("plan.dot", DotExport(&scheduler).to_string())?;
//! // $ dot -Tsvg plan.dot -o plan.svg
//! ```
//!
//! The produced graph uses one `subgraph cluster_<i>` per CPU/GPU stage with
//! each member system rendered as a labelled box, and diamond nodes for
//! boundary stages annotated with their finalised channel IDs. Invisible
//! "anchor" nodes between clusters carry the stage-ordering edges so the
//! rendered graph reads top-to-bottom in execution order (`rankdir=TB`).
//!
//! The actual formatting lives on [`Scheduler::fmt_dot`]; this wrapper only
//! exposes it through `fmt::Display` so callers can use the standard
//! formatting machinery.

use std::fmt;

use crate::engine::scheduler::Scheduler;

/// Display wrapper that renders a compiled scheduler plan as a Graphviz DOT
/// graph.
///
/// The wrapper borrows the scheduler for the duration of the formatting call
/// only; it adds no state of its own. Construct it inline at the call site:
///
/// ```ignore
/// let dot = DotExport(&scheduler).to_string();
/// ```
///
/// The output includes the `digraph ... { ... }` wrapper, so the result is
/// a self-contained DOT program. Pipe it to `dot -Tsvg` or similar.
///
/// # Staleness note
///
/// If the scheduler has been mutated since its last `rebuild` or `run`, the
/// exported graph reflects the last compiled state, not the pending one. Call
/// [`Scheduler::rebuild`] first if up-to-date output is required.
pub struct DotExport<'a>(pub &'a Scheduler);

impl<'a> fmt::Display for DotExport<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_dot(f)
    }
}
