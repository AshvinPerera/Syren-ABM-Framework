//! Human-readable export of a compiled scheduler plan.
//!
//! [`PlanDisplay`] is a thin wrapper around [`Scheduler`] that implements
//! [`std::fmt::Display`], allowing the compiled execution plan to be rendered
//! directly with `println!`, `write!`, or [`ToString::to_string`]:
//!
//! ```ignore
//! use abm_framework::engine::plan_display::PlanDisplay;
//! println!("{}", PlanDisplay(&scheduler));
//! ```
//!
//! The actual formatting logic lives on [`Scheduler::fmt_plan`]; this wrapper
//! only connects that method to the standard `Display` trait so that plans
//! can flow through any `fmt::Write` sink (log output, string buffers, test
//! snapshots) without the caller having to juggle `fmt::Formatter` manually.
//!
//! Each line of the output describes one stage: boundary stages list
//! finalised channel IDs; CPU and GPU stages list their member systems along
//! with any `produces` / `consumes` channel declarations. The format is
//! stable enough for human reading and test assertions but is **not** a
//! machine-parseable interchange format — use [`DotExport`](crate::engine::dot_export::DotExport)
//! for programmatic consumption.

use std::fmt;

use crate::engine::scheduler::Scheduler;

/// Display wrapper that renders a compiled scheduler plan as a text table.
///
/// The wrapper borrows the scheduler for the duration of the formatting call
/// only; it adds no state of its own. Construct it inline at the call site:
///
/// ```ignore
/// let plan_text = PlanDisplay(&scheduler).to_string();
/// ```
///
/// # Staleness note
///
/// If the scheduler has been mutated since its last `rebuild` or `run`, the
/// displayed plan reflects the last compiled state, not the pending one. Call
/// [`Scheduler::rebuild`] first if up-to-date output is required.
pub struct PlanDisplay<'a>(pub &'a Scheduler);

impl<'a> fmt::Display for PlanDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_plan(f)
    }
}