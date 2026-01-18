/// Chrome Trace (flame-style) execution profiler.
///
/// This module provides a **feature-gated, zero-overhead (when disabled)**
/// profiling API for the ECS runtime. When enabled, it records structured
/// execution spans and emits a **Chrome Trace Event JSON** file that can be
/// inspected using:
///
/// - `chrome://tracing`
/// - <https://ui.perfetto.dev>
///
/// ## Feature flag
///
/// This module is only compiled when the `profiling` feature is enabled:
///
/// ```bash
/// cargo run --features profiling
/// ```
///
/// When the feature is disabled, all profiling calls compile to no-ops and
/// impose **zero runtime overhead** (no allocations, no atomics, no branches).
///
/// ## Usage
///
/// ```no_run
/// use abm_framework::profiler;
///
/// profiler::init("profile/trace.json");
///
/// {
///     let _g = profiler::span("Scheduler::run");
///     // run ECS / simulation tick
/// }
///
/// profiler::shutdown();
/// ```
///
/// ## Design notes
///
/// - Spans are recorded using RAII guards (`SpanGuard`)
/// - Events are timestamped using a monotonic clock
/// - Each OS thread is assigned a stable logical thread ID
/// - Output format follows the Chrome Trace `"X"` (complete event) specification
///
/// This profiler is intended for **performance analysis and optimization**
/// of ECS scheduling, system execution, query iteration, and GPU dispatch.

pub mod profiler;