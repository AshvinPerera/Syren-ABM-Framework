# Profiling

_Requires the `profiling` feature._

Syren can emit a timeline of where a tick spends its time as a **Chrome Trace**
file, viewable in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev).

## Spans compile away when off

The framework and model code annotate work with tracing spans. When the
`profiling` feature is off, those spans compile to nothing, so a build without
the feature has no span overhead and span annotations can stay in model code.

## Capturing a trace

Initialise the profiler with an output path before running, and shut it down
afterwards so the trace is flushed:

```rust,ignore
syren::init("profile/run.json");
// ... run the model ...
syren::shutdown();
```

The framework's own systems and boundaries are already instrumented, so a capture
shows the stages of each tick and the time inside them. Add spans in your systems
to attribute time to specific model phases.

## Reading a trace

Open the JSON file in `chrome://tracing` or Perfetto. Each tick appears as a run
of stages, with the systems in a stage executing in parallel. A trace shows
whether time is spent in one system, in a stage that parallelises poorly, or in
structural mutation at a scheduler boundary.

See [collect results and profiles](../how-to/results-profiles.md) and
[performance methodology](../reference/performance.md).
