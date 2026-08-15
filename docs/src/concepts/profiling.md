# Profiling

_Requires the `profiling` feature._

Syren can emit a timeline of where a tick spends its time as a **Chrome Trace**
file, viewable in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev).

## Spans compile away when off

The framework and your model annotate work with tracing spans. When the
`profiling` feature is off, those spans compile to nothing, so an ordinary build
pays no runtime cost for them. This means you can leave span annotations in
model code permanently.

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
of stages; within a stage you can see systems executing in parallel. This is the
tool to reach for when a model is slower than expected: it shows whether time is
in one system, in a poorly parallelised stage, or in structural mutation at a
scheduler boundary.

See [collect results and profiles](../how-to/results-profiles.md) and
[performance methodology](../reference/performance.md).
