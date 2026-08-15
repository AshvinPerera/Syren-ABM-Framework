# Collect results and profiles

## Collect results

Read per-agent data with a reduction and model-wide data from the environment;
both are covered in [reading and recording results](../getting-started/results.md).
To record a trajectory, read after each tick and append a row.

Give every output schema a single source of truth. Define the column names once,
next to the function that formats a row, and derive the header from the same
list. The macroeconomy example does this in `examples/macroeconomy/output.rs`:
the headline, aggregate-trace, and firm-trace schemas each have a column list and
a row builder in one place, and a test asserts that each header is a single line,
has unique names, and has the same field count as its row.

A sketch:

```rust,ignore
const COLUMNS: &[&str] = &["tick", "mean", "variance"];

fn header() -> String {
    COLUMNS.join(",")
}

fn row(tick: u64, stats: &Welford) -> String {
    format!("{},{:.6},{:.6}", tick, stats.mean, stats.variance())
}
```

Writing the header and row through the same column list keeps them from drifting
apart, and makes the "does the header match the row?" check a unit test rather
than a manual review.

## Collect a profile

_Requires the `profiling` feature._

Capture where a tick spends its time as a Chrome Trace:

```rust,ignore
syren::init("profile/run.json");
model.run(ticks)?;
syren::shutdown();
```

Open the JSON in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev). The
framework's stages and boundaries are already instrumented; add spans in your
systems to attribute time to model phases. When the `profiling` feature is off,
the spans compile away, so instrumentation costs nothing in a normal run.

See [profiling](../concepts/profiling.md) and [performance
methodology](../reference/performance.md).
