# Running ticks

A built [`Model`] advances one quarter, step, or generation at a time. What a
"tick" means is up to your model; the framework only guarantees the order in
which the parts of a tick run.

## Advancing the model

```rust,ignore
model.tick()?;        // advance exactly one tick
model.run(50)?;       // advance 50 ticks
let done = model.tick_count();
```

[`tick`][Model::tick] runs one tick; [`run`][Model::run] runs a fixed number.
Both return an [`ECSResult`], so a failure inside any system stops the tick and
surfaces the error rather than leaving the world half-updated.

## What happens in a tick

Each tick runs, in order:

1. **Sub-schedulers**, in the order they were added. A sub-scheduler shares the
   model's world but has its own systems and stages.
2. **Nested models**, each of which completes its own `tick` and then runs its
   bridge to write parent-facing effects.
3. **The root scheduler**, which runs the model's own systems.

Within a scheduler, systems are grouped into stages. Systems in the same stage
have non-conflicting access and run in parallel; stages run one after another.
See [scheduling](../concepts/scheduling.md).

## Determinism across ticks

The tick counter and the model seed feed each system's run context. Draws taken
through `DetRng::from_context` therefore depend on `(seed, tick, system, salt)`
and nothing else — not on the number of worker threads. Running the same model
with the same seed produces the same trajectory whether it runs on one core or
many. See [reproducibility](../science/reproducibility.md).

[`Model`]: https://docs.rs/syren/latest/syren/model/struct.Model.html
[Model::tick]: https://docs.rs/syren/latest/syren/model/struct.Model.html#method.tick
[Model::run]: https://docs.rs/syren/latest/syren/model/struct.Model.html#method.run
[`ECSResult`]: https://docs.rs/syren/latest/syren/type.ECSResult.html
