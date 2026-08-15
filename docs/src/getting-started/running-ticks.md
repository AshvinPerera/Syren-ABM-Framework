# Running ticks

A built [`Model`] advances one tick at a time. A tick has no fixed meaning; the
framework defines the order in which the parts of a tick run.

## Advancing the model

```rust,ignore
model.tick()?;        // advance one tick
model.run(50)?;       // advance 50 ticks
let done = model.tick_count();
```

[`tick`][Model::tick] runs one tick; [`run`][Model::run] runs a fixed number.
Both return an [`ECSResult`]. A failure in any system stops the tick and returns
the error; the world is not left partially updated.

## What happens in a tick

Each tick runs, in order:

1. **Sub-schedulers**, in the order they were added. A sub-scheduler shares the
   model's world and has its own systems and stages.
2. **Nested models**, each of which completes its own `tick` and then runs its
   bridge to write parent-facing effects.
3. **The root scheduler**, which runs the model's own systems.

Within a scheduler, systems are grouped into stages. Systems in the same stage
have non-conflicting access and run in parallel; stages run in sequence. See
[scheduling](../concepts/scheduling.md).

## Determinism across ticks

Each system's run context carries the model seed and the tick counter. Draws
taken through `DetRng::from_context` depend on `(seed, tick, system_id, salt)`,
so they do not vary with the worker-thread count. The same model and seed
produce the same trajectory on one core or many. See
[reproducibility](../reproducibility/guarantees.md).

[`Model`]: https://docs.rs/syren/latest/syren/model/struct.Model.html
[Model::tick]: https://docs.rs/syren/latest/syren/model/struct.Model.html#method.tick
[Model::run]: https://docs.rs/syren/latest/syren/model/struct.Model.html#method.run
[`ECSResult`]: https://docs.rs/syren/latest/syren/type.ECSResult.html
