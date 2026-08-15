# Use DetRng

[`DetRng`] is the sanctioned random number generator for models. It is what makes
a model's randomness reproducible independently of how work is scheduled across
threads.

## Why not a thread-local RNG

A generator that carries mutable state (a thread-local, or one stored on the
world) produces draws in the order rows are visited. Under Rayon's work stealing
that order changes with the thread count, so the trajectory would change too.
`DetRng` avoids this by **keying** each draw on coordinates that do not depend on
scheduling.

## Keying on the run context

Inside a system, open a stream from the run context and a salt:

```rust,ignore
let context = ecs.run_context();
let mut rng = DetRng::from_context(context, salt);
let u = rng.next_u64();
let f = rng.next_f64();          // in [0, 1)
let k = rng.next_below(6);       // in [0, 6)
```

`from_context` folds `(simulation_seed, tick, system_id, salt)`. Two draws with
the same coordinates produce the same value; different salts give independent
streams. The `simulation_seed` comes from `ModelBuilder::with_seed`.

## Per-agent streams

When a loop over agents draws randomness, salt the stream with the agent's
identity so the draw depends on the agent, not on the position at which it is
visited:

```rust,ignore
ecs.for_each_entity_w1::<Position>(query, move |entity, pos| {
    let mut rng = DetRng::from_context(context, u64::from(entity.index()));
    pos.x += if rng.next_below(2) == 0 { -1 } else { 1 };
})?;
```

This is the pattern the `first_model` example uses. It is what keeps the walk
identical at one thread and at eight.

## Draw helpers

- `next_u64`, `next_f64`, `next_f32` — raw draws.
- `next_below(upper)` — a `u64` in `[0, upper)`.
- `next_index(len)` — a `usize` in `[0, len)`, for choosing an element.

## Randomness outside a system

Population generation happens before any tick, so there is no run context. Seed a
separate `DetRng::from_seed(seed)` for that, keyed on the model's configured seed,
and keep it distinct from the per-tick streams.

[`DetRng`]: https://docs.rs/syren/latest/syren/struct.DetRng.html
