# Errors and failure boundaries

Syren reports failures as values, not panics, on its normal paths. Most
operations return an [`ECSResult`] (an alias for `Result` with the crate error
type), and a failure inside a tick stops the tick and surfaces the error rather
than leaving the world partly updated.

## Error categories

The crate error type composes the errors from each subsystem. The main
categories you will encounter:

- **Registration** — a component or resource used before it was registered, or a
  type mismatch. Freeze the registry after registering all components.
- **Query shape** — a query passed to a method that expects a different shape (for
  example, `reduce_read` requires exactly one read and no writes).
- **Borrow conflict** — two systems, or two accesses, that alias a component
  incompatibly. This is the runtime backstop behind the scheduler's access
  analysis.
- **Stale entity** — an operation on an entity handle whose slot was recycled; the
  version no longer matches live storage.
- **Spawn and structural** — a failed batch spawn or despawn. Batches are atomic:
  on error the whole batch rolls back.
- **Model build** — a validation failure in `ModelBuilder::build`, such as
  duplicate sub-scheduler names or a channel used out of scope.
- **Messaging, environment, and space** — subsystem-specific failures such as an
  invalid message layout or a bad environment key type.

## Failure boundaries

Failures are contained at well-defined boundaries:

- **Within a tick**, the first system error stops execution and is returned; the
  world is not left half-updated by continuing.
- **Structural mutation** (spawns, despawns, migrations) applies at the scheduler
  boundary and is atomic per batch, so a failure rolls the batch back rather than
  leaving a partial cohort.
- **Build-time validation** rejects an inconsistent model before it ever runs.

## Handling errors

Propagate errors with `?` and decide at the top of your run loop whether a failed
tick is recoverable. Because the world is not left in a torn state, you can
inspect it after a failed tick to diagnose the cause.

[`ECSResult`]: https://docs.rs/syren/latest/syren/type.ECSResult.html
