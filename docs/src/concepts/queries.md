# Queries and borrows

A **query** selects the entities that have a set of components and declares
whether each is read or written. You build one with [`QueryBuilder`], naming the
registry it resolves component types against:

```rust,ignore
let query = QueryBuilder::with_registry(registry)
    .read::<Position>()?
    .write::<Velocity>()?
    .build()?;
```

The result is a [`BuiltQuery`]: a resolved signature of component identifiers and
access modes. A built query is cheap to clone and can be reused across ticks.

## Iterating

A query is run against an [`ECSReference`] (a handle to the world for the current
system). The `for_each` family walks every matching chunk:

- `for_each` iterates the declared components.
- The `for_each_entity_*` variants also give you the [`Entity`] for each row —
  use these when a per-agent computation needs the entity's identity, for
  example to salt a deterministic RNG.
- The `reduce_read` family folds a read query into an accumulator; see [reading
  results](../getting-started/results.md).

Iteration parallelises across chunks. Because each closure sees only the rows in
one chunk, and reductions combine per-chunk partials, the work distributes across
threads without changing the result.

## Runtime borrow checks

Rust's compile-time borrow checker cannot see which components a query touches,
so Syren enforces the aliasing rules at runtime. A borrow tracker records the
read and write borrows a system holds and rejects a conflicting one — two writes
to the same component, or a read overlapping a write — with an error rather than
undefined behaviour. This is why systems declare access: the scheduler uses the
declaration to avoid ever placing conflicting systems in the same parallel stage,
and the runtime check is the backstop.

## Query shape and reductions

Some methods require a particular query shape. `reduce_read`, for instance,
expects exactly one read component and no writes, and returns a shape error if
the query does not match. These checks keep a mis-built query from reading the
wrong column.

[`QueryBuilder`]: https://docs.rs/syren/latest/syren/struct.QueryBuilder.html
[`BuiltQuery`]: https://docs.rs/syren/latest/syren/struct.BuiltQuery.html
[`ECSReference`]: https://docs.rs/syren/latest/syren/struct.ECSReference.html
[`Entity`]: https://docs.rs/syren/latest/syren/struct.Entity.html
