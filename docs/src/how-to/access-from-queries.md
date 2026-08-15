# Derive access sets from queries

A system must declare the components it reads and writes. The robust way is to
let the declaration follow the queries the system runs, so the two cannot drift
apart.

## Use `from_queries`

Build the query, then pass it both to `FnSystem::from_queries` (to derive the
access) and into the closure (to iterate):

```rust,ignore
let step_query = QueryBuilder::with_registry(registry)
    .write::<Position>()?
    .build()?;
let run_query = step_query.clone();

let system = FnSystem::from_queries(
    0,
    "random_walk",
    &[&step_query],
    move |ecs| {
        ecs.for_each_entity_w1::<Position>(run_query.clone(), |entity, pos| {
            // mutate pos
        })
    },
);
```

Change the query — say, add a `read::<Velocity>()` — and the access set follows
automatically. There is no separate hand-written access list to keep in sync.

## Multiple queries

If a system runs more than one query, list them all so the derived access covers
everything it touches:

```rust,ignore
FnSystem::from_queries(id, name, &[&query_a, &query_b], move |ecs| { /* ... */ })
```

## When to write access by hand

Occasionally a system's access is not captured by the queries it runs — for
example, it reads a boundary in a way the query shape does not express. In that
case construct an `AccessSets` directly. Prefer the derived path wherever a query
captures the access; the manual path is the exception, and it is the one place
where the declaration can drift from reality.

## Why it matters

The scheduler uses the declared access to place non-conflicting systems in the
same parallel stage. If a declaration understates what a system touches, the
runtime borrow check catches the conflict and returns an error rather than
risking a data race — but an accurate, query-derived declaration lets the
scheduler parallelise correctly in the first place.
