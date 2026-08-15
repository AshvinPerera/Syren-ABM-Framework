# Entities, components, and archetypes

## Components

A **component** is a plain Rust type, usually a small `Copy` struct, that holds
one part of an agent's state. Components carry no behaviour; systems act on them.
A type becomes a component when it is registered with a [`ComponentRegistry`],
which assigns it a `ComponentID`.

The registry is filled up front and then **frozen**. Freezing fixes the set of
component types and their identifiers for the life of the world, so storage
layouts are decided once.

## Entities

An **entity** is a compact handle carrying an index and a version. The version
distinguishes a live entity from a recycled slot: when an entity is despawned its
slot may be reused, and the version marks earlier handles for that slot as stale.

## Archetypes and chunks

An **archetype** is the set of entities that have exactly the same components.
Storage is organised by archetype: each archetype keeps one columnar array (an
`Attribute<T>`) per component, split into fixed-size **chunks**.

This is a struct-of-arrays layout, so iterating one component over a large
population walks contiguous memory. Adding or removing a component **migrates**
the entity to a different archetype, copying its columns to the destination.

## Shards

Entities are distributed across **shards** ([`EntityShards`]), typically one per
worker thread, so spawns and despawns on different shards do not contend on a
single structure.

## Where this lives

The registry, archetypes, chunked attributes, and shards form the core ECS. A
model reaches them indirectly, through component registration, queries, and
iteration. The lower-level types are exposed in the [`advanced`] module.

[`ComponentRegistry`]: https://docs.rs/syren/latest/syren/struct.ComponentRegistry.html
[`EntityShards`]: https://docs.rs/syren/latest/syren/advanced/struct.EntityShards.html
[`advanced`]: https://docs.rs/syren/latest/syren/advanced/index.html
