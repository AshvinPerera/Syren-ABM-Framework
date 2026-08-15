# Entities, components, and archetypes

## Components

A **component** is a plain Rust type — usually a small `Copy` struct — that holds
one facet of an agent's state. Components carry no behaviour; systems act on
them. A type becomes a component when it is registered with a
[`ComponentRegistry`], which assigns it a `ComponentID`.

The registry is filled up front and then **frozen**. Freezing fixes the set of
component types and their identifiers for the life of the world, which is what
lets storage layouts be decided once rather than renegotiated at runtime.

## Entities

An **entity** is an identifier, not an object. It is a compact handle carrying an
index and a version. The version distinguishes a live entity from a recycled
slot: when an entity is despawned its slot may be reused, and the version lets
stale handles be detected instead of silently pointing at a different agent.

## Archetypes and chunks

An **archetype** is the set of entities that have exactly the same set of
components. Storage is organised by archetype: each archetype keeps one columnar
array (an `Attribute<T>`) per component, split into fixed-size **chunks**.

This is a struct-of-arrays layout. Iterating one component over a large
population walks contiguous memory, which is cache-friendly and vectorises well.
Adding or removing a component moves an entity from one archetype to another —
a **migration** that copies its columns to the destination archetype.

## Shards

Entities are distributed across **shards** ([`EntityShards`]), typically one per
worker thread, so that spawns and despawns on different shards proceed without
contending on a single global structure.

## Where this lives

The registry, archetypes, chunked attributes, and shards are the core ECS. Most
of them are reached indirectly: you register components, build queries, and let
the framework place and iterate entities. The lower-level types are available
through the [`advanced`] module when you need them.

[`ComponentRegistry`]: https://docs.rs/syren/latest/syren/struct.ComponentRegistry.html
[`EntityShards`]: https://docs.rs/syren/latest/syren/advanced/struct.EntityShards.html
[`advanced`]: https://docs.rs/syren/latest/syren/advanced/index.html
