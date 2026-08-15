# The Syren Guide

Syren is a parallel Rust framework for agent-based models. It stores agents in
an archetype entity-component-system (ECS), runs systems over them through a
deterministic stage scheduler on top of [Rayon], and adds agent, environment,
messaging, and optional GPU layers behind Cargo features.

This guide is the user and contributor manual. The Rust API reference lives on
[docs.rs](https://docs.rs/syren); this guide explains how the pieces fit
together and how to use them.

## Who this is for

Syren is for researchers and engineers who write agent-based models in Rust and
care about reproducibility and scale. You should be comfortable with Rust and
Cargo. There is no separate model-definition language: a model is a Rust program
that uses the library.

## What the framework provides

- **Archetype-ECS storage** — components stored in chunked, columnar arrays,
  iterated over contiguous memory.
- **Deterministic scheduling** — systems declare their data access; the
  scheduler packs non-conflicting systems into parallel stages and preserves a
  reproducible activation order.
- **Query-derived access** — a system's read/write set is derived from the
  queries it runs, so the declared access cannot drift from what it touches.
- **Deterministic randomness** — `DetRng` keys draws on the run context and a
  salt, so results do not depend on which worker thread visits which rows.
- **Model layer** (`model` feature) — `ModelBuilder`, agent templates,
  environments, sub-schedulers, and nested models.
- **Messaging** (`messaging` feature) — four message specialisations
  (brute-force, bucketed, spatial, targeted).
- **Optional GPU execution** (`gpu` feature) — mirror component columns to the
  GPU and dispatch compute systems through [wgpu].

## How to read this guide

- **Getting started** takes you from an empty project to a running model.
- **Core concepts** explains each part of the framework and how it relates to
  the source.
- **How-to** gives task-focused recipes.
- **Reproducibility** covers the reproducibility guarantees, verification,
  provenance, and citation.
- **Reference** collects the feature matrix, compatibility policy, error model,
  performance methodology, and a glossary.
- **Contributing** documents the engine internals and the development process.

[Rayon]: https://docs.rs/rayon
[wgpu]: https://docs.rs/wgpu
