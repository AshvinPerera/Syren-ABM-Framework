# Syren ABM Framework

Syren is a Rust framework for agent-based models. It combines a custom
archetype ECS, declared system access sets, optional model-level composition,
typed environments, typed per-tick messaging, and an optional `wgpu` compute
backend.

The crate is maintained with a documented public API, feature-gated subsystems,
and correctness-oriented integration tests.

## Status

- Crate name: `abm_framework`
- Rust edition: 2021
- MSRV: 1.87
- Library artifact: `rlib`
- Current focus: ECS storage, scheduler behavior, model composition,
  messaging, GPU dispatch, documentation, and release hardening

Determinism is provided by explicit scheduling and access declarations. Model
determinism still depends on the systems you write, especially RNG and floating
point behavior.

## Architecture Overview

Syren's core is an archetype-based Entity–Component System (ECS). Entities that
share the same set of component types are grouped into archetypes, where each
component is stored in a dense, columnar array. This layout maximises cache
locality during iteration.

Simulation logic is expressed as systems. Each system declares an `AccessSets`
value that lists the components it reads and writes, the channels it produces
and consumes, and any GPU resources it touches. The scheduler compiles these
declarations into a deterministic execution plan of stages. Systems whose
access sets do not conflict run in parallel within a stage (via `rayon`);
boundary stages sit between parallel stages and handle deferred command
application, channel finalisation, and GPU synchronisation.

Channels are the ordering primitive. A system that *produces* a channel must
complete before any system that *consumes* it. Boundary resources — objects that
implement the `BoundaryResource` trait — hook into the per-tick lifecycle
(`begin_tick` → `finalise` → `end_tick`) and own one or more channels. The
environment store and the message buffer set are both boundary resources.

The optional higher-level layers build on top of the raw ECS. The `agents`
feature provides `AgentTemplate`, `AgentSpawner`, `AgentBatch`, and lifecycle
hooks. The `environment` feature adds a typed key-value parameter store with
dirty-channel tracking. The `messaging` feature provides typed per-tick message
buffers in four specialisations: brute-force, bucket, spatial, and targeted.
The `model` feature ties these together behind `ModelBuilder`, which wires up
component registration, agent populations, environments, messaging, and
scheduling in a single fluent API.

For performance-critical paths the engine includes a thread-local xorshift64*
RNG. Each thread owns an independent, lock-free state seeded with a fixed
constant, giving deterministic output per thread with zero synchronisation
overhead.

## Feature Flags

| Flag | Enables |
| --- | --- |
| `agents` | Agent templates, spawners, lifecycle hooks, and agent handles |
| `environment` | Typed model environment values and dirty-channel boundary tracking |
| `messaging` | Typed brute-force, bucket, spatial, and targeted per-tick messages |
| `gpu` | `wgpu` compute systems, GPU-safe components, and CPU/GPU mirroring |
| `messaging_gpu` | GPU-backed message resources and GPU finalisation; includes `messaging` and `gpu` |
| `model` | High-level `ModelBuilder`, agent registry integration, environments, sub-schedulers, and nested models; includes `agents` and `environment` |
| `profiling` | Chrome Trace output for scheduler and ECS spans |
| `gpu_profiling` | `gpu` plus `profiling` |
| `all` | Every optional subsystem |

## Dependencies

The framework's runtime dependencies are deliberately small. `rayon` drives
parallel system execution. `parking_lot` (with `arc_lock`) provides fast
reader-writer locking for borrow tracking and boundary access. `smallvec`
keeps small component sets inline. `thiserror` structures error types. The
optional `gpu` feature pulls in `wgpu` 29.x for the compute backend,
`bytemuck` for safe Pod casts, and `pollster` for blocking on GPU futures.
`criterion` is used as a dev-dependency for benchmarks.

## Quickstart

Add the crate from this repository:

```toml
[dependencies]
abm_framework = { git = "https://github.com/AshvinPerera/ABM-Framework.git" }
```

### Low-level ECS

At the ECS level, simulations register component types, spawn bundles through
deferred commands, and run systems through a scheduler:

```rust
use std::sync::{Arc, RwLock};
use abm_framework::{
    advanced::EntityShards, AccessSets, Bundle, Command, ComponentRegistry,
    ECSManager, ECSResult, FnSystem, Read, Scheduler,
};

#[derive(Clone, Copy)]
struct Wealth {
    value: f64,
}

fn main() -> ECSResult<()> {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let wealth_id = registry.write().unwrap().register::<Wealth>().unwrap();
    registry.write().unwrap().freeze();

    let ecs = ECSManager::with_registry(EntityShards::new(1)?, registry);
    let world = ecs.world_ref();

    let mut bundle = Bundle::new();
    bundle.insert(wealth_id, Wealth { value: 10.0 });
    world.defer(Command::Spawn { bundle })?;
    ecs.apply_deferred_commands()?;

    let mut access = AccessSets::default();
    access.read.set(wealth_id);
    let mut scheduler = Scheduler::new();
    scheduler.add_system(FnSystem::new(0, "observe_wealth", access, move |ecs| {
        let q = ecs.query()?.read::<Wealth>()?.build()?;
        ecs.for_each::<(Read<Wealth>,)>(q, &|wealth| {
            let _ = wealth.0.value;
        })
    }));

    ecs.run(&mut scheduler)
}
```

### Model-level API

When the `model` feature is enabled, prefer `ModelBuilder` and `AgentTemplate`
over the raw ECS API. `ModelBuilder` handles component registration, agent
templates, environments, messaging, and scheduling in one fluent chain.

For large initial populations, use the bulk path (`with_agent_population`)
instead of calling `AgentSpawner` once per entity. For finer-grained control
over multi-component batches, see `AgentBatch`.

```rust
# #[cfg(feature = "model")]
# fn bulk_population_example() -> Result<(), Box<dyn std::error::Error>> {
# use std::sync::{Arc, RwLock};
# use abm_framework::advanced::EntityShards;
# use abm_framework::agents::AgentTemplate;
# use abm_framework::model::ModelBuilder;
# use abm_framework::ComponentRegistry;
# #[derive(Clone, Copy, Default)]
# struct Household { cash: u32 }
let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
let household_id = registry.write().unwrap().register::<Household>()?;
registry.write().unwrap().freeze();

let households: Vec<Household> = (0..1_000_000)
    .map(|_| Household { cash: 100 })
    .collect();

let model = ModelBuilder::new()
    .with_component_registry(registry)
    .with_shards(EntityShards::new(4)?)
    .with_agent_template(
        AgentTemplate::builder("household")
            .with_component::<Household>(household_id)?
            .with_capacity(households.len())
            .build(),
    )?
    .with_agent_population("household", household_id, households)?
    .build()?;
# let _ = model;
# Ok(())
# }
```

## The `advanced` Module

The `advanced` module re-exports storage and scheduling internals that are
deliberately kept out of the root API: `Archetype`, `ChunkBorrow`,
`EntityShards`, `ECSData`, `ChannelAllocator`, and raw typed-slice casts
(`cast_slice`, `cast_slice_mut`, `Attribute`, `TypeErasedAttribute`).

`EntityShards` controls how entity IDs are partitioned across allocation
shards and is required by both the low-level `ECSManager::with_registry` path
and `ModelBuilder::with_shards`. The remaining types expose archetype storage
internals; callers using them directly must uphold the ECS storage invariants
that the public API normally enforces automatically.

## Current Examples And Tests

The repository uses integration tests as executable examples:

| Test | Features | What it covers |
| --- | --- | --- |
| `tests/closed_market_economy.rs` | `model messaging` | A small closed labor/goods market with household agents, firm agents, wage payments, goods orders, receipts, and accounting invariants |
| `tests/economy_messaging_gpu.rs` | `model messaging_gpu` | GPU household order emission, CPU market clearing, CPU receipt emission, and GPU receipt consumption |
| `tests/sugarscape_axtell.rs` | `model` or `gpu` | Epstein-Axtell Chapter 2 style Sugarscape fixtures with toroidal terrain, vision, metabolism, death/replacement, and CPU/GPU equality checks |
| `tests/scheduler_graph.rs` | (none) | Scheduler dependency-graph construction, stage packing, activation ordering, and boundary channel routing |
| `tests/gpu_dispatch_params.rs` | `gpu` | GPU dispatch parameter passing, write-back, and multi-archetype filtering |
| `tests/engine_boundary_lifecycle.rs` | (none) | Boundary resource per-slot locking, channel routing, registration collisions, and fallible parallel iteration |
| `tests/mem_layout.rs` | (none) | Memory layout and alignment assertions for archetype columnar storage, chunk capacity, and typed slice casts |

Useful targeted commands:

```bash
cargo test --features "model messaging" --test closed_market_economy
cargo test --features "model messaging_gpu" --test economy_messaging_gpu
cargo test --features model --test sugarscape_axtell
cargo test --features gpu --test sugarscape_axtell
```

Broader validation commands:

```bash
cargo test --no-default-features
cargo test --features all --lib
cargo test --features all --tests
cargo bench --no-run --features all
```

## Benchmarks

Criterion benchmarks live under `benches/`. The full set of targets:

| Bench | Required features | Area |
| --- | --- | --- |
| `spawn` | — | Core ECS entity spawning |
| `iterate` | — | Archetype iteration throughput |
| `tick` | — | Full scheduler tick cost |
| `reduce` | — | Parallel reduction primitives |
| `query_matching` | — | Query/signature matching |
| `scheduler_packing` | — | Stage packing and plan compilation |
| `scheduler_execution` | — | End-to-end scheduler execution |
| `structural_mutation` | — | Add/remove component structural changes |
| `environment_dirty_tracking` | `environment` | Environment dirty-channel overhead |
| `messaging_finalisation` | `messaging` | Message buffer finalisation |
| `message_specialisations` | `messaging` | Brute-force, bucket, spatial, targeted dispatch |
| `model_agent` | `model` | Agent template and spawner throughput |
| `gpu_startup` | `gpu` | GPU device and pipeline initialisation |
| `gpu_dispatch` | `gpu` | GPU compute dispatch |
| `gpu_tick` | `gpu` | GPU-inclusive tick cost |
| `gpu_dispatch_poll` | `gpu` | GPU dispatch with poll-based completion |
| `gpu_readback` | `gpu` | GPU buffer readback latency |
| `gpu_bind_group_creation` | `gpu` | Bind group creation overhead |
| `gpu_message_finalisation` | `model messaging_gpu` | GPU message buffer finalisation |

Run all benchmarks with:

```bash
cargo bench --features all
```

## Profiling

Enable `profiling` to emit Chrome Trace Event Format JSON that can be
inspected in `chrome://tracing` or Perfetto. The trace captures per-system
execution spans, boundary finalisation, deferred command application, and
(with `gpu_profiling`) GPU dispatch timing.

```rust
abm_framework::init("profile/trace.json");
// run simulation work
abm_framework::shutdown();
```

## Project Layout

```
src/
  lib.rs                — public API re-exports and prelude
  engine/               — core ECS: archetypes, entities, components, scheduler,
                          queries, commands, boundaries, borrow tracking, workers
    random.rs           — thread-local xorshift64* RNG
  agents/               — agent templates, spawners, batch spawning, handles, registry
  environment/          — typed key-value parameter store with dirty-channel tracking
  messaging/            — per-tick typed message buffers (brute-force, bucket,
                          spatial, targeted) and optional GPU message resources
  model/                — ModelBuilder, Model, nested models, sub-schedulers
  gpu/                  — wgpu compute pipeline, GPU-safe components, CPU/GPU mirroring
  profiling/            — Chrome Trace output
tests/                  — integration tests (executable examples)
benches/                — Criterion benchmarks
```

## License

This project is licensed under the [MIT License](LICENSE).
