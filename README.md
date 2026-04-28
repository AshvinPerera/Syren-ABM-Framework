# Syren ABM Framework

Syren is an experimental Rust framework for agent-based models. It combines a
custom archetype ECS, declared system access sets, optional model-level
composition, typed environments, typed per-tick messaging, and an optional
`wgpu` compute backend.

This project is still a hobby and learning project. It is useful for exploring
the internals of scalable ABM runtimes and for building early economic model
experiments, but it should be treated as evolving infrastructure rather than a
finished research platform.

## Status

- Crate name: `abm_framework`
- Rust edition: 2021
- MSRV: 1.76
- Library artifact: `rlib`
- API stability: not guaranteed yet
- Current focus: ECS storage, scheduler behavior, model composition,
  messaging, GPU dispatch, and correctness-oriented model tests

Determinism is provided by explicit scheduling and access declarations. Model
determinism still depends on the systems you write, especially RNG and floating
point behavior.

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

## Quickstart

Add the crate from this repository:

```toml
[dependencies]
abm_framework = { git = "https://github.com/AshvinPerera/ABM-Framework.git" }
```

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

For higher-level model tests, prefer `ModelBuilder` and `AgentTemplate` when the
feature set includes `model`.

## Current Examples And Tests

The repository uses integration tests as executable examples:

| Test | Features | What it covers |
| --- | --- | --- |
| `tests/closed_market_economy.rs` | `model messaging` | A small closed labor/goods market with household agents, firm agents, wage payments, goods orders, receipts, and accounting invariants |
| `tests/economy_messaging_gpu.rs` | `model messaging_gpu` | GPU household order emission, CPU market clearing, CPU receipt emission, and GPU receipt consumption |
| `tests/sugarscape_axtell.rs` | `model` or `gpu` | Epstein-Axtell Chapter 2 style Sugarscape fixtures with toroidal terrain, vision, metabolism, death/replacement, and CPU/GPU equality checks |

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

Criterion benchmarks live under `benches/`.

Representative split benchmark targets:

```bash
cargo bench --features all --bench query_matching
cargo bench --features all --bench scheduler_packing
cargo bench --features all --bench environment_dirty_tracking
cargo bench --features all --bench messaging_finalisation
cargo bench --features all --bench structural_mutation
cargo bench --features all --bench model_agent
cargo bench --features all --bench message_specialisations
cargo bench --features all --bench gpu_message_finalisation
```

## Profiling

Enable `profiling` to emit Chrome Trace JSON that can be inspected in
`chrome://tracing` or Perfetto.

```rust
abm_framework::init("profile/trace.json");
// run simulation work
abm_framework::shutdown();
```

## Current Limitations

- The public API is still changing.
- The GPU backend is optional and depends on local adapter/driver support.
- GPU and CPU paths are intended to be rule-equivalent where tests assert it,
  but arbitrary user systems must still manage deterministic ordering carefully.
- The example economy is deliberately small and closed. It is a correctness
  fixture, not a macroeconomic model with credit, bankruptcy, unemployment
  dynamics, or policy behavior.
- Sugarscape coverage targets the Epstein-Axtell Chapter 2 wealth-distribution
  mechanics used by the tests, not every later extension of the Sugarscape
  family.

## License

This project is licensed under the [MIT License](LICENSE).
