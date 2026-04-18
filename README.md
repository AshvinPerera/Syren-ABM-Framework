# Syren ABM Framework

Syren is a high-performance, data-oriented Agent-Based Modeling (ABM) framework built on a custom archetype-based Entity–Component System (ECS) in Rust. It is designed for large-scale agent-based economic models with millions of agents, employing archetype-based Structure-of-Arrays storage for cache-efficient data layout, deterministic phase-based scheduling for reproducible simulation runs, and an optional GPU compute backend via `wgpu`.

The library targets economists, social scientists, and researchers who need a scalable simulation toolkit for agent-based modeling (especially economic systems). The crate builds as an `rlib`.

**Note:**
- This is currently a hobby project to both 1. learn the internals of a scalable agent-based modeling framework, and 2. develop my own economic agent-based models.
- Determinism is guaranteed at the scheduling and data-access level; numerical determinism depends on system logic and RNG usage.

---

## Installation

Syren targets Rust 2021 (MSRV 1.76) and later. Add the crate to your project via your `Cargo.toml`:

```toml
[dependencies]
abm_framework = { git = "https://github.com/AshvinPerera/ABM-Framework.git" }
```

The framework has no additional OS requirements beyond standard Rust support. It uses Rayon internally for parallel CPU execution.

### Feature Flags

| Flag | Description |
|---|---|
| `gpu` | Enables the GPU compute backend via `wgpu` (Vulkan, Metal, DX12, WebGPU) |
| `profiling` | Enables the Chrome Trace flame-graph profiler (zero overhead when disabled) |
| `gpu_profiling` | Enables both `gpu` and `profiling` together |

---

## Usage

Basic usage involves creating a `ComponentRegistry`, registering your component types, building an `ECSManager`, spawning entities, and running systems through a `Scheduler`. The repository includes two example simulations.

**`toy_economy_test.rs`** – a simple toy economy with households and firms demonstrating core ECS workflow.

**`sugarscape_basic.rs`** – a full Sugarscape model with a 600×600 grid, supporting both a CPU path and an optional GPU path (enabled with `--features gpu`).

Both examples are implemented as integration tests and can be run via `cargo test`.

### Example Workflow

The following steps outline the ECS workflow.

**1. Create a Component Registry.** Define your component types and register each one before building the world. The registry must be frozen before use.

```rust
let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
{
    let mut reg = registry.write().unwrap();
    reg.register::<Position>().unwrap();
    reg.register::<Wealth>().unwrap();
    // ... register other components ...
    reg.freeze();
}
```

**2. Initialize the ECS.** Create the ECS manager with a chosen number of shards (worker threads) and the shared registry.

```rust
let shards = EntityShards::new(4)?;
let ecs = ECSManager::with_registry(shards, registry);
```

**3. Spawn Entities.** Build component bundles and defer spawn commands to add entities.

```rust
let world = ecs.world_ref();
world.with_exclusive(|_| {
    let mut bundle = Bundle::new();
    bundle.insert(pos_id, Position { x: 0.0, y: 0.0 });
    bundle.insert(wealth_id, Wealth { value: 100.0 });
    world.defer(Command::Spawn { bundle })?;
    Ok(())
})?;
ecs.apply_deferred_commands()?;
```

**4. Define Systems and a Scheduler.** Implement your logic as structs implementing the `System` trait, then add them to a `Scheduler`. The scheduler automatically groups systems into parallel stages respecting declared read/write access sets.

```rust
let mut scheduler = Scheduler::new();
scheduler.add_system(ProductionSystem::new(&reg));
scheduler.add_system(WagePaymentSystem::new(&reg));
```

**5. Run the Simulation Loop.** Call `ecs.run(&mut scheduler)` each tick to execute all systems.

```rust
for _step in 0..1000 {
    ecs.run(&mut scheduler)?;
}
```

### GPU Systems

Systems can be executed on the GPU by implementing both `System` and the `GpuSystem` trait. A GPU system provides a WGSL compute shader, an entry point, and a workgroup size. The scheduler automatically dispatches GPU systems to the GPU backend when the `gpu` feature is enabled.

```rust
impl GpuSystem for MetabolismSystem {
    fn shader(&self) -> &'static str { include_str!("shaders/metabolism.wgsl") }
    fn entry_point(&self) -> &'static str { "main" }
    fn workgroup_size(&self) -> u32 { 256 }
}
```

GPU execution follows a four-stage pipeline: upload component columns to GPU buffers, dispatch compute shaders per archetype, synchronize, then download mutated data back to ECS storage. All GPU execution occurs inside an exclusive ECS phase, preventing concurrent CPU iteration or structural mutation.

### Profiling

Enable the `profiling` feature to emit a Chrome Trace JSON file inspectable in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev). The profiler is zero-overhead when the feature is disabled.

```rust
abm_framework::init("profile/trace.json");
// ... run simulation ...
abm_framework::shutdown();
```

---

## Key Features

- Archetype-based Structure-of-Arrays (SoA) storage
- Cache-friendly chunk iteration
- Parallel CPU execution via Rayon
- Phase-based scheduling with automatic system grouping
- Explicit read/write access declarations enforced at runtime
- Non-overlapping write guarantees
- Deferred structural mutations (spawn, despawn, component moves)
- Optional GPU compute backend via `wgpu` (Vulkan, Metal, DX12, WebGPU)
- Optional Chrome Trace profiler (zero overhead when disabled)
- Per-archetype GPU dispatch with explicit CPU↔GPU data mirroring
- GPU resource registry for shared grid/buffer data across GPU systems
- Instance-owned component registries (multiple independent ECS worlds supported)

---

## Example Simulations

| Simulation | File | Description |
|---|---|---|
| Toy Economy | `tests/toy_economy_test.rs` | Households and firms; demonstrates core ECS API |
| Sugarscape | `tests/sugarscape_basic.rs` | 600×600 grid; CPU and GPU execution paths |

Run with:

```bash
# CPU only
cargo test --test sugarscape_basic -- --nocapture

# With GPU backend
cargo test --features gpu --test sugarscape_basic -- --nocapture

# With profiling
cargo test --features profiling --test sugarscape_basic -- --nocapture

# GPU + profiling
cargo test --features gpu_profiling --test sugarscape_basic -- --nocapture
```

---

## Future Development

- Scripting language for quick simulation design
- Message passing
- Social networks

---

## License

This project is licensed under the [MIT License](LICENSE)