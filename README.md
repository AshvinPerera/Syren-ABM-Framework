# Syren ABM Framework
Syren is a high-performance, data-oriented Agent-Based Modeling (ABM) framework built on a custom archetype-based Entity–Component System (ECS) in Rust.
It is designed for large-scale agent-based economic models with millions of agents and employs archetype-based storage for a cache-efficient data layout and 
deterministic scheduling to ensure reproducible simulation runs. The library is written in Rust, leveraging safe, explicit data access to ensure memory safety.

The library targtes economists, social scientists, and researchers who need a scalable simulation toolkit for agent-based modeling (especially economic systems). 
The crate builds as both an rlib and a C-compatible cdylib for FFI use, making it flexible for integration into varied workflows.

Note that this is currently a hobby project of mine to both 1. learn the internals of a scalable agent based modeling framework, and 2. to develop my own 
economic agent based models.

## Installation
Add the crate to your project by including it in your Cargo.toml

``` rust
[dependencies]
abm_framework = { git = "https://github.com/AshvinPerera/ABM-Framework.git" }
```

The framework has no additional OS requirements beyond standard Rust support, and it uses Rayon internally for parallel execution.

## Usage
Basic usage involves defining your data components, registering them with the ECS, creating an `ECSManager`, spawning entities, and 
running systems in a schedule. The repository includes one example simulation at the momemt.

`toy_economy_test.rs` – a full example using abm_framework’s ECS to implement a simple toy economy with households and firms.

You can run these examples via `cargo test`. The following steps outline the ECS workflow, referencing the `toy_economy_test.rs` example

- Register Components: Define your component types and register each one before building the world.
``` rust
register_component::<Cash>();
register_component::<Hunger>();
register_component::<Inventory>();
// ... register other components ...
freeze_components();
```

- Initialize the ECS: Create the ECS manager with a chosen number of shards (worker threads).
``` rust
let ecs = ECSManager::new(EntityShards::new(4));
```

- Spawn Entities: Build component bundles and defer spawn commands to add entities.
``` rust
let mut bundle = Bundle::new();
bundle.insert(component_id_of::<Cash>(), Cash(100.0));
bundle.insert(component_id_of::<Inventory>(), Inventory(50.0));
// ... insert other components ...
data.defer(Command::Spawn { bundle });
```

- Define Systems and Schedule: Implement your logic as structs that implement the `System` trait. Collect them into a stage schedule.
``` rust
let systems: Vec<Box<dyn System>> = vec![
    Box::new(ProductionSystem),
    Box::new(WagePaymentSystem),
    // ...
];
let stages = make_stages(systems)
```

- Run the Simulation Loop: Call `run_schedule(&mut ecs, &stages)` in each step to execute all systems in parallel.
``` rust
for _step in 0..1000 {
    run_schedule(&mut ecs, &stages);
}
```

The included examples demonstrate these steps end-to-end.

## Key Features
- Structure-of-Arrays (SoA) storage
- Cache-friendly chunk iteration
- Parallel execution using Rayon
- Read/write access declarations
- Deferred structural mutations
- Non-overlapping write guarantees
- Phase-based scheduling

## In Development
- Component validation using a component registry

## Future Development
- GPU integration
- Scripting language for quick simulation design
- Message passing
- Social networks 

## License
This project is licensed under the [MIT License](LICENSE)
