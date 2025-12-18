# Syren ABM Framework
Syren is a high-performance, data-oriented Agent-Based Modeling (ABM) framework built on a custom archetype-based Entity–Component System (ECS) in Rust.

Syren is designed for large-scale, parallel ABMs with millions of agents, offering explicit control over memory layout, synchronization, and execution. 
The framework is inspired by FLAME, and modern ECS designs.

## Key Features
- Structure-of-Arrays (SoA) storage
- Cache-friendly chunk iteration
- Explicit parallel execution using Rayon
- Explicit read/write access declarations
- Deferred structural mutations
- Non-overlapping write guarantees
- Phase-based scheduling

## Philosophy
Syren is built around three core principles:

1. Data-oriented design
Agent behavior emerges from data layout and execution order, not object hierarchies.

2. Explicitness over abstraction
All concurrency, mutation, and synchronization points are visible and controlled by the user.

3. Performance as a first-class concern
ABMs should scale to millions of agents without changing model semantics.

