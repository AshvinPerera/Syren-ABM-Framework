# Syren

Syren is a parallel Rust framework for agent-based models. It stores agents in an
archetype entity-component-system (ECS), runs systems over them through a
deterministic stage scheduler on top of [Rayon], and adds agent, environment,
messaging, and optional GPU layers behind Cargo features.

It is for researchers and engineers who write agent-based models in Rust and care
about reproducibility and scale. A model is a Rust program that uses the library;
there is no separate model-definition language.

## Status

| | |
| --- | --- |
| Package | `syren` |
| Version | `0.6.0-rc.1` |
| MSRV | Rust 1.87 |
| License | [MIT](LICENSE) |
| Guide | <https://ashvinperera.github.io/Syren-ABM-Framework/> |
| API reference | <https://docs.rs/syren> |

Syren is pre-1.0; see the [compatibility policy](docs/src/reference/compatibility.md).

> **Note:** the guide and API-reference links above are the destinations they
> will have once published. The guide goes live on GitHub Pages when this work
> merges to `master` and Pages is enabled; `docs.rs` and `crates.io` resolve
> once the crate is published. Until then, read the guide sources under
> [`docs/src`](docs/src).

## Capabilities

- **Archetype-ECS storage** — components in chunked, columnar arrays for
  cache-friendly iteration over large populations.
- **Deterministic scheduling** — systems declare their data access; the scheduler
  packs non-conflicting systems into parallel stages and keeps a reproducible
  activation order.
- **Query-derived access** — a system's read/write set is derived from the queries
  it runs, so the declaration cannot drift from what it touches.
- **Deterministic randomness** — `DetRng` keys draws on the run context and a salt,
  so results do not depend on which worker thread visits which rows.
- **Model layer** (`model`) — `ModelBuilder`, agent templates, environments,
  sub-schedulers, and nested models.
- **Messaging** (`messaging`) — brute-force, bucketed, spatial, and targeted
  message specialisations.
- **Optional GPU execution** (`gpu`) — mirror component columns to the GPU and
  dispatch compute systems through [wgpu].

## Installation

```toml
[dependencies]
syren = { version = "0.6.0-rc.1", features = ["model"] }
```

Syren has no default features; enable the ones your model needs. See the [feature
matrix](https://ashvinperera.github.io/Syren-ABM-Framework/reference/features.html).

## A first taste

A component is a plain `Copy` struct; a model is assembled with `ModelBuilder` and
advanced with `tick`:

```rust,ignore
#[derive(Clone, Copy, Default)]
struct Position {
    x: i64,
}

let mut model = ModelBuilder::new()
    .with_seed(42)
    .with_component_registry(Arc::clone(&registry))
    .with_shards(EntityShards::new(1)?)
    .with_agent_template(
        AgentTemplate::builder("walker")
            .with_component::<Position>(position_id)?
            .with_capacity(walkers.len())
            .build(),
    )?
    .with_agent_population("walker", position_id, walkers)?
    .with_system(system)
    .build()?;

model.run(50)?;
```

The full, compiled version is the `first_model` example, which the guide walks
through step by step.

## Examples

- [`first_model`](examples/first_model.rs) — the smallest complete model
  (`cargo run --example first_model --features model`).
- [Sugarscape](examples/sugarscape/) — a large grid-based model with an optional
  GPU path.
- [Macroeconomy](examples/macroeconomy/) — a fully documented, calibrated
  macroeconomic model.

## Documentation

- **Guide** — <https://ashvinperera.github.io/Syren-ABM-Framework/> — installation,
  concepts, how-to recipes, the science of reproducibility, and contributor docs.
- **API reference** — <https://docs.rs/syren>.

## Contributing and citation

- [Contributing guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).
- [Security policy](SECURITY.md).
- If you use Syren in academic work, please cite it — see
  [`CITATION.cff`](CITATION.cff).

## License

Licensed under the [MIT License](LICENSE).

[Rayon]: https://docs.rs/rayon
[wgpu]: https://docs.rs/wgpu
