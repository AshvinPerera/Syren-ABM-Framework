# Feature matrix

Syren has no default features. Enable the ones your model needs. Features are
additive: enabling one never removes capability, and combinations compose.

## Features

| Feature | Enables | Implies |
| --- | --- | --- |
| _(none)_ | Core ECS: components, queries, systems, scheduler. | — |
| `agents` | Agent templates and lifecycle hooks. | — |
| `environment` | Typed model-wide environment values. | — |
| `messaging` | The four message specialisations. | — |
| `model` | The `ModelBuilder` layer. | `agents`, `environment` |
| `gpu` | GPU state mirroring and compute dispatch (wgpu). | — |
| `messaging_gpu` | GPU-resident message buffers. | `messaging`, `gpu` |
| `profiling` | Tracing spans and Chrome Trace output. | — |
| `gpu_profiling` | Convenience aggregate. | `gpu`, `profiling` |
| `all` | Everything above. | all |

The `space` module (discrete grid and continuous space) is available with the
`environment` feature, and therefore with `model`.

## Combinations checked in CI

The continuous-integration pipeline checks each of these explicitly:

- no features
- `agents`
- `environment`
- `messaging`
- `model`
- `model messaging`
- `gpu`
- `model messaging_gpu`
- `all`

Test runs cover the no-features build, `model messaging`, and the all-features
library. Integration tests are compiled with all features.

## Choosing features

- A model that uses `ModelBuilder`, agents, and environments: `model`.
- Add message passing: `model messaging`.
- Run systems on the GPU: add `gpu` (or `messaging_gpu` for GPU messages).
- Capture profiles: add `profiling`.
