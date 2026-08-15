# Installation

Syren's minimum supported Rust version (MSRV) is **Rust 1.87**. Newer toolchains
also work. The development toolchain is pinned in `rust-toolchain.toml`.

## Add the dependency

Syren has no default features. Enable the features your model uses:

```toml
[dependencies]
syren = { version = "0.6.0", features = ["model", "messaging"] }
```

## Feature selection

Features are additive. Each enables a module and its dependencies.

| Feature | Enables |
| --- | --- |
| _(none)_ | The core ECS: components, queries, systems, and the scheduler. |
| `agents` | Agent templates and lifecycle hooks. |
| `environment` | Typed, model-wide environment values. |
| `messaging` | The four message specialisations. |
| `model` | The `ModelBuilder` layer. Implies `agents` and `environment`. |
| `gpu` | GPU state mirroring and compute dispatch through wgpu. |
| `messaging_gpu` | GPU-resident message buffers. Implies `messaging` and `gpu`. |
| `profiling` | Tracing spans and Chrome Trace output. |
| `all` | Everything above. |

A model built with `ModelBuilder` requires `model`. Add `messaging` for message
passing, and `gpu` or `messaging_gpu` for GPU execution.

The [feature matrix](../reference/features.md) lists the supported combinations.

## Verify the toolchain

```bash
cargo build
```

The `gpu` feature requires only that the wgpu crate compiles; running a GPU
system additionally requires a working graphics adapter. See [CPU and GPU
state](../concepts/cpu-gpu.md).
