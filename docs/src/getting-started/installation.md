# Installation

Syren requires a Rust toolchain that meets the library's minimum supported
version (MSRV), **Rust 1.87**. Newer toolchains work; the development toolchain
is pinned in `rust-toolchain.toml`.

## Add the dependency

Add Syren to your `Cargo.toml`. It has no default features, so enable the ones
your model needs:

```toml
[dependencies]
syren = { version = "0.6.0-rc.1", features = ["model", "messaging"] }
```

## Feature selection

Features are additive. Enable only what you use; each unlocks a module and its
dependencies.

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

Most models want `model`. Add `messaging` if agents exchange messages, and
`gpu` (or `messaging_gpu`) for GPU execution.

See the [feature matrix](../reference/features.md) for the supported
combinations.

## Verify the toolchain

```bash
cargo build
```

If you use GPU features, note that building the `gpu` feature only requires the
wgpu crate to compile; **running** a GPU system additionally needs a working
graphics adapter. See [CPU and GPU state](../concepts/cpu-gpu.md).
