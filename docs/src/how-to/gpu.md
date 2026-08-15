# Add a GPU component and system

_Requires the `gpu` feature._

A GPU system runs as a compute shader over mirrored component columns. Getting a
component onto the GPU has two parts: making the component GPU-safe, and writing
a system that declares the GPU backend.

## Make the component GPU-safe

A GPU component must be plain-old-data with a fixed layout. Implement [`GPUPod`]
for it and register it with [`register_gpu_component`] instead of the plain
registration:

```rust,ignore
let velocity_id = register_gpu_component::<Velocity>()?;
```

`register_gpu_component` records that the type can be mirrored to a GPU buffer.
Keep GPU components small and `#[repr(C)]`-compatible; the mirror copies their
bytes directly.

## Write a GPU system

A GPU system implements the [`GpuSystem`] trait: it names the resources it uses,
supplies the compute shader, and provides the dispatch parameters (workgroup
sizing). The scheduler treats it like any other system for ordering — it takes
part in stages and channels — but dispatches it to the device.

Add it to the model the same way as a CPU system. The framework mirrors the
component columns the system needs, dispatches the shader over the archetypes,
and reads results back when the CPU next reads those columns.

## GPU messaging

With `messaging_gpu`, message buffers can live on the GPU, so a GPU system both
produces and consumes messages without copying to the CPU between stages.
Register GPU message resources through the builder's GPU message methods.

## Fallback and testing

Building the `gpu` feature only needs wgpu to compile; running a GPU system needs
a real adapter. Where none is present, GPU execution is unavailable — provide a
CPU path if the model must run on such machines. The GPU execution tests report a
skip when no adapter is found, and the manual `GPU tests` workflow runs them on
hardware. See [CPU and GPU state](../concepts/cpu-gpu.md).

For a model that uses a GPU system with a CPU fallback, see the metabolism system
in the Sugarscape example.

[`GPUPod`]: https://docs.rs/syren/latest/syren/struct.GPUPod.html
[`register_gpu_component`]: https://docs.rs/syren/latest/syren/fn.register_gpu_component.html
[`GpuSystem`]: https://docs.rs/syren/latest/syren/struct.GpuSystem.html
