# CPU and GPU state

_Requires the `gpu` feature._

Syren can run selected systems on the GPU while the rest of the model runs on the
CPU. The model stays the source of truth; the GPU holds mirrored copies of the
component columns a GPU system needs.

## GPU components

A component that a GPU system reads or writes must be **GPU-safe**: a
fixed-layout, plain-old-data type. Such a type implements [`GPUPod`] and is
registered with [`register_gpu_component`] instead of the plain registration, so
the framework knows it can be mirrored to a GPU buffer.

## Mirroring and dispatch

For a GPU system, the framework **mirrors** the relevant component columns into
GPU buffers, dispatches the compute shader over the archetypes, and reads results
back when the CPU next needs them. Mirror buffers, parameter buffers, and bind
groups are cached and invalidated by generation, so steady-state ticks reuse GPU
resources rather than rebuilding them each tick.

## GPU systems

A GPU system implements the [`GpuSystem`] trait: it names the resources it uses
and provides the shader and dispatch parameters. The scheduler treats it like any
other system for ordering — it participates in stages and channels — but
dispatches it to the device rather than running a CPU closure.

## Visibility points

The CPU and GPU views of a column are synchronised at defined points: the
scheduler boundary, and explicit sync or readback calls. Between those points a
column may be resident on the GPU. Keeping visibility at boundaries is what lets
GPU dispatch avoid a blocking poll on the hot path.

## Building versus running

Enabling the `gpu` feature only requires the wgpu crate to **compile**. Actually
**running** a GPU system needs a working graphics adapter. Where no adapter is
available, GPU execution is unavailable; the test suite reports a skip rather
than failing. Plan for a CPU fallback path if your model must run on machines
without a GPU.

See [add a GPU component and system](../how-to/gpu.md).

[`GPUPod`]: https://docs.rs/syren/latest/syren/struct.GPUPod.html
[`register_gpu_component`]: https://docs.rs/syren/latest/syren/fn.register_gpu_component.html
[`GpuSystem`]: https://docs.rs/syren/latest/syren/struct.GpuSystem.html
