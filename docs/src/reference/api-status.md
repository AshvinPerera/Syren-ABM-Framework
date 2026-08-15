# API status

Not every public item carries the same stability. Syren groups its surface into
tiers so you know how much notice a change will get.

## Stable public API

The types and functions re-exported from the crate root and from the `model`,
`agents`, `environment`, `messaging`, and `space` modules are the intended public
API. These follow the [compatibility policy](compatibility.md): stable across
patch releases, changed only with migration notes across pre-1.0 minor releases.

This is the surface the guide teaches and the one the rustdoc reference documents
as primary.

## The `advanced` module

The [`advanced`] module exposes lower-level building blocks — entity shards,
archetypes, chunk borrows, type-erased attributes, and the worker staging
primitive. They exist for models and extensions that need to reach beneath the
high-level API. They are more likely to change than the stable surface and are
documented as such. Reach for them deliberately.

## GPU API

The GPU types (`GpuSystem`, `GPUPod`, `register_gpu_component`, and the GPU
messaging surface) are functional but lower-level, and they depend on wgpu. They
are marked separately and may change with less notice than the CPU API, partly
because they track an external dependency.

## Experimental items

Anything documented as experimental may change or be removed without the usual
notice. Where an item is experimental, its rustdoc says so.

## Deprecation

Where practical, an item is deprecated for one minor release before removal, so
dependent code gets a compiler warning and a migration path rather than a sudden
break.

[`advanced`]: https://docs.rs/syren/latest/syren/advanced/index.html
