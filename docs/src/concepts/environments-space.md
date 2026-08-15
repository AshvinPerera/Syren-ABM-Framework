# Environments and space

## Environments

_Requires the `environment` feature (implied by `model`)._

An **environment** holds model-wide values — state that is not per-agent, such as
a price index, a policy rate, or an audit record. Values are keyed by name and
type. You register a key up front and get a typed handle ([`EnvKey`]):

```rust,ignore
let key = builder.register_environment::<Prices>("prices", Prices::default())?;
```

Each environment key owns a **channel**, so a system that reads an environment
value is automatically ordered after the systems that write it (see
[scheduling](scheduling.md)). Reads and writes go through the environment store
by name and type:

```rust,ignore
let prices: Prices = model.environment().get::<Prices>("prices")?;
```

Because the environment is a boundary, writes made during a parallel stage are
finalised deterministically at the stage edge. See [use environment
values](../how-to/environment.md).

## Space

_Requires the `environment` feature (implied by `model`)._

The **space** layer provides spatial structure — a discrete grid and a continuous
2-D space — for models where agents have positions and interact by locality. A
[`SpaceHandle`] owns a channel, so systems that update and read space are ordered
correctly relative to one another.

Grid geometry uses saturating integer conversions, returns empty ranges for
queries that do not intersect the space, and wraps explicitly on a torus where
toroidal boundaries are requested. The same cell math backs the spatial message
specialisation, so spatial messaging and spatial queries agree on which cell a
position falls in.

See [use grids and continuous space](../how-to/space.md).

[`EnvKey`]: https://docs.rs/syren/latest/syren/environment/struct.EnvKey.html
[`SpaceHandle`]: https://docs.rs/syren/latest/syren/space/struct.SpaceHandle.html
