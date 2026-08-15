# Use grids and continuous space

_Requires the `environment` feature (implied by `model`)._

The space layer indexes agents by position so systems can find neighbours. There
are two spaces, both built from a [`GridGeometry`] (the cell dimensions and
whether boundaries wrap) and a channel id:

- [`GridSpace2D`] — a discrete grid of cells.
- [`ContinuousSpace2D`] — a continuous 2-D plane, indexed by an underlying grid
  for range queries.

A [`SpaceHandle`] owns the channel that orders the systems that build and read
the space.

## Discrete grid

Populate the grid by staging each agent's cell, then query neighbours:

```rust,ignore
grid.stage(entity, col, row);
// ...after the staging stage...
let here = grid.occupants(col, row);              // entities in a cell
let neighbours = grid.moore_neighborhood(col, row, radius);
let adjacent = grid.von_neumann_neighborhood(col, row, radius);
```

For movement where several agents compete for the same destination cell, use the
claims mechanism ([`GridClaims`]): agents `bid` for a cell and the `winner`
resolves deterministically, so contested moves do not depend on visitation order.
This is how the Sugarscape example moves agents.

## Continuous space

Continuous space answers radius queries with exact distances, honouring toroidal
wrapping where the geometry requests it:

```rust,ignore
for (entity, x, y) in space.neighbors_within(qx, qy, radius) {
    // agents within `radius` of (qx, qy)
}
```

## Geometry and wrapping

`GridGeometry` uses saturating conversions from floating-point positions,
returns empty ranges for queries that do not intersect the space, and wraps
explicitly on a torus. The same geometry backs the spatial message
specialisation, so spatial messaging and spatial queries agree on cells.

See the Sugarscape example (`examples/sugarscape/`) for a grid-based model.

[`GridGeometry`]: https://docs.rs/syren/latest/syren/space/struct.GridGeometry.html
[`GridSpace2D`]: https://docs.rs/syren/latest/syren/space/struct.GridSpace2D.html
[`ContinuousSpace2D`]: https://docs.rs/syren/latest/syren/space/struct.ContinuousSpace2D.html
[`SpaceHandle`]: https://docs.rs/syren/latest/syren/space/struct.SpaceHandle.html
[`GridClaims`]: https://docs.rs/syren/latest/syren/space/struct.GridClaims.html
