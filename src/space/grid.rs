//! Discrete cell-grid agent space with occupancy queries and deterministic
//! parallel movement claims.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::boundary::{BoundaryContext, BoundaryResource};
use crate::engine::entity::Entity;
use crate::engine::error::ECSResult;
use crate::engine::types::ChannelID;
use crate::engine::worker_stage::WorkerStage;

use super::geometry::GridGeometry;
use super::SpaceError;

/// Per-tick occupancy index over agents on discrete grid cells.
///
/// Register via `ModelBuilder::register_grid_space` (which wires the reindex
/// system) or manually at the ECS level. Same snapshot semantics as
/// [`ContinuousSpace2D`](super::ContinuousSpace2D): queries observe cell
/// occupancy as of the reindex stage.
///
/// The embedded [`GridClaims`] table supports the classic parallel-movement
/// pattern (Sugarscape-style "move to the best free cell"): agents **bid**
/// for target cells inside a parallel system, the lowest entity id wins each
/// cell deterministically, and winners apply their moves. Claims clear at
/// `begin_tick`.
pub struct GridSpace2D {
    geometry: GridGeometry,
    channels: [ChannelID; 1],
    deterministic: bool,

    stage: WorkerStage<(Entity, u32, u32)>,
    scratch: Vec<(Entity, u32, u32)>,
    counts: Vec<u32>,

    /// CSR: `cell_starts[c]..cell_starts[c + 1]` indexes `occupants`.
    cell_starts: Vec<u32>,
    occupants: Vec<Entity>,

    claims: GridClaims,
}

impl GridSpace2D {
    /// Creates a grid space over `geometry` (typically
    /// [`GridGeometry::cells`]), finalising on `channel`.
    pub fn new(geometry: GridGeometry, channel: ChannelID) -> Result<Self, SpaceError> {
        geometry.validate()?;
        let total_cells = geometry.total_cells();
        Ok(Self {
            geometry,
            channels: [channel],
            deterministic: true,
            stage: WorkerStage::new(),
            scratch: Vec::new(),
            counts: vec![0; total_cells],
            cell_starts: vec![0; total_cells + 1],
            occupants: Vec::new(),
            claims: GridClaims::new(geometry.cols(), geometry.rows()),
        })
    }

    /// Enables or disables the per-cell entity-id sort (default: enabled).
    #[must_use]
    pub fn with_determinism(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    /// The scheduler channel this space finalises on.
    #[inline]
    pub fn channel(&self) -> ChannelID {
        self.channels[0]
    }

    /// The space's geometry.
    #[inline]
    pub fn geometry(&self) -> &GridGeometry {
        &self.geometry
    }

    /// Stages one agent's cell for the next index build (Phase A).
    ///
    /// Out-of-range cells wrap on torus geometries and clamp on bounded ones.
    #[inline]
    pub fn stage(&self, entity: Entity, col: u32, row: u32) {
        self.stage.push((entity, col, row));
    }

    /// Number of agents in the current snapshot.
    #[inline]
    pub fn len(&self) -> usize {
        self.occupants.len()
    }

    /// `true` if the snapshot contains no agents.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.occupants.is_empty()
    }

    /// Occupants of one cell, sorted by entity id when determinism is on.
    pub fn occupants(&self, col: u32, row: u32) -> &[Entity] {
        if col >= self.geometry.cols() || row >= self.geometry.rows() {
            return &[];
        }
        let cell = self.geometry.cell_index(col, row) as usize;
        let start = self.cell_starts[cell] as usize;
        let end = self.cell_starts[cell + 1] as usize;
        &self.occupants[start..end]
    }

    /// Occupant count of one cell.
    #[inline]
    pub fn count(&self, col: u32, row: u32) -> usize {
        self.occupants(col, row).len()
    }

    /// `true` if the cell has no occupants in the snapshot.
    #[inline]
    pub fn is_cell_empty(&self, col: u32, row: u32) -> bool {
        self.count(col, row) == 0
    }

    /// Iterates the cells of the Moore neighbourhood (Chebyshev distance
    /// `<= radius`, centre excluded), torus-wrapped or clipped per geometry.
    pub fn moore_neighborhood(
        &self,
        col: u32,
        row: u32,
        radius: u32,
    ) -> impl Iterator<Item = (u32, u32)> + '_ {
        let radius = radius as i64;
        let centre = (col as i64, row as i64);
        (-radius..=radius)
            .flat_map(move |dr| (-radius..=radius).map(move |dc| (dc, dr)))
            .filter(|&(dc, dr)| dc != 0 || dr != 0)
            .filter_map(move |(dc, dr)| self.geometry.wrap_cell(centre.0 + dc, centre.1 + dr))
    }

    /// Iterates the cells of the von Neumann neighbourhood (Manhattan
    /// distance `<= radius`, centre excluded).
    pub fn von_neumann_neighborhood(
        &self,
        col: u32,
        row: u32,
        radius: u32,
    ) -> impl Iterator<Item = (u32, u32)> + '_ {
        let radius = radius as i64;
        let centre = (col as i64, row as i64);
        (-radius..=radius)
            .flat_map(move |dr| (-radius..=radius).map(move |dc| (dc, dr)))
            .filter(move |&(dc, dr)| (dc != 0 || dr != 0) && dc.abs() + dr.abs() <= radius)
            .filter_map(move |(dc, dr)| self.geometry.wrap_cell(centre.0 + dc, centre.1 + dr))
    }

    /// The claims table for deterministic parallel movement.
    #[inline]
    pub fn claims(&self) -> &GridClaims {
        &self.claims
    }

    fn rebuild(&mut self) {
        self.scratch.clear();
        self.stage.drain_into(&mut self.scratch);

        let cols = self.geometry.cols();
        let rows = self.geometry.rows();
        for item in &mut self.scratch {
            item.1 = item.1.min(cols - 1);
            item.2 = item.2.min(rows - 1);
        }

        let total_cells = self.geometry.total_cells();
        self.counts.clear();
        self.counts.resize(total_cells, 0);
        for &(_, col, row) in &self.scratch {
            self.counts[self.geometry.cell_index(col, row) as usize] += 1;
        }

        self.cell_starts.clear();
        self.cell_starts.resize(total_cells + 1, 0);
        for cell in 0..total_cells {
            self.cell_starts[cell + 1] = self.cell_starts[cell] + self.counts[cell];
        }

        self.occupants.clear();
        self.occupants
            .resize(self.scratch.len(), Entity::PLACEHOLDER);
        let mut cursor: Vec<u32> = self.cell_starts[..total_cells].to_vec();
        for &(entity, col, row) in &self.scratch {
            let cell = self.geometry.cell_index(col, row) as usize;
            let slot = cursor[cell] as usize;
            self.occupants[slot] = entity;
            cursor[cell] += 1;
        }

        if self.deterministic {
            for cell in 0..total_cells {
                let start = self.cell_starts[cell] as usize;
                let end = self.cell_starts[cell + 1] as usize;
                self.occupants[start..end].sort_unstable_by_key(|entity| entity.to_raw());
            }
        }
    }
}

impl BoundaryResource for GridSpace2D {
    fn name(&self) -> &str {
        "GridSpace2D"
    }

    fn channels(&self) -> &[ChannelID] {
        &self.channels
    }

    fn begin_tick(&mut self, _ctx: &mut BoundaryContext<'_>) -> ECSResult<()> {
        self.stage.clear();
        self.claims.clear();
        Ok(())
    }

    fn finalise(
        &mut self,
        _ctx: &mut BoundaryContext<'_>,
        channels: &[ChannelID],
    ) -> ECSResult<()> {
        if channels.contains(&self.channels[0]) {
            self.rebuild();
        }
        Ok(())
    }

    fn end_tick(&mut self, _ctx: &mut BoundaryContext<'_>) -> ECSResult<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Deterministic parallel movement-conflict resolution: agents **bid** for
/// cells from inside a parallel system; the **lowest entity id wins** each
/// cell.
///
/// The two-pass pattern:
///
/// ```text
/// // Pass 1 (parallel system): every agent bids for its chosen cell.
/// grid.claims().bid(target_col, target_row, entity);
/// // Pass 2 (later system, same or next stage): winners apply the move.
/// if grid.claims().won(target_col, target_row, entity) {
///     position.set(target_col, target_row);
/// }
/// ```
///
/// `bid` is an atomic `fetch_min` on the entity's raw id - an
/// order-independent reduction, so the winner is identical for a fixed seed
/// at **any** thread count, unlike first-come-first-served CAS. Cleared
/// automatically at `begin_tick` when embedded in [`GridSpace2D`].
pub struct GridClaims {
    cols: u32,
    rows: u32,
    /// One slot per cell holding the lowest bidding entity's raw id;
    /// `Entity::PLACEHOLDER` (all-ones) means unclaimed, which is exactly
    /// what `fetch_min` wants as the identity element.
    cells: Vec<AtomicU64>,
}

impl GridClaims {
    /// Creates a cleared claims table for a `cols x rows` grid.
    pub fn new(cols: u32, rows: u32) -> Self {
        let total = cols as usize * rows as usize;
        Self {
            cols,
            rows,
            cells: (0..total)
                .map(|_| AtomicU64::new(Entity::PLACEHOLDER.to_raw()))
                .collect(),
        }
    }

    #[inline]
    fn index(&self, col: u32, row: u32) -> Option<usize> {
        (col < self.cols && row < self.rows).then(|| (row * self.cols + col) as usize)
    }

    /// Bids for a cell on behalf of `entity`. Lowest entity id wins;
    /// out-of-range cells ignore the bid.
    #[inline]
    pub fn bid(&self, col: u32, row: u32, entity: Entity) {
        if let Some(index) = self.index(col, row) {
            self.cells[index].fetch_min(entity.to_raw(), Ordering::Relaxed);
        }
    }

    /// The winning bidder of a cell, if any.
    #[inline]
    pub fn winner(&self, col: u32, row: u32) -> Option<Entity> {
        let index = self.index(col, row)?;
        let raw = self.cells[index].load(Ordering::Relaxed);
        (raw != Entity::PLACEHOLDER.to_raw()).then(|| Entity::from_raw(raw))
    }

    /// `true` if `entity` holds the winning bid for the cell.
    #[inline]
    pub fn won(&self, col: u32, row: u32, entity: Entity) -> bool {
        self.winner(col, row) == Some(entity)
    }

    /// Resets every cell to unclaimed.
    pub fn clear(&self) {
        let unclaimed = Entity::PLACEHOLDER.to_raw();
        for cell in &self.cells {
            cell.store(unclaimed, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(raw: u64) -> Entity {
        Entity::from_raw(raw)
    }

    fn grid() -> GridSpace2D {
        GridSpace2D::new(GridGeometry::cells(10, 10, true), 3).unwrap()
    }

    #[test]
    fn occupancy_snapshot_counts_and_sorts() {
        let mut g = grid();
        g.stage(entity(5), 2, 3);
        g.stage(entity(1), 2, 3);
        g.stage(entity(9), 7, 7);
        g.rebuild();

        assert_eq!(g.len(), 3);
        assert_eq!(g.count(2, 3), 2);
        assert!(g.is_cell_empty(0, 0));
        let ids: Vec<u64> = g.occupants(2, 3).iter().map(|e| e.to_raw()).collect();
        assert_eq!(ids, vec![1, 5], "occupants sorted by entity id");
    }

    #[test]
    fn neighborhoods_respect_topology() {
        let g = grid();
        // Torus: corner cell has full neighbourhoods.
        assert_eq!(g.moore_neighborhood(0, 0, 1).count(), 8);
        assert_eq!(g.von_neumann_neighborhood(0, 0, 1).count(), 4);

        let bounded = GridSpace2D::new(GridGeometry::cells(10, 10, false), 3).unwrap();
        assert_eq!(bounded.moore_neighborhood(0, 0, 1).count(), 3);
        assert_eq!(bounded.von_neumann_neighborhood(0, 0, 1).count(), 2);
    }

    #[test]
    fn claims_lowest_entity_wins_regardless_of_bid_order() {
        let claims = GridClaims::new(4, 4);
        // Bids arrive in "bad" order; fetch_min is order-independent.
        for &raw in &[900u64, 3, 512, 44] {
            claims.bid(1, 1, entity(raw));
        }
        assert_eq!(claims.winner(1, 1), Some(entity(3)));
        assert!(claims.won(1, 1, entity(3)));
        assert!(!claims.won(1, 1, entity(44)));
        assert_eq!(claims.winner(2, 2), None);

        claims.clear();
        assert_eq!(claims.winner(1, 1), None);
    }

    #[test]
    fn parallel_bids_are_deterministic() {
        let claims = GridClaims::new(8, 8);
        rayon::scope(|s| {
            for raw in 1..=512u64 {
                let claims = &claims;
                s.spawn(move |_| {
                    // Everyone bids for the cell keyed by their id.
                    claims.bid((raw % 8) as u32, ((raw / 8) % 8) as u32, entity(raw));
                });
            }
        });
        // Winner of cell (c, r) is the smallest id mapping there.
        for row in 0..8u32 {
            for col in 0..8u32 {
                let expected = (1..=512u64)
                    .filter(|raw| (raw % 8) as u32 == col && ((raw / 8) % 8) as u32 == row)
                    .min()
                    .map(entity);
                assert_eq!(claims.winner(col, row), expected);
            }
        }
    }
}
