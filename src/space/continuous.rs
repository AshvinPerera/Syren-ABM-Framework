//! Continuous 2-D agent space backed by a uniform-grid CSR index.

use crate::engine::boundary::{BoundaryContext, BoundaryResource};
use crate::engine::entity::Entity;
use crate::engine::error::ECSResult;
use crate::engine::types::ChannelID;
use crate::engine::worker_stage::WorkerStage;

use super::geometry::GridGeometry;
use super::SpaceError;

/// One indexed agent: entity plus its normalised position.
type Item = (Entity, f32, f32);

/// Per-tick spatial index over agents at continuous 2-D positions.
///
/// Register on a model via
/// `ModelBuilder::register_continuous_space` (which also wires the reindex
/// system), or at the ECS level by constructing one, registering it with
/// [`ECSManager::register_boundary`](crate::ECSManager::register_boundary),
/// and staging positions from your own system via [`stage`](Self::stage).
///
/// Queries observe the snapshot built at the boundary after the reindex
/// system ran; see the [module docs](super) for the data-flow contract.
pub struct ContinuousSpace2D {
    geometry: GridGeometry,
    channels: [ChannelID; 1],
    deterministic: bool,

    stage: WorkerStage<Item>,
    /// Reused drain buffer.
    scratch: Vec<Item>,
    /// Reused counting buffer, one slot per cell.
    counts: Vec<u32>,

    /// CSR: `cell_starts[c]..cell_starts[c + 1]` indexes `items` for cell `c`.
    cell_starts: Vec<u32>,
    items: Vec<Item>,
}

impl ContinuousSpace2D {
    /// Creates a space with the given geometry, producing/finalising on
    /// `channel`.
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
            items: Vec::new(),
        })
    }

    /// Enables or disables the per-cell entity-id sort that makes iteration
    /// order independent of thread scheduling (default: enabled).
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

    /// Stages one agent's position for the next index build.
    ///
    /// Called from the reindex system (Phase A); lock-free per worker.
    /// Positions are normalised (torus wrap / plane clamp) at finalise.
    #[inline]
    pub fn stage(&self, entity: Entity, x: f32, y: f32) {
        self.stage.push((entity, x, y));
    }

    /// Number of agents in the current snapshot.
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// `true` if the current snapshot contains no agents.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Agents in one cell of the current snapshot.
    pub fn cell_items(&self, col: u32, row: u32) -> &[Item] {
        if col >= self.geometry.cols() || row >= self.geometry.rows() {
            return &[];
        }
        let cell = self.geometry.cell_index(col, row) as usize;
        let start = self.cell_starts[cell] as usize;
        let end = self.cell_starts[cell + 1] as usize;
        &self.items[start..end]
    }

    /// Iterates agents within `radius` of `(x, y)` (exact distance,
    /// torus-aware), yielding `(entity, (x, y))` in deterministic cell-major
    /// order. Zero allocations beyond the inline candidate-rect buffer.
    pub fn neighbors_within(
        &self,
        x: f32,
        y: f32,
        radius: f32,
    ) -> impl Iterator<Item = (Entity, (f32, f32))> + '_ {
        let (qx, qy) = self.geometry.normalise(x, y);
        let radius2 = radius * radius;
        self.geometry
            .candidate_rects(qx, qy, radius)
            .into_iter()
            .flat_map(move |(col_lo, col_hi, row_lo, row_hi)| {
                (row_lo..=row_hi).flat_map(move |row| {
                    (col_lo..=col_hi).flat_map(move |col| self.cell_items(col, row).iter())
                })
            })
            .filter(move |&&(_, ax, ay)| self.geometry.distance2(qx, qy, ax, ay) <= radius2)
            .map(|&(entity, ax, ay)| (entity, (ax, ay)))
    }

    /// Collects the `k` nearest agents to `(x, y)` into `out` as
    /// `(entity, squared_distance)`, nearest first (ties broken by entity
    /// id). `out` is cleared first and used as scratch, so a reused buffer
    /// makes repeated queries allocation-free.
    ///
    /// Expanding-ring search: cell rings are visited outward until `k`
    /// candidates are held and the next ring cannot beat the current k-th
    /// distance.
    pub fn nearest_k(&self, x: f32, y: f32, k: usize, out: &mut Vec<(Entity, f32)>) {
        out.clear();
        if k == 0 || self.items.is_empty() {
            return;
        }

        let (qx, qy) = self.geometry.normalise(x, y);
        let (centre_col, centre_row) = self.geometry.cell_of(qx, qy);
        let cols = self.geometry.cols() as i64;
        let rows = self.geometry.rows() as i64;
        // Beyond this ring every cell has been visited (torus rings start
        // re-wrapping onto themselves; planar rings fall fully outside).
        let max_ring = if self.geometry.torus {
            cols.max(rows) / 2 + 1
        } else {
            cols.max(rows)
        };

        for ring in 0..=max_ring {
            // Once k candidates are held, stop when even the nearest point of
            // this ring cannot beat the current k-th best.
            if out.len() >= k {
                let ring_min = ((ring - 1).max(0) as f32) * self.geometry.cell_size;
                let kth = out[k - 1].1;
                if ring_min * ring_min > kth {
                    break;
                }
            }

            let mut any_cell = false;
            self.for_each_ring_cell(centre_col as i64, centre_row as i64, ring, |col, row| {
                any_cell = true;
                for &(entity, ax, ay) in self.cell_items(col, row) {
                    let d2 = self.geometry.distance2(qx, qy, ax, ay);
                    out.push((entity, d2));
                }
            });
            if !any_cell && ring > 0 && !self.geometry.torus {
                break; // Bounded plane: ring fully outside the grid.
            }

            // Keep the best k: sort by (distance, entity id) - the id
            // tie-break also makes torus wrap double-visits of one entity
            // adjacent, so dedup removes them.
            out.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.to_raw().cmp(&b.0.to_raw()))
            });
            out.dedup_by_key(|(entity, _)| entity.to_raw());
            out.truncate(k);
        }
    }

    /// Visits every cell on the square ring at Chebyshev distance `ring`
    /// from the centre (torus-wrapped; out-of-bounds cells skipped on
    /// bounded planes). Small torii may re-visit a wrapped cell; callers
    /// dedup by entity.
    fn for_each_ring_cell(
        &self,
        centre_col: i64,
        centre_row: i64,
        ring: i64,
        mut visit: impl FnMut(u32, u32),
    ) {
        let visit_cell = |col: i64, row: i64, visit: &mut dyn FnMut(u32, u32)| {
            if let Some((c, r)) = self.geometry.wrap_cell(col, row) {
                visit(c, r);
            }
        };
        if ring == 0 {
            visit_cell(centre_col, centre_row, &mut visit);
            return;
        }
        for col in (centre_col - ring)..=(centre_col + ring) {
            visit_cell(col, centre_row - ring, &mut visit);
            visit_cell(col, centre_row + ring, &mut visit);
        }
        for row in (centre_row - ring + 1)..=(centre_row + ring - 1) {
            visit_cell(centre_col - ring, row, &mut visit);
            visit_cell(centre_col + ring, row, &mut visit);
        }
    }

    /// Rebuilds the CSR index from staged positions (counting sort +
    /// optional per-cell determinism sort).
    fn rebuild(&mut self) {
        self.scratch.clear();
        self.stage.drain_into(&mut self.scratch);

        // Normalise positions once, here.
        for item in &mut self.scratch {
            let (x, y) = self.geometry.normalise(item.1, item.2);
            item.1 = x;
            item.2 = y;
        }

        let total_cells = self.geometry.total_cells();
        self.counts.clear();
        self.counts.resize(total_cells, 0);
        for &(_, x, y) in &self.scratch {
            let (col, row) = self.geometry.cell_of(x, y);
            self.counts[self.geometry.cell_index(col, row) as usize] += 1;
        }

        self.cell_starts.clear();
        self.cell_starts.resize(total_cells + 1, 0);
        for cell in 0..total_cells {
            self.cell_starts[cell + 1] = self.cell_starts[cell] + self.counts[cell];
        }

        self.items.clear();
        self.items
            .resize(self.scratch.len(), (Entity::PLACEHOLDER, 0.0, 0.0));
        let mut cursor: Vec<u32> = self.cell_starts[..total_cells].to_vec();
        for &(entity, x, y) in &self.scratch {
            let (col, row) = self.geometry.cell_of(x, y);
            let cell = self.geometry.cell_index(col, row) as usize;
            let slot = cursor[cell] as usize;
            self.items[slot] = (entity, x, y);
            cursor[cell] += 1;
        }

        if self.deterministic {
            for cell in 0..total_cells {
                let start = self.cell_starts[cell] as usize;
                let end = self.cell_starts[cell + 1] as usize;
                self.items[start..end].sort_unstable_by_key(|&(entity, _, _)| entity.to_raw());
            }
        }
    }
}

impl BoundaryResource for ContinuousSpace2D {
    fn name(&self) -> &str {
        "ContinuousSpace2D"
    }

    fn channels(&self) -> &[ChannelID] {
        &self.channels
    }

    fn begin_tick(&mut self, _ctx: &mut BoundaryContext<'_>) -> ECSResult<()> {
        // Defensive: the stage is normally drained by finalise; the snapshot
        // itself persists until the next reindex so pre-reindex systems see
        // last tick's positions (documented semantics).
        self.stage.clear();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn space(torus: bool) -> ContinuousSpace2D {
        ContinuousSpace2D::new(
            GridGeometry {
                width: 100.0,
                height: 100.0,
                cell_size: 10.0,
                torus,
            },
            7,
        )
        .unwrap()
    }

    fn entity(raw: u64) -> Entity {
        Entity::from_raw(raw)
    }

    fn rebuild(space: &mut ContinuousSpace2D) {
        space.rebuild();
    }

    #[test]
    fn radius_query_matches_brute_force_including_torus_seam() {
        for torus in [false, true] {
            let mut s = space(torus);
            // Deterministic pseudo-random layout.
            let mut rng = crate::engine::random::DetRng::from_seed(42);
            let mut points = Vec::new();
            for raw in 0..500u64 {
                let x = rng.next_f32() * 100.0;
                let y = rng.next_f32() * 100.0;
                points.push((entity(raw), x, y));
                s.stage(entity(raw), x, y);
            }
            rebuild(&mut s);

            for &(qx, qy, r) in &[
                (5.0f32, 5.0f32, 12.0f32),
                (99.0, 1.0, 15.0),
                (50.0, 50.0, 7.5),
            ] {
                let mut expected: Vec<u64> = points
                    .iter()
                    .filter(|&&(_, x, y)| s.geometry().distance2(qx, qy, x, y) <= r * r)
                    .map(|&(e, _, _)| e.to_raw())
                    .collect();
                expected.sort_unstable();
                let mut observed: Vec<u64> = s
                    .neighbors_within(qx, qy, r)
                    .map(|(e, _)| e.to_raw())
                    .collect();
                observed.sort_unstable();
                assert_eq!(observed, expected, "torus={torus} query=({qx},{qy},{r})");
            }
        }
    }

    #[test]
    fn nearest_k_returns_sorted_nearest() {
        let mut s = space(false);
        for raw in 0..10u64 {
            s.stage(entity(raw), raw as f32 * 5.0, 0.0); // along a line
        }
        rebuild(&mut s);

        let mut out = Vec::new();
        s.nearest_k(0.0, 0.0, 3, &mut out);
        let ids: Vec<u64> = out.iter().map(|(e, _)| e.to_raw()).collect();
        assert_eq!(ids, vec![0, 1, 2]);
        assert!(out[0].1 <= out[1].1 && out[1].1 <= out[2].1);
    }

    #[test]
    fn snapshot_is_deterministic_within_cells() {
        let mut s = space(false);
        // Same cell, staged in scrambled order.
        for &raw in &[9u64, 3, 7, 1, 5] {
            s.stage(entity(raw), 1.0 + raw as f32 * 0.1, 1.0);
        }
        rebuild(&mut s);
        let ids: Vec<u64> = s
            .cell_items(0, 0)
            .iter()
            .map(|&(e, _, _)| e.to_raw())
            .collect();
        assert_eq!(ids, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn rebuild_replaces_the_previous_snapshot() {
        let mut s = space(false);
        s.stage(entity(1), 5.0, 5.0);
        rebuild(&mut s);
        assert_eq!(s.len(), 1);

        // Next tick: nothing staged -> empty snapshot.
        rebuild(&mut s);
        assert!(s.is_empty());
        assert_eq!(s.neighbors_within(5.0, 5.0, 50.0).count(), 0);
    }
}
