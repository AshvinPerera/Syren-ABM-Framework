//! Grid geometry shared by the discrete and continuous spaces.
//!
//! Deliberately self-contained (the ~40 lines of cell math are duplicated
//! from messaging's `SpatialConfig` rather than shared) so the `space` and
//! `messaging` features stay independent. The cell math uses saturating
//! float→int casts, returns empty ranges for non-intersecting queries, and
//! wraps explicitly on the torus.

use smallvec::SmallVec;

use super::SpaceError;

/// Rectangular cell range, inclusive on all four bounds.
pub(crate) type CellRect = (u32, u32, u32, u32); // (col_lo, col_hi, row_lo, row_hi)

/// Geometry of a 2-D cell grid: extent, cell size, and edge topology.
#[derive(Clone, Copy, Debug)]
pub struct GridGeometry {
    /// World width in world units.
    pub width: f32,
    /// World height in world units.
    pub height: f32,
    /// Cell edge length in world units (must be positive and finite).
    pub cell_size: f32,
    /// Whether edges wrap (torus) or clamp (bounded plane).
    pub torus: bool,
}

impl GridGeometry {
    /// Geometry for a discrete `cols x rows` grid (cell size 1).
    pub fn cells(cols: u32, rows: u32, torus: bool) -> Self {
        Self {
            width: cols as f32,
            height: rows as f32,
            cell_size: 1.0,
            torus,
        }
    }

    /// Validates extents and cell size.
    pub fn validate(&self) -> Result<(), SpaceError> {
        let ok = self.width.is_finite()
            && self.height.is_finite()
            && self.cell_size.is_finite()
            && self.width > 0.0
            && self.height > 0.0
            && self.cell_size > 0.0;
        if ok {
            Ok(())
        } else {
            Err(SpaceError::InvalidGeometry {
                width: self.width,
                height: self.height,
                cell_size: self.cell_size,
            })
        }
    }

    /// Number of cells along the X axis.
    #[inline]
    pub fn cols(&self) -> u32 {
        (self.width / self.cell_size).ceil().max(1.0) as u32
    }

    /// Number of cells along the Y axis.
    #[inline]
    pub fn rows(&self) -> u32 {
        (self.height / self.cell_size).ceil().max(1.0) as u32
    }

    /// Total cell count.
    #[inline]
    pub fn total_cells(&self) -> usize {
        self.cols() as usize * self.rows() as usize
    }

    /// Flat index of a cell.
    #[inline]
    pub fn cell_index(&self, col: u32, row: u32) -> u32 {
        row * self.cols() + col
    }

    /// Canonicalises a position: torus geometries wrap into
    /// `[0, width) x [0, height)`; bounded geometries clamp just inside the
    /// extent. NaN coordinates collapse to the origin.
    #[inline]
    pub fn normalise(&self, x: f32, y: f32) -> (f32, f32) {
        if self.torus {
            (
                self.wrap_axis(x, self.width),
                self.wrap_axis(y, self.height),
            )
        } else {
            (clamp_finite(x, self.width), clamp_finite(y, self.height))
        }
    }

    #[inline]
    fn wrap_axis(&self, value: f32, extent: f32) -> f32 {
        if !value.is_finite() {
            return 0.0;
        }
        let wrapped = value.rem_euclid(extent);
        // rem_euclid can return `extent` itself for tiny negatives.
        if wrapped >= extent {
            0.0
        } else {
            wrapped
        }
    }

    /// Cell containing a (normalised) position; out-of-range inputs clamp to
    /// edge cells, NaN collapses to cell (0, 0).
    #[inline]
    pub fn cell_of(&self, x: f32, y: f32) -> (u32, u32) {
        let (x, y) = self.normalise(x, y);
        let col = ((x / self.cell_size) as u32).min(self.cols() - 1);
        let row = ((y / self.cell_size) as u32).min(self.rows() - 1);
        (col, row)
    }

    /// Wraps a discrete cell coordinate onto the grid (torus) or returns
    /// `None` when it falls outside (bounded plane).
    #[inline]
    pub fn wrap_cell(&self, col: i64, row: i64) -> Option<(u32, u32)> {
        let cols = self.cols() as i64;
        let rows = self.rows() as i64;
        if self.torus {
            Some((col.rem_euclid(cols) as u32, row.rem_euclid(rows) as u32))
        } else if col >= 0 && col < cols && row >= 0 && row < rows {
            Some((col as u32, row as u32))
        } else {
            None
        }
    }

    /// Squared distance between two (normalised) points, using the minimum
    /// image convention on torus geometries.
    #[inline]
    pub fn distance2(&self, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
        let mut dx = (ax - bx).abs();
        let mut dy = (ay - by).abs();
        if self.torus {
            if dx > self.width - dx {
                dx = self.width - dx;
            }
            if dy > self.height - dy {
                dy = self.height - dy;
            }
        }
        dx * dx + dy * dy
    }

    /// Decomposes the set of cells whose extent could intersect a circle at
    /// `(cx, cy)` with radius `r` into at most four in-bounds rectangles
    /// (torus wrap splits each axis at most once). Bounded geometries whose
    /// circle misses the grid entirely yield no rectangles.
    pub(crate) fn candidate_rects(&self, cx: f32, cy: f32, r: f32) -> SmallVec<[CellRect; 4]> {
        let mut rects: SmallVec<[CellRect; 4]> = SmallVec::new();
        let cols = self.cols();
        let rows = self.rows();
        let radius = if r.is_finite() { r.max(0.0) } else { 0.0 };

        if !self.torus {
            // Bounded plane: single clamped rect, empty when disjoint.
            let intersects = cx + radius >= 0.0
                && cy + radius >= 0.0
                && cx - radius < self.width
                && cy - radius < self.height;
            if !intersects {
                return rects;
            }
            let col_lo = ((((cx - radius) / self.cell_size).floor().max(0.0)) as u32).min(cols - 1);
            let col_hi = ((cx + radius) / self.cell_size)
                .ceil()
                .min((cols - 1) as f32) as u32;
            let row_lo = ((((cy - radius) / self.cell_size).floor().max(0.0)) as u32).min(rows - 1);
            let row_hi = ((cy + radius) / self.cell_size)
                .ceil()
                .min((rows - 1) as f32) as u32;
            rects.push((col_lo, col_hi, row_lo, row_hi));
            return rects;
        }

        // Torus: work in wrapped centre cell +/- cell radius, splitting each
        // axis into at most two contiguous in-bounds ranges.
        let (centre_col, centre_row) = self.cell_of(cx, cy);
        let cell_radius = (radius / self.cell_size).ceil() as i64 + 1;

        let col_ranges = axis_ranges(centre_col as i64, cell_radius, cols as i64);
        let row_ranges = axis_ranges(centre_row as i64, cell_radius, rows as i64);
        for &(col_lo, col_hi) in &col_ranges {
            for &(row_lo, row_hi) in &row_ranges {
                rects.push((col_lo as u32, col_hi as u32, row_lo as u32, row_hi as u32));
            }
        }
        rects
    }
}

/// Splits `centre +/- radius` on a wrapped axis of `extent` cells into one
/// or two contiguous in-bounds inclusive ranges.
fn axis_ranges(centre: i64, radius: i64, extent: i64) -> SmallVec<[(i64, i64); 2]> {
    let mut ranges: SmallVec<[(i64, i64); 2]> = SmallVec::new();
    if 2 * radius + 1 >= extent {
        ranges.push((0, extent - 1));
        return ranges;
    }
    let lo = centre - radius;
    let hi = centre + radius;
    if lo < 0 {
        ranges.push((0, hi));
        ranges.push((lo.rem_euclid(extent), extent - 1));
    } else if hi >= extent {
        ranges.push((lo, extent - 1));
        ranges.push((0, hi - extent));
    } else {
        ranges.push((lo, hi));
    }
    ranges
}

#[inline]
fn clamp_finite(value: f32, extent: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    // Clamp just inside the extent so `cell_of` maps to the last cell.
    value.clamp(0.0, f32::from_bits(extent.to_bits() - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn torus() -> GridGeometry {
        GridGeometry {
            width: 100.0,
            height: 50.0,
            cell_size: 10.0,
            torus: true,
        }
    }

    fn plane() -> GridGeometry {
        GridGeometry {
            torus: false,
            ..torus()
        }
    }

    #[test]
    fn validation_rejects_degenerate_geometry() {
        for bad in [
            GridGeometry {
                width: 0.0,
                ..plane()
            },
            GridGeometry {
                cell_size: -1.0,
                ..plane()
            },
            GridGeometry {
                height: f32::NAN,
                ..plane()
            },
        ] {
            assert!(bad.validate().is_err());
        }
        assert!(plane().validate().is_ok());
    }

    #[test]
    fn torus_normalise_wraps_and_plane_clamps() {
        let (x, y) = torus().normalise(-5.0, 55.0);
        assert!((x - 95.0).abs() < 1e-4);
        assert!((y - 5.0).abs() < 1e-4);

        let (x, y) = plane().normalise(-5.0, 55.0);
        assert_eq!(x, 0.0);
        assert!(y < 50.0 && y > 49.9);
    }

    #[test]
    fn torus_distance_uses_minimum_image() {
        let g = torus();
        // 2 units apart across the seam, not 98.
        let d2 = g.distance2(1.0, 0.0, 99.0, 0.0);
        assert!((d2 - 4.0).abs() < 1e-3, "got {d2}");
        let d2 = plane().distance2(1.0, 0.0, 99.0, 0.0);
        assert!((d2 - 98.0f32.powi(2)).abs() < 1e-1);
    }

    #[test]
    fn candidate_rects_wrap_across_the_seam() {
        let g = torus();
        // Query near the left edge: columns wrap to the right side.
        let rects = g.candidate_rects(1.0, 25.0, 10.0);
        let mut cols: Vec<u32> = Vec::new();
        for (cl, ch, _, _) in rects {
            cols.extend(cl..=ch);
        }
        cols.sort_unstable();
        cols.dedup();
        assert!(cols.contains(&0));
        assert!(cols.contains(&9), "wrapped column expected, got {cols:?}");
    }

    #[test]
    fn bounded_disjoint_query_yields_no_rects() {
        let g = plane();
        assert!(g.candidate_rects(150.0, 25.0, 5.0).is_empty());
        assert!(g.candidate_rects(-20.0, 25.0, 5.0).is_empty());
        assert!(!g.candidate_rects(99.0, 25.0, 5.0).is_empty());
    }

    #[test]
    fn wrap_cell_matches_topology() {
        assert_eq!(torus().wrap_cell(-1, 0), Some((9, 0)));
        assert_eq!(torus().wrap_cell(10, 5), Some((0, 0)));
        assert_eq!(plane().wrap_cell(-1, 0), None);
        assert_eq!(plane().wrap_cell(3, 2), Some((3, 2)));
    }
}
