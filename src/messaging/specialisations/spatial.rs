//! [`SpatialBuffer`] - a counting-sort index over 2-D grid cells.
//!
//! Messages are sorted by their world-space `position()` into a flat grid
//! of cells, enabling O(cells_in_radius x occupancy) radius queries with no
//! per-query allocations.
//!
//! # Query algorithm
//!
//! [`SpatialQueryIter`] iterates over the bounding box of cells that
//! intersect the query circle, yielding each message in those cells.  The
//! caller is responsible for filtering by exact radius if needed.

use crate::messaging::aligned_buffer::AlignedBuffer;
use crate::messaging::error::MessagingError;
use crate::messaging::message::Message;
use crate::messaging::registry::{ErasedFns, SpatialConfig};
use crate::ECSResult;

// -----------------------------------------------------------------------------
// Buffer
// -----------------------------------------------------------------------------

/// Stores all messages for a `Spatial`-specialised type, sorted by grid cell.
pub(crate) struct SpatialBuffer {
    /// Sorted message storage (valid after [`finalise`](SpatialBuffer::finalise)).
    pub(crate) data: AlignedBuffer,
    /// `cell_starts[c]` = index of the first message in cell `c`.
    /// Length is `total_cells + 1`.
    pub(crate) cell_starts: Vec<u32>,
    /// Grid configuration.
    pub(crate) config: SpatialConfig,
    item_size: usize,
}

impl SpatialBuffer {
    pub(crate) fn new(
        item_size: usize,
        item_align: usize,
        config: SpatialConfig,
        capacity: usize,
    ) -> Self {
        let total_cells = config.total_cells();
        SpatialBuffer {
            data: AlignedBuffer::with_capacity(item_size, item_align, capacity),
            cell_starts: vec![0u32; total_cells + 1],
            config,
            item_size,
        }
    }

    /// Clears for a new tick.
    pub(crate) fn begin_tick(&mut self) {
        self.data.clear();
        self.cell_starts.fill(0);
    }

    /// Drains `raw` into sorted `self.data`, building `cell_starts`.
    ///
    /// # Safety
    ///
    /// `fns.position` must be `Some` and must read items of the registered type.
    pub(crate) unsafe fn finalise(
        &mut self,
        raw: &AlignedBuffer,
        fns: &ErasedFns,
    ) -> ECSResult<()> {
        let n = raw.len();
        if n == 0 {
            return Ok(());
        }

        let position_fn = fns.position.ok_or(MessagingError::MissingErasedFunction {
            specialisation: "Spatial",
            function: "position",
        })?;
        let total_cells = self.config.total_cells();

        // -- 1. Count ---------------------------------------------------------
        let mut counts: Vec<u32> = vec![0u32; total_cells];
        for i in 0..n {
            let ptr = unsafe { raw.as_ptr_at(i) };
            let (x, y) = unsafe { position_fn(ptr) };
            let cell = self.config.cell_id_of(x, y) as usize;
            counts[cell] += 1;
        }

        // -- 2. Prefix-sum -----------------------------------------------------
        self.cell_starts[0] = 0;
        for c in 0..total_cells {
            self.cell_starts[c + 1] = self.cell_starts[c] + counts[c];
        }

        // -- 3. Scatter --------------------------------------------------------
        self.data.reserve(n);
        unsafe { self.data.set_len(n) };

        let mut scatter_cursor = self.cell_starts[..total_cells].to_vec();

        for i in 0..n {
            let src = unsafe { raw.as_ptr_at(i) };
            let (x, y) = unsafe { position_fn(src) };
            let cell = self.config.cell_id_of(x, y) as usize;
            let dst_idx = scatter_cursor[cell] as usize;
            let dst = unsafe { self.data.as_mut_ptr_at(dst_idx) };
            unsafe { std::ptr::copy_nonoverlapping(src, dst, self.item_size) };
            scatter_cursor[cell] += 1;
        }
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Iterator
// -----------------------------------------------------------------------------

/// An iterator over all messages whose grid cell intersects a query circle.
///
/// Messages are returned in cell-major order.  The iterator does **not**
/// apply exact distance filtering - that is left to the caller.
///
/// Produced by [`MessageBufferSet::spatial`](crate::messaging::MessageBufferSet::spatial).
pub struct SpatialQueryIter<'a, M> {
    /// The full sorted data slice.
    data: &'a [M],
    /// Cell starts array (length = total_cells + 1).
    cell_starts: &'a [u32],
    /// Grid configuration (needed to enumerate cells from bounding box).
    config: SpatialConfig,
    /// Bounding box of query (inclusive column/row ranges).
    col_lo: u32,
    col_hi: u32,
    row_hi: u32,
    /// Current position within the iterator.
    cur_col: u32,
    cur_row: u32,
    /// Slice within the current cell we are reading from.
    cell_slice: &'a [M],
    cell_index: usize,
    /// Set to true once we've exhausted the bounding box.
    done: bool,
}

impl<'a, M: Message> SpatialQueryIter<'a, M> {
    pub(crate) fn new(buf: &'a SpatialBuffer, cx: f32, cy: f32, r: f32) -> Self {
        if buf.data.is_empty() {
            return Self::empty_with_config(buf.config);
        }

        // SAFETY: M matches the buffer's type.
        let data: &'a [M] = unsafe { buf.data.as_slice() };
        let (col_lo, col_hi, row_lo, row_hi) = buf.config.cell_range_for_radius(cx, cy, r);

        let mut iter = SpatialQueryIter {
            data,
            cell_starts: &buf.cell_starts,
            config: buf.config,
            col_lo,
            col_hi,
            row_hi,
            cur_col: col_lo,
            cur_row: row_lo,
            cell_slice: &[],
            cell_index: 0,
            done: false,
        };
        iter.load_cell(col_lo, row_lo);
        iter
    }

    fn empty_with_config(config: SpatialConfig) -> Self {
        // We can't borrow a temporary slice from nothing; use a static empty
        // slice for the data and cell_starts.
        SpatialQueryIter {
            data: &[],
            cell_starts: &[],
            config,
            col_lo: 0,
            col_hi: 0,
            row_hi: 0,
            cur_col: 0,
            cur_row: 0,
            cell_slice: &[],
            cell_index: 0,
            done: true,
        }
    }

    pub(crate) fn empty() -> Self {
        SpatialQueryIter {
            data: &[],
            cell_starts: &[],
            config: SpatialConfig {
                width: 1.0,
                height: 1.0,
                cell_size: 1.0,
            },
            col_lo: 0,
            col_hi: 0,
            row_hi: 0,
            cur_col: 0,
            cur_row: 0,
            cell_slice: &[],
            cell_index: 0,
            done: true,
        }
    }

    /// Advances to the next non-empty cell, or sets `done`.
    fn advance_cell(&mut self) {
        loop {
            // Advance column
            if self.cur_col < self.col_hi {
                self.cur_col += 1;
            } else if self.cur_row < self.row_hi {
                self.cur_col = self.col_lo;
                self.cur_row += 1;
            } else {
                self.done = true;
                return;
            }
            self.load_cell(self.cur_col, self.cur_row);
            if !self.cell_slice.is_empty() {
                return;
            }
        }
    }

    fn load_cell(&mut self, col: u32, row: u32) {
        let cell = (row * self.config.cols() + col) as usize;
        if cell + 1 >= self.cell_starts.len() {
            self.cell_slice = &[];
            self.cell_index = 0;
            return;
        }
        let start = self.cell_starts[cell] as usize;
        let end = self.cell_starts[cell + 1] as usize;
        self.cell_slice = if start < end {
            &self.data[start..end]
        } else {
            &[]
        };
        self.cell_index = 0;
    }
}

impl<'a, M: Message> Iterator for SpatialQueryIter<'a, M> {
    type Item = M;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.done {
                return None;
            }
            if self.cell_index < self.cell_slice.len() {
                let item = self.cell_slice[self.cell_index];
                self.cell_index += 1;
                return Some(item);
            }
            self.advance_cell();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::error::ECSError;
    use crate::messaging::registry::ErasedFns;

    #[derive(Clone, Copy)]
    struct TestMsg {
        _value: u32,
    }

    #[test]
    fn missing_position_accessor_returns_error() {
        let mut raw = AlignedBuffer::with_capacity(
            std::mem::size_of::<TestMsg>(),
            std::mem::align_of::<TestMsg>(),
            1,
        );
        unsafe { raw.push(TestMsg { _value: 1 }) };
        let mut buf = SpatialBuffer::new(
            std::mem::size_of::<TestMsg>(),
            std::mem::align_of::<TestMsg>(),
            SpatialConfig {
                width: 10.0,
                height: 10.0,
                cell_size: 1.0,
            },
            1,
        );
        let fns = ErasedFns {
            bucket_key: None,
            position: None,
            recipient: None,
        };

        let err = unsafe { buf.finalise(&raw, &fns) }.unwrap_err();
        assert!(matches!(
            err,
            ECSError::Messaging(MessagingError::MissingErasedFunction {
                specialisation: "Spatial",
                function: "position"
            })
        ));
    }
}
