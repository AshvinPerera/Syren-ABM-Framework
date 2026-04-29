//! [`BucketBuffer`] - a counting-sort index over integer bucket keys.
//!
//! Messages are sorted by their `bucket_key()` into contiguous slices within
//! a single [`AlignedBuffer`], allowing O(1) per-bucket access after an
//! O(n) scatter pass.
//!
//! # Scatter algorithm
//!
//! 1. **Count**: walk the raw (unsorted) buffer, increment `counts[key]` for
//!    each message.
//! 2. **Prefix-sum**: convert `counts` into exclusive prefix sums stored in
//!    `bucket_starts`.  After this, `bucket_starts[k]` is the index of the
//!    first message for bucket `k` in the sorted output.
//! 3. **Scatter**: allocate a new sorted buffer; use a temporary
//!    `scatter_cursor` array (independent copy of `bucket_starts`) to place
//!    each message at the correct destination and advance the cursor.

use crate::messaging::aligned_buffer::AlignedBuffer;
use crate::messaging::error::MessagingError;
use crate::messaging::message::Message;
use crate::messaging::registry::ErasedFns;
use crate::ECSResult;

// -----------------------------------------------------------------------------
// Buffer
// -----------------------------------------------------------------------------

/// Stores all messages for a `Bucket`-specialised type, sorted by bucket key.
pub(crate) struct BucketBuffer {
    /// Sorted message storage (valid after [`finalise`](BucketBuffer::finalise)).
    pub(crate) data: AlignedBuffer,
    /// `bucket_starts[k]` = index of the first message in bucket `k`.
    /// Length is `max_buckets + 1` (the last element equals `data.len()`).
    pub(crate) bucket_starts: Vec<u32>,
    /// Maximum number of distinct bucket keys (exclusive upper bound).
    pub(crate) max_buckets: u32,
    /// Size of one item; stored for scatter allocations.
    item_size: usize,
}

impl BucketBuffer {
    pub(crate) fn new(
        item_size: usize,
        item_align: usize,
        max_buckets: u32,
        capacity: usize,
    ) -> Self {
        BucketBuffer {
            data: AlignedBuffer::with_capacity(item_size, item_align, capacity),
            bucket_starts: vec![0u32; max_buckets as usize + 1],
            max_buckets,
            item_size,
        }
    }

    /// Clears for a new tick.
    pub(crate) fn begin_tick(&mut self) {
        self.data.clear();
        self.bucket_starts.fill(0);
    }

    /// Drains `raw` (unsorted arrivals) into `self.data` via counting sort,
    /// then builds `bucket_starts`.
    ///
    /// # Safety
    ///
    /// `fns.bucket_key` must be `Some` and must read items of the type this
    /// buffer was created for.
    pub(crate) unsafe fn finalise(
        &mut self,
        raw: &AlignedBuffer,
        fns: &ErasedFns,
    ) -> ECSResult<()> {
        let n = raw.len();
        if n == 0 {
            return Ok(());
        }

        let bucket_key_fn = fns
            .bucket_key
            .ok_or(MessagingError::MissingErasedFunction {
                specialisation: "Bucket",
                function: "bucket_key",
            })?;
        let max = self.max_buckets as usize;

        // -- 1. Count ---------------------------------------------------------
        let mut counts: Vec<u32> = vec![0u32; max];
        for i in 0..n {
            // SAFETY: i < raw.len(); ptr points to a valid item.
            let ptr = unsafe { raw.as_ptr_at(i) };
            let key = unsafe { bucket_key_fn(ptr) } as usize;
            if key >= max {
                return Err(MessagingError::BucketKeyOutOfRange {
                    key: key as u32,
                    max: self.max_buckets,
                }
                .into());
            }
            counts[key] += 1;
        }

        // -- 2. Prefix-sum into bucket_starts ---------------------------------
        self.bucket_starts[0] = 0;
        for (k, count) in counts.iter().enumerate().take(max) {
            self.bucket_starts[k + 1] = self.bucket_starts[k] + *count;
        }

        // -- 3. Scatter --------------------------------------------------------
        self.data.reserve(n);
        // SAFETY: we just reserved n items; set_len after scatter.
        unsafe { self.data.set_len(n) };

        let mut scatter_cursor = self.bucket_starts[..max].to_vec();

        for i in 0..n {
            let src = unsafe { raw.as_ptr_at(i) };
            let key = unsafe { bucket_key_fn(src) } as usize;
            let dst_idx = scatter_cursor[key] as usize;
            let dst = unsafe { self.data.as_mut_ptr_at(dst_idx) };
            // SAFETY: src and dst point to valid, non-overlapping item slots.
            unsafe { std::ptr::copy_nonoverlapping(src, dst, self.item_size) };
            scatter_cursor[key] += 1;
        }
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Iterator
// -----------------------------------------------------------------------------

/// An iterator over all messages in a single bucket of a `BucketBuffer`.
pub struct BucketIter<'a, M> {
    slice: &'a [M],
    index: usize,
}

impl<'a, M: Message> BucketIter<'a, M> {
    /// Creates an iterator over bucket `key` of `buf`.
    ///
    /// # Safety
    ///
    /// `M` must be the type this buffer was created for.
    pub(crate) fn new(buf: &'a BucketBuffer, key: u32) -> Self {
        let key = key as usize;
        if key >= buf.max_buckets as usize || buf.data.is_empty() {
            return BucketIter {
                slice: &[],
                index: 0,
            };
        }
        let start = buf.bucket_starts[key] as usize;
        let end = buf.bucket_starts[key + 1] as usize;
        if start >= end {
            return BucketIter {
                slice: &[],
                index: 0,
            };
        }
        // SAFETY: [start, end) is a valid sub-range of sorted data.
        let full: &'a [M] = unsafe { buf.data.as_slice() };
        BucketIter {
            slice: &full[start..end],
            index: 0,
        }
    }

    /// Empty iterator.
    pub(crate) fn empty() -> Self {
        BucketIter {
            slice: &[],
            index: 0,
        }
    }
}

impl<'a, M: Copy> Iterator for BucketIter<'a, M> {
    type Item = M;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.slice.len() {
            let item = self.slice[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.slice.len() - self.index;
        (r, Some(r))
    }
}

impl<'a, M: Copy> ExactSizeIterator for BucketIter<'a, M> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::error::ECSError;

    #[derive(Clone, Copy)]
    struct TestMsg {
        _value: u32,
    }

    #[test]
    fn missing_bucket_accessor_returns_error() {
        let mut raw = AlignedBuffer::with_capacity(
            std::mem::size_of::<TestMsg>(),
            std::mem::align_of::<TestMsg>(),
            1,
        );
        unsafe { raw.push(TestMsg { _value: 1 }) };
        let mut buf = BucketBuffer::new(
            std::mem::size_of::<TestMsg>(),
            std::mem::align_of::<TestMsg>(),
            2,
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
                specialisation: "Bucket",
                function: "bucket_key"
            })
        ));
    }
}
