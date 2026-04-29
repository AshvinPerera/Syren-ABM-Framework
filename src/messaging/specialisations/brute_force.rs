//! [`BruteForceBuffer`] - a flat, unsorted message store.
//!
//! Every consumer iterates all messages in arrival order.  This is the
//! simplest specialisation and has the best cache behaviour when the
//! consumer-to-message ratio is high (e.g. broadcast notifications).

use crate::messaging::aligned_buffer::AlignedBuffer;
use crate::messaging::message::Message;

// -----------------------------------------------------------------------------
// Buffer
// -----------------------------------------------------------------------------

/// Stores all messages for a `BruteForce`-specialised type.
pub(crate) struct BruteForceBuffer {
    /// Raw message storage.
    pub(crate) data: AlignedBuffer,
    /// Set to `true` after [`finalise`](BruteForceBuffer::finalise) is called,
    /// i.e. after all worker buffers have been drained into `data`.
    pub(crate) finalised: bool,
}

impl BruteForceBuffer {
    /// Creates a new buffer pre-allocated for `capacity` messages.
    pub(crate) fn new(item_size: usize, item_align: usize, capacity: usize) -> Self {
        BruteForceBuffer {
            data: AlignedBuffer::with_capacity(item_size, item_align, capacity),
            finalised: false,
        }
    }

    /// Prepares the buffer for a new tick: clears stored messages and resets
    /// the finalised flag.
    pub(crate) fn begin_tick(&mut self) {
        self.data.clear();
        self.finalised = false;
    }

    /// Marks the buffer as finalised after all worker drains have been
    /// completed.  Must be called before any consumer reads.
    pub(crate) fn finalise(&mut self) {
        self.finalised = true;
    }
}

// -----------------------------------------------------------------------------
// Iterator
// -----------------------------------------------------------------------------

/// An iterator over all messages in a `BruteForceBuffer`.
///
/// Produced by [`MessageBufferSet::brute_force`](crate::messaging::MessageBufferSet::brute_force).
pub struct BruteForceIter<'a, M> {
    slice: &'a [M],
    index: usize,
}

impl<'a, M: Message> BruteForceIter<'a, M> {
    pub(crate) fn new(buf: &'a BruteForceBuffer) -> Self {
        // SAFETY: M matches the type for which the buffer was created (enforced
        // by the buffer-set access methods).
        let slice: &'a [M] = if buf.data.is_empty() {
            &[]
        } else {
            unsafe { buf.data.as_slice::<M>() }
        };
        BruteForceIter { slice, index: 0 }
    }

    /// Returns an empty iterator (used when the buffer does not exist).
    pub(crate) fn empty() -> Self {
        BruteForceIter {
            slice: &[],
            index: 0,
        }
    }
}

impl<'a, M: Copy> Iterator for BruteForceIter<'a, M> {
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
        let remaining = self.slice.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, M: Copy> ExactSizeIterator for BruteForceIter<'a, M> {}
