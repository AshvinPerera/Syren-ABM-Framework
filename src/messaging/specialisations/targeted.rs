//! [`TargetedBuffer`] — an inbox index keyed on the full [`Entity`].
//!
//! Messages are sorted so that each entity's messages occupy a contiguous
//! slice, enabling O(1) inbox lookup after an O(n) scatter pass.
//!
//! # Key correctness
//!
//! The key is the full packed [`Entity`] value (shard + index + version bits),
//! *not* `entity.index()` alone.  In multi-shard simulations two entities from
//! different shards can have the same 22-bit index; keying only on the index
//! would collide their inboxes.

use std::collections::HashMap;
use std::ops::Range;

use crate::engine::entity::Entity;
use crate::messaging::aligned_buffer::AlignedBuffer;
use crate::messaging::message::Message;
use crate::messaging::registry::ErasedFns;

// ─────────────────────────────────────────────────────────────────────────────
// Buffer
// ─────────────────────────────────────────────────────────────────────────────

/// Stores all messages for a `Targeted`-specialised type, sorted by recipient.
pub(crate) struct TargetedBuffer {
    /// Sorted message storage (valid after [`finalise`](TargetedBuffer::finalise)).
    pub(crate) data: AlignedBuffer,
    /// Maps each recipient [`Entity`] to its contiguous slice in `data`.
    pub(crate) inbox_index: HashMap<Entity, Range<u32>>,
    item_size: usize,
}

impl TargetedBuffer {
    pub(crate) fn new(item_size: usize, item_align: usize, capacity: usize) -> Self {
        TargetedBuffer {
            data: AlignedBuffer::with_capacity(item_size, item_align, capacity),
            inbox_index: HashMap::new(),
            item_size,
        }
    }

    /// Clears for a new tick.
    pub(crate) fn begin_tick(&mut self) {
        self.data.clear();
        self.inbox_index.clear();
    }

    /// Drains `raw` into sorted `self.data`, building `inbox_index`.
    ///
    /// # Safety
    ///
    /// `fns.recipient` must be `Some` and must read items of the registered type.
    pub(crate) unsafe fn finalise(&mut self, raw: &AlignedBuffer, fns: &ErasedFns) {
        let n = raw.len();
        if n == 0 {
            return;
        }

        let recipient_fn = fns.recipient.expect("TargetedBuffer: missing recipient_fn");

        // ── 1. Count per-entity ───────────────────────────────────────────────
        // We use a deterministic two-pass approach to avoid per-entity Vec
        // allocations: first count, then prefix-sum into a sorted order.

        // First pass: collect (entity, original_index) pairs and count.
        let mut entity_pairs: Vec<Entity> = Vec::with_capacity(n);
        let mut counts: HashMap<Entity, u32> = HashMap::with_capacity(n / 4 + 1);

        for i in 0..n {
            let ptr = unsafe { raw.as_ptr_at(i) };
            let entity = unsafe { recipient_fn(ptr) };
            entity_pairs.push(entity);
            *counts.entry(entity).or_insert(0) += 1;
        }

        // ── 2. Assign contiguous ranges ───────────────────────────────────────
        // Sort entities for deterministic output (matches scheduler
        // reproducibility goals).
        let mut entities: Vec<Entity> = counts.keys().copied().collect();
        entities.sort_unstable_by_key(|e| e.to_raw());

        self.inbox_index.clear();
        let mut cursor: u32 = 0;
        for &entity in &entities {
            let cnt = counts[&entity];
            let start = cursor;
            cursor += cnt;
            self.inbox_index.insert(entity, start..cursor);
        }

        // ── 3. Scatter ────────────────────────────────────────────────────────
        self.data.reserve(n);
        unsafe { self.data.set_len(n) };

        // scatter_cursor[entity] = next write position for that entity.
        let mut scatter_cursor: HashMap<Entity, u32> = self
            .inbox_index
            .iter()
            .map(|(&e, r)| (e, r.start))
            .collect();

        for (i, &entity) in entity_pairs.iter().enumerate() {
            let src = unsafe { raw.as_ptr_at(i) };
            let dst_idx = *scatter_cursor
                .get(&entity)
                .expect("entity missing from scatter_cursor") as usize;
            let dst = unsafe { self.data.as_mut_ptr_at(dst_idx) };
            unsafe { std::ptr::copy_nonoverlapping(src, dst, self.item_size) };
            *scatter_cursor.get_mut(&entity).unwrap() += 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Iterator
// ─────────────────────────────────────────────────────────────────────────────

/// An iterator over all messages addressed to a specific [`Entity`].
///
/// Produced by [`MessageBufferSet::inbox`](crate::messaging::MessageBufferSet::inbox).
pub struct InboxIter<'a, M> {
    slice: &'a [M],
    index: usize,
}

impl<'a, M: Message> InboxIter<'a, M> {
    /// Creates an inbox iterator for `recipient`.
    ///
    /// # Safety
    ///
    /// `M` must be the type this buffer was created for.
    pub(crate) fn new(buf: &'a TargetedBuffer, recipient: Entity) -> Self {
        let range = match buf.inbox_index.get(&recipient) {
            Some(r) => r.clone(),
            None => {
                return InboxIter {
                    slice: &[],
                    index: 0,
                }
            }
        };
        if range.is_empty() || buf.data.is_empty() {
            return InboxIter {
                slice: &[],
                index: 0,
            };
        }
        let full: &'a [M] = unsafe { buf.data.as_slice() };
        let start = range.start as usize;
        let end = range.end as usize;
        InboxIter {
            slice: &full[start..end],
            index: 0,
        }
    }

    pub(crate) fn empty() -> Self {
        InboxIter {
            slice: &[],
            index: 0,
        }
    }
}

impl<'a, M: Copy> Iterator for InboxIter<'a, M> {
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

impl<'a, M: Copy> ExactSizeIterator for InboxIter<'a, M> {}
