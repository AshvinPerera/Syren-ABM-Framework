//! Central allocator for scheduler channel IDs.
//!
//! A [`ChannelAllocator`] hands out fresh [`ChannelID`]s from a single opaque
//! monotonic counter. One allocator exists per `Model` (owned by `ModelBuilder`
//! during construction). Extension modules that register channel-carrying
//! entities call [`alloc`](ChannelAllocator::alloc) at registration time to
//! obtain a unique channel ID, store it, and record it in
//! [`AccessSets::produces`](crate::engine::systems::AccessSets::produces) /
//! [`AccessSets::consumes`](crate::engine::systems::AccessSets::consumes) on
//! the systems that interact with it.
//!
//! ## Opaque IDs
//!
//! The engine never interprets a [`ChannelID`]'s meaning. It is purely a
//! scheduling tag that the stage packer uses to enforce producer-before-consumer
//! ordering. Messaging and environment share one ID space via one allocator so
//! the scheduler can reason about both uniformly.
//!
//! ## Overflow
//!
//! A [`ChannelID`] is `u32`. Overflow from calling [`alloc`] more than
//! `u32::MAX + 1` times panics via `checked_add`. This is not a practical
//! concern for any realistic simulation.

use crate::engine::types::ChannelID;


/// Monotonic, infallible allocator for [`ChannelID`]s.
///
/// One instance exists per `Model`, shared between messaging and environment
/// during construction so that all channel IDs live in one flat `u32` space.
/// After construction is complete, the allocator is no longer needed and may
/// be dropped or archived for diagnostics via [`peek_next`](ChannelAllocator::peek_next).
#[derive(Debug, Default)]
pub struct ChannelAllocator {
    next: ChannelID,
}

impl ChannelAllocator {
    /// Creates a new allocator starting at channel ID 0.
    pub fn new() -> Self {
        Self { next: 0 }
    }

    /// Allocates and returns the next available [`ChannelID`].
    ///
    /// # Panics
    ///
    /// Panics if more than `u32::MAX + 1` channels are allocated, indicating a
    /// programming error (no realistic simulation needs that many channels).
    pub fn alloc(&mut self) -> ChannelID {
        let id = self.next;
        self.next = self.next
            .checked_add(1)
            .expect("ChannelID overflow: more than u32::MAX channels allocated");
        id
    }

    /// Returns the next ID that *would* be allocated without consuming it.
    ///
    /// Useful for diagnostics and for sizing boundary-resource bookkeeping
    /// arrays at model freeze time (the total channel count equals `peek_next()`
    /// once all modules have registered their channels).
    pub fn peek_next(&self) -> ChannelID {
        self.next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_is_monotonic() {
        let mut a = ChannelAllocator::new();
        assert_eq!(a.alloc(), 0);
        assert_eq!(a.alloc(), 1);
        assert_eq!(a.alloc(), 2);
        assert_eq!(a.peek_next(), 3);
    }

    #[test]
    fn peek_does_not_advance() {
        let a = ChannelAllocator::new();
        assert_eq!(a.peek_next(), 0);
        assert_eq!(a.peek_next(), 0);
    }
}
