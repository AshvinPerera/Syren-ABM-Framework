//! Core message traits and typed message handles.

use crate::engine::entity::Entity;
use crate::engine::types::ChannelID;
use std::marker::PhantomData;

// -----------------------------------------------------------------------------
// Base trait
// -----------------------------------------------------------------------------

/// A value that can be sent as a message between agents.
///
/// Messages must be:
/// - [`Copy`] - emitted by value, stored in a plain byte buffer.
/// - [`Send`] + [`Sync`] - safe to produce on one thread and consume on
///   another.
/// - `'static` - no borrowed data; message lifetimes are not tracked.
///
/// The optional [`GPU_SAFE`](Message::GPU_SAFE) associated constant signals
/// that the message's memory layout is compatible with GPU buffers (i.e. the
/// type is `#[repr(C)]` with only primitive fields).  It is `false` by
/// default.
pub trait Message: Copy + Send + Sync + 'static {
    /// Whether this message type's layout is safe to copy into a GPU buffer.
    const GPU_SAFE: bool = false;
}

// -----------------------------------------------------------------------------
// Specialisation traits
// -----------------------------------------------------------------------------

/// A [`Message`] that is consumed by iterating over *all* messages of this
/// type in arrival order (linear scan, no index).
///
/// This is the simplest and most cache-friendly specialisation when every
/// consumer needs to see every message.
pub trait BruteForceMessage: Message {}

/// A [`Message`] that is sorted into discrete integer *buckets*, allowing
/// consumers to iterate only messages for a specific key in O(bucket_size).
///
/// The key returned by [`BucketMessage::bucket_key`] must be in
/// `[0, max_buckets)` where `max_buckets` is set at registration time.
pub trait BucketMessage: Message {
    /// Returns the bucket this message belongs to.
    fn bucket_key(&self) -> u32;
}

/// A [`Message`] that can be located in 2-D world space, enabling
/// radius-based spatial queries in O(cells_in_radius * occupancy).
///
/// World coordinates returned by [`SpatialMessage::position`] must lie
/// within the grid bounds declared at registration time; messages outside
/// the grid are clamped to the nearest cell.
pub trait SpatialMessage: Message {
    /// Returns the (x, y) world position of this message.
    fn position(&self) -> (f32, f32);
}

/// A [`Message`] directed to a specific [`Entity`] recipient, enabling
/// O(1) inbox lookup per entity.
///
/// The recipient returned by [`TargetedMessage::recipient`] must be the
/// full packed [`Entity`] value (shard + index + version); keying on the
/// index alone is incorrect in multi-shard simulations.
pub trait TargetedMessage: Message {
    /// Returns the intended recipient entity.
    fn recipient(&self) -> Entity;
}

// -----------------------------------------------------------------------------
// Opaque identifier
// -----------------------------------------------------------------------------

/// A typed handle to a registered message channel.
///
/// Systems store this handle and use [`channel_id`](Self::channel_id) in their
/// [`AccessSets`](crate::AccessSets) declarations. Runtime emission and reads
/// pass the same handle back to [`MessageBufferSet`](crate::messaging::MessageBufferSet),
/// which keeps the raw type index private to the messaging module.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct MessageHandle<M: Message> {
    pub(crate) message_type_id: MessageTypeID,
    pub(crate) registry_id: u64,
    channel_id: ChannelID,
    _marker: PhantomData<fn() -> M>,
}

impl<M: Message> MessageHandle<M> {
    pub(crate) fn new(
        message_type_id: MessageTypeID,
        registry_id: u64,
        channel_id: ChannelID,
    ) -> Self {
        Self {
            message_type_id,
            registry_id,
            channel_id,
            _marker: PhantomData,
        }
    }

    /// Returns the scheduler channel associated with this message type.
    #[inline]
    pub fn channel_id(self) -> ChannelID {
        self.channel_id
    }
}

impl<M: Message> Clone for MessageHandle<M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M: Message> Copy for MessageHandle<M> {}

/// An opaque, dense index assigned to each message type at registration time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MessageTypeID(pub(crate) u32);

impl MessageTypeID {
    /// Returns the raw index.  This is `pub(crate)` so external users cannot
    /// construct arbitrary IDs.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
