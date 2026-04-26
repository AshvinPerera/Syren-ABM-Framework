//! Typed, per-tick messaging for agent-based simulations.
//!
//! Message types are registered during model construction and return a
//! [`MessageHandle<M>`]. Systems declare the handle's channel in
//! [`AccessSets`](crate::AccessSets), then emit or read messages through the
//! model-owned [`MessageBufferSet`] boundary resource.

pub(crate) mod aligned_buffer;
pub(crate) mod buffer_set;
pub mod error;
pub(crate) mod message;
pub(crate) mod registry;
pub(crate) mod specialisations;
pub(crate) mod thread_local_emit;

pub use error::MessagingError;

pub use message::{
    BruteForceMessage, BucketMessage, Message, MessageHandle, SpatialMessage, TargetedMessage,
};

pub use registry::{Capacity, MessageDescriptor, MessageRegistry, SpatialConfig, Specialisation};

pub use buffer_set::MessageBufferSet;

pub use specialisations::{BruteForceIter, BucketIter, InboxIter, SpatialQueryIter};
