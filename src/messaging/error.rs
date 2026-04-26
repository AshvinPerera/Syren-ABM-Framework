//! Error types for the messaging module.

use crate::engine::types::ChannelID;

/// Errors that can occur during messaging registration or runtime use.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MessagingError {
    /// A message type with the same [`TypeId`] has already been registered.
    #[error("message type already registered: {0}")]
    AlreadyRegistered(&'static str),

    /// Attempted to register a new message type after the registry was frozen.
    #[error("cannot register after message registry is frozen")]
    RegistryFrozen,

    /// Attempted to look up or access a message type that was never registered.
    #[error("message type not registered: {0}")]
    NotRegistered(&'static str),

    /// The caller requested a specialisation that does not match what was
    /// registered for this message type.
    #[error("wrong specialisation: requested {requested} but type is registered as {actual}")]
    WrongSpecialisation {
        /// The specialisation the caller asked for.
        requested: &'static str,
        /// The specialisation the type was registered under.
        actual: &'static str,
    },

    /// A per-tick emit capacity limit was exceeded for the given channel.
    #[error("emit capacity exceeded for channel {channel_id:?}: len={len}, max={max}")]
    EmitCapacityExceeded {
        /// Channel whose per-tick hard cap was exceeded.
        channel_id: ChannelID,
        /// Number of emitted messages.
        len: usize,
        /// Configured hard cap.
        max: usize,
    },

    /// Bucket registration used an invalid bucket count.
    #[error("bucket config invalid: max_buckets must be greater than zero")]
    InvalidBucketConfig,

    /// The spatial configuration provided is invalid (e.g. non-positive
    /// cell_size, or zero-area world).
    #[error("spatial config invalid: cell_size={cell_size}, width={width}, height={height}")]
    InvalidSpatialConfig {
        /// Size of a spatial cell in world units.
        cell_size: f32,
        /// World width in world units.
        width: f32,
        /// World height in world units.
        height: f32,
    },

    /// A bucket key was out of range for the registered `max_buckets` value.
    #[error("bucket key {key} out of range (max_buckets={max})")]
    BucketKeyOutOfRange {
        /// The bucket key that was out of range.
        key: u32,
        /// The maximum number of buckets registered.
        max: u32,
    },
}
