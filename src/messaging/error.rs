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

    /// Attempted to construct runtime buffers before freezing the registry.
    #[error("message registry must be frozen before creating a MessageBufferSet")]
    RegistryNotFrozen,

    /// Attempted to look up or access a message type that was never registered.
    #[error("message type not registered: {0}")]
    NotRegistered(&'static str),

    /// A handle from one registry/runtime was used with another buffer set.
    #[error("message handle belongs to a different registry")]
    WrongRegistry,

    /// A messaging synchronization primitive was poisoned.
    #[error("messaging lock poisoned: {0}")]
    LockPoisoned(&'static str),

    /// Channel ID allocation exhausted the shared channel namespace.
    #[error("channel ID allocation overflowed")]
    ChannelAllocationOverflow,

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

    /// GPU message metadata does not match the payload layout.
    #[cfg(feature = "messaging_gpu")]
    #[error("gpu message layout invalid for {type_name}: {reason}")]
    InvalidGpuMessageLayout {
        /// Message type name.
        type_name: &'static str,
        /// Human-readable validation failure.
        reason: String,
    },

    /// GPU message emission exceeded the registered capacity.
    #[cfg(feature = "messaging_gpu")]
    #[error("gpu emit capacity exceeded for {resource}: len={len}, max={max}")]
    GpuEmitCapacityExceeded {
        /// Resource name.
        resource: String,
        /// Attempted or merged message count.
        len: usize,
        /// Configured hard capacity.
        max: usize,
    },

    /// GPU message finalisation reported invalid data.
    #[cfg(feature = "messaging_gpu")]
    #[error("gpu message finalise failed for {resource}: {reason}")]
    GpuFinaliseFailed {
        /// Resource name.
        resource: String,
        /// Human-readable failure reason.
        reason: String,
    },
}
