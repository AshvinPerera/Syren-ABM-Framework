//! Errors produced while building a [`Model`](crate::model::Model).

use crate::engine::error::ECSError;
use crate::engine::types::ChannelID;

/// Errors that can occur during model construction.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// ECS-level setup or scheduler validation failed.
    #[error(transparent)]
    Ecs(#[from] ECSError),

    /// Agent template registration failed.
    #[error(transparent)]
    Agent(#[from] crate::agents::AgentError),

    /// Message registration failed.
    #[cfg(feature = "messaging")]
    #[error(transparent)]
    Messaging(#[from] crate::messaging::MessagingError),

    /// Environment registration failed.
    #[error(transparent)]
    Environment(#[from] crate::environment::EnvironmentError),

    /// Default entity shards could not be created.
    #[error(transparent)]
    Spawn(#[from] crate::engine::error::SpawnError),

    /// A transient message channel crosses scheduler boundaries.
    #[error(
        "message channel {channel_id} crosses scheduler boundary from {producer_scope} to {consumer_scope}"
    )]
    CrossSchedulerMessageChannel {
        /// Channel that crossed scopes.
        channel_id: ChannelID,
        /// Scope producing the channel.
        producer_scope: String,
        /// Scope consuming the channel.
        consumer_scope: String,
    },

    /// An environment channel flows backward relative to model tick order.
    #[error(
        "environment channel {channel_id} flows backward from {producer_scope} to {consumer_scope}"
    )]
    BackwardEnvironmentChannel {
        /// Channel with invalid flow.
        channel_id: ChannelID,
        /// Scope producing the channel.
        producer_scope: String,
        /// Scope consuming the channel.
        consumer_scope: String,
    },

    /// Two sub-schedulers were registered with the same display name.
    #[error("duplicate sub-scheduler name: {name}")]
    DuplicateSubSchedulerName {
        /// Duplicate name.
        name: String,
    },

    /// A lock needed during build was poisoned.
    #[error("model build lock poisoned: {0}")]
    LockPoisoned(&'static str),
}
