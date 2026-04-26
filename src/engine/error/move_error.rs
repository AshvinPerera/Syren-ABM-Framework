//! Error types for archetype-to-archetype entity migration.
//!
//! This module defines [`MoveError`], which is returned by archetype migration
//! logic when transferring component rows between archetypes during add/remove
//! component operations.
//!
//! These errors typically reflect internal invariant violations — storage
//! misalignment, inconsistent swap metadata, or failed component transfers —
//! rather than recoverable user-facing conditions.

use std::fmt;

use crate::engine::types::{ArchetypeID, ChunkID, ComponentID, EntityID, RowID};

use super::attribute::AttributeError;

/// Errors that can occur while moving an entity between archetypes.
///
/// ## Context
/// `MoveError` is used by archetype migration logic when transferring
/// component rows between archetypes during add/remove operations.
///
/// ## Notes
/// These errors generally indicate internal inconsistencies or violated
/// invariants rather than recoverable user-facing failures.

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveError {
    /// Component storage layouts were inconsistent between archetypes.
    InconsistentStorage,

    /// Failed to move component data from the source archetype.
    PushFromFailed {
        /// Component being transferred.
        component_id: ComponentID,

        /// Underlying attribute error.
        source_error: AttributeError,
    },

    /// Component columns disagreed on the destination row.
    RowMisalignment {
        /// Expected `(chunk, row)` position.
        expected: (ChunkID, RowID),

        /// Actual `(chunk, row)` encountered.
        got: (ChunkID, RowID),

        /// Component whose storage was misaligned.
        component_id: ComponentID,
    },

    /// No components were transferred during the move.
    NoComponentsMoved,

    /// The destination signature would not change because the component exists already.
    ComponentAlreadyPresent {
        /// Component already present on the source entity.
        component_id: ComponentID,
    },

    /// Failed while inserting component data into the destination archetype.
    PushFailed {
        /// Component being inserted.
        component_id: ComponentID,

        /// Underlying attribute error.
        source_error: AttributeError,
    },

    /// Failed while removing component data from the source archetype.
    SwapRemoveError {
        /// Component being removed.
        component_id: ComponentID,

        /// Underlying attribute error.
        source_error: AttributeError,
    },

    /// Swap-remove operations yielded inconsistent metadata.
    InconsistentSwapInfo,

    /// Entity metadata could not be updated consistently after the move.
    MetadataFailure {
        /// The entity being moved, if known.
        entity: Option<EntityID>,

        /// The archetype the entity was being moved from, if known.
        source_archetype: Option<ArchetypeID>,

        /// The archetype the entity was being moved into, if known.
        destination_archetype: Option<ArchetypeID>,
    },
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MoveError::InconsistentStorage => {
                f.write_str("component storage layouts are inconsistent between archetypes")
            }

            MoveError::PushFromFailed {
                component_id,
                source_error,
            } => {
                write!(
                    f,
                    "failed to move component {} from source archetype: {}",
                    component_id, source_error
                )
            }

            MoveError::RowMisalignment {
                expected,
                got,
                component_id,
            } => {
                write!(
                    f,
                    "component {} storage misaligned: expected position {:?}, got {:?}",
                    component_id, expected, got
                )
            }

            MoveError::NoComponentsMoved => {
                f.write_str("no components were moved during archetype transition")
            }

            MoveError::ComponentAlreadyPresent { component_id } => {
                write!(f, "component {} is already present", component_id)
            }

            MoveError::PushFailed {
                component_id,
                source_error,
            } => {
                write!(
                    f,
                    "failed to insert component {} into destination archetype: {}",
                    component_id, source_error
                )
            }

            MoveError::SwapRemoveError {
                component_id,
                source_error,
            } => {
                write!(
                    f,
                    "failed to remove component {} from source archetype: {}",
                    component_id, source_error
                )
            }

            MoveError::InconsistentSwapInfo => {
                f.write_str("swap-remove produced inconsistent metadata")
            }

            MoveError::MetadataFailure {
                entity,
                source_archetype,
                destination_archetype,
            } => {
                write!(f, "failed to update entity metadata after archetype move")?;
                if let Some(e) = entity {
                    write!(f, " (entity: {})", e)?;
                }
                if let Some(src) = source_archetype {
                    write!(f, " (source archetype: {})", src)?;
                }
                if let Some(dst) = destination_archetype {
                    write!(f, " (destination archetype: {})", dst)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for MoveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MoveError::PushFromFailed { source_error, .. } => Some(source_error),
            MoveError::PushFailed { source_error, .. } => Some(source_error),
            MoveError::SwapRemoveError { source_error, .. } => Some(source_error),
            _ => None,
        }
    }
}
