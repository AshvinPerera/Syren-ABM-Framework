//! Component registry errors and result types.
//!
//! This module defines [`RegistryError`], the error type returned by the global
//! component registry and its associated factories, along with the
//! [`RegistryResult`] convenience alias.
//!
//! # Error Variants
//!
//! | Variant | When it occurs |
//! |---|---|
//! | [`RegistryError::Frozen`] | A registration is attempted after the registry has been frozen |
//! | [`RegistryError::CapacityExceeded`] | The registry has run out of available [`ComponentID`]s |
//! | [`RegistryError::NotRegistered`] | A component type is looked up but was never registered |
//! | [`RegistryError::MissingFactory`] | A [`ComponentID`] exists in the registry but has no associated factory |
//! | [`RegistryError::ZeroSizedComponent`] | A zero-sized type is passed to registration |
//! | [`RegistryError::PoisonedLock`] | An internal registry or factory table [`Mutex`](std::sync::Mutex) was poisoned |

use std::any::TypeId;
use std::fmt;

use crate::engine::types::ComponentID;

/// Errors from the global component registry and its factories.
///
/// These cover:
/// - attempting to register after freeze
/// - exceeding the component ID capacity
/// - missing registrations / factories
/// - lock poisoning within the registry internals

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// Registry has been frozen; registrations are not allowed.
    Frozen,

    /// Component capacity exceeded (no more IDs available).
    CapacityExceeded {
        /// Capacity
        cap: usize,
    },

    /// A type was requested but never registered.
    NotRegistered {
        /// Type ID
        type_id: TypeId,
    },

    /// A factory for a registered component ID is missing.
    MissingFactory {
        /// Component ID of missing factory
        component_id: ComponentID,
    },

    /// A component ID exceeded the configured component bitset capacity.
    InvalidComponentId {
        /// Component ID supplied by the caller.
        component_id: ComponentID,
        /// Maximum number of component IDs supported by this build.
        cap: usize,
    },

    /// A component ID was in range but has not been registered in this registry.
    ComponentIdNotRegistered {
        /// Component ID supplied by the caller.
        component_id: ComponentID,
    },

    /// Zero sized component registered
    ZeroSizedComponent {
        /// type id of the zero sized component
        type_id: TypeId,
    },

    /// Registry or factory table lock was poisoned.
    PoisonedLock,
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::Frozen => f.write_str("component registry is frozen"),
            RegistryError::CapacityExceeded { cap } => {
                write!(f, "component registry capacity exceeded (cap {cap})")
            }
            RegistryError::NotRegistered { type_id } => {
                write!(f, "component type is not registered: {:?}", type_id)
            }
            RegistryError::MissingFactory { component_id } => write!(
                f,
                "missing component factory for component id {}",
                component_id
            ),
            RegistryError::InvalidComponentId { component_id, cap } => write!(
                f,
                "invalid component id {}; valid component ids are < {}",
                component_id, cap
            ),
            RegistryError::ComponentIdNotRegistered { component_id } => {
                write!(f, "component id {} is not registered", component_id)
            }
            RegistryError::ZeroSizedComponent { type_id } => {
                write!(f, "zero-sized component is not allowed: {:?}", type_id)
            }
            RegistryError::PoisonedLock => f.write_str("component registry lock poisoned"),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Result type used by the component registry and factories.
pub type RegistryResult<T> = Result<T, RegistryError>;
