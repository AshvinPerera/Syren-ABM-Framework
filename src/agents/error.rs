//! Error types for the `agents` module.
//!
//! ## Design
//!
//! The `agents` module operates on top of the engine's public API where
//! agent-specific failures are reported as `Result<T, AgentError>` at all
//! public module boundaries.
//!
//! When results must cross into scheduler or manager APIs that require
//! [`crate::ECSResult`], callers can use the provided [`From<AgentError> for ECSError`]
//! implementation, which maps agent errors into
//! `ECSError::Execute(ExecutionError::SchedulerInvariantViolation)` - the
//! closest existing unit-variant that signals a module-level invariant
//! violation at runtime.
//!
//! ## Error boundary rule
//!
//! * Inside the `agents` module, all fallible methods return
//!   `Result<T, AgentError>`.
//! * When an `agents` result crosses a scheduler/system boundary, convert with
//!   `map_err(ECSError::from)?`.

use std::fmt;

use crate::engine::error::ECSError;
use crate::engine::types::{ComponentID, COMPONENT_CAP};

/// Errors specific to the `agents` domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentError {
    /// A named template was requested but not found in the [`crate::agents::AgentRegistry`].
    TemplateNotFound(String),

    /// A named template was registered more than once.
    DuplicateTemplate(String),

    /// A [`ComponentID`] was referenced that is not part of the template's
    /// declared [`crate::Signature`].
    MissingComponent(ComponentID),

    /// An [`crate::agents::AgentHandle`] was used after the underlying entity was despawned.
    StaleHandle,

    /// The template already has a component registered under this [`ComponentID`].
    DuplicateComponent(ComponentID),

    /// A [`ComponentID`] is outside the supported component bitset range.
    InvalidComponentId {
        /// Component ID supplied by the caller.
        component_id: ComponentID,
        /// Maximum number of component IDs supported by this build.
        cap: usize,
    },

    /// Batch spawning was requested from a template that has not been registered.
    UnregisteredTemplate(String),

    /// An attempt was made to register a template after the registry was sealed.
    RegistrySealed,

    /// A batch column length did not match the declared batch size.
    BatchLengthMismatch {
        /// Component column with the mismatch.
        component_id: ComponentID,
        /// Expected row count.
        expected: usize,
        /// Actual row count.
        actual: usize,
    },
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::TemplateNotFound(name) => write!(f, "agent template not found: {name}"),
            AgentError::DuplicateTemplate(name) => {
                write!(f, "agent template already registered: {name}")
            }
            AgentError::MissingComponent(id) => {
                write!(f, "component {id} is not in the template signature")
            }
            AgentError::StaleHandle => write!(f, "stale agent handle: entity no longer exists"),
            AgentError::DuplicateComponent(id) => {
                write!(f, "component {id} is already registered in this template")
            }
            AgentError::InvalidComponentId { component_id, cap } => write!(
                f,
                "invalid component id {component_id}; valid component ids are < {cap}"
            ),
            AgentError::UnregisteredTemplate(name) => {
                write!(f, "agent template '{name}' has not been registered")
            }
            AgentError::RegistrySealed => write!(
                f,
                "agent registry is sealed; no further templates may be registered"
            ),
            AgentError::BatchLengthMismatch {
                component_id,
                expected,
                actual,
            } => write!(
                f,
                "batch column {component_id} has length {actual}, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for AgentError {}

impl AgentError {
    pub(crate) fn invalid_component_id(component_id: ComponentID) -> Self {
        AgentError::InvalidComponentId {
            component_id,
            cap: COMPONENT_CAP,
        }
    }
}

/// Converts an [`AgentError`] into an [`ECSError`].
impl From<AgentError> for ECSError {
    fn from(e: AgentError) -> Self {
        ECSError::Agent(e)
    }
}

/// Convenience alias for agent-domain results.
pub type AgentResult<T> = Result<T, AgentError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_template_not_found() {
        let e = AgentError::TemplateNotFound("Sheep".into());
        assert!(e.to_string().contains("Sheep"));
    }

    #[test]
    fn display_missing_component() {
        let e = AgentError::MissingComponent(42);
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn display_stale_handle() {
        let e = AgentError::StaleHandle;
        assert!(e.to_string().contains("stale"));
    }

    #[test]
    fn display_duplicate_component() {
        let e = AgentError::DuplicateComponent(7);
        assert!(e.to_string().contains("7"));
    }

    #[test]
    fn display_invalid_component_id() {
        let e = AgentError::InvalidComponentId {
            component_id: COMPONENT_CAP as ComponentID,
            cap: COMPONENT_CAP,
        };
        assert!(e.to_string().contains(&COMPONENT_CAP.to_string()));
    }

    #[test]
    fn display_unregistered_template() {
        let e = AgentError::UnregisteredTemplate("Fox".into());
        assert!(e.to_string().contains("Fox"));
    }

    #[test]
    fn display_registry_sealed() {
        let e = AgentError::RegistrySealed;
        assert!(e.to_string().contains("sealed"));
    }

    #[test]
    fn into_ecs_error_does_not_panic() {
        let e = AgentError::TemplateNotFound("X".into());
        let ecs: ECSError = e.into();
        assert!(matches!(
            ecs,
            ECSError::Agent(AgentError::TemplateNotFound(_))
        ));
    }
}
