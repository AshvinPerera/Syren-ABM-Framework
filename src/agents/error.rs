//! Error types for the `agents` module.
//!
//! ## Design
//!
//! The `agents` module operates on top of the engine's public API and cannot
//! add variants to [`ECSError`]. Instead, agent-specific failures are reported
//! as `Result<T, AgentError>` at all public module boundaries.
//!
//! When results must cross into scheduler or manager APIs that require
//! [`ECSResult`], callers can use the provided [`From<AgentError> for ECSError`]
//! implementation, which maps agent errors into
//! `ECSError::Execute(ExecutionError::SchedulerInvariantViolation)` — the
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

use crate::engine::error::{ECSError, ExecutionError};
use crate::engine::types::ComponentID;

/// Errors specific to the `agents` domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentError {
    /// A named template was requested but not found in the [`AgentRegistry`].
    TemplateNotFound(String),

    /// A [`ComponentID`] was referenced that is not part of the template's
    /// declared [`Signature`].
    MissingComponent(ComponentID),

    /// An [`AgentHandle`] was used after the underlying entity was despawned.
    StaleHandle,

    /// The template already has a component registered under this [`ComponentID`].
    DuplicateComponent(ComponentID),

    /// An attempt was made to register a template after the registry was sealed.
    RegistrySealed,
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::TemplateNotFound(name) => write!(f, "agent template not found: {name}"),
            AgentError::MissingComponent(id) => {
                write!(f, "component {id} is not in the template signature")
            }
            AgentError::StaleHandle => write!(f, "stale agent handle: entity no longer exists"),
            AgentError::DuplicateComponent(id) => {
                write!(f, "component {id} is already registered in this template")
            }
            AgentError::RegistrySealed => write!(
                f,
                "agent registry is sealed; no further templates may be registered"
            ),
        }
    }
}

impl std::error::Error for AgentError {}

/// Converts an [`AgentError`] into an [`ECSError`].
///
/// Because [`ECSError`] has no generic agent-error variant (engine files cannot
/// be modified per the constraint), this maps through
/// `ExecutionError::SchedulerInvariantViolation`. Callers that need to
/// distinguish agent errors should match `AgentError` before converting.
impl From<AgentError> for ECSError {
    fn from(_e: AgentError) -> Self {
        ECSError::Execute(ExecutionError::SchedulerInvariantViolation)
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
            ECSError::Execute(ExecutionError::SchedulerInvariantViolation)
        ));
    }
}
