//! Typed wrapper around [`Entity`] for single-entity CPU component access.
//!
//! ## Warning
//!
//! [`AgentHandle`] is for **single-entity CPU lookups only** — for example,
//! a market-clearing step that needs to inspect one specific agent.
//!
//! ## Read implementation
//!
//! [`AgentHandle::read`] delegates to
//! [`ECSReference::read_entity_component`], which performs a point read
//! under a shared phase lock. This works both during and outside of
//! iteration because it acquires only the column-level `RwLock` (the same
//! lock that `for_each` uses for concurrent readers) and does not require
//! exclusive access to `ECSData`.
//!
//! ## Write implementation
//!
//! [`AgentHandle::write`] enqueues a deferred `Command::Add` via
//! [`ECSReference::defer`]. Write is applied at the next
//! `apply_deferred_commands` boundary — it is not an immediate in-place
//! mutation.

use crate::engine::entity::Entity;
use crate::engine::types::ComponentID;
use crate::engine::manager::ECSReference;
use crate::engine::commands::Command;

use super::error::{AgentError};

/// A typed wrapper around [`Entity`] for single-agent component access.
///
/// # Warning
///
/// This type is for **single-entity CPU lookups only** (e.g. market-clearing
/// logic that inspects one specific agent). Never use in hot iteration paths —
/// use `ECSReference::for_each_*` instead.
///
/// For reading other agents' components *during* iteration (e.g. social
/// network lookups), use [`ECSReference::read_entity_component`] directly
/// with the stored `Entity` handle. `AgentHandle` adds a component-membership
/// guard on top of that primitive.
///
/// # Obtaining a handle
///
/// Handles are created by the model layer after a spawn command has been
/// resolved and the entity handle is available (returned by
/// `apply_deferred_commands`):
///
/// ```ignore
/// let ids: Vec<ComponentID> = template.signature()
///     .iterate_over_components()
///     .collect();
/// let handle = AgentHandle::new(entity, ids);
/// ```
pub struct AgentHandle {
    entity: Entity,
    /// Component IDs from the template's signature, cached for membership
    /// checks without re-deriving the signature on every access.
    component_ids: Vec<ComponentID>,
}

impl AgentHandle {
    /// Creates a new handle for `entity`.
    ///
    /// `component_ids` should be the [`ComponentID`]s declared in the agent
    /// template's [`Signature`](crate::engine::component::Signature).
    pub fn new(entity: Entity, component_ids: Vec<ComponentID>) -> Self {
        Self { entity, component_ids }
    }

    /// Returns the underlying [`Entity`] handle.
    #[inline]
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// Returns `true` if `id` is in the template signature cached by this handle.
    #[inline]
    pub fn has_component(&self, id: ComponentID) -> bool {
        self.component_ids.contains(&id)
    }

    /// Reads a component value for this entity by [`ComponentID`].
    ///
    /// Delegates to [`ECSReference::read_entity_component`] after verifying
    /// that `id` is part of this handle's template signature.
    ///
    /// # Errors
    ///
    /// * [`AgentHandleError::Agent(AgentError::MissingComponent)`] — `id` not
    ///   in template signature.
    /// * [`AgentHandleError::ECS`] — entity is stale, component is missing
    ///   from the archetype, the column is currently write-locked by the
    ///   calling system, the requested type does not match the stored type,
    ///   or a lock is poisoned.
    pub fn read<T: 'static + Clone>(
        &self,
        id: ComponentID,
        ecs: ECSReference<'_>,
    ) -> Result<T, AgentHandleError> {
        if !self.component_ids.contains(&id) {
            return Err(AgentHandleError::Agent(AgentError::MissingComponent(id)));
        }
        ecs.read_entity_component::<T>(self.entity, id)
            .map_err(AgentHandleError::ECS)
    }

    /// Enqueues a deferred write of `value` to component `id` for this entity.
    ///
    /// The write is applied at the next `apply_deferred_commands` boundary via
    /// `Command::Add`. This does **not** perform an immediate in-place write.
    ///
    /// # Errors
    ///
    /// * [`AgentHandleError::Agent(AgentError::MissingComponent)`] — `id` is
    ///   not in the template signature cached by this handle.
    /// * [`AgentHandleError::ECS`] — the deferred command queue lock is
    ///   poisoned.
    pub fn write<T: 'static + Send>(
        &self,
        id: ComponentID,
        value: T,
        ecs: ECSReference<'_>,
    ) -> Result<(), AgentHandleError> {
        if !self.component_ids.contains(&id) {
            return Err(AgentHandleError::Agent(AgentError::MissingComponent(id)));
        }
        ecs.defer(Command::Add {
            entity: self.entity,
            component_id: id,
            value: Box::new(value),
        })
        .map_err(AgentHandleError::ECS)
    }
}

// ── Error type ───────────────────────────────────────────────────────────────

/// Combined error returned by [`AgentHandle::read`] and [`AgentHandle::write`].
///
/// Both methods can fail for two orthogonal reasons:
///
/// * An agent-domain invariant violation (wrong component ID, stale entity) →
///   [`AgentHandleError::Agent`].
/// * An ECS-level failure (lock poison, structural mutation during iteration)
///   → [`AgentHandleError::ECS`].
///
/// Collapsing both into `AgentError` would lose ECS diagnostic detail, so
/// they are kept as separate variants.
#[derive(Debug)]
pub enum AgentHandleError {
    /// An agent-domain invariant was violated.
    Agent(AgentError),
    /// An ECS-level failure occurred (lock poison, scope violation, etc.).
    ECS(crate::engine::error::ECSError),
}

impl std::fmt::Display for AgentHandleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentHandleError::Agent(e) => write!(f, "agent error: {e}"),
            AgentHandleError::ECS(e) => write!(f, "ecs error: {e}"),
        }
    }
}

impl std::error::Error for AgentHandleError {}

impl From<AgentError> for AgentHandleError {
    fn from(e: AgentError) -> Self {
        AgentHandleError::Agent(e)
    }
}

impl From<crate::engine::error::ECSError> for AgentHandleError {
    fn from(e: crate::engine::error::ECSError) -> Self {
        AgentHandleError::ECS(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::entity::Entity;

    fn dummy_entity() -> Entity {
        Entity::from_raw(0u64)
    }

    #[test]
    fn has_component_returns_correct_values() {
        let handle = AgentHandle::new(dummy_entity(), vec![3, 7, 11]);
        assert!(handle.has_component(3));
        assert!(handle.has_component(7));
        assert!(handle.has_component(11));
        assert!(!handle.has_component(0));
        assert!(!handle.has_component(5));
    }

    #[test]
    fn missing_component_detected() {
        // The component-membership guard fires before any ECS access.
        // We can test it without a live ECSManager by checking
        // has_component directly.
        let handle = AgentHandle::new(dummy_entity(), vec![0, 1]);
        assert!(!handle.has_component(99));
        assert!(handle.has_component(0));
        assert!(handle.has_component(1));
    }

    #[test]
    fn entity_accessor_returns_correct_entity() {
        let e = Entity::from_raw(42u64);
        let handle = AgentHandle::new(e, vec![]);
        assert_eq!(handle.entity().to_raw(), 42u64);
    }
}
