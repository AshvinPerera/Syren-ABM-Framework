//! Named, typed agent-class definitions.
//!
//! ## Design
//!
//! [`AgentTemplate`] is the central descriptor for a class of agents. It records:
//!
//! * a human-readable name,
//! * the [`Signature`] of components every agent of this class carries,
//! * a default-value factory for every component in the signature,
//! * optional lifecycle [`SpawnHook`] and [`DespawnHook`] callbacks.
//!
//! Templates are constructed with [`AgentTemplateBuilder`] (via
//! [`AgentTemplate::builder`]) and registered in an [`AgentRegistry`] before
//! simulation start. After registration the template is treated as immutable.
//!
//! ## No new storage
//!
//! `AgentTemplate` adds no ECS storage of its own. Every agent it describes
//! is a plain ECS entity living in the archetype determined by the template's
//! [`Signature`].

use std::any::Any;
use std::collections::HashMap;

use crate::engine::component::Signature;
use crate::engine::error::ECSResult;
use crate::engine::manager::ECSReference;
use crate::engine::types::ComponentID;
use crate::Entity;

use super::error::{AgentError, AgentResult};
use super::hooks::{DespawnHook, SpawnHook};

/// Factory closure that produces a heap-allocated default component value.
///
/// Stored per-[`ComponentID`] inside [`AgentTemplate`]. Invoked during spawn
/// to fill in component slots that the caller did not explicitly override.
///
/// The returned [`Box<dyn Any + Send>`] must contain the concrete component
/// type registered for the corresponding [`ComponentID`].
pub type DefaultFactory = Box<dyn Fn() -> Box<dyn Any + Send> + Send + Sync>;

/// Named, typed descriptor for a class of agents.
///
/// Registered once at world setup via
/// [`AgentRegistry::register`](super::registry::AgentRegistry::register).
/// After registration the template is treated as read-only.
///
/// # Invariants
///
/// * Every [`ComponentID`] set in `signature` has a corresponding entry in
///   `defaults`.
/// * `name` is unique within an [`AgentRegistry`] (enforced by the registry).
pub struct AgentTemplate {
    pub(crate) name: String,
    pub(crate) signature: Signature,
    /// Maps each `ComponentID` in `signature` to a factory that produces the
    /// default component value for that slot.
    pub(crate) defaults: HashMap<ComponentID, DefaultFactory>,
    pub(crate) on_spawn: Option<SpawnHook>,
    pub(crate) on_despawn: Option<DespawnHook>,
}

impl AgentTemplate {
    /// Returns a builder for a new template with the given name.
    ///
    /// Call [`AgentTemplateBuilder::build`] to finalise.
    pub fn builder(name: impl Into<String>) -> AgentTemplateBuilder {
        AgentTemplateBuilder {
            name: name.into(),
            signature: Signature::default(),
            defaults: HashMap::new(),
            on_spawn: None,
            on_despawn: None,
        }
    }

    /// Returns an [`AgentSpawner`](super::spawner::AgentSpawner) seeded with
    /// this template's defaults.
    ///
    /// The spawner accepts per-instance overrides before issuing a deferred
    /// `Command::Spawn`.
    pub fn spawner(&self) -> super::spawner::AgentSpawner<'_> {
        super::spawner::AgentSpawner::new(self)
    }

    /// Enqueues a tagged despawn for an entity that belongs to this template.
    ///
    /// The entity is removed when deferred commands are drained. If this
    /// template has an `on_despawn` hook, the model invokes it after the drain
    /// reports that the tagged despawn was applied.
    pub fn despawn(&self, ecs: ECSReference<'_>, entity: Entity) -> ECSResult<()> {
        ecs.defer(crate::engine::commands::Command::DespawnTagged {
            entity,
            tag: self.name.clone(),
        })
    }

    /// Returns the template's human-readable name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the [`Signature`] describing every component an agent of this
    /// class carries.
    #[inline]
    pub fn signature(&self) -> &Signature {
        &self.signature
    }

    /// Returns `true` if `component_id` is part of this template's signature.
    #[inline]
    pub fn has_component(&self, component_id: ComponentID) -> bool {
        self.signature.has(component_id)
    }

    /// Returns the spawn hook, if one was registered.
    #[inline]
    pub fn on_spawn(&self) -> Option<&SpawnHook> {
        self.on_spawn.as_ref()
    }

    /// Returns the despawn hook, if one was registered.
    #[inline]
    pub fn on_despawn(&self) -> Option<&DespawnHook> {
        self.on_despawn.as_ref()
    }
}

// Manual Debug - DefaultFactory is not Debug.
impl std::fmt::Debug for AgentTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTemplate")
            .field("name", &self.name)
            .field("component_count", &self.defaults.len())
            .finish_non_exhaustive()
    }
}

// -- Builder ------------------------------------------------------------------

/// Fluent constructor for [`AgentTemplate`].
///
/// Obtain via [`AgentTemplate::builder`].
pub struct AgentTemplateBuilder {
    name: String,
    signature: Signature,
    defaults: HashMap<ComponentID, DefaultFactory>,
    on_spawn: Option<SpawnHook>,
    on_despawn: Option<DespawnHook>,
}

impl AgentTemplateBuilder {
    /// Registers a component in the template's signature using the type's
    /// [`Default`] implementation as the factory.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::DuplicateComponent`] if `id` is already in the
    /// signature.
    pub fn with_component<T>(mut self, id: ComponentID) -> AgentResult<Self>
    where
        T: Any + Default + Send + 'static,
    {
        if self.signature.has(id) {
            return Err(AgentError::DuplicateComponent(id));
        }
        self.signature.set(id);
        self.defaults.insert(
            id,
            Box::new(|| Box::new(T::default()) as Box<dyn Any + Send>),
        );
        Ok(self)
    }

    /// Registers a component with an explicit factory closure.
    ///
    /// Use this when the component's meaningful default differs from its
    /// [`Default`] implementation, or when [`Default`] is not implemented.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::DuplicateComponent`] if `id` is already in the
    /// signature.
    pub fn with_component_factory(
        mut self,
        id: ComponentID,
        factory: DefaultFactory,
    ) -> AgentResult<Self> {
        if self.signature.has(id) {
            return Err(AgentError::DuplicateComponent(id));
        }
        self.signature.set(id);
        self.defaults.insert(id, factory);
        Ok(self)
    }

    /// Attaches a [`SpawnHook`] invoked after each agent of this class is
    /// resolved from a `Command::Spawn`.
    pub fn on_spawn(mut self, hook: SpawnHook) -> Self {
        self.on_spawn = Some(hook);
        self
    }

    /// Attaches a [`DespawnHook`] invoked before each agent of this class is
    /// despawned.
    pub fn on_despawn(mut self, hook: DespawnHook) -> Self {
        self.on_despawn = Some(hook);
        self
    }

    /// Consumes the builder and returns the completed [`AgentTemplate`].
    pub fn build(self) -> AgentTemplate {
        AgentTemplate {
            name: self.name,
            signature: self.signature,
            defaults: self.defaults,
            on_spawn: self.on_spawn,
            on_despawn: self.on_despawn,
        }
    }
}

impl std::fmt::Debug for AgentTemplateBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTemplateBuilder")
            .field("name", &self.name)
            .field("component_count", &self.defaults.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default, Clone)]
    struct Health(f32);

    #[derive(Default, Clone)]
    struct Wealth {
        _value: f32,
    }

    #[test]
    fn builder_registers_components() {
        let tmpl = AgentTemplate::builder("Sheep")
            .with_component::<Health>(0)
            .unwrap()
            .with_component::<Wealth>(1)
            .unwrap()
            .build();

        assert_eq!(tmpl.name(), "Sheep");
        assert!(tmpl.has_component(0));
        assert!(tmpl.has_component(1));
        assert!(!tmpl.has_component(2));
        assert_eq!(tmpl.defaults.len(), 2);
    }

    #[test]
    fn duplicate_component_is_rejected() {
        let res = AgentTemplate::builder("X")
            .with_component::<Health>(0)
            .unwrap()
            .with_component::<Health>(0);
        assert_eq!(res.unwrap_err(), AgentError::DuplicateComponent(0));
    }

    #[test]
    fn default_factory_produces_value_of_correct_type() {
        let tmpl = AgentTemplate::builder("A")
            .with_component::<Health>(0)
            .unwrap()
            .build();
        let boxed = (tmpl.defaults[&0])();
        // Must downcast back to Health.
        assert!(boxed.downcast::<Health>().is_ok());
    }

    #[test]
    fn explicit_factory_is_stored() {
        let tmpl = AgentTemplate::builder("B")
            .with_component_factory(
                7,
                Box::new(|| Box::new(Health(42.0)) as Box<dyn Any + Send>),
            )
            .unwrap()
            .build();
        assert!(tmpl.has_component(7));
        let boxed = (tmpl.defaults[&7])();
        let h = boxed.downcast::<Health>().unwrap();
        assert!((h.0 - 42.0).abs() < f32::EPSILON);
    }

    #[test]
    fn spawner_is_created_from_template() {
        let tmpl = AgentTemplate::builder("C")
            .with_component::<Health>(0)
            .unwrap()
            .build();
        // Just check it constructs without panicking.
        let _ = tmpl.spawner();
    }
}
