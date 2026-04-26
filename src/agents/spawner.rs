//! Builder for spawning agents from a template with per-field overrides.
//!
//! ## Design
//!
//! [`AgentSpawner`] is a short-lived builder scoped to one spawn operation.
//! It starts from the defaults defined in an [`AgentTemplate`], lets the
//! caller override individual component values via [`AgentSpawner::set`], and
//! then emits a single tagged deferred spawn command via
//! [`AgentSpawner::spawn`].
//!
//! ## Bundle construction
//!
//! [`DefaultFactory`] produces `Box<dyn Any + Send>` containing the concrete
//! component value. These pre-boxed values are inserted into a [`Bundle`] via
//! [`Bundle::insert_boxed`], which stores them flat. This avoids the
//! double-boxing that would occur if we passed them through
//! [`Bundle::insert<T>`] (which would produce `Box<Box<dyn Any + Send>>`,
//! causing `push_dyn`'s `downcast::<T>()` to fail at runtime).
//!
//! ## Hook timing
//!
//! If the template has an `on_spawn` hook, it is **not** called inside
//! [`AgentSpawner::spawn`]. It must be invoked by the model layer (e.g.
//! `AgentRegistry::flush_spawn_hooks`) after `apply_deferred_commands` resolves
//! the entity handle and returns it to the caller.
//!
//! [`DefaultFactory`]: super::template::DefaultFactory
//! [`Bundle::insert_boxed`]: crate::engine::component::Bundle::insert_boxed

use std::any::Any;

use crate::engine::commands::Command;
use crate::engine::component::Bundle;
use crate::engine::error::ECSResult;
use crate::engine::manager::ECSReference;
use crate::engine::types::ComponentID;

use super::error::{AgentError, AgentResult};
use super::template::AgentTemplate;

// ── AgentSpawner ─────────────────────────────────────────────────────────────

/// Builder for spawning one agent from a template with per-field overrides.
///
/// Created via [`AgentTemplate::spawner`]. Override specific component values
/// with [`AgentSpawner::set`], then call [`AgentSpawner::spawn`] to enqueue a
/// deferred tagged spawn.
pub struct AgentSpawner<'t> {
    template: &'t AgentTemplate,
    overrides: Vec<(ComponentID, Box<dyn Any + Send>)>,
}

impl<'t> AgentSpawner<'t> {
    /// Creates a spawner seeded with `template`'s defaults.
    pub fn new(template: &'t AgentTemplate) -> Self {
        Self {
            template,
            overrides: Vec::new(),
        }
    }

    /// Overrides the value for component `id` in this spawn instance.
    ///
    /// If `set` is called multiple times with the same `id`, the last value
    /// wins.
    pub fn set<T: Any + Send + 'static>(mut self, id: ComponentID, value: T) -> AgentResult<Self> {
        if !self.template.signature.has(id) {
            return Err(AgentError::MissingComponent(id));
        }
        self.overrides.retain(|(cid, _)| *cid != id);
        self.overrides.push((id, Box::new(value)));
        Ok(self)
    }

    /// Enqueues a tagged spawn command that materialises the agent.
    ///
    /// # Algorithm
    ///
    /// 1. Build a [`Bundle`] from the template's default factories using
    ///    [`Bundle::insert_boxed`] to avoid double-boxing.
    /// 2. Apply per-instance overrides (last write wins).
    /// 3. Defer a tagged spawn via [`ECSReference::defer`].
    ///
    /// The entity is not created immediately. It is materialised when
    /// `apply_deferred_commands` processes the command queue at the next
    /// sync boundary. The returned `Vec<Entity>` from that call provides
    /// the resolved entity handle.
    ///
    /// # Errors
    ///
    /// Propagates lock-poisoning errors from [`ECSReference::defer`].
    pub fn spawn(self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let mut bundle = Bundle::new();

        // 1. Template defaults.
        for cid in self.template.signature.iterate_over_components() {
            if let Some(factory) = self.template.defaults.get(&cid) {
                bundle.insert_boxed(cid, factory());
            }
        }

        // 2. Per-instance overrides.
        for (cid, value) in self.overrides {
            bundle.insert_boxed(cid, value);
        }

        // 3. Enqueue deferred tagged spawn.
        ecs.defer(Command::SpawnTagged {
            bundle,
            tag: self.template.name().to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::agents::error::AgentError;
    use crate::agents::template::AgentTemplate;

    #[allow(dead_code)]
    #[derive(Default, Clone)]
    struct Wealth(f64);

    #[test]
    fn spawner_set_overrides_are_stored() {
        let tmpl = AgentTemplate::builder("A")
            .with_component::<Wealth>(0)
            .unwrap()
            .build();
        let spawner = tmpl.spawner().set::<Wealth>(0, Wealth(99.0)).unwrap();
        assert_eq!(spawner.overrides.len(), 1);
        let (cid, _) = &spawner.overrides[0];
        assert_eq!(*cid, 0);
    }

    #[test]
    fn spawner_duplicate_set_last_wins() {
        let tmpl = AgentTemplate::builder("B")
            .with_component::<Wealth>(0)
            .unwrap()
            .build();
        let spawner = tmpl
            .spawner()
            .set::<Wealth>(0, Wealth(1.0))
            .unwrap()
            .set::<Wealth>(0, Wealth(2.0))
            .unwrap();
        assert_eq!(spawner.overrides.len(), 1);
    }

    #[test]
    fn spawner_rejects_unknown_component() {
        let tmpl = AgentTemplate::builder("C")
            .with_component::<Wealth>(0)
            .unwrap()
            .build();
        let result = tmpl.spawner().set::<Wealth>(99, Wealth(1.0));
        assert!(matches!(result, Err(AgentError::MissingComponent(99))));
    }
}
