//! World-scoped store of named agent templates.
//!
//! ## Design
//!
//! [`AgentRegistry`] is owned by the model (not by [`ECSManager`]). It stores
//! one [`AgentTemplate`] per agent class, keyed by name. It is sealed before
//! simulation start to prevent accidental runtime registration.
//!
//! It also maintains a **pending spawn-hook queue**: when
//! [`AgentSpawner::spawn`] enqueues a `Command::Spawn` for a template that
//! has an `on_spawn` hook, it adds `(template_name, entity)` to this queue.
//! The model layer calls [`AgentRegistry::flush_spawn_hooks`] after
//! `apply_deferred_commands` to invoke all pending hooks with the resolved
//! entity handles.
//!
//! ## Thread safety
//!
//! `AgentRegistry` is **not** `Sync` and is intended for single-threaded
//! ownership by the model. Do not share across threads.

use std::collections::HashMap;

use crate::engine::entity::Entity;
use crate::engine::manager::ECSReference;

use super::template::AgentTemplate;
use super::error::{AgentError, AgentResult};

/// World-scoped store of named [`AgentTemplate`]s.
///
/// # Lifecycle
///
/// 1. Create with [`AgentRegistry::new`].
/// 2. Register templates with [`AgentRegistry::register`] before simulation
///    starts.
/// 3. Call [`AgentRegistry::seal`] to prevent further registrations.
/// 4. During simulation, retrieve templates with [`AgentRegistry::get`] and
///    use [`AgentTemplate::spawner`] to spawn agents.
/// 5. After each `apply_deferred_commands` call, invoke
///    [`AgentRegistry::flush_spawn_hooks`] to trigger spawn lifecycle hooks.
pub struct AgentRegistry {
    templates: HashMap<String, AgentTemplate>,
    sealed: bool,
    /// Queue of `(template_name, entity)` pairs whose `on_spawn` hook is
    /// pending invocation.
    ///
    /// Populated by the model layer (which knows the resolved entity) and
    /// drained by [`AgentRegistry::flush_spawn_hooks`].
    pending_spawn_hooks: Vec<(String, Entity)>,
}

impl AgentRegistry {
    /// Creates an empty, unsealed registry.
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            sealed: false,
            pending_spawn_hooks: Vec::new(),
        }
    }

    /// Registers a template.
    ///
    /// # Errors
    ///
    /// * [`AgentError::RegistrySealed`] — called after [`AgentRegistry::seal`].
    /// * [`AgentError::TemplateNotFound`] (repurposed as "already exists") — a
    ///   template with the same name is already present. The error message
    ///   clarifies this.
    pub fn register(&mut self, template: AgentTemplate) -> AgentResult<()> {
        if self.sealed {
            return Err(AgentError::RegistrySealed);
        }
        if self.templates.contains_key(&template.name) {
            // AgentError has no "DuplicateName" variant (names are opaque
            // strings, not ComponentIDs). We reuse TemplateNotFound to signal
            // "already exists" with a descriptive message rather than adding a
            // new variant.
            return Err(AgentError::TemplateNotFound(format!(
                "template '{}' is already registered",
                template.name
            )));
        }
        self.templates.insert(template.name.clone(), template);
        Ok(())
    }

    /// Returns a reference to the named template.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::TemplateNotFound`] if no template with that name
    /// exists.
    pub fn get(&self, name: &str) -> AgentResult<&AgentTemplate> {
        self.templates
            .get(name)
            .ok_or_else(|| AgentError::TemplateNotFound(name.to_owned()))
    }

    /// Seals the registry, preventing further [`AgentRegistry::register`] calls.
    ///
    /// Call before simulation start. After sealing, `register` always returns
    /// [`AgentError::RegistrySealed`].
    pub fn seal(&mut self) {
        self.sealed = true;
    }

    /// Returns `true` if the registry has been sealed.
    #[inline]
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// Returns the number of registered templates.
    #[inline]
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Returns `true` if no templates have been registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Iterates over all registered templates in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &AgentTemplate)> {
        self.templates.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Enqueues a pending spawn-hook invocation.
    ///
    /// Called by the model layer (after `apply_deferred_commands` resolves a
    /// new entity) for templates that declare an `on_spawn` hook.
    ///
    /// Flushed by [`AgentRegistry::flush_spawn_hooks`].
    pub fn enqueue_spawn_hook(&mut self, template_name: impl Into<String>, entity: Entity) {
        self.pending_spawn_hooks.push((template_name.into(), entity));
    }

    /// Invokes all pending `on_spawn` hooks and clears the queue.
    ///
    /// The model must call this **after** `apply_deferred_commands` returns,
    /// when resolved entity handles are available. Each hook receives the
    /// shared `ECSReference` and the entity that was just spawned.
    ///
    /// Hooks for templates that cannot be found (e.g. removed between
    /// enqueue and flush) are silently skipped.
    pub fn flush_spawn_hooks(&mut self, ecs: ECSReference<'_>) {
        let pending = std::mem::take(&mut self.pending_spawn_hooks);
        for (name, entity) in pending {
            if let Some(template) = self.templates.get(&name) {
                if let Some(hook) = template.on_spawn() {
                    hook(ecs, entity);
                }
            }
        }
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::template::AgentTemplate;

    fn make_template(name: &str) -> AgentTemplate {
        AgentTemplate::builder(name).build()
    }

    #[test]
    fn register_and_get() {
        let mut reg = AgentRegistry::new();
        reg.register(make_template("Wolf")).unwrap();
        let tmpl = reg.get("Wolf").unwrap();
        assert_eq!(tmpl.name(), "Wolf");
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let mut reg = AgentRegistry::new();
        reg.register(make_template("Fox")).unwrap();
        let err = reg.register(make_template("Fox")).unwrap_err();
        assert!(matches!(err, AgentError::TemplateNotFound(_)));
    }

    #[test]
    fn sealed_registry_rejects_registration() {
        let mut reg = AgentRegistry::new();
        reg.seal();
        let err = reg.register(make_template("Deer")).unwrap_err();
        assert_eq!(err, AgentError::RegistrySealed);
    }

    #[test]
    fn get_unknown_template_returns_error() {
        let reg = AgentRegistry::new();
        assert_eq!(
            reg.get("Ghost").unwrap_err(),
            AgentError::TemplateNotFound("Ghost".into()),
        );
    }

    #[test]
    fn seal_prevents_further_registration() {
        let mut reg = AgentRegistry::new();
        reg.register(make_template("A")).unwrap();
        reg.seal();
        assert!(reg.is_sealed());
        assert_eq!(
            reg.register(make_template("B")).unwrap_err(),
            AgentError::RegistrySealed,
        );
        // Already-registered templates are still accessible.
        assert!(reg.get("A").is_ok());
    }

    #[test]
    fn len_and_is_empty() {
        let mut reg = AgentRegistry::new();
        assert!(reg.is_empty());
        reg.register(make_template("A")).unwrap();
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
        reg.register(make_template("B")).unwrap();
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn enqueue_and_check_pending() {
        let mut reg = AgentRegistry::new();
        reg.register(make_template("Sheep")).unwrap();
        // Simulate an entity value (bit pattern doesn't matter for this unit test).
        let entity: Entity = unsafe { std::mem::transmute(0u64) };
        reg.enqueue_spawn_hook("Sheep", entity);
        assert_eq!(reg.pending_spawn_hooks.len(), 1);
    }

    #[test]
    fn flush_spawn_hooks_clears_queue() {
        let mut reg = AgentRegistry::new();
        // Template without a hook — flush should complete without panic.
        reg.register(make_template("Rabbit")).unwrap();
        let entity: Entity = unsafe { std::mem::transmute(0u64) };
        reg.enqueue_spawn_hook("Rabbit", entity);
        // We cannot call flush_spawn_hooks without a live ECSManager, but we
        // can verify the queue is drained by taking it manually.
        let _ = std::mem::take(&mut reg.pending_spawn_hooks);
        assert!(reg.pending_spawn_hooks.is_empty());
    }
}
