//! Isolated child model support.
//!
//! A [`NestedModel`] owns a complete child [`Model`] with its own ECS world,
//! environment, agents, messages, and GPU resources. It is intentionally
//! separate from [`SubScheduler`](super::SubScheduler), which shares the root
//! model state.

use crate::ECSResult;

use super::model::Model;

/// Explicit bridge from an isolated child model back into its parent.
pub type NestedBridge = Box<dyn FnMut(&mut Model, &Model) -> ECSResult<()> + Send>;

/// Isolated child model executed as part of a parent model tick.
pub struct NestedModel {
    name: String,
    model: Box<Model>,
    bridge: Option<NestedBridge>,
}

impl NestedModel {
    /// Creates a named nested model without a parent bridge.
    pub fn new(name: impl Into<String>, model: Model) -> Self {
        Self {
            name: name.into(),
            model: Box::new(model),
            bridge: None,
        }
    }

    /// Installs a parent bridge that runs after the child model ticks.
    pub fn with_bridge(
        mut self,
        bridge: impl FnMut(&mut Model, &Model) -> ECSResult<()> + Send + 'static,
    ) -> Self {
        self.bridge = Some(Box::new(bridge));
        self
    }

    /// Returns the nested model display name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the child model.
    #[inline]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Returns the child model mutably.
    #[inline]
    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    pub(crate) fn tick_and_bridge(&mut self, parent: &mut Model) -> ECSResult<()> {
        self.model.tick()?;
        if let Some(bridge) = &mut self.bridge {
            bridge(parent, &self.model)?;
        }
        Ok(())
    }
}
