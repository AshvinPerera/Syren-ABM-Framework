//! Builder for spawning many agents from one template.

use std::any::Any;
use std::collections::HashMap;

use crate::engine::commands::{BatchColumn, Command, SpawnBatch};
use crate::engine::error::ECSResult;
use crate::engine::manager::ECSReference;
use crate::engine::types::{AgentTemplateId, ComponentID};

use super::error::{AgentError, AgentResult};
use super::template::AgentTemplate;

/// Deferred batch spawn builder for a single agent template.
pub struct AgentBatch<'t> {
    template: &'t AgentTemplate,
    template_id: AgentTemplateId,
    count: usize,
    columns: HashMap<ComponentID, Vec<Box<dyn Any + Send>>>,
}

impl<'t> AgentBatch<'t> {
    /// Creates a new batch builder.
    pub(crate) fn new(
        template: &'t AgentTemplate,
        template_id: AgentTemplateId,
        count: usize,
    ) -> Self {
        Self {
            template,
            template_id,
            count,
            columns: HashMap::new(),
        }
    }

    /// Sets an entire component column for the batch.
    pub fn set_column<T: Any + Send + 'static>(
        mut self,
        component_id: ComponentID,
        values: Vec<T>,
    ) -> AgentResult<Self> {
        if !self
            .template
            .signature
            .try_has(component_id)
            .map_err(|_| AgentError::invalid_component_id(component_id))?
        {
            return Err(AgentError::MissingComponent(component_id));
        }
        if values.len() != self.count {
            return Err(AgentError::BatchLengthMismatch {
                component_id,
                expected: self.count,
                actual: values.len(),
            });
        }
        self.columns.insert(
            component_id,
            values
                .into_iter()
                .map(|value| Box::new(value) as Box<dyn Any + Send>)
                .collect(),
        );
        Ok(self)
    }

    pub(crate) fn set_boxed_column(
        mut self,
        component_id: ComponentID,
        values: Vec<Box<dyn Any + Send>>,
    ) -> AgentResult<Self> {
        if !self
            .template
            .signature
            .try_has(component_id)
            .map_err(|_| AgentError::invalid_component_id(component_id))?
        {
            return Err(AgentError::MissingComponent(component_id));
        }
        if values.len() != self.count {
            return Err(AgentError::BatchLengthMismatch {
                component_id,
                expected: self.count,
                actual: values.len(),
            });
        }
        self.columns.insert(component_id, values);
        Ok(self)
    }

    /// Converts this builder into an engine batch payload.
    pub fn into_spawn_batch(mut self) -> SpawnBatch {
        let mut columns = Vec::new();
        for component_id in self.template.signature.iterate_over_components() {
            let values = self.columns.remove(&component_id).unwrap_or_else(|| {
                let factory = &self.template.defaults[&component_id];
                (0..self.count).map(|_| factory()).collect()
            });
            columns.push(BatchColumn {
                component_id,
                values,
            });
        }
        SpawnBatch {
            count: self.count,
            signature: self.template.signature,
            columns,
        }
    }

    /// Enqueues this batch as a template-id tagged spawn command.
    pub fn spawn(self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let template_id = self.template_id;
        ecs.defer(Command::SpawnBatchTagged {
            batch: self.into_spawn_batch(),
            template_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::registry::AgentRegistry;

    #[test]
    fn set_column_rejects_invalid_component_id() {
        let mut registry = AgentRegistry::new();
        registry
            .register(
                AgentTemplate::builder("Sheep")
                    .with_component::<u32>(0)
                    .unwrap()
                    .build(),
            )
            .unwrap();
        let template = registry.get("Sheep").unwrap();
        let invalid = crate::engine::types::COMPONENT_CAP as ComponentID;
        let result = template
            .batch(1)
            .unwrap()
            .set_column::<u32>(invalid, vec![1]);

        match result {
            Err(err) => assert_eq!(err, AgentError::invalid_component_id(invalid)),
            Ok(_) => panic!("expected invalid component id error"),
        }
    }
}
