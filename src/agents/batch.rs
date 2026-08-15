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
///
/// Columns are stored **columnar**: each `set_column` call keeps the caller's
/// `Vec<T>` intact behind one type-erased box, and unset components are
/// filled from the template's column factories. No per-value boxing occurs
/// anywhere on this path; the storage layer bulk-copies each column in
/// chunk-sized runs at apply time.
pub struct AgentBatch<'t> {
    template: &'t AgentTemplate,
    template_id: AgentTemplateId,
    count: usize,
    /// `component_id -> (type-erased Vec<T>, element count)`.
    columns: HashMap<ComponentID, (Box<dyn Any + Send>, usize)>,
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
    ///
    /// The vector is stored whole (one allocation for the column); its
    /// element type must match the storage type registered for
    /// `component_id`.
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
        let len = values.len();
        self.columns.insert(component_id, (Box::new(values), len));
        Ok(self)
    }

    /// Sets a pre-erased column (`Box<dyn Any + Send>` containing a `Vec<T>`).
    ///
    /// Used by model-level plumbing that has already erased the column type.
    pub(crate) fn set_erased_column(
        mut self,
        component_id: ComponentID,
        values: Box<dyn Any + Send>,
        len: usize,
    ) -> AgentResult<Self> {
        if !self
            .template
            .signature
            .try_has(component_id)
            .map_err(|_| AgentError::invalid_component_id(component_id))?
        {
            return Err(AgentError::MissingComponent(component_id));
        }
        if len != self.count {
            return Err(AgentError::BatchLengthMismatch {
                component_id,
                expected: self.count,
                actual: len,
            });
        }
        self.columns.insert(component_id, (values, len));
        Ok(self)
    }

    /// Converts this builder into an engine batch payload.
    ///
    /// Components not set explicitly are filled from the template's column
    /// factories (a single `Vec<T>` of defaults per component).
    pub fn into_spawn_batch(mut self) -> SpawnBatch {
        let mut columns = Vec::new();
        for component_id in self.template.signature.iterate_over_components() {
            let (values, len) = self.columns.remove(&component_id).unwrap_or_else(|| {
                let factory = &self.template.column_defaults[&component_id];
                (factory(self.count), self.count)
            });
            columns.push(BatchColumn {
                component_id,
                values,
                len,
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

    #[test]
    fn unset_columns_fill_from_column_defaults() {
        let mut registry = AgentRegistry::new();
        registry
            .register(
                AgentTemplate::builder("Sheep")
                    .with_component::<u32>(0)
                    .unwrap()
                    .with_component_factory(1, || 7u64)
                    .unwrap()
                    .build(),
            )
            .unwrap();
        let template = registry.get("Sheep").unwrap();
        let batch = template
            .batch(3)
            .unwrap()
            .set_column::<u32>(0, vec![1, 2, 3])
            .unwrap()
            .into_spawn_batch();

        assert_eq!(batch.count, 3);
        assert_eq!(batch.columns.len(), 2);
        for column in &batch.columns {
            assert_eq!(column.len, 3);
        }
        let defaults = batch
            .columns
            .into_iter()
            .find(|column| column.component_id == 1)
            .unwrap();
        let values = defaults.values.downcast::<Vec<u64>>().unwrap();
        assert_eq!(*values, vec![7, 7, 7]);
    }
}
