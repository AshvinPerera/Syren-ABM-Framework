//! Runtime model owner.
//!
//! A [`Model`] owns the root ECS world plus model-level resources such as the
//! environment, agent registry, root scheduler, optional shared sub-schedulers,
//! and isolated nested child models.
//!
//! Tick order is explicit:
//!
//! 1. Begin all boundary resources on the root ECS world.
//! 2. Run shared sub-schedulers against the root ECS world, flushing agent
//!    lifecycle hooks after each deferred-command drain.
//! 3. Tick each isolated nested child model. After a child tick succeeds, run
//!    its bridge into the parent before moving to the next child.
//! 4. Run the root scheduler.
//! 5. Drain tail deferred commands, flush agent hooks, then end root
//!    boundaries.

use std::any::Any;
use std::fmt::Write;
use std::sync::Arc;

use crate::agents::AgentRegistry;
use crate::engine::commands::Command;
use crate::engine::commands::CommandEvents;
use crate::engine::dot_export::DotExport;
use crate::engine::manager::ECSManager;
use crate::engine::manager::ECSReference;
use crate::engine::plan_display::PlanDisplay;
use crate::engine::scheduler::Scheduler;
use crate::engine::types::BoundaryID;
use crate::environment::Environment;
use crate::ECSResult;
use crate::{AgentTemplateId, Entity};

use super::nested::NestedModel;
use super::sub_scheduler::SubScheduler;

/// Top-level simulation model.
pub struct Model {
    pub(crate) ecs: ECSManager,
    pub(crate) environment: Arc<Environment>,
    pub(crate) agents: AgentRegistry,
    pub(crate) scheduler: Scheduler,
    /// Global RNG seed configured through [`ModelBuilder::with_seed`].
    ///
    /// [`ModelBuilder::with_seed`]: crate::model::ModelBuilder::with_seed
    pub(crate) seed: u64,
    pub(crate) sub_schedulers: Vec<SubScheduler>,
    pub(crate) nested_models: Vec<NestedModel>,
    pub(crate) environment_boundary_id: BoundaryID,
    #[cfg(feature = "messaging")]
    pub(crate) message_boundary_id: BoundaryID,
    #[cfg(feature = "messaging")]
    pub(crate) message_registry: Arc<crate::messaging::MessageRegistry>,
    pub(crate) tick_count: u64,
}

impl Model {
    /// Returns the global RNG seed configured through
    /// [`ModelBuilder::with_seed`](crate::model::ModelBuilder::with_seed).
    ///
    /// This is the value delivered to systems as `RunContext::simulation_seed`
    /// on the root scheduler and every shared sub-scheduler. Nested models are
    /// isolated worlds and report the seed configured by their own builders.
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Runs one simulation tick.
    ///
    /// Nested models are isolated worlds. Each nested child completes its own
    /// `tick`, then its bridge writes any parent-facing effects, and only then
    /// does the root scheduler run. This keeps child state evolution and
    /// parent observation order deterministic.
    pub fn tick(&mut self) -> ECSResult<()> {
        self.ecs.begin_tick()?;

        let tick_result = (|| {
            for sub in &mut self.sub_schedulers {
                let ecs = self.ecs.world_ref();
                let agents = &mut self.agents;
                let seed = sub.scheduler().global_seed();
                sub.scheduler_mut().run_with_context_and_lifecycle_events(
                    ecs,
                    seed,
                    self.tick_count,
                    |events| Self::flush_agent_hooks(agents, ecs, events),
                )?;
            }

            let mut nested_models = std::mem::take(&mut self.nested_models);
            let nested_result: ECSResult<()> = (|| {
                for nested in &mut nested_models {
                    nested.tick_and_bridge(self)?;
                }
                Ok(())
            })();
            self.nested_models = nested_models;
            nested_result?;

            let ecs = self.ecs.world_ref();
            let agents = &mut self.agents;
            let seed = self.scheduler.global_seed();
            self.scheduler.run_with_context_and_lifecycle_events(
                ecs,
                seed,
                self.tick_count,
                |events| Self::flush_agent_hooks(agents, ecs, events),
            )?;
            self.ecs.world_ref().clear_borrows();
            match self.ecs.apply_deferred_commands_with_events() {
                Ok(events) => {
                    Self::flush_agent_hooks(&mut self.agents, self.ecs.world_ref(), &events)?;
                    Ok(())
                }
                Err(failure) => {
                    Self::flush_agent_hooks(
                        &mut self.agents,
                        self.ecs.world_ref(),
                        &failure.events,
                    )?;
                    Err(failure.error)
                }
            }
        })();

        self.ecs.world_ref().clear_borrows();
        let end_result = self.ecs.end_tick();
        match (tick_result, end_result) {
            (Ok(()), Ok(())) => {
                self.tick_count += 1;
                Ok(())
            }
            (Err(error), _) => Err(error),
            (Ok(()), Err(error)) => Err(error),
        }
    }

    fn flush_agent_hooks(
        agents: &mut AgentRegistry,
        ecs: ECSReference<'_>,
        events: &CommandEvents,
    ) -> ECSResult<()> {
        for batch in &events.spawned_batches {
            agents.enqueue_spawn_batch_hook(batch.template_id, batch.entities.clone())?;
            agents.enqueue_spawn_hooks_by_id(batch.template_id, &batch.entities)?;
        }
        for batch in &events.despawned_batches {
            agents.enqueue_despawn_batch_hook(batch.template_id, batch.entities.clone())?;
            agents.enqueue_despawn_hooks_by_id(batch.template_id, &batch.entities)?;
        }
        for event in &events.spawned {
            if event.template_id.is_some() {
                continue;
            }
            if let Some(tag) = &event.tag {
                agents.enqueue_spawn_hook(tag.clone(), event.entity);
            }
        }
        for event in &events.despawned {
            if event.template_id.is_some() {
                continue;
            }
            if let Some(tag) = &event.tag {
                agents.enqueue_despawn_hook(tag.clone(), event.entity);
            }
        }
        agents.flush_spawn_hooks(ecs);
        agents.flush_despawn_hooks(ecs);
        Ok(())
    }

    /// Runs `n` ticks, stopping on the first error.
    pub fn run(&mut self, n: u64) -> ECSResult<()> {
        for _ in 0..n {
            self.tick()?;
        }
        Ok(())
    }

    /// Returns the ECS manager.
    pub fn ecs(&self) -> &ECSManager {
        &self.ecs
    }

    /// Returns the shared environment.
    pub fn environment(&self) -> &Arc<Environment> {
        &self.environment
    }

    /// Returns the agent registry.
    pub fn agents(&self) -> &AgentRegistry {
        &self.agents
    }

    /// Spawns a single-column batch of agents and returns the created entities.
    ///
    /// The column is carried as one `Vec<T>` end to end - no per-value boxing
    /// - and applied through the columnar bulk-spawn path.
    pub fn spawn_agent_batch<T>(
        &mut self,
        template_name: &str,
        component_id: crate::ComponentID,
        values: Vec<T>,
    ) -> Result<Vec<Entity>, super::error::ModelError>
    where
        T: Any + Send + 'static,
    {
        let len = values.len();
        self.spawn_agent_batch_erased(template_name, vec![(component_id, Box::new(values))], len)
    }

    /// Spawns one batch of agents, setting every supplied component column on
    /// the same entities.
    ///
    /// Taking the columns together is what allows an agent to be modelled as
    /// several small components rather than one wide struct: a query then names
    /// only the columns it reads, instead of dragging every field of a combined
    /// struct through cache. Spawning one batch per column would create one set
    /// of entities per column instead.
    pub(crate) fn spawn_agent_batch_erased(
        &mut self,
        template_name: &str,
        columns: Vec<(crate::ComponentID, Box<dyn Any + Send>)>,
        len: usize,
    ) -> Result<Vec<Entity>, super::error::ModelError> {
        let template_id = self.agents.id(template_name)?;
        let mut builder = self.agents.get(template_name)?.batch(len)?;
        for (component_id, values) in columns {
            builder = builder.set_erased_column(component_id, values, len)?;
        }
        let batch = builder.into_spawn_batch();
        self.ecs
            .world_ref()
            .defer(Command::SpawnBatchTagged { batch, template_id })?;
        let events = self.ecs.apply_deferred_commands()?;
        let spawned = entities_for_template(&events.spawned_batches, template_id);
        Self::flush_agent_hooks(&mut self.agents, self.ecs.world_ref(), &events)?;
        Ok(spawned)
    }

    /// Despawns a batch of agents associated with one template.
    pub fn despawn_agent_batch(
        &mut self,
        template_name: &str,
        entities: Vec<Entity>,
    ) -> Result<(), super::error::ModelError> {
        let template_id = self.agents.id(template_name)?;
        self.ecs.world_ref().defer(Command::DespawnBatchTagged {
            entities,
            template_id,
        })?;
        let events = self.ecs.apply_deferred_commands()?;
        Self::flush_agent_hooks(&mut self.agents, self.ecs.world_ref(), &events)?;
        Ok(())
    }

    /// Returns isolated nested child models.
    pub fn nested_models(&self) -> &[NestedModel] {
        &self.nested_models
    }

    /// Returns successful tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Returns the environment boundary ID.
    pub fn environment_boundary_id(&self) -> BoundaryID {
        self.environment_boundary_id
    }

    /// Returns the messaging boundary ID.
    #[cfg(feature = "messaging")]
    pub fn message_boundary_id(&self) -> BoundaryID {
        self.message_boundary_id
    }

    /// Returns the message registry.
    #[cfg(feature = "messaging")]
    pub fn message_registry(&self) -> &Arc<crate::messaging::MessageRegistry> {
        &self.message_registry
    }

    /// Returns true when any root or sub-scheduler system uses the GPU backend.
    pub fn has_gpu_systems(&self) -> bool {
        self.scheduler.has_gpu_systems()
            || self
                .sub_schedulers
                .iter()
                .any(|sub| sub.scheduler().has_gpu_systems())
            || self
                .nested_models
                .iter()
                .any(|nested| nested.model().has_gpu_systems())
    }

    /// Renders root and sub-scheduler plans as human-readable text.
    pub fn execution_plan_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(&mut out, "[root]");
        let _ = writeln!(&mut out, "{}", PlanDisplay(&self.scheduler));
        for sub in &self.sub_schedulers {
            let _ = writeln!(&mut out, "[{}]", sub.name());
            let _ = writeln!(&mut out, "{}", PlanDisplay(sub.scheduler()));
        }
        out
    }

    /// Renders root and sub-scheduler plans as DOT text.
    pub fn execution_plan_dot(&self) -> String {
        let mut out = String::from("digraph model_execution_plan {\n");
        let _ = writeln!(&mut out, "  subgraph cluster_root {{");
        let root = DotExport(&self.scheduler).to_string();
        for line in root
            .lines()
            .filter(|line| !line.starts_with("digraph") && *line != "}")
        {
            let _ = writeln!(&mut out, "  {line}");
        }
        let _ = writeln!(&mut out, "  }}");
        for (idx, sub) in self.sub_schedulers.iter().enumerate() {
            let _ = writeln!(&mut out, "  subgraph cluster_sub_{idx} {{");
            let _ = writeln!(&mut out, "    label=\"{}\";", dot_escape(sub.name()));
            let dot = DotExport(sub.scheduler()).to_string();
            for line in dot
                .lines()
                .filter(|line| !line.starts_with("digraph") && *line != "}")
            {
                let _ = writeln!(&mut out, "  {line}");
            }
            let _ = writeln!(&mut out, "  }}");
        }
        out.push_str("}\n");
        out
    }
}

fn entities_for_template(
    batches: &[crate::engine::commands::TemplateLifecycleBatch],
    template_id: AgentTemplateId,
) -> Vec<Entity> {
    batches
        .iter()
        .filter(|batch| batch.template_id == template_id)
        .flat_map(|batch| batch.entities.iter().copied())
        .collect()
}

fn dot_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}
