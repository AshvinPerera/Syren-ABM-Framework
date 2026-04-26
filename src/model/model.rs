//! Runtime model owner.

use std::fmt::Write;
use std::sync::Arc;

use crate::agents::AgentRegistry;
use crate::engine::dot_export::DotExport;
use crate::engine::manager::ECSManager;
use crate::engine::plan_display::PlanDisplay;
use crate::engine::scheduler::Scheduler;
use crate::engine::types::BoundaryID;
use crate::environment::Environment;
use crate::ECSResult;

use super::sub_scheduler::SubScheduler;

/// Top-level simulation model.
pub struct Model {
    pub(crate) ecs: ECSManager,
    pub(crate) environment: Arc<Environment>,
    pub(crate) agents: AgentRegistry,
    pub(crate) scheduler: Scheduler,
    pub(crate) sub_schedulers: Vec<SubScheduler>,
    pub(crate) environment_boundary_id: BoundaryID,
    #[cfg(feature = "messaging")]
    pub(crate) message_boundary_id: BoundaryID,
    #[cfg(feature = "messaging")]
    pub(crate) message_registry: Arc<crate::messaging::MessageRegistry>,
    pub(crate) tick_count: u64,
}

impl Model {
    /// Runs one simulation tick.
    pub fn tick(&mut self) -> ECSResult<()> {
        self.ecs.begin_tick()?;

        for sub in &mut self.sub_schedulers {
            sub.scheduler_mut().run(self.ecs.world_ref())?;
        }

        self.scheduler.run(self.ecs.world_ref())?;
        self.ecs.world_ref().clear_borrows();
        self.ecs.apply_deferred_commands()?;
        self.ecs.end_tick()?;
        self.tick_count += 1;
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
        for line in root.lines().filter(|line| !line.starts_with("digraph") && *line != "}") {
            let _ = writeln!(&mut out, "  {line}");
        }
        let _ = writeln!(&mut out, "  }}");
        for sub in &self.sub_schedulers {
            let _ = writeln!(&mut out, "  subgraph cluster_{} {{", sub.name());
            let _ = writeln!(&mut out, "    label=\"{}\";", sub.name());
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
