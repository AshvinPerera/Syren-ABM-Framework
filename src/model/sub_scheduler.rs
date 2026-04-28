//! Named scheduler scope inside a [`Model`](crate::model::Model).

use crate::engine::scheduler::Scheduler;

/// A named scheduler that shares the model's ECS world and boundary resources.
pub struct SubScheduler {
    name: String,
    scheduler: Scheduler,
}

impl SubScheduler {
    /// Creates an empty sub-scheduler.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            scheduler: Scheduler::new(),
        }
    }

    /// Creates a sub-scheduler from an existing scheduler.
    pub fn with_scheduler(name: impl Into<String>, scheduler: Scheduler) -> Self {
        Self {
            name: name.into(),
            scheduler,
        }
    }

    /// Returns the sub-scheduler name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the scheduler.
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Returns the scheduler mutably.
    pub fn scheduler_mut(&mut self) -> &mut Scheduler {
        &mut self.scheduler
    }
}
