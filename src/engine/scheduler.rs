//! ECS system scheduling and execution.
//!
//! This module is responsible for:
//! * grouping systems into execution stages based on access compatibility,
//! * running compatible systems in parallel using Rayon,
//! * enforcing structural synchronization points between stages.
//!
//! ## Scheduling model
//!
//! Systems are assigned to **stages** such that:
//! * systems within the same stage do **not** conflict on component access,
//! * all systems in a stage may run in parallel,
//! * stages are executed sequentially.
//!
//! This allows maximal parallelism while preserving safety guarantees
//! derived from declared read/write access sets, in conjunction with
//! execution-phase discipline enforced by the ECS manager.
//!
//! ## Structural synchronization
//!
//! Deferred ECS commands (spawns, despawns, component mutations) are applied
//! at explicit synchronization points controlled by the ECS manager,
//! typically between scheduler stages.
//! 
//! ## Safety note
//!
//! This module assumes that systems are executed only within the
//! appropriate ECS execution phases; violating phase discipline
//! may result in undefined behavior.

use rayon::prelude::*;

use crate::engine::manager::ECSReference;
use crate::engine::systems::{AccessSets, System, SystemBackend};
use crate::engine::component::Signature;
use crate::engine::error::{ECSResult, ECSError};


/// A logical execution stage used by [`Scheduler`] during planning.
///
/// Stores *indices* into the scheduler's system list. This lets the
/// scheduler:
/// - keep systems registered for repeated ticks,
/// - rebuild plans when systems are added/removed,
/// - evolve into a CPU/GPU multi-backend dispatcher.
#[derive(Clone, Debug, Default)]
pub struct Stage {
    /// Indices of systems that can execute in parallel.
    pub system_indices: Vec<usize>,
    /// Aggregate access sets of systems in this stage (used for fast conflict checks
    /// during plan construction).
    aggregate_access: AccessSets,
}

impl Stage {
    /// Returns true if `access` does NOT conflict with anything already in this stage.
    #[inline]
    pub fn can_accept(&self, access: &AccessSets) -> bool {
        !access.conflicts_with(&self.aggregate_access)
    }

    /// Adds a system index to this stage and merges its access into the aggregate.
    #[inline]
    pub fn push(&mut self, idx: usize, access: &AccessSets) {
        self.system_indices.push(idx);
        or_signature_in_place(&mut self.aggregate_access.read, &access.read);
        or_signature_in_place(&mut self.aggregate_access.write, &access.write);
    }

    /// Returns true if this stage is acting as a boundary marker.
    #[inline]
    pub fn is_boundary(&self) -> bool {
        self.system_indices.is_empty()
    }
}

/// Scheduler that stores systems, compiles them into conflict-free execution stages
/// based on declared access sets, and executes stages with parallelism.
pub struct Scheduler {
    systems: Vec<Box<dyn System>>,
    /// Cached CPU stages.
    cpu_stages: Vec<Stage>,
    /// Whether `cpu_stages` needs rebuilding.
    dirty: bool,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler {
    /// Creates an empty scheduler.
    #[inline]
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            cpu_stages: Vec::new(),
            dirty: true,
        }
    }

    /// Returns the number of registered systems.
    #[inline]
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// Returns `true` if no systems are registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// Removes all systems and stages.
    #[inline]
    pub fn clear(&mut self) {
        self.systems.clear();
        self.cpu_stages.clear();
        self.dirty = true;
    }

    /// Registers a boxed system.
    #[inline]
    pub fn add_boxed(&mut self, system: Box<dyn System>) {
        self.systems.push(system);
        self.dirty = true;
    }

    /// Registers a concrete system.
    #[inline]
    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.add_boxed(Box::new(system));
    }

    /// Registers a function-backed system.
    pub fn add_fn_system<F>(
        &mut self,
        system: crate::engine::systems::FnSystem<F>,
    )
    where
        F: Fn(crate::engine::manager::ECSReference<'_>) -> ECSResult<()>
            + Send
            + Sync
            + 'static,
    {
        self.add_system(system);
    }

    /// Convenience helper to build-and-register an [`FnSystem`](crate::engine::systems::FnSystem)
    #[inline]
    pub fn add_fn<F>(
        &mut self,
        id: crate::engine::types::SystemID,
        name: &'static str,
        access: AccessSets,
        f: F,
    )
    where
        F: Fn(crate::engine::manager::ECSReference<'_>) -> ECSResult<()>
            + Send
            + Sync
            + 'static,
    {
        self.add_fn_system(crate::engine::systems::FnSystem::new(id, name, access, f));
    }

    /// Registers an infallible function-backed system.
    /// The function is automatically wrapped to return `Ok(())`.
    
    #[inline]
    pub fn add_fn_infallible<F>(
        &mut self,
        id: crate::engine::types::SystemID,
        name: &'static str,
        access: AccessSets,
        f: F,
    )
    where
        F: Fn(crate::engine::manager::ECSReference<'_>) + Send + Sync + 'static,
    {
        self.add_fn(id, name, access, move |world| {
            f(world);
            Ok::<(), ECSError>(())
        });
    }

    /// Ensures stages are up to date.
    pub fn rebuild(&mut self) {
        if !self.dirty {
            return;
        }

        // Deterministic: sort indices by system ID.
        let mut indices: Vec<usize> = (0..self.systems.len()).collect();
        indices.sort_by_key(|&i| self.systems[i].id());

        self.cpu_stages.clear();

        for idx in indices {
            let sys = &self.systems[idx];

            // For GPU scaling: keep GPU systems as hard boundaries.
            if sys.backend() == SystemBackend::GPU {
                // Flush any pending CPU stage and create a boundary stage.
                self.cpu_stages.push(Stage::default());

                let mut stage = Stage::default();
                stage.push(idx, &sys.access());
                self.cpu_stages.push(stage);

                continue;
            }

            let access = sys.access();

            // Greedy packing into the first compatible stage.
            let mut placed = false;
            for stage in self.cpu_stages.iter_mut() {
                if stage.system_indices.is_empty() {
                    continue;
                }
                if stage.can_accept(&access) {
                    stage.push(idx, &access);
                    placed = true;
                    break;
                }
            }
            if !placed {
                let mut stage = Stage::default();
                stage.push(idx, &access);
                self.cpu_stages.push(stage);
            }
        }

        self.dirty = false;
    }

    /// Runs the schedule once.
    ///
    /// This will:
    /// 1) rebuild the execution plan if needed,
    /// 2) execute each stage sequentially,
    /// 3) run systems within a stage in parallel.
    ///
    /// Structural synchronization is expected to be handled
    /// by the ECS manager at higher-level execution boundaries
    
    pub fn run(&mut self, ecs: ECSReference<'_>) -> ECSResult<()> {
        self.rebuild();

        for stage in &self.cpu_stages {
            if stage.system_indices.is_empty() {
                continue;
            }

            stage
                .system_indices
                .par_iter()
                .try_for_each(|&system_idx| -> ECSResult<()> {
                    self.systems[system_idx].run(ecs)
                })?;

            ecs.clear_borrows();
        }

        Ok(())
    }
}

#[inline]
fn or_signature_in_place(
    dst: &mut Signature,
    src: &Signature
) {
    for (d, s) in dst.components.iter_mut().zip(src.components.iter()) {
        *d |= *s;
    }
}
