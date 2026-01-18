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
use crate::engine::error::{ECSResult, ECSError, ExecutionError};
use crate::profiling::profiler;

#[cfg(feature = "gpu")]
use crate::gpu;


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageType {
    BOUNDARY,
    CPU,
    GPU,
}

impl Default for StageType {
    fn default() -> Self {
        StageType::CPU
    }
}

/// A logical execution stage used by [`Scheduler`] during planning.
///
/// Stores *indices* into the scheduler's system list. This lets the
/// scheduler:
/// - keep systems registered for repeated ticks,
/// - rebuild plans when systems are added/removed,
/// - evolve into a CPU/GPU multi-backend dispatcher.
#[derive(Clone, Debug, Default)]
pub struct Stage {
    stage_type: StageType,
    /// Indices of systems that can execute in parallel.
    pub system_indices: Vec<usize>,
    /// Aggregate access sets of systems in this stage (CPU)
    aggregate_access: AccessSets,
}

impl Stage {
    fn boundary() -> Self {
        Self { 
            stage_type: StageType::BOUNDARY, 
            system_indices: Vec::new(), 
            aggregate_access: AccessSets::default() 
        }
    }

    fn cpu() -> Self {
        Self { 
            stage_type: StageType::CPU, 
            system_indices: Vec::new(), 
            aggregate_access: AccessSets::default() 
        }
    }

    fn gpu() -> Self {
        Self { 
            stage_type: StageType::GPU, 
            system_indices: Vec::new(), 
            aggregate_access: AccessSets::default() 
        }
    }

    /// Returns true if `access` does NOT conflict with anything already in this stage.
    #[inline]
    pub fn can_accept(&self, access: &AccessSets) -> bool {
        debug_assert_eq!(self.stage_type, StageType::CPU);
        !access.conflicts_with(&self.aggregate_access)
    }

    /// Adds a system index to this stage and merges its access into the aggregate (CPU only).
    #[inline]
    pub fn push_cpu(&mut self, index: usize, access: &AccessSets) {
        debug_assert_eq!(self.stage_type, StageType::CPU);
        self.system_indices.push(index);
        or_signature_in_place(&mut self.aggregate_access.read, &access.read);
        or_signature_in_place(&mut self.aggregate_access.write, &access.write);
    }

    /// Adds a system index to this stage (GPU only; executed sequentially).
    #[inline]
    pub fn push_gpu(&mut self, index: usize) {
        debug_assert_eq!(self.stage_type, StageType::GPU);
        self.system_indices.push(index);
    }

    /// Returns true if this stage is acting as a boundary marker.
    #[inline]
    pub fn is_boundary(&self) -> bool {
        self.stage_type == StageType::BOUNDARY
    }
}

/// Scheduler that stores systems, compiles them into conflict-free execution stages
/// based on declared access sets, and executes stages with parallelism.
pub struct Scheduler {
    systems: Vec<Box<dyn System>>,
    /// Cached stages.
    plan: Vec<Stage>,
    /// Whether `plan` needs rebuilding.
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
            plan: Vec::new(),
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
        self.plan.clear();
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

    /// Ensures stages are up to date.
    pub fn rebuild(&mut self) {
        if !self.dirty {
            return;
        }

        let mut indices: Vec<usize> = (0..self.systems.len()).collect();
        indices.sort_by_key(|&i| self.systems[i].id());

        self.plan.clear();

        // Track whether currently inside a GPU run.
        let mut in_gpu_run = false;

        for index in indices {
            let system = &self.systems[index];
            let access = system.access();

            match system.backend() {
                SystemBackend::CPU => {
                    // If leaving a GPU run, close it with a boundary.
                    if in_gpu_run {
                        self.plan.push(Stage::boundary());
                        in_gpu_run = false;
                    }

                    // Place into the first compatible CPU stage.
                    let mut placed = false;

                    for stage in self.plan.iter_mut().rev() {
                        if stage.stage_type == StageType::BOUNDARY {
                            break;
                        }
                        if stage.stage_type != StageType::CPU {
                            continue;
                        }
                        if stage.can_accept(&access) {
                            stage.push_cpu(index, &access);
                            placed = true;
                            break;
                        }
                    }

                    if !placed {
                        let mut stage = Stage::cpu();
                        stage.push_cpu(index, &access);
                        self.plan.push(stage);
                    }
                }

                SystemBackend::GPU => {
                    // If entering a GPU run, open it with a boundary.
                    if !in_gpu_run {
                        self.plan.push(Stage::boundary());
                        self.plan.push(Stage::gpu());
                        in_gpu_run = true;
                    }

                    // Push into the current GPU stage.
                    let last = self.plan.last_mut().expect("plan must have a GPU stage");
                    debug_assert_eq!(last.stage_type, StageType::GPU);
                    last.push_gpu(index);
                }
            }
        }

        // Close final GPU run with a boundary.
        if in_gpu_run {
            self.plan.push(Stage::boundary());
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
        let _g = profiler::span("Scheduler::run");
        
        self.rebuild();

        for stage in &self.plan {
            match stage.stage_type {
                StageType::BOUNDARY => {
                    let _s = profiler::span("Stage::BOUNDARY");

                    ecs.clear_borrows();

                    #[cfg(feature = "gpu")]
                    {
                        let _g0 = profiler::span("GPU::sync_pending_to_cpu");
                        // Ensures ECS CPU storage is consistent before commands or CPU stages.
                        gpu::sync_pending_to_cpu(ecs)?;
                    }

                    let _g1 = profiler::span("ECS::apply_deferred_commands");

                    ecs.apply_deferred_commands()?;
                }

                StageType::CPU => {
                    let _s = profiler::span("Stage::CPU")
                        .arg("systems", profiler::Arg::U64(stage.system_indices.len() as u64));
                    
                    stage.system_indices
                        .par_iter()
                        .try_for_each(|&i| {
                            let sys = &self.systems[i];

                            profiler::next_arg("system_id", profiler::Arg::U64(sys.id() as u64));
                            profiler::next_arg("backend", profiler::Arg::Str("CPU".to_string()));

                            let _sg = profiler::span_fmt(format_args!("System::{}", sys.name()));

                            sys.run(ecs)
                        })?;

                    ecs.clear_borrows();
                }

                StageType::GPU => {
                    let _s = profiler::span("Stage::GPU")
                        .arg("systems", profiler::Arg::U64(stage.system_indices.len() as u64));

                    #[cfg(not(feature = "gpu"))]
                    {
                        let _ = &stage;
                        return Err(ECSError::from(ExecutionError::GpuNotEnabled));
                    }

                    #[cfg(feature = "gpu")]
                    {
                        // Sequential execution on GPU.
                        for &idx in &stage.system_indices {
                            let system = &self.systems[idx];

                            profiler::next_arg("system_id", profiler::Arg::U64(system.id() as u64));
                            profiler::next_arg("backend", profiler::Arg::Str("GPU".to_string()));

                            let _sg = profiler::span_fmt(format_args!("System::{}", system.name()));

                            let gpu_cap = system.gpu().ok_or_else(|| {
                                ECSError::from(ExecutionError::SchedulerInvariantViolation)
                            })?;

                            {
                                let _g_exec = profiler::span("GPU::execute_gpu_system");
                                gpu::execute_gpu_system(ecs, system.as_ref(), gpu_cap)?;
                            }

                            ecs.clear_borrows();
                        }

                    }
                }
            }
        }

        Ok(())
    }
}

#[inline]
fn or_signature_in_place(
    destination: &mut Signature,
    source: &Signature
) {
    for (d, s) in destination.components.iter_mut().zip(source.components.iter()) {
        *d |= *s;
    }
}
