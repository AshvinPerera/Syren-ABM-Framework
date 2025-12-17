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
//! derived from declared read/write access sets.
//!
//! ## Structural synchronization
//!
//! Deferred ECS commands (spawns, despawns, component mutations) are applied:
//! * **before** each stage begins,
//! * **after** each stage completes.
//!
//! This ensures that structural changes do not race with system execution.


use rayon::prelude::*;
use crate::engine::systems::System;
use crate::engine::manager::ECSManager;


/// A group of systems that can be executed in parallel.
///
/// ## Invariants
/// * All systems within a `Stage` have **non-conflicting access sets**
/// * Systems in a stage may safely run concurrently
///
/// Stages themselves must be executed sequentially.

pub struct Stage {
    /// Systems scheduled to run in this stage.
    pub systems: Vec<Box<dyn System>>,
}

/// Partitions a list of systems into parallel execution stages.
///
/// ## Algorithm
/// Systems are processed in deterministic order (by system ID) and assigned
/// greedily:
/// * Each system is placed into the first stage where it does not conflict
///   with existing systems.
/// * If no such stage exists, a new stage is created.
///
/// ## Determinism
/// Sorting by system ID ensures that stage construction is stable and
/// reproducible across runs.
///
/// ## Complexity
/// * O(n²) in the worst case (pathological conflict patterns)
/// * Expected to be small for typical ECS workloads
///
/// ## Returns
/// A vector of stages, each containing systems that may run in parallel.

pub fn make_stages(mut systems: Vec<Box<dyn System>>) -> Vec<Stage> {
    let mut stages: Vec<Stage> = Vec::new();

    systems.sort_by_key(|s| s.id());

    'next_system: for sys in systems.into_iter() {
        for stage in stages.iter_mut() {
            let conflict = stage.systems.iter()
                .any(|other| sys.access().conflicts_with(&other.access()));
            if !conflict {
                stage.systems.push(sys);
                continue 'next_system;
            }
        }
        stages.push(Stage { systems: vec![sys] });
    }
    stages
}

/// Partitions a list of systems into parallel execution stages.
///
/// ## Algorithm
/// Systems are processed in deterministic order (by system ID) and assigned
/// greedily:
/// * Each system is placed into the first stage where it does not conflict
///   with existing systems.
/// * If no such stage exists, a new stage is created.
///
/// ## Determinism
/// Sorting by system ID ensures that stage construction is stable and
/// reproducible across runs.
///
/// ## Complexity
/// * O(n²) in the worst case (pathological conflict patterns)
/// * Expected to be small for typical ECS workloads
///
/// ## Returns
/// A vector of stages, each containing systems that may run in parallel.

pub fn run_schedule(ecs_manager: &mut ECSManager, stages: &[Stage]) {
    for stage in stages {
        ecs_manager.apply_deferred_commands();

        stage.systems.par_iter().for_each(|sys| {
            let ecs_reference = ecs_manager.world_ref();
            sys.run(ecs_reference);
        });

        ecs_manager.apply_deferred_commands();
    }
}
