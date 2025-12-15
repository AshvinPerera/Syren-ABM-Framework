use rayon::prelude::*;
use crate::systems::System;
use crate::types::{SystemId, AccessSets};
use crate::world::World;

/// A "stage" = a set of systems that can run together safely.
pub struct Stage {
    pub systems: Vec<Box<dyn System>>,
}

/// Given systems, partition them into stages with no conflicts inside a stage (greedy coloring).
pub fn make_stages(mut systems: Vec<Box<dyn System>>) -> Vec<Stage> {
    let mut stages: Vec<Stage> = Vec::new();

    systems.sort_by_key(|s| s.id()); // deterministic

    'next_system: for sys in systems.into_iter() {
        // Try to place into an existing stage
        for stage in stages.iter_mut() {
            let conflict = stage.systems.iter()
                .any(|other| sys.access().conflicts_with(&other.access()));
            if !conflict {
                stage.systems.push(sys);
                continue 'next_system;
            }
        }
        // Otherwise start a new stage
        stages.push(Stage { systems: vec![sys] });
    }
    stages
}

/// Execute all stages sequentially; systems inside a stage run in parallel.
/// Each system is free to use the world's Phase-6 parallel chunk iterators.
pub fn run_schedule(world: &mut World, stages: &[Stage]) {
    for (i, stage) in stages.iter().enumerate() {
        // Start-of-stage safepoint: apply deferred commands from previous stage
        world.apply_deferred_commands();

        // Run systems in this stage in parallel
        stage.systems.par_iter().for_each(|sys| {
            // Each system gets a &mut World — but we can't hand out multiple &mut at once.
            // Pattern: lock a single short-lived mutex inside World to extract parallel chunk tasks
            // OR make each system's run function re-borrow &mut world internally under a lock
            // kept off the hot loop (shown in World below).
            world.with_exclusive(|w| {
                // The system builds its queries & kickoffs parallel chunk loops from Phase 6
                sys.run(w);
            });
        });

        // End-of-stage safepoint: apply any structural edits queued by systems during this stage
        world.apply_deferred_commands();
        // (Optional) metrics: world.end_stage(i);
    }
}
