mod sugarscape;

#[cfg(feature = "model")]
use std::collections::HashSet;

#[cfg(any(feature = "model", feature = "gpu"))]
use abm_framework::ECSResult;
#[cfg(feature = "gpu")]
use abm_framework::{ECSError, ExecutionError};
use sugarscape::axtell::*;

#[cfg(feature = "model")]
fn assert_unique_occupancy(agents: &[SugarAgent], width: u32, height: u32) {
    let mut seen = HashSet::new();
    for agent in agents {
        assert!(agent.x < width);
        assert!(agent.y < height);
        assert!(seen.insert((agent.x, agent.y)), "duplicate cell occupancy");
    }
}

#[cfg(feature = "model")]
#[test]
fn axtell_chapter2_cpu_model_preserves_core_invariants() -> ECSResult<()> {
    let config = SugarscapeConfig::default();
    let (mut model, state, _) = build_model(config, false);

    for _ in 0..8 {
        model.tick()?;
    }

    let agents = model_agents(&model)?;
    assert_eq!(agents.len(), config.population);
    assert_unique_occupancy(&agents, config.width, config.height);
    assert!(agents.iter().all(|agent| (1..=6).contains(&agent.vision)));
    assert!(agents
        .iter()
        .all(|agent| (1..=4).contains(&agent.metabolism)));
    assert!(agents
        .iter()
        .all(|agent| (60..=100).contains(&agent.max_age)));

    let grid = state.lock().unwrap();
    assert_eq!(grid.cells.len(), (config.width * config.height) as usize);
    assert!(grid
        .cells
        .iter()
        .all(|cell| cell.sugar <= cell.capacity && cell.capacity <= 4));
    Ok(())
}

#[cfg(feature = "model")]
#[test]
fn axtell_replaces_dead_agents_with_newborns() -> ECSResult<()> {
    let config = SugarscapeConfig {
        population: 16,
        seed: 7,
        ..SugarscapeConfig::default()
    };
    let (mut model, _state, agent_id) = build_model(config, false);

    let world = model.ecs().world_ref();
    let q = world.query()?.write::<SugarAgent>()?.build()?;
    world.for_each::<(abm_framework::Write<SugarAgent>,)>(q, &|agent| {
        if agent.0.id == 0 {
            agent.0.wealth = 0;
            agent.0.max_age = 0;
        }
    })?;

    model.tick()?;
    let agents = model_agents(&model)?;
    let newborn = agents.iter().find(|agent| agent.id == 0).unwrap();
    assert_eq!(newborn.age, 0);
    assert!((5..=25).contains(&newborn.wealth));
    assert_unique_occupancy(&agents, config.width, config.height);

    let _ = agent_id;
    Ok(())
}

#[test]
fn movement_uses_torus_cardinal_vision_and_random_ties() {
    let width = 5;
    let height = 5;
    let mut cells = vec![Cell::default(); (width * height) as usize];
    let idx = |x: u32, y: u32| (y * width + x) as usize;
    cells[idx(4, 0)] = Cell {
        sugar: 4,
        capacity: 4,
    };
    cells[idx(0, 4)] = Cell {
        sugar: 4,
        capacity: 4,
    };
    cells[idx(1, 1)] = Cell {
        sugar: 9,
        capacity: 9,
    };
    let occupied = vec![false; cells.len()];
    let agent = SugarAgent {
        id: 0,
        x: 0,
        y: 0,
        wealth: 10,
        metabolism: 1,
        vision: 1,
        age: 0,
        max_age: 80,
    };
    let mut rng = 1;
    let destination = choose_destination(agent, &cells, width, height, &occupied, &mut rng);
    assert!(
        destination == (4, 0) || destination == (0, 4),
        "diagonal sugar must not be visible, and torus cardinal cells must be"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_metabolism_path_matches_cpu_oracle() -> ECSResult<()> {
    let config = SugarscapeConfig {
        population: 64,
        seed: 123,
        ..SugarscapeConfig::default()
    };

    let (cpu_world, mut cpu_scheduler, _cpu_state, _) = build_raw_world(config, false)?;
    let (gpu_world, mut gpu_scheduler, _gpu_state, _) = build_raw_world(config, true)?;

    for _ in 0..4 {
        cpu_world.run(&mut cpu_scheduler)?;
        match gpu_world.run(&mut gpu_scheduler) {
            Ok(()) => {}
            Err(ECSError::Execute(ExecutionError::GpuInitFailed { message })) => {
                eprintln!("skipping Sugarscape GPU equality test: {message}");
                return Ok(());
            }
            Err(err) => return Err(err),
        }
    }

    assert_eq!(raw_agents(&cpu_world)?, raw_agents(&gpu_world)?);
    Ok(())
}
