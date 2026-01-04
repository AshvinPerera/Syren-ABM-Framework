// Run (CPU-only):
//   cargo test --test sugarscape_basic -- --nocapture
//
// Run with GPU backend enabled:
//   cargo test --features gpu --test sugarscape_basic -- --nocapture
//
// CPU path:
//   - External Grid (Arc<Mutex<Grid>>)
//   - CPU movement + harvest
//   - CPU metabolism + death
//
// GPU path:
//   - World-owned SugarGrid GPUResource
//   - GPU movement + harvest
//   - GPU metabolism + death
//   - CPU reductions after explicit sync

mod sugarscape;

use std::sync::Once;

use sugarscape::components::*;
use sugarscape::cpu::*;

use abm_framework::engine::component::{
    Bundle,
    component_id_of,
    register_component,
    freeze_components,
};

#[cfg(feature = "gpu")]
use abm_framework::engine::component::register_gpu_component;

use abm_framework::engine::entity::EntityShards;
use abm_framework::engine::manager::{ECSData, ECSManager};
use abm_framework::engine::scheduler::Scheduler;
use abm_framework::engine::commands::Command;
use abm_framework::engine::reduce::{Count, Sum};
use abm_framework::engine::error::ECSResult;

#[cfg(feature = "gpu")]
use abm_framework::gpu;

#[cfg(feature = "gpu")]
use sugarscape::gpu_resources::{SugarGrid, AgentIntentBuffers};

#[cfg(feature = "gpu")]
use sugarscape::gpu::{
    MetabolismGpuSystem,
    DeathGpuSystem,
    AgentIntentGpuSystem,
    ResolveHarvestGpuSystem,
    SugarRegrowthGpuSystem,
    ClearOccupancyGpuSystem
};


/// One-time component registration
static INIT: Once = Once::new();

fn init_components() -> ECSResult<()> {
    let mut out = Ok(());

    INIT.call_once(|| {
        out = (|| {
            register_component::<AgentTag>()?;

            #[cfg(feature = "gpu")]
            {
                register_gpu_component::<Position>()?;
                register_gpu_component::<Vision>()?;
                register_gpu_component::<RNG>()?; 
                register_gpu_component::<Sugar>()?;
                register_gpu_component::<Metabolism>()?;
                register_gpu_component::<Alive>()?;
            }

            #[cfg(not(feature = "gpu"))]
            {
                register_component::<Position>()?;
                register_component::<Vision>()?;
                register_component::<RNG>()?;               
                register_component::<Sugar>()?;
                register_component::<Metabolism>()?;
                register_component::<Alive>()?;
            }

            freeze_components()?;
            Ok(())
        })();
    });

    out
}

/// Test

#[test]
fn sugarscape_basic_abm() -> ECSResult<()> {
    init_components()?;

    let w = 600;
    let h = 600;

    let shards = EntityShards::new(4);
    let ecs = ECSManager::new(ECSData::new(shards));
    let world = ecs.world_ref();

    // CPU grid

    #[cfg(not(feature = "gpu"))]
    let grid = std::sync::Arc::new(std::sync::Mutex::new(Grid::new(w, h)));

    // GPU grid

    #[cfg(feature = "gpu")]
    let (sugar_grid_id, intent_id) = {
        let mut capacity = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                capacity.push(sugar_capacity_hills(x, y, w, h));
            }
        }

        let sugar_grid_id = world.register_gpu_resource(
            SugarGrid::new(w as u32, h as u32, capacity)
        )?;
        let intent_id = world.register_gpu_resource(
            AgentIntentBuffers::new(200_000)
        )?;

        (sugar_grid_id, intent_id)
    };

    // Spawn agents

    world.with_exclusive(|_| {
        for i in 0..1_000_000u32 {
            let mut seed = i as u64 ^ 0x9E3779B97F4A7C15;
            let x = rng_range(&mut seed, w as u32) as i32;
            let y = rng_range(&mut seed, h as u32) as i32;

            let mut b = Bundle::new();
            b.insert(component_id_of::<AgentTag>()?, AgentTag(0));
            b.insert(component_id_of::<Position>()?, Position { x, y });
            b.insert(component_id_of::<Sugar>()?, Sugar(1.0));
            b.insert(component_id_of::<Metabolism>()?, Metabolism(0.1));
            b.insert(component_id_of::<Vision>()?, Vision(2));
            b.insert(component_id_of::<RNG>()?, RNG { state: seed });
            b.insert(component_id_of::<Alive>()?, Alive(1));

            world.defer(Command::Spawn { bundle: b })?;
        }
        Ok(())
    })?;

    ecs.apply_deferred_commands()?;

    // Scheduler

    let mut scheduler = Scheduler::new();

    #[cfg(not(feature = "gpu"))]
    {
        scheduler.add_system(MoveAndHarvestSystem {
            grid: grid.clone(),
        });

        scheduler.add_system(SugarRegrowthSystem {
            grid: grid.clone(),
            rate: 4.0
        });

        scheduler.add_system(MetabolismSystem);

        scheduler.add_system(DeathSystem);
    }

    #[cfg(feature = "gpu")]
    {
        scheduler.add_system(ClearOccupancyGpuSystem::new(sugar_grid_id));
        scheduler.add_system(AgentIntentGpuSystem::new(sugar_grid_id, intent_id));
        scheduler.add_system(ResolveHarvestGpuSystem::new(sugar_grid_id, intent_id));
        scheduler.add_system(SugarRegrowthGpuSystem::new(sugar_grid_id));
        scheduler.add_system(MetabolismGpuSystem);
        scheduler.add_system(DeathGpuSystem);
    }

    // Run simulation

    for step in 0..20 {
        ecs.run(&mut scheduler)?;

        #[cfg(feature = "gpu")]
        gpu::sync_pending_to_cpu(world)?;

        let q = world.query()?.read::<Sugar>()?.read::<Alive>()?.build()?;

        let sum = world.reduce_read2::<Sugar, Alive, Sum>(
            q.clone(),
            Sum::default,
            |acc, s, alive| if alive.0 == 1 { acc.0 += s.0 as f64 },
            |a, b| a.0 += b.0,
        )?;

        let count = world.reduce_read2::<Sugar, Alive, Count>(
            q,
            Count::default,
            |acc, _, alive| if alive.0 == 1 { acc.0 += 1 },
            |a, b| a.0 += b.0,
        )?;

        let avg = if count.0 > 0 {
            sum.0 / count.0 as f64
        } else {
            0.0
        };

        println!("{step},{avg},{}", count.0);

        if count.0 == 0 {
            break;
        }
    }

    Ok(())
}
