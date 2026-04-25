// Run (CPU-only):
//   cargo test --test sugarscape_basic -- --nocapture
//
// Run with GPU backend enabled:
//   cargo test --features gpu --test sugarscape_basic -- --nocapture
//
// Run with profiling:
//   cargo test --features profiling --test sugarscape_basic -- --nocapture
//
// Run GPU + profiling:
//   cargo test --features gpu_profiling --test sugarscape_basic -- --nocapture

mod sugarscape;

use std::sync::{Arc, RwLock};

use sugarscape::components::*;

use sugarscape::cpu::*;

use abm_framework::{
    Bundle, ComponentRegistry, ECSData, ECSManager,
    EntityShards, Scheduler, Command, Count, Sum, ECSResult,
};

use abm_framework::{span, Arg};

#[cfg(feature = "profiling")]
use abm_framework::{init, shutdown, thread_name};

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

/// Build a frozen component registry with all sugarscape components.
///
/// When the `gpu` feature is enabled, components that participate in GPU
/// compute shaders are registered via [`ComponentRegistry::register_gpu`]
/// so that their `gpu_usage` flag is set and the GPU mirror layer accepts
/// them.
fn make_registry() -> Arc<RwLock<ComponentRegistry>> {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    {
        let mut reg = registry.write().unwrap();

        // Components that are CPU-only regardless of feature flags.
        reg.register::<AgentTag>().unwrap();

        // Components that participate in GPU compute shaders when the gpu
        // feature is active.  On CPU-only builds they are registered normally.
        #[cfg(feature = "gpu")]
        {
            reg.register_gpu::<Position>().unwrap();
            reg.register_gpu::<Vision>().unwrap();
            reg.register_gpu::<RNG>().unwrap();
            reg.register_gpu::<Sugar>().unwrap();
            reg.register_gpu::<Metabolism>().unwrap();
            reg.register_gpu::<Alive>().unwrap();
        }

        #[cfg(not(feature = "gpu"))]
        {
            reg.register::<Position>().unwrap();
            reg.register::<Vision>().unwrap();
            reg.register::<RNG>().unwrap();
            reg.register::<Sugar>().unwrap();
            reg.register::<Metabolism>().unwrap();
            reg.register::<Alive>().unwrap();
        }

        reg.freeze();
    }
    registry
}

/// Total agent population for this test.
///
/// This is a single source of truth used for both the spawn loop and any
/// per-agent GPU buffer allocations (e.g. [`AgentIntentBuffers`]). Keeping
/// these in lockstep avoids out-of-bounds storage-buffer writes on the GPU,
/// which on D3D12/Vulkan typically surface asynchronously as
/// `Device::poll: device lost` (the queue is killed by the driver, then the
/// next blocking poll observes the loss).
const N_AGENTS: u32 = 1_000_000;

#[test]
fn sugarscape_basic_abm() -> ECSResult<()> {
    // ─── PROFILER SETUP ───────────────────────────────────────────────────────
    #[cfg(all(feature = "profiling", not(feature = "gpu")))]
    {
        init("profile/sugarscape_cpu_trace.json");
        thread_name("Main");
    }

    #[cfg(all(feature = "profiling", feature = "gpu"))]
    {
        init("profile/sugarscape_gpu_trace.json");
        thread_name("Main");
    }

    let registry = make_registry();
    let reg = registry.read().unwrap();

    let w = 600;
    let h = 600;

    let shards = EntityShards::new(4)?;
    let ecs = ECSManager::new(ECSData::new(shards, registry.clone()));
    let world = ecs.world_ref();

    // ─── GRID SETUP ───────────────────────────────────────────────────────────
    #[cfg(not(feature = "gpu"))]
    let grid = {
        let _g = span("setup::cpu_grid");
        std::sync::Arc::new(std::sync::Mutex::new(Grid::new(w, h)))
    };

    #[cfg(feature = "gpu")]
    let (sugar_grid_id, intent_id) = {
        let _g = span("setup::gpu_grid");

        let mut capacity = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                capacity.push(sugar_capacity_hills(x, y, w, h));
            }
        }

        let sugar_grid_id = world.register_gpu_resource(
            SugarGrid::new(w as u32, h as u32, capacity)
        )?;
        // IMPORTANT: AgentIntentBuffers must be sized to the maximum agent
        // population, not a smaller arbitrary constant. Compute shaders are
        // dispatched with `entity_len` equal to the live agent count, and
        // every thread `i < entity_len` writes to `agent_target[i]` /
        // `agent_score[i]`. If the buffer is shorter than `entity_len`,
        // those writes go out-of-bounds, the GPU queue is killed
        // asynchronously, and the next `Device::poll` panics with
        // "Parent device is lost".
        let intent_id = world.register_gpu_resource(
            AgentIntentBuffers::new(N_AGENTS as usize)
        )?;

        (sugar_grid_id, intent_id)
    };

    // ─── SPAWN AGENTS ─────────────────────────────────────────────────────────
    {
        let _g = span("setup::spawn_agents");

        let agent_tag_id = reg.id_of::<AgentTag>().unwrap();
        let position_id = reg.id_of::<Position>().unwrap();
        let sugar_id = reg.id_of::<Sugar>().unwrap();
        let metabolism_id = reg.id_of::<Metabolism>().unwrap();
        let vision_id = reg.id_of::<Vision>().unwrap();
        let rng_id = reg.id_of::<RNG>().unwrap();
        let alive_id = reg.id_of::<Alive>().unwrap();

        world.with_exclusive(|_| {
            for i in 0..N_AGENTS {
                let mut seed = i as u64 ^ 0x9E3779B97F4A7C15;
                let x = rng_range(&mut seed, w as u32) as i32;
                let y = rng_range(&mut seed, h as u32) as i32;

                let mut b = Bundle::new();
                b.insert(agent_tag_id, AgentTag(0));
                b.insert(position_id, Position { x, y });
                b.insert(sugar_id, Sugar(1.0));
                b.insert(metabolism_id, Metabolism(0.1));
                b.insert(vision_id, Vision(2));
                b.insert(rng_id, RNG { state: seed });
                b.insert(alive_id, Alive(1));

                world.defer(Command::Spawn { bundle: b })?;
            }
            Ok(())
        })?;
    }

    ecs.apply_deferred_commands()?;

    // ─── SCHEDULER ────────────────────────────────────────────────────────────
    let mut scheduler = Scheduler::new();

    #[cfg(not(feature = "gpu"))]
    {
        scheduler.add_system(MoveAndHarvestSystem::new(grid.clone(), &reg));
        scheduler.add_system(SugarRegrowthSystem::new(grid.clone(), 4.0));
        scheduler.add_system(MetabolismSystem::new(&reg));
        scheduler.add_system(DeathSystem::new(&reg));
    }

    #[cfg(feature = "gpu")]
    {
        scheduler.add_system(ClearOccupancyGpuSystem::new(sugar_grid_id));
        scheduler.add_system(AgentIntentGpuSystem::new(sugar_grid_id, intent_id, &reg));
        scheduler.add_system(ResolveIntentCpuSystem::new(sugar_grid_id, intent_id));
        scheduler.add_system(ResolveHarvestGpuSystem::new(sugar_grid_id, intent_id, &reg));
        scheduler.add_system(SugarRegrowthGpuSystem::new(sugar_grid_id));
        scheduler.add_system(MetabolismGpuSystem::new(&reg));
        scheduler.add_system(DeathGpuSystem::new(&reg));
    }

    drop(reg);

    // ─── SIMULATION LOOP ──────────────────────────────────────────────────────
    for step in 0..20 {
        let _tick = span("tick")
            .arg("step", Arg::U64(step as u64));

        {
            let _run = span("ecs::run");
            ecs.run(&mut scheduler)?;
        }

        #[cfg(feature = "gpu")]
        {
            let _g = span("gpu::sync_pending_to_cpu");
            gpu::sync_pending_to_cpu(world, &[])?;
        }

        let q = world.query()?.read::<Sugar>()?.read::<Alive>()?.build()?;

        let sum = {
            let _g = span("reduce::sum_sugar_alive");
            world.reduce_read2::<Sugar, Alive, Sum>(
                q.clone(),
                Sum::default,
                |acc, s, alive| if alive.0 == 1 { acc.0 += s.0 as f64 },
                |a, b| a.0 += b.0,
            )?
        };

        let count = {
            let _g = span("reduce::count_alive");
            world.reduce_read2::<Sugar, Alive, Count>(
                q,
                Count::default,
                |acc, _, alive| if alive.0 == 1 { acc.0 += 1 },
                |a, b| a.0 += b.0,
            )?
        };

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

    #[cfg(feature = "profiling")]
    shutdown();

    Ok(())
}
