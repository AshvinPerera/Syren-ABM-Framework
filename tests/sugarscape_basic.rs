// tests/sugarscape_basic.rs
//
// Run (CPU-only):
//   cargo test --test sugarscape_basic -- --nocapture
//
// Run with GPU backend enabled:
//   cargo test --features gpu --test sugarscape_basic -- --nocapture
//
// Notes:
// - This implementation keeps the “grid” as an external resource (Arc<Mutex<Grid>>), so 
// movement/harvest stays CPU (it has mutex contention + irregular access).
// - Metabolism + Death are executed on GPU as compute shaders.
// - After ecs.run(), call gpu::sync_pending_to_cpu(...) so the CPU reductions see the
//   latest Sugar/Alive values written on GPU.

use std::sync::{Arc, Mutex, Once};

use abm_framework::engine::component::{
    Bundle, component_id_of, freeze_components, register_component,
};
use abm_framework::engine::entity::EntityShards;
use abm_framework::engine::manager::{ECSData, ECSManager, ECSReference};
use abm_framework::engine::scheduler::Scheduler;
use abm_framework::engine::systems::{AccessSets, System, SystemBackend};
use abm_framework::engine::commands::Command;
use abm_framework::engine::error::{ECSError, ECSResult};
use abm_framework::engine::reduce::{Count, Sum};

#[cfg(feature = "gpu")]
use abm_framework::engine::component::{register_gpu_component, GPUPod};

#[cfg(feature = "gpu")]
use abm_framework::engine::systems::GpuSystem;

#[cfg(feature = "gpu")]
use abm_framework::gpu;

/// Components

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct AgentTag(pub u8);

#[repr(C)]
#[derive(Clone, Copy)]
struct Position {
    x: i32,
    y: i32,
}

// GPU-safe, single-field POD wrappers:
#[repr(transparent)]
#[derive(Clone, Copy)]
struct Sugar(pub f32);

#[repr(transparent)]
#[derive(Clone, Copy)]
struct Metabolism(pub f32);

#[repr(transparent)]
#[derive(Clone, Copy)]
struct Vision(pub i32);

#[repr(C)]
#[derive(Clone, Copy)]
struct RNG {
    state: u64,
}

#[repr(transparent)]
#[derive(Clone, Copy)]
struct Alive(pub u32); // 1 = alive, 0 = dead

#[cfg(feature = "gpu")]
unsafe impl GPUPod for Sugar {}
#[cfg(feature = "gpu")]
unsafe impl GPUPod for Metabolism {}
#[cfg(feature = "gpu")]
unsafe impl GPUPod for Alive {}

/// Deterministic RNG (xorshift64*)

#[inline]
fn rng_next_u32(state: &mut u64) -> u32 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    ((x.wrapping_mul(0x2545F4914F6CDD1D)) >> 32) as u32
}

#[inline]
fn rng_range(state: &mut u64, n: u32) -> u32 {
    if n <= 1 { return 0; }
    rng_next_u32(state) % n
}

/// External Grid Resource

#[derive(Clone, Copy)]
struct Cell {
    current: f32,
    capacity: f32,
    occupant: bool,
}

struct Grid {
    w: i32,
    h: i32,
    cells: Vec<Cell>,
}

impl Grid {
    fn new(w: i32, h: i32) -> Self {
        let mut cells = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let cap = sugar_capacity_hills(x, y, w, h);
                cells.push(Cell { current: cap, capacity: cap, occupant: false });
            }
        }
        Self { w, h, cells }
    }

    #[inline]
    fn idx(&self, x: i32, y: i32) -> usize {
        (y * self.w + x) as usize
    }

    #[inline]
    fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && x < self.w && y >= 0 && y < self.h
    }

    fn regrow(&mut self, rate: f32) {
        for c in &mut self.cells {
            c.current = (c.current + rate).min(c.capacity);
        }
    }

    fn clear_occupancy(&mut self) {
        for c in &mut self.cells {
            c.occupant = false;
        }
    }

    fn set_occupant(&mut self, x: i32, y: i32) {
        let i = self.idx(x, y);
        self.cells[i].occupant = true;
    }

    fn is_free(&self, x: i32, y: i32) -> bool {
        !self.cells[self.idx(x, y)].occupant
    }

    fn sugar_at(&self, x: i32, y: i32) -> f32 {
        self.cells[self.idx(x, y)].current
    }

    fn harvest(&mut self, x: i32, y: i32) -> f32 {
        let i = self.idx(x, y);
        let s = self.cells[i].current;
        self.cells[i].current = 0.0;
        s
    }
}

/// Two-hill Sugarscape landscape
fn sugar_capacity_hills(x: i32, y: i32, w: i32, h: i32) -> f32 {
    let (cx1, cy1) = (w / 4, h / 4);
    let (cx2, cy2) = (3 * w / 4, 3 * h / 4);

    let d1 = (x - cx1).abs() + (y - cy1).abs();
    let d2 = (x - cx2).abs() + (y - cy2).abs();
    let d = d1.min(d2) as f32;

    (10.0 - 0.2 * d).max(1.0)
}

/// Systems

/// Grid regrowth is CPU-only.
struct SugarRegrowthSystem {
    grid: Arc<Mutex<Grid>>,
    rate: f32,
}

impl System for SugarRegrowthSystem {
    fn id(&self) -> u16 { 1 }

    fn backend(&self) -> SystemBackend { SystemBackend::CPU }

    fn access(&self) -> AccessSets {
        // Only touches external grid resource
        AccessSets::default()
    }

    fn run(&self, _ecs: ECSReference<'_>) -> ECSResult<()> {
        let mut g = self.grid.lock().unwrap();
        g.regrow(self.rate);
        Ok(())
    }
}

/// CPU movement + harvest (irregular grid access + mutex).
struct MoveAndHarvestSystem {
    grid: Arc<Mutex<Grid>>,
}

impl System for MoveAndHarvestSystem {
    fn id(&self) -> u16 { 2 }

    fn backend(&self) -> SystemBackend { SystemBackend::CPU }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();

        a.read.set(component_id_of::<Vision>().unwrap());

        a.write.set(component_id_of::<Position>().unwrap());
        a.write.set(component_id_of::<Sugar>().unwrap());
        a.write.set(component_id_of::<RNG>().unwrap());
        a.write.set(component_id_of::<Alive>().unwrap());

        a
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        // Clear occupancy
        {
            let mut g = self.grid.lock().unwrap();
            g.clear_occupancy();
        }

        // Mark occupancy from current positions of living agents
        {
            let grid = self.grid.clone();
            let q = ecs.query()?
                .read::<Position>()?
                .read::<Alive>()?
                .build()?;

            ecs.for_each_read2::<Position, Alive>(q, move |pos, alive| {
                if alive.0 == 0 { return; }
                let mut g = grid.lock().unwrap();
                if g.in_bounds(pos.x, pos.y) {
                    g.set_occupant(pos.x, pos.y);
                }
            })?;
        }

        // Move + harvest:
        // Reads: Vision, Position
        // Writes: Position, Sugar, RNG, Alive
        let grid = self.grid.clone();

        let query = ecs.query()?
            .read::<Vision>()?
            .write::<Position>()?
            .write::<Sugar>()?
            .write::<RNG>()?
            .write::<Alive>()?
            .build()?;

        ecs.for_each_abstraction(query, move |reads, writes| unsafe {
            let vision = abm_framework::engine::storage::cast_slice::<Vision>(reads[0].as_ptr(), reads[0].len());
            let pos = abm_framework::engine::storage::cast_slice_mut::<Position>(writes[0].as_mut_ptr(), writes[0].len());
            let sugar= abm_framework::engine::storage::cast_slice_mut::<Sugar>(writes[1].as_mut_ptr(), writes[1].len());
            let rng = abm_framework::engine::storage::cast_slice_mut::<RNG>(writes[2].as_mut_ptr(), writes[2].len());
            let alive = abm_framework::engine::storage::cast_slice_mut::<Alive>(writes[3].as_mut_ptr(), writes[3].len());

            let dirs = [(1,0), (-1,0), (0,1), (0,-1)];

            for i in 0..vision.len() {
                if alive[i].0 == 0 { continue; }
                let (x0, y0) = (pos[i].x, pos[i].y);

                let mut g = grid.lock().unwrap();
                if !g.in_bounds(x0, y0) { continue; }

                let v = vision[i].0.max(1).min(50);
                let mut best = -1.0;
                let mut best_xy = (x0, y0);

                for (dx, dy) in dirs {
                    for s in 1..=v {
                        let (nx, ny) = (x0 + dx * s, y0 + dy * s);
                        if !g.in_bounds(nx, ny) || !g.is_free(nx, ny) { break; }
                        let sc = g.sugar_at(nx, ny);
                        if sc > best {
                            best = sc;
                            best_xy = (nx, ny);
                        }
                    }
                }

                pos[i].x = best_xy.0;
                pos[i].y = best_xy.1;
                sugar[i].0 += g.harvest(best_xy.0, best_xy.1);
                rng_next_u32(&mut rng[i].state);
            }
        })?;

        Ok(())
    }
}


/// GPU Systems
/// These are executed by the scheduler via abm_framework::gpu::execute_gpu_system.

#[cfg(feature = "gpu")]
struct MetabolismGpuSystem;

#[cfg(feature = "gpu")]
impl System for MetabolismGpuSystem {
    fn id(&self) -> u16 { 3 }

    fn backend(&self) -> SystemBackend { SystemBackend::GPU }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Metabolism>().unwrap());
        a.read.set(component_id_of::<Alive>().unwrap());
        a.write.set(component_id_of::<Sugar>().unwrap());
        a
    }

    // Scheduler won't call this in GPU stage, but trait requires it.
    fn run(&self, _ecs: ECSReference<'_>) -> ECSResult<()> { Ok(()) }

    fn gpu(&self) -> Option<&dyn GpuSystem> { Some(self) }
}

#[cfg(feature = "gpu")]
impl GpuSystem for MetabolismGpuSystem {
    fn shader(&self) -> &'static str {
        r#"
struct Params {
  entity_len: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read> metab : array<f32>;
@group(0) @binding(1) var<storage, read> alive : array<u32>;
@group(0) @binding(2) var<storage, read_write> sugar : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.entity_len) { return; }
  if (alive[i] == 0u) { return; }
  sugar[i] = sugar[i] - metab[i];
}
"#
    }

    fn entry_point(&self) -> &'static str { "main" }

    fn workgroup_size(&self) -> u32 { 256 }
}

#[cfg(feature = "gpu")]
struct DeathGpuSystem;

#[cfg(feature = "gpu")]
impl System for DeathGpuSystem {
    fn id(&self) -> u16 { 4 }

    fn backend(&self) -> SystemBackend { SystemBackend::GPU }

    fn access(&self) -> AccessSets {
        let mut a = AccessSets::default();
        a.read.set(component_id_of::<Sugar>().unwrap());
        a.write.set(component_id_of::<Alive>().unwrap());
        a
    }

    fn run(&self, _ecs: ECSReference<'_>) -> ECSResult<()> { Ok(()) }

    fn gpu(&self) -> Option<&dyn GpuSystem> { Some(self) }
}

#[cfg(feature = "gpu")]
impl GpuSystem for DeathGpuSystem {
    fn shader(&self) -> &'static str {
        r#"
struct Params {
  entity_len: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read> sugar : array<f32>;
@group(0) @binding(1) var<storage, read_write> alive : array<u32>;
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.entity_len) { return; }
  if (alive[i] == 0u) { return; }
  if (sugar[i] <= 0.0) {
    alive[i] = 0u;
  }
}
"#
    }

    fn entry_point(&self) -> &'static str { "main" }

    fn workgroup_size(&self) -> u32 { 256 }
}

/// One-time component initialization 

static INIT: Once = Once::new();

fn init_components() -> ECSResult<()> {
    let mut out: ECSResult<()> = Ok(());
    INIT.call_once(|| {
        out = (|| {
            // Deterministic registration order.
            register_component::<AgentTag>()?;
            register_component::<Position>()?;

            // Mark the components used by GPU stages as GPU-safe:
            #[cfg(feature = "gpu")]
            {
                register_gpu_component::<Sugar>()?;
                register_gpu_component::<Metabolism>()?;
                register_component::<Vision>()?; // CPU-only
                register_component::<RNG>()?; // CPU-only
                register_gpu_component::<Alive>()?;
            }

            #[cfg(not(feature = "gpu"))]
            {
                register_component::<Sugar>()?;
                register_component::<Metabolism>()?;
                register_component::<Vision>()?;
                register_component::<RNG>()?;
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

    let w = 200;
    let h = 200;

    let grid = Arc::new(Mutex::new(Grid::new(w, h)));

    // ECS setup
    let shards = EntityShards::new(4);
    let ecs = ECSManager::new(ECSData::new(shards));
    let world = ecs.world_ref();

    // Spawn agents
    world.with_exclusive(|_| -> Result<(), ECSError> {
        for i in 0..200_000u32 {
            let mut seed = (i as u64).wrapping_add(0x9E3779B97F4A7C15);

            let x = rng_range(&mut seed, w as u32) as i32;
            let y = rng_range(&mut seed, h as u32) as i32;

            let vision = 1 + (rng_range(&mut seed, 6) as i32);
            let metab  = 1.0 + (rng_range(&mut seed, 16) as f32) * 0.2;

            let mut b = Bundle::new();
            b.insert(component_id_of::<AgentTag>()?, AgentTag(0));
            b.insert(component_id_of::<Position>()?, Position { x, y });
            b.insert(component_id_of::<Sugar>()?, Sugar(10.0));
            b.insert(component_id_of::<Metabolism>()?, Metabolism(metab));
            b.insert(component_id_of::<Vision>()?, Vision(vision));
            b.insert(component_id_of::<RNG>()?, RNG { state: seed ^ 0xD1B54A32D192ED03 });
            b.insert(component_id_of::<Alive>()?, Alive(1));

            world.defer(Command::Spawn { bundle: b })?;
        }
        Ok(())
    })?;

    ecs.apply_deferred_commands()?;

    // Scheduler plan:
    // CPU: regrow grid
    // CPU: move+harvest (writes Sugar/Alive/etc on CPU, which marks dirty chunks under gpu feature)
    // GPU: metabolism (writes Sugar)
    // GPU: death (writes Alive)
    let mut scheduler = Scheduler::new();
    scheduler.add_system(SugarRegrowthSystem { grid: grid.clone(), rate: 1.0 });
    scheduler.add_system(MoveAndHarvestSystem { grid: grid.clone() });

    #[cfg(feature = "gpu")]
    {
        scheduler.add_system(MetabolismGpuSystem);
        scheduler.add_system(DeathGpuSystem);
    }

    // run without --features gpu, fall back to CPU implementations:
    #[cfg(not(feature = "gpu"))]
    {
        // CPU metabolism
        struct MetabolismCpu;
        impl System for MetabolismCpu {
            fn id(&self) -> u16 { 3 }
            fn access(&self) -> AccessSets {
                let mut a = AccessSets::default();
                a.read.set(component_id_of::<Metabolism>().unwrap());
                a.read.set(component_id_of::<Alive>().unwrap());
                a.write.set(component_id_of::<Sugar>().unwrap());
                a
            }
            fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
                let q = ecs.query()?
                    .read::<Metabolism>()?
                    .read::<Alive>()?
                    .write::<Sugar>()?
                    .build()?;
                ecs.for_each_read2_write1::<Metabolism, Alive, Sugar>(q, |m, alive, s| {
                    if alive.0 == 0 { return; }
                    s.0 -= m.0;
                })?;
                Ok(())
            }
        }

        // CPU death
        struct DeathCpu;
        impl System for DeathCpu {
            fn id(&self) -> u16 { 4 }
            fn access(&self) -> AccessSets {
                let mut a = AccessSets::default();
                a.read.set(component_id_of::<Sugar>().unwrap());
                a.write.set(component_id_of::<Alive>().unwrap());
                a
            }
            fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
                let q = ecs.query()?
                    .read::<Sugar>()?
                    .write::<Alive>()?
                    .build()?;
                ecs.for_each_read_write::<Sugar, Alive>(q, |s, alive| {
                    if alive.0 == 1 && s.0 <= 0.0 { alive.0 = 0; }
                })?;
                Ok(())
            }
        }

        scheduler.add_system(MetabolismCpu);
        scheduler.add_system(DeathCpu);
    }

    // Run simulation + metrics
    for step in 0..300 {
        ecs.run(&mut scheduler)?;

        // If the last scheduler stage was GPU, the CPU-side arrays may be stale until download.
        #[cfg(feature = "gpu")]
        {
            gpu::sync_pending_to_cpu(ecs.world_ref())?;
        }

        let world = ecs.world_ref();
        let q = world.query()?.read::<Sugar>()?.read::<Alive>()?.build()?;

        let sum = world.reduce_read2::<Sugar, Alive, Sum>(
            q.clone(),
            Sum::default,
            |acc, s, alive| if alive.0 == 1 { acc.0 += s.0 as f64; },
            |a, b| a.0 += b.0,
        )?;

        let count = world.reduce_read2::<Sugar, Alive, Count>(
            q,
            Count::default,
            |acc, _, alive| if alive.0 == 1 { acc.0 += 1; },
            |a, b| a.0 += b.0,
        )?;

        let avg = if count.0 > 0 { sum.0 / count.0 as f64 } else { 0.0 };
        println!("{step},{avg},{alive}", alive = count.0);

        if count.0 == 0 { break; }
    }

    Ok(())
}
