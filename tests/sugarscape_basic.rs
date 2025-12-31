use std::sync::{Arc, Mutex, Once};

use abm_framework::engine::component::{
    Bundle, register_component, freeze_components, component_id_of,
};
use abm_framework::engine::entity::EntityShards;
use abm_framework::engine::manager::{ECSManager, ECSReference, ECSData};
use abm_framework::engine::systems::{AccessSets, System};
use abm_framework::engine::scheduler::Scheduler;
use abm_framework::engine::commands::Command;
use abm_framework::engine::error::{ECSError, ECSResult};
use abm_framework::engine::reduce::{Count, Sum};

/// ===== Components (FULLY SoA) =====

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct AgentTag(pub u8);

#[derive(Clone, Copy)]
struct Position {
    x: i32,
    y: i32,
}

#[derive(Clone, Copy)]
struct Sugar(pub f32);

#[derive(Clone, Copy)]
struct Metabolism(pub f32);

#[derive(Clone, Copy)]
struct Vision(pub i32);

#[derive(Clone, Copy)]
struct RNG {
    state: u64,
}

#[derive(Clone, Copy)]
struct Alive(pub bool);

/// ===== Deterministic RNG =====

#[inline]
fn rng_next_u32(state: &mut u64) -> u32 {
    // xorshift64*
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

/// ===== External Grid Resource (O(1) lookup by x,y) =====

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
                cells.push(Cell {
                    current: cap,
                    capacity: cap,
                    occupant: false,
                });
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

/// Two-hill landscape
fn sugar_capacity_hills(x: i32, y: i32, w: i32, h: i32) -> f32 {
    let (cx1, cy1) = (w / 4, h / 4);
    let (cx2, cy2) = (3 * w / 4, 3 * h / 4);

    let d1 = (x - cx1).abs() + (y - cy1).abs();
    let d2 = (x - cx2).abs() + (y - cy2).abs();
    let d = d1.min(d2) as f32;

    (10.0 - 0.2 * d).max(1.0)
}

/// ===== Systems =====

struct SugarRegrowthSystem {
    grid: Arc<Mutex<Grid>>,
    rate: f32,
}

impl System for SugarRegrowthSystem {
    fn id(&self) -> u16 { 1 }

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

/// Sugarscape movement:
/// - rebuild occupancy from current positions
/// - for each agent: scan N/S/E/W up to vision
/// - choose max sugar among free cells
/// - break ties randomly using per-agent RNG
/// - move + harvest
///
/// IMPORTANT (ECS rule):
/// Position is WRITE-ONLY here (you must not read+write the same component in one query).
/// We read the "old" position from the write slice before updating it.
struct MoveAndHarvestSystem {
    grid: Arc<Mutex<Grid>>,
}

impl System for MoveAndHarvestSystem {
    fn id(&self) -> u16 { 2 }

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
        // 1) Clear occupancy
        {
            let mut g = self.grid.lock().unwrap();
            g.clear_occupancy();
        }

        // 2) Mark occupancy for living agents (Position + Alive) using a READ-ONLY pass
        {
            let grid = self.grid.clone();
            let q = ecs.query()?
                .read::<Position>()?
                .read::<Alive>()?
                .build()?;

            ecs.for_each_read2::<Position, Alive>(q, move |pos, alive| {
                if !alive.0 { return; }
                let mut g = grid.lock().unwrap();
                if g.in_bounds(pos.x, pos.y) {
                    g.set_occupant(pos.x, pos.y);
                }
            })?;
        }

        // 3) Move+Harvest
        //
        // Reads:  Vision
        // Writes: Position, Sugar, RNG, Alive
        //
        // NOTE: Position is WRITE-ONLY in the query. We read old Position from the write slice.
        let grid = self.grid.clone();

        let qb = ecs.query()?;
        let query = qb
            .read::<Vision>()?
            .write::<Position>()?
            .write::<Sugar>()?
            .write::<RNG>()?
            .write::<Alive>()?
            .build()?;

        ecs.for_each_abstraction(query, move |reads, writes| unsafe {
            let vision = abm_framework::engine::storage::cast_slice::<Vision>(
                reads[0].as_ptr(),
                reads[0].len(),
            );

            let pos = abm_framework::engine::storage::cast_slice_mut::<Position>(
                writes[0].as_mut_ptr(),
                writes[0].len(),
            );
            let sugar = abm_framework::engine::storage::cast_slice_mut::<Sugar>(
                writes[1].as_mut_ptr(),
                writes[1].len(),
            );
            let rng = abm_framework::engine::storage::cast_slice_mut::<RNG>(
                writes[2].as_mut_ptr(),
                writes[2].len(),
            );
            let alive = abm_framework::engine::storage::cast_slice_mut::<Alive>(
                writes[3].as_mut_ptr(),
                writes[3].len(),
            );

            debug_assert_eq!(vision.len(), pos.len());
            debug_assert_eq!(vision.len(), sugar.len());
            debug_assert_eq!(vision.len(), rng.len());
            debug_assert_eq!(vision.len(), alive.len());

            let dirs = [(1,0), (-1,0), (0,1), (0,-1)];

            for i in 0..vision.len() {
                if !alive[i].0 { continue; }

                let (x0, y0) = (pos[i].x, pos[i].y);

                let mut g = grid.lock().unwrap();
                if !g.in_bounds(x0, y0) {
                    pos[i].x = x0.clamp(0, g.w - 1);
                    pos[i].y = y0.clamp(0, g.h - 1);
                    continue;
                }

                let v = vision[i].0.max(1).min(50);
                let mut best = -1.0f32;

                // store up to 64 ties
                let mut ties: [(i32, i32); 64] = [(0,0); 64];
                let mut ties_len = 0usize;

                for (dx, dy) in dirs {
                    for step in 1..=v {
                        let nx = x0 + dx * step;
                        let ny = y0 + dy * step;
                        if !g.in_bounds(nx, ny) { break; }
                        if !g.is_free(nx, ny) { continue; }

                        let s = g.sugar_at(nx, ny);
                        if s > best {
                            best = s;
                            ties_len = 0;
                            ties[ties_len] = (nx, ny);
                            ties_len = 1;
                        } else if (s - best).abs() <= f32::EPSILON && ties_len < ties.len() {
                            ties[ties_len] = (nx, ny);
                            ties_len += 1;
                        }
                    }
                }

                // If no candidate, harvest current cell
                if ties_len == 0 {
                    sugar[i].0 += g.harvest(x0, y0);
                    continue;
                }

                // Tie-break using per-agent RNG
                let pick = rng_range(&mut rng[i].state, ties_len as u32) as usize;
                let (tx, ty) = ties[pick];

                // Move + harvest
                pos[i].x = tx;
                pos[i].y = ty;
                sugar[i].0 += g.harvest(tx, ty);

                // Advance RNG
                let _ = rng_next_u32(&mut rng[i].state);
            }
        })?;

        Ok(())
    }
}

struct MetabolismSystem;

impl System for MetabolismSystem {
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
            if !alive.0 { return; }
            s.0 -= m.0;
        })?;

        Ok(())
    }
}

struct DeathSystem;

impl System for DeathSystem {
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
            if alive.0 && s.0 <= 0.0 {
                alive.0 = false;
            }
        })?;

        Ok(())
    }
}

static INIT: Once = Once::new();

fn init_components() -> ECSResult<()> {
    let mut out: ECSResult<()> = Ok(());
    INIT.call_once(|| {
        out = (|| {
            register_component::<AgentTag>()?;
            register_component::<Position>()?;
            register_component::<Sugar>()?;
            register_component::<Metabolism>()?;
            register_component::<Vision>()?;
            register_component::<RNG>()?;
            register_component::<Alive>()?;
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
        for i in 0..1_000_000u32 {
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
            b.insert(component_id_of::<Alive>()?, Alive(true));

            world.defer(Command::Spawn { bundle: b })?;
        }
        Ok(())
    })?;

    ecs.apply_deferred_commands()?;

    // Scheduler
    let mut scheduler = Scheduler::new();
    scheduler.add_system(SugarRegrowthSystem { grid: grid.clone(), rate: 1.0 });
    scheduler.add_system(MoveAndHarvestSystem { grid: grid.clone() });
    scheduler.add_system(MetabolismSystem);
    scheduler.add_system(DeathSystem);

    // Run simulation
    for step in 0..100 {
        ecs.run(&mut scheduler)?;

        let world = ecs.world_ref();
        let q = world.query()?.read::<Sugar>()?.read::<Alive>()?.build()?;

        let sum = world.reduce_read2::<Sugar, Alive, Sum>(
            q.clone(),
            Sum::default,
            |acc, s, alive| if alive.0 { acc.0 += s.0 as f64; },
            |a, b| a.0 += b.0,
        )?;

        let count = world.reduce_read2::<Sugar, Alive, Count>(
            q,
            Count::default,
            |acc, _, alive| if alive.0 { acc.0 += 1; },
            |a, b| a.0 += b.0,
        )?;

        let avg = if count.0 > 0 { sum.0 / count.0 as f64 } else { 0.0 };
        println!("{step},{avg},{alive}", alive = count.0);

        if count.0 == 0 { break; }
    }

    Ok(())
}
