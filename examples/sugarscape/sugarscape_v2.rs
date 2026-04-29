use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::time::Instant;

use rayon::prelude::*;

use abm_framework::{
    advanced::EntityShards, AccessSets, Arg, Bundle, Command, ComponentID, ComponentRegistry,
    ECSError, ECSManager, ECSReference, ECSResult, ExecutionError, Scheduler, System,
};

#[cfg(feature = "gpu")]
use abm_framework::{GPUPod, GpuSystem, SystemBackend};

const DEFAULT_AGENTS: usize = 1_000_000;
const DEFAULT_WIDTH: u32 = 4096;
const DEFAULT_HEIGHT: u32 = 4096;
const DEFAULT_TICKS: u32 = 20;
const DEFAULT_SEED: u64 = 0xA57E11;
const DEFAULT_GROWTH_RATE: u32 = 1;
#[cfg(feature = "profiling")]
const DEFAULT_PROFILE: &str = "profile/sugarscape_axtell_large.json";

const GROW_DONE: u32 = 0;
const MOVE_DONE: u32 = 1;
const METABOLISM_DONE: u32 = 2;
const REPLACEMENT_DONE: u32 = 3;

#[derive(Clone, Debug)]
struct Cli {
    config: SugarscapeConfig,
    ticks: u32,
    force_cpu: bool,
    profile_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
struct SugarscapeConfig {
    width: u32,
    height: u32,
    population: usize,
    growth_rate: u32,
    seed: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SugarAgent {
    id: u32,
    x: u32,
    y: u32,
    wealth: u32,
    metabolism: u32,
    vision: u32,
    age: u32,
    max_age: u32,
}

#[cfg(feature = "gpu")]
unsafe impl GPUPod for SugarAgent {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Cell {
    sugar: u32,
    capacity: u32,
}

#[derive(Clone, Debug)]
struct SugarscapeState {
    config: SugarscapeConfig,
    cells: Vec<Cell>,
    rng: u64,
    occupancy: Vec<bool>,
    order: Vec<u32>,
    dead: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetabolismBackend {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu,
}

impl MetabolismBackend {
    fn as_str(self) -> &'static str {
        match self {
            MetabolismBackend::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            MetabolismBackend::Gpu => "gpu",
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = match Cli::parse()? {
        Some(cli) => cli,
        None => return Ok(()),
    };

    #[cfg(feature = "profiling")]
    if let Some(path) = &cli.profile_path {
        abm_framework::init(path);
        eprintln!("profile trace: {}", path.display());
    }

    #[cfg(not(feature = "profiling"))]
    if cli.profile_path.is_some() {
        eprintln!("profiling requested, but trace output requires --features profiling");
    }

    abm_framework::thread_name("sugarscape-main");

    let result = run(cli);

    abm_framework::flush_thread();
    abm_framework::shutdown();

    result
}

fn run(cli: Cli) -> Result<(), Box<dyn Error>> {
    let backend = select_backend(cli.force_cpu);
    let started = Instant::now();

    let _run_span = abm_framework::span("Sugarscape::run")
        .arg("agents", Arg::U64(cli.config.population as u64))
        .arg("width", Arg::U64(cli.config.width as u64))
        .arg("height", Arg::U64(cli.config.height as u64))
        .arg("ticks", Arg::U64(cli.ticks as u64))
        .arg("backend", Arg::Str(backend.as_str().to_string()));

    eprintln!(
        "setup: agents={} grid={}x{} ticks={} backend={}",
        cli.config.population,
        cli.config.width,
        cli.config.height,
        cli.ticks,
        backend.as_str()
    );

    let setup_started = Instant::now();
    let (ecs, mut scheduler) = {
        let _setup_span = abm_framework::span("Sugarscape::setup");
        build_simulation(cli.config, backend)?
    };
    eprintln!(
        "setup complete in {:.3}s",
        setup_started.elapsed().as_secs_f64()
    );

    println!(
        "tick,agents,avg_sugar_per_tile,wealth_mean,wealth_p50,wealth_p90,wealth_p99,wealth_max,wealth_gini"
    );

    for tick in 1..=cli.ticks {
        let tick_started = Instant::now();
        {
            let _tick_span = abm_framework::span("Sugarscape::tick")
                .arg("tick", Arg::U64(tick as u64))
                .arg("agents", Arg::U64(cli.config.population as u64))
                .arg("width", Arg::U64(cli.config.width as u64))
                .arg("height", Arg::U64(cli.config.height as u64))
                .arg("backend", Arg::Str(backend.as_str().to_string()));
            ecs.run(&mut scheduler)?;
        }
        abm_framework::flush_thread();
        eprintln!(
            "tick {tick} complete in {:.3}s",
            tick_started.elapsed().as_secs_f64()
        );
    }

    eprintln!("total elapsed {:.3}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn select_backend(force_cpu: bool) -> MetabolismBackend {
    if force_cpu {
        return MetabolismBackend::Cpu;
    }

    #[cfg(feature = "gpu")]
    {
        match abm_framework::gpu::GPUContext::new() {
            Ok(_) => MetabolismBackend::Gpu,
            Err(err) => {
                eprintln!("GPU preflight failed; falling back to CPU metabolism: {err}");
                MetabolismBackend::Cpu
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("gpu feature is not enabled; using CPU metabolism");
        MetabolismBackend::Cpu
    }
}

fn build_simulation(
    config: SugarscapeConfig,
    backend: MetabolismBackend,
) -> Result<(ECSManager, Scheduler), Box<dyn Error>> {
    validate_config(config)?;

    let (registry, agent_id) = register_components()?;
    let mut state = SugarscapeState::new(config)?;
    let agents = initial_agents(&mut state)?;
    let state = Arc::new(Mutex::new(state));

    let shards = abm_framework::max_workers().max(1) as usize;
    let ecs = ECSManager::with_registry(EntityShards::new(shards)?, registry);
    spawn_agents(&ecs, agent_id, agents)?;
    let scheduler = make_scheduler(state, agent_id, backend);
    Ok((ecs, scheduler))
}

fn validate_config(config: SugarscapeConfig) -> Result<(), Box<dyn Error>> {
    if config.width == 0 || config.height == 0 {
        return Err("width and height must both be greater than zero".into());
    }
    let cells = checked_cell_count(config.width, config.height)?;
    if config.population == 0 {
        return Err("agents must be greater than zero".into());
    }
    if config.population > cells {
        return Err(format!(
            "agents ({}) must not exceed grid cells ({cells})",
            config.population
        )
        .into());
    }
    if config.population > u32::MAX as usize {
        return Err("agents must fit in u32 agent ids".into());
    }
    if cells > u32::MAX as usize {
        return Err("width * height must fit in u32 grid indexing for this example".into());
    }
    Ok(())
}

fn checked_cell_count(width: u32, height: u32) -> Result<usize, Box<dyn Error>> {
    let cells = (width as u64)
        .checked_mul(height as u64)
        .ok_or("width * height overflowed")?;
    if cells > usize::MAX as u64 {
        return Err("width * height does not fit in usize".into());
    }
    Ok(cells as usize)
}

fn register_components() -> ECSResult<(Arc<RwLock<ComponentRegistry>>, ComponentID)> {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let agent_id = {
        let mut registry = registry.write().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "component registry",
            })
        })?;
        #[cfg(feature = "gpu")]
        let agent_id = registry.register_gpu::<SugarAgent>()?;
        #[cfg(not(feature = "gpu"))]
        let agent_id = registry.register::<SugarAgent>()?;
        registry.freeze();
        agent_id
    };
    Ok((registry, agent_id))
}

impl SugarscapeState {
    fn new(config: SugarscapeConfig) -> Result<Self, Box<dyn Error>> {
        let cell_count = checked_cell_count(config.width, config.height)?;
        let mut cells = Vec::with_capacity(cell_count);
        for y in 0..config.height {
            for x in 0..config.width {
                let capacity = sugar_capacity(x, y, config.width, config.height);
                cells.push(Cell {
                    sugar: capacity,
                    capacity,
                });
            }
        }

        Ok(Self {
            config,
            cells,
            rng: config.seed,
            occupancy: vec![false; cell_count],
            order: Vec::with_capacity(config.population),
            dead: Vec::new(),
        })
    }

    #[inline]
    fn index(&self, x: u32, y: u32) -> usize {
        (y * self.config.width + x) as usize
    }

    fn growback(&mut self) {
        let growth = self.config.growth_rate;
        self.cells.par_iter_mut().for_each(|cell| {
            cell.sugar = cell.sugar.saturating_add(growth).min(cell.capacity);
        });
    }

    fn harvest(&mut self, x: u32, y: u32) -> u32 {
        let idx = self.index(x, y);
        let sugar = self.cells[idx].sugar;
        self.cells[idx].sugar = 0;
        sugar
    }
}

fn sugar_capacity(x: u32, y: u32, width: u32, height: u32) -> u32 {
    let peaks = [
        (width.saturating_mul(3) / 10, height.saturating_mul(3) / 10),
        (width.saturating_mul(7) / 10, height.saturating_mul(7) / 10),
    ];
    let d = peaks
        .iter()
        .map(|(px, py)| x.abs_diff(*px) + y.abs_diff(*py))
        .min()
        .unwrap_or(0);
    let scale = width.min(height).max(1);
    let normalized = ((d as u64) * 50 / (scale as u64)) as u32;
    match normalized {
        0..=3 => 4,
        4..=6 => 3,
        7..=9 => 2,
        10..=12 => 1,
        _ => 0,
    }
}

fn rng_next_u32(state: &mut u64) -> u32 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    ((x.wrapping_mul(0x2545F4914F6CDD1D)) >> 32) as u32
}

fn rng_range(state: &mut u64, n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        rng_next_u32(state) % n
    }
}

fn rng_range_inclusive(state: &mut u64, lo: u32, hi: u32) -> u32 {
    lo + rng_range(state, hi - lo + 1)
}

fn shuffle(values: &mut [u32], rng: &mut u64) {
    for i in (1..values.len()).rev() {
        let j = rng_range(rng, (i + 1) as u32) as usize;
        values.swap(i, j);
    }
}

fn wrap_add(value: u32, delta: i32, limit: u32) -> u32 {
    let limit = limit as i64;
    (value as i64 + delta as i64).rem_euclid(limit) as u32
}

fn random_unoccupied_cell(width: u32, height: u32, occupied: &[bool], rng: &mut u64) -> (u32, u32) {
    let len = width * height;
    for _ in 0..len.saturating_mul(2).max(1) {
        let idx = rng_range(rng, len) as usize;
        if !occupied[idx] {
            return ((idx as u32) % width, (idx as u32) / width);
        }
    }
    let idx = occupied
        .iter()
        .position(|occupied| !*occupied)
        .expect("population was validated to fit in the grid");
    ((idx as u32) % width, (idx as u32) / width)
}

fn random_agent(
    width: u32,
    height: u32,
    occupied: &mut [bool],
    rng: &mut u64,
    id: u32,
) -> SugarAgent {
    let (x, y) = random_unoccupied_cell(width, height, occupied, rng);
    occupied[(y * width + x) as usize] = true;
    SugarAgent {
        id,
        x,
        y,
        wealth: rng_range_inclusive(rng, 5, 25),
        metabolism: rng_range_inclusive(rng, 1, 4),
        vision: rng_range_inclusive(rng, 1, 6),
        age: 0,
        max_age: rng_range_inclusive(rng, 60, 100),
    }
}

fn choose_destination(
    agent: SugarAgent,
    cells: &[Cell],
    width: u32,
    height: u32,
    occupied: &[bool],
    rng: &mut u64,
) -> (u32, u32) {
    let mut best_sugar = 0;
    let mut best_distance = u32::MAX;
    let mut ties = Vec::<(u32, u32)>::with_capacity((agent.vision as usize) * 4 + 1);

    let mut consider = |x: u32, y: u32, distance: u32| {
        let idx = (y * width + x) as usize;
        if occupied[idx] && !(x == agent.x && y == agent.y) {
            return;
        }
        let sugar = cells[idx].sugar;
        if sugar > best_sugar || (sugar == best_sugar && distance < best_distance) {
            best_sugar = sugar;
            best_distance = distance;
            ties.clear();
            ties.push((x, y));
        } else if sugar == best_sugar && distance == best_distance {
            ties.push((x, y));
        }
    };

    consider(agent.x, agent.y, 0);
    for distance in 1..=agent.vision {
        consider(wrap_add(agent.x, distance as i32, width), agent.y, distance);
        consider(
            wrap_add(agent.x, -(distance as i32), width),
            agent.y,
            distance,
        );
        consider(
            agent.x,
            wrap_add(agent.y, distance as i32, height),
            distance,
        );
        consider(
            agent.x,
            wrap_add(agent.y, -(distance as i32), height),
            distance,
        );
    }

    let selected = rng_range(rng, ties.len() as u32) as usize;
    ties[selected]
}

fn initial_agents(state: &mut SugarscapeState) -> Result<Vec<SugarAgent>, Box<dyn Error>> {
    let config = state.config;
    let mut occupied = vec![false; state.cells.len()];
    let mut agents = Vec::with_capacity(config.population);
    for id in 0..config.population {
        agents.push(random_agent(
            config.width,
            config.height,
            &mut occupied,
            &mut state.rng,
            id as u32,
        ));
    }
    Ok(agents)
}

fn spawn_agents(ecs: &ECSManager, agent_id: ComponentID, agents: Vec<SugarAgent>) -> ECSResult<()> {
    let _span = abm_framework::span("Sugarscape::spawn_agents")
        .arg("agents", Arg::U64(agents.len() as u64));
    let world = ecs.world_ref();
    world.with_exclusive(|_| {
        for agent in agents {
            let mut bundle = Bundle::new();
            bundle.insert(agent_id, agent);
            world.defer(Command::Spawn { bundle })?;
        }
        Ok(())
    })?;
    ecs.apply_deferred_commands()?;
    Ok(())
}

fn make_scheduler(
    state: Arc<Mutex<SugarscapeState>>,
    agent_id: ComponentID,
    backend: MetabolismBackend,
) -> Scheduler {
    let mut scheduler = Scheduler::new();
    scheduler.add_system(GrowbackSystem::new(Arc::clone(&state), GROW_DONE));
    scheduler.add_system(MoveHarvestAgeSystem::new(
        Arc::clone(&state),
        agent_id,
        GROW_DONE,
        MOVE_DONE,
    ));
    match backend {
        MetabolismBackend::Cpu => {
            scheduler.add_system(CpuMetabolismSystem::new(
                agent_id,
                MOVE_DONE,
                METABOLISM_DONE,
            ));
        }
        #[cfg(feature = "gpu")]
        MetabolismBackend::Gpu => {
            scheduler.add_system(GpuMetabolismSystem::new(
                agent_id,
                MOVE_DONE,
                METABOLISM_DONE,
            ));
        }
    }
    scheduler.add_system(ReplacementSystem::new(
        Arc::clone(&state),
        agent_id,
        METABOLISM_DONE,
        REPLACEMENT_DONE,
    ));
    scheduler.add_system(StatsSystem::new(state, agent_id, REPLACEMENT_DONE));
    scheduler
}

fn collect_agents(ecs: ECSReference<'_>) -> ECSResult<Vec<SugarAgent>> {
    let q = ecs.query()?.read::<SugarAgent>()?.build()?;
    let mut agents = ecs.reduce_read::<SugarAgent, Vec<SugarAgent>>(
        q,
        Vec::new,
        |acc, agent| acc.push(*agent),
        |acc, mut other| acc.append(&mut other),
    )?;
    agents.sort_unstable_by_key(|agent| agent.id);
    Ok(agents)
}

fn write_agents(ecs: ECSReference<'_>, agents: Vec<SugarAgent>) -> ECSResult<()> {
    let q = ecs.query()?.write::<SugarAgent>()?.build()?;
    ecs.for_each_w1::<SugarAgent>(q, move |agent| {
        *agent = agents[agent.id as usize];
    })
}

fn lock_state(state: &Arc<Mutex<SugarscapeState>>) -> ECSResult<MutexGuard<'_, SugarscapeState>> {
    state.lock().map_err(|_| {
        ECSError::from(ExecutionError::LockPoisoned {
            what: "sugarscape state",
        })
    })
}

struct GrowbackSystem {
    state: Arc<Mutex<SugarscapeState>>,
    access: AccessSets,
}

impl GrowbackSystem {
    fn new(state: Arc<Mutex<SugarscapeState>>, produces: u32) -> Self {
        let mut access = AccessSets::default();
        access.produces.insert(produces);
        Self { state, access }
    }
}

impl System for GrowbackSystem {
    fn name(&self) -> &str {
        "Sugarscape::growback"
    }

    fn id(&self) -> abm_framework::SystemID {
        1
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, _world: ECSReference<'_>) -> ECSResult<()> {
        let _span = abm_framework::span("Sugarscape::growback");
        lock_state(&self.state)?.growback();
        Ok(())
    }
}

struct MoveHarvestAgeSystem {
    state: Arc<Mutex<SugarscapeState>>,
    access: AccessSets,
}

impl MoveHarvestAgeSystem {
    fn new(
        state: Arc<Mutex<SugarscapeState>>,
        agent_id: ComponentID,
        consumes: u32,
        produces: u32,
    ) -> Self {
        let mut access = AccessSets::default();
        access.write.set(agent_id);
        access.consumes.insert(consumes);
        access.produces.insert(produces);
        Self { state, access }
    }
}

impl System for MoveHarvestAgeSystem {
    fn name(&self) -> &str {
        "Sugarscape::move_harvest_age"
    }

    fn id(&self) -> abm_framework::SystemID {
        2
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let _span = abm_framework::span("Sugarscape::move_harvest_age");
        let mut agents = collect_agents(ecs)?;
        let mut state = lock_state(&self.state)?;
        let width = state.config.width;
        let height = state.config.height;

        state.occupancy.fill(false);
        for agent in &agents {
            let idx = state.index(agent.x, agent.y);
            state.occupancy[idx] = true;
        }

        state.order.clear();
        state.order.extend(0..agents.len() as u32);
        let mut rng = state.rng;
        shuffle(&mut state.order, &mut rng);

        for pos in 0..state.order.len() {
            let id = state.order[pos] as usize;
            let agent = agents[id];
            let old_idx = state.index(agent.x, agent.y);
            state.occupancy[old_idx] = false;

            let (x, y) = choose_destination(
                agent,
                &state.cells,
                width,
                height,
                &state.occupancy,
                &mut rng,
            );
            let harvested = state.harvest(x, y);
            let new_idx = state.index(x, y);
            state.occupancy[new_idx] = true;

            let updated = &mut agents[id];
            updated.x = x;
            updated.y = y;
            updated.wealth = updated.wealth.saturating_add(harvested);
            updated.age = updated.age.saturating_add(1);
        }

        state.rng = rng;
        drop(state);
        write_agents(ecs, agents)
    }
}

struct CpuMetabolismSystem {
    access: AccessSets,
}

impl CpuMetabolismSystem {
    fn new(agent_id: ComponentID, consumes: u32, produces: u32) -> Self {
        let mut access = AccessSets::default();
        access.write.set(agent_id);
        access.consumes.insert(consumes);
        access.produces.insert(produces);
        Self { access }
    }
}

impl System for CpuMetabolismSystem {
    fn name(&self) -> &str {
        "Sugarscape::metabolism_cpu"
    }

    fn id(&self) -> abm_framework::SystemID {
        3
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let _span = abm_framework::span("Sugarscape::metabolism_cpu");
        let q = ecs.query()?.write::<SugarAgent>()?.build()?;
        ecs.for_each_w1::<SugarAgent>(q, |agent| {
            agent.wealth = agent.wealth.saturating_sub(agent.metabolism);
        })
    }
}

#[cfg(feature = "gpu")]
struct GpuMetabolismSystem {
    access: AccessSets,
}

#[cfg(feature = "gpu")]
impl GpuMetabolismSystem {
    fn new(agent_id: ComponentID, consumes: u32, produces: u32) -> Self {
        let mut access = AccessSets::default();
        access.write.set(agent_id);
        access.consumes.insert(consumes);
        access.produces.insert(produces);
        Self { access }
    }
}

#[cfg(feature = "gpu")]
impl System for GpuMetabolismSystem {
    fn name(&self) -> &str {
        "Sugarscape::metabolism_gpu"
    }

    fn id(&self) -> abm_framework::SystemID {
        3
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn backend(&self) -> SystemBackend {
        SystemBackend::GPU
    }

    fn run(&self, _world: ECSReference<'_>) -> ECSResult<()> {
        Ok(())
    }

    fn gpu(&self) -> Option<&dyn GpuSystem> {
        Some(self)
    }
}

#[cfg(feature = "gpu")]
impl GpuSystem for GpuMetabolismSystem {
    fn shader(&self) -> &'static str {
        r#"
struct Agent {
    id: u32,
    x: u32,
    y: u32,
    wealth: u32,
    metabolism: u32,
    vision: u32,
    age: u32,
    max_age: u32,
};

struct Params {
    entity_len: u32,
    archetype_base: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.entity_len) { return; }
    var agent = agents[i];
    if (agent.wealth > agent.metabolism) {
        agent.wealth = agent.wealth - agent.metabolism;
    } else {
        agent.wealth = 0u;
    }
    agents[i] = agent;
}
"#
    }

    fn workgroup_size(&self) -> u32 {
        256
    }
}

struct ReplacementSystem {
    state: Arc<Mutex<SugarscapeState>>,
    access: AccessSets,
}

impl ReplacementSystem {
    fn new(
        state: Arc<Mutex<SugarscapeState>>,
        agent_id: ComponentID,
        consumes: u32,
        produces: u32,
    ) -> Self {
        let mut access = AccessSets::default();
        access.write.set(agent_id);
        access.consumes.insert(consumes);
        access.produces.insert(produces);
        Self { state, access }
    }
}

impl System for ReplacementSystem {
    fn name(&self) -> &str {
        "Sugarscape::replacement"
    }

    fn id(&self) -> abm_framework::SystemID {
        4
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let _span = abm_framework::span("Sugarscape::replacement");
        let mut agents = collect_agents(ecs)?;
        let mut state = lock_state(&self.state)?;
        let width = state.config.width;
        let height = state.config.height;

        state.occupancy.fill(false);
        state.dead.clear();

        for agent in &agents {
            if agent.wealth > 0 && agent.age < agent.max_age {
                let idx = state.index(agent.x, agent.y);
                state.occupancy[idx] = true;
            } else {
                state.dead.push(agent.id);
            }
        }

        let mut rng = state.rng;
        shuffle(&mut state.dead, &mut rng);
        for pos in 0..state.dead.len() {
            let id = state.dead[pos];
            agents[id as usize] = random_agent(width, height, &mut state.occupancy, &mut rng, id);
        }
        state.rng = rng;

        drop(state);
        write_agents(ecs, agents)
    }
}

struct StatsSystem {
    state: Arc<Mutex<SugarscapeState>>,
    access: AccessSets,
    next_tick: AtomicU32,
}

#[derive(Clone, Copy, Debug, Default)]
struct WealthStats {
    mean: f64,
    p50: u32,
    p90: u32,
    p99: u32,
    max: u32,
    gini: f64,
}

impl StatsSystem {
    fn new(state: Arc<Mutex<SugarscapeState>>, agent_id: ComponentID, consumes: u32) -> Self {
        let mut access = AccessSets::default();
        access.read.set(agent_id);
        access.consumes.insert(consumes);
        Self {
            state,
            access,
            next_tick: AtomicU32::new(1),
        }
    }
}

impl System for StatsSystem {
    fn name(&self) -> &str {
        "Sugarscape::stats"
    }

    fn id(&self) -> abm_framework::SystemID {
        5
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let _span = abm_framework::span("Sugarscape::stats");
        let q = ecs.query()?.read::<SugarAgent>()?.build()?;
        let mut wealths = ecs.reduce_read::<SugarAgent, Vec<u32>>(
            q,
            Vec::new,
            |acc, agent| acc.push(agent.wealth),
            |acc, mut other| acc.append(&mut other),
        )?;
        let agent_count = wealths.len() as u64;
        let wealth = wealth_stats(&mut wealths);

        let avg_sugar = {
            let state = lock_state(&self.state)?;
            let total_sugar: u64 = state.cells.par_iter().map(|cell| cell.sugar as u64).sum();
            total_sugar as f64 / state.cells.len() as f64
        };

        let tick = self.next_tick.fetch_add(1, Ordering::Relaxed);
        println!(
            "{tick},{agent_count},{avg_sugar:.6},{:.6},{},{},{},{},{:.6}",
            wealth.mean, wealth.p50, wealth.p90, wealth.p99, wealth.max, wealth.gini
        );
        Ok(())
    }
}

fn wealth_stats(wealths: &mut [u32]) -> WealthStats {
    if wealths.is_empty() {
        return WealthStats::default();
    }

    wealths.par_sort_unstable();
    let n = wealths.len();
    let sum: u128 = wealths.iter().map(|&v| v as u128).sum();
    let mean = sum as f64 / n as f64;
    let max = wealths[n - 1];

    let p50 = wealths[nearest_rank_index(n, 50)];
    let p90 = wealths[nearest_rank_index(n, 90)];
    let p99 = wealths[nearest_rank_index(n, 99)];

    let gini = if sum == 0 {
        0.0
    } else {
        let weighted_sum: f64 = wealths
            .iter()
            .enumerate()
            .map(|(idx, &wealth)| (idx as f64 + 1.0) * wealth as f64)
            .sum();
        (2.0 * weighted_sum) / (n as f64 * sum as f64) - (n as f64 + 1.0) / n as f64
    };

    WealthStats {
        mean,
        p50,
        p90,
        p99,
        max,
        gini,
    }
}

fn nearest_rank_index(len: usize, percentile: usize) -> usize {
    let rank = (percentile * len).div_ceil(100);
    rank.saturating_sub(1).min(len - 1)
}

impl Cli {
    fn parse() -> Result<Option<Self>, Box<dyn Error>> {
        let mut cli = Cli {
            config: SugarscapeConfig {
                width: DEFAULT_WIDTH,
                height: DEFAULT_HEIGHT,
                population: DEFAULT_AGENTS,
                growth_rate: DEFAULT_GROWTH_RATE,
                seed: DEFAULT_SEED,
            },
            ticks: DEFAULT_TICKS,
            force_cpu: false,
            profile_path: default_profile_path(),
        };

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    println!("{}", usage());
                    return Ok(None);
                }
                "--agents" => cli.config.population = parse_next(&mut args, "--agents")?,
                "--width" => cli.config.width = parse_next(&mut args, "--width")?,
                "--height" => cli.config.height = parse_next(&mut args, "--height")?,
                "--ticks" => cli.ticks = parse_next(&mut args, "--ticks")?,
                "--seed" => cli.config.seed = parse_next_u64(&mut args, "--seed")?,
                "--growth-rate" => cli.config.growth_rate = parse_next(&mut args, "--growth-rate")?,
                "--cpu" => cli.force_cpu = true,
                "--profile" => {
                    let path = args.next().ok_or("--profile requires a path argument")?;
                    cli.profile_path = Some(PathBuf::from(path));
                }
                "--no-profile" => cli.profile_path = None,
                other => return Err(format!("unknown argument: {other}\n\n{}", usage()).into()),
            }
        }

        Ok(Some(cli))
    }
}

fn default_profile_path() -> Option<PathBuf> {
    #[cfg(feature = "profiling")]
    {
        Some(PathBuf::from(DEFAULT_PROFILE))
    }
    #[cfg(not(feature = "profiling"))]
    {
        None
    }
}

fn parse_next<T>(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    let value = args
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))?;
    Ok(value.parse::<T>()?)
}

fn parse_next_u64(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<u64, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))?;
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        Ok(u64::from_str_radix(hex, 16)?)
    } else {
        Ok(value.parse::<u64>()?)
    }
}

fn usage() -> &'static str {
    "Usage: cargo run --release --features \"gpu profiling\" --example sugarscape_axtell_large -- [options]\n\
Options:\n\
  --agents N          Number of agents (default: 1000000)\n\
  --width W           Grid width (default: 4096)\n\
  --height H          Grid height (default: 4096)\n\
  --ticks N           Number of ticks (default: 20)\n\
  --seed N            RNG seed, decimal or 0x-prefixed hex (default: 0xA57E11)\n\
  --growth-rate N     Sugar growback per tick (default: 1)\n\
  --cpu               Force CPU metabolism even when GPU is available\n\
  --profile PATH      Write Chrome Trace JSON to PATH when built with profiling\n\
  --no-profile        Disable profiling trace initialization\n\
  -h, --help          Print this help"
}
