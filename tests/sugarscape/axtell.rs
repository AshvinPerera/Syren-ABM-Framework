#![allow(dead_code)]

use std::sync::{Arc, Mutex, RwLock};

#[cfg(any(feature = "model", feature = "gpu"))]
use abm_framework::advanced::EntityShards;
use abm_framework::{AccessSets, ComponentRegistry, ECSReference, ECSResult, Read, System, Write};

#[cfg(feature = "model")]
use abm_framework::agents::AgentTemplate;
#[cfg(feature = "model")]
use abm_framework::model::ModelBuilder;

#[cfg(feature = "gpu")]
use abm_framework::{
    Bundle, Command, ECSManager, GPUPod, GpuSystem, Scheduler, Signature, SystemBackend,
};

pub const AXTELL_WIDTH: u32 = 50;
pub const AXTELL_HEIGHT: u32 = 50;
pub const AXTELL_POPULATION: usize = 250;

#[derive(Clone, Copy, Debug)]
pub struct SugarscapeConfig {
    pub width: u32,
    pub height: u32,
    pub population: usize,
    pub growth_rate: u32,
    pub seed: u64,
}

impl Default for SugarscapeConfig {
    fn default() -> Self {
        Self {
            width: AXTELL_WIDTH,
            height: AXTELL_HEIGHT,
            population: AXTELL_POPULATION,
            growth_rate: 1,
            seed: 0xA57E11,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SugarAgent {
    pub id: u32,
    pub x: u32,
    pub y: u32,
    pub wealth: u32,
    pub metabolism: u32,
    pub vision: u32,
    pub age: u32,
    pub max_age: u32,
}

#[cfg(feature = "gpu")]
unsafe impl GPUPod for SugarAgent {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cell {
    pub sugar: u32,
    pub capacity: u32,
}

#[derive(Clone, Debug)]
pub struct SugarscapeState {
    pub config: SugarscapeConfig,
    pub cells: Vec<Cell>,
    pub rng: u64,
}

impl SugarscapeState {
    pub fn new(config: SugarscapeConfig) -> Self {
        let mut cells = Vec::with_capacity((config.width * config.height) as usize);
        for y in 0..config.height {
            for x in 0..config.width {
                let capacity = sugar_capacity(x, y, config.width, config.height);
                cells.push(Cell {
                    sugar: capacity,
                    capacity,
                });
            }
        }
        Self {
            rng: config.seed,
            config,
            cells,
        }
    }

    pub fn index(&self, x: u32, y: u32) -> usize {
        (y * self.config.width + x) as usize
    }

    fn growback(&mut self) {
        for cell in &mut self.cells {
            cell.sugar = (cell.sugar + self.config.growth_rate).min(cell.capacity);
        }
    }

    fn harvest(&mut self, x: u32, y: u32) -> u32 {
        let idx = self.index(x, y);
        let sugar = self.cells[idx].sugar;
        self.cells[idx].sugar = 0;
        sugar
    }

    fn random_unoccupied_cell(&mut self, occupied: &[bool]) -> (u32, u32) {
        let len = occupied.len() as u32;
        for _ in 0..(len * 2).max(1) {
            let idx = rng_range(&mut self.rng, len) as usize;
            if !occupied[idx] {
                return (
                    (idx as u32) % self.config.width,
                    (idx as u32) / self.config.width,
                );
            }
        }
        let idx = occupied
            .iter()
            .position(|occupied| !*occupied)
            .expect("Sugarscape population must not exceed cell count");
        (
            (idx as u32) % self.config.width,
            (idx as u32) / self.config.width,
        )
    }

    fn random_agent(&mut self, id: u32, occupied: &mut [bool]) -> SugarAgent {
        let (x, y) = self.random_unoccupied_cell(occupied);
        let idx = self.index(x, y);
        occupied[idx] = true;
        SugarAgent {
            id,
            x,
            y,
            wealth: rng_range_inclusive(&mut self.rng, 5, 25),
            metabolism: rng_range_inclusive(&mut self.rng, 1, 4),
            vision: rng_range_inclusive(&mut self.rng, 1, 6),
            age: 0,
            max_age: rng_range_inclusive(&mut self.rng, 60, 100),
        }
    }
}

pub fn sugar_capacity(x: u32, y: u32, width: u32, height: u32) -> u32 {
    let peaks = [
        (width.saturating_mul(3) / 10, height.saturating_mul(3) / 10),
        (width.saturating_mul(7) / 10, height.saturating_mul(7) / 10),
    ];
    let d = peaks
        .iter()
        .map(|(px, py)| x.abs_diff(*px) + y.abs_diff(*py))
        .min()
        .unwrap_or(0);
    match d {
        0..=3 => 4,
        4..=6 => 3,
        7..=9 => 2,
        10..=12 => 1,
        _ => 0,
    }
}

pub fn rng_next_u32(state: &mut u64) -> u32 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    ((x.wrapping_mul(0x2545F4914F6CDD1D)) >> 32) as u32
}

pub fn rng_range(state: &mut u64, n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        rng_next_u32(state) % n
    }
}

pub fn rng_range_inclusive(state: &mut u64, lo: u32, hi: u32) -> u32 {
    lo + rng_range(state, hi - lo + 1)
}

fn shuffle(values: &mut [u32], rng: &mut u64) {
    for i in (1..values.len()).rev() {
        let j = rng_range(rng, (i + 1) as u32) as usize;
        values.swap(i, j);
    }
}

fn wrap_add(value: u32, delta: i32, limit: u32) -> u32 {
    let limit = limit as i32;
    ((value as i32 + delta).rem_euclid(limit)) as u32
}

pub fn choose_destination(
    agent: SugarAgent,
    cells: &[Cell],
    width: u32,
    height: u32,
    occupied: &[bool],
    rng: &mut u64,
) -> (u32, u32) {
    let mut best_sugar = 0;
    let mut best_distance = u32::MAX;
    let mut ties = Vec::<(u32, u32)>::new();

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

fn collect_agents(ecs: ECSReference<'_>) -> ECSResult<Vec<SugarAgent>> {
    let agents = Arc::new(Mutex::new(Vec::<SugarAgent>::new()));
    let agents_for_query = Arc::clone(&agents);
    let q = ecs.query()?.read::<SugarAgent>()?.build()?;
    ecs.for_each::<(Read<SugarAgent>,)>(q, &move |agent| {
        agents_for_query.lock().unwrap().push(*agent.0);
    })?;
    let mut agents = agents.lock().unwrap().clone();
    agents.sort_by_key(|agent| agent.id);
    Ok(agents)
}

fn write_agents(ecs: ECSReference<'_>, agents: &[SugarAgent]) -> ECSResult<()> {
    let agents = agents.to_vec();
    let q = ecs.query()?.write::<SugarAgent>()?.build()?;
    ecs.for_each::<(Write<SugarAgent>,)>(q, &move |agent| {
        *agent.0 = agents[agent.0.id as usize];
    })
}

pub struct GrowbackSystem {
    state: Arc<Mutex<SugarscapeState>>,
    access: AccessSets,
}

impl GrowbackSystem {
    pub fn new(state: Arc<Mutex<SugarscapeState>>, produces: u32) -> Self {
        let mut access = AccessSets::default();
        access.produces.insert(produces);
        Self { state, access }
    }
}

impl System for GrowbackSystem {
    fn id(&self) -> abm_framework::SystemID {
        1
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, _world: ECSReference<'_>) -> ECSResult<()> {
        self.state.lock().unwrap().growback();
        Ok(())
    }
}

pub struct MoveHarvestAgeSystem {
    state: Arc<Mutex<SugarscapeState>>,
    access: AccessSets,
}

impl MoveHarvestAgeSystem {
    pub fn new(
        state: Arc<Mutex<SugarscapeState>>,
        agent_id: abm_framework::ComponentID,
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
    fn id(&self) -> abm_framework::SystemID {
        2
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let mut agents = collect_agents(ecs)?;
        let mut state = self.state.lock().unwrap();
        let width = state.config.width;
        let height = state.config.height;
        let mut occupied = vec![false; (width * height) as usize];
        for agent in &agents {
            occupied[state.index(agent.x, agent.y)] = true;
        }

        let mut order: Vec<u32> = (0..agents.len() as u32).collect();
        shuffle(&mut order, &mut state.rng);

        for id in order {
            let agent = agents[id as usize];
            let old_idx = state.index(agent.x, agent.y);
            occupied[old_idx] = false;
            let mut rng = state.rng;
            let (x, y) =
                choose_destination(agent, &state.cells, width, height, &occupied, &mut rng);
            state.rng = rng;
            let harvested = state.harvest(x, y);
            let new_idx = state.index(x, y);
            occupied[new_idx] = true;
            let updated = &mut agents[id as usize];
            updated.x = x;
            updated.y = y;
            updated.wealth += harvested;
            updated.age += 1;
        }

        drop(state);
        write_agents(ecs, &agents)
    }
}

pub struct CpuMetabolismSystem {
    access: AccessSets,
}

impl CpuMetabolismSystem {
    pub fn new(agent_id: abm_framework::ComponentID, consumes: u32, produces: u32) -> Self {
        let mut access = AccessSets::default();
        access.write.set(agent_id);
        access.consumes.insert(consumes);
        access.produces.insert(produces);
        Self { access }
    }
}

impl System for CpuMetabolismSystem {
    fn id(&self) -> abm_framework::SystemID {
        3
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let q = ecs.query()?.write::<SugarAgent>()?.build()?;
        ecs.for_each::<(Write<SugarAgent>,)>(q, &|agent| {
            agent.0.wealth = agent.0.wealth.saturating_sub(agent.0.metabolism);
        })
    }
}

#[cfg(feature = "gpu")]
pub struct GpuMetabolismSystem {
    access: AccessSets,
}

#[cfg(feature = "gpu")]
impl GpuMetabolismSystem {
    pub fn new(agent_id: abm_framework::ComponentID, consumes: u32, produces: u32) -> Self {
        let mut write = Signature::default();
        write.set(agent_id);
        let mut access = AccessSets {
            write,
            ..AccessSets::default()
        };
        access.consumes.insert(consumes);
        access.produces.insert(produces);
        Self { access }
    }
}

#[cfg(feature = "gpu")]
impl System for GpuMetabolismSystem {
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

@compute @workgroup_size(64)
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
        64
    }
}

pub struct ReplacementSystem {
    state: Arc<Mutex<SugarscapeState>>,
    access: AccessSets,
}

impl ReplacementSystem {
    pub fn new(
        state: Arc<Mutex<SugarscapeState>>,
        agent_id: abm_framework::ComponentID,
        consumes: u32,
    ) -> Self {
        let mut access = AccessSets::default();
        access.write.set(agent_id);
        access.consumes.insert(consumes);
        Self { state, access }
    }
}

impl System for ReplacementSystem {
    fn id(&self) -> abm_framework::SystemID {
        4
    }

    fn access(&self) -> &AccessSets {
        &self.access
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let mut agents = collect_agents(ecs)?;
        let mut state = self.state.lock().unwrap();
        let mut occupied = vec![false; (state.config.width * state.config.height) as usize];
        for agent in &agents {
            if agent.wealth > 0 && agent.age < agent.max_age {
                occupied[state.index(agent.x, agent.y)] = true;
            }
        }

        let mut dead: Vec<u32> = agents
            .iter()
            .filter(|agent| agent.wealth == 0 || agent.age >= agent.max_age)
            .map(|agent| agent.id)
            .collect();
        shuffle(&mut dead, &mut state.rng);

        for id in dead {
            agents[id as usize] = state.random_agent(id, &mut occupied);
        }

        drop(state);
        write_agents(ecs, &agents)
    }
}

pub fn register_components() -> (Arc<RwLock<ComponentRegistry>>, abm_framework::ComponentID) {
    let registry = Arc::new(RwLock::new(ComponentRegistry::new()));
    let agent_id = {
        let mut reg = registry.write().unwrap();
        #[cfg(feature = "gpu")]
        let id = reg.register_gpu::<SugarAgent>().unwrap();
        #[cfg(not(feature = "gpu"))]
        let id = reg.register::<SugarAgent>().unwrap();
        id
    };
    (registry, agent_id)
}

pub fn initial_agents(state: &mut SugarscapeState) -> Vec<SugarAgent> {
    let mut occupied = vec![false; (state.config.width * state.config.height) as usize];
    (0..state.config.population as u32)
        .map(|id| state.random_agent(id, &mut occupied))
        .collect()
}

#[cfg(feature = "gpu")]
pub fn spawn_agents_raw(
    ecs: &ECSManager,
    agent_id: abm_framework::ComponentID,
    agents: &[SugarAgent],
) -> ECSResult<()> {
    let world = ecs.world_ref();
    for agent in agents {
        let mut bundle = Bundle::new();
        bundle.insert(agent_id, *agent);
        world.defer(Command::Spawn { bundle })?;
    }
    ecs.apply_deferred_commands()?;
    Ok(())
}

#[cfg(feature = "gpu")]
pub fn make_scheduler(
    state: Arc<Mutex<SugarscapeState>>,
    agent_id: abm_framework::ComponentID,
    use_gpu_metabolism: bool,
) -> Scheduler {
    let grow_done = 0;
    let move_done = 1;
    let metabolism_done = 2;
    let mut scheduler = Scheduler::new();
    scheduler.add_system(GrowbackSystem::new(Arc::clone(&state), grow_done));
    scheduler.add_system(MoveHarvestAgeSystem::new(
        Arc::clone(&state),
        agent_id,
        grow_done,
        move_done,
    ));

    #[cfg(feature = "gpu")]
    if use_gpu_metabolism {
        scheduler.add_system(GpuMetabolismSystem::new(
            agent_id,
            move_done,
            metabolism_done,
        ));
    } else {
        scheduler.add_system(CpuMetabolismSystem::new(
            agent_id,
            move_done,
            metabolism_done,
        ));
    }

    #[cfg(not(feature = "gpu"))]
    {
        let _ = use_gpu_metabolism;
        scheduler.add_system(CpuMetabolismSystem::new(
            agent_id,
            move_done,
            metabolism_done,
        ));
    }

    scheduler.add_system(ReplacementSystem::new(state, agent_id, metabolism_done));
    scheduler
}

#[cfg(feature = "gpu")]
pub fn build_raw_world(
    config: SugarscapeConfig,
    use_gpu_metabolism: bool,
) -> ECSResult<(
    ECSManager,
    Scheduler,
    Arc<Mutex<SugarscapeState>>,
    abm_framework::ComponentID,
)> {
    let (registry, agent_id) = register_components();
    let mut state = SugarscapeState::new(config);
    let agents = initial_agents(&mut state);
    let state = Arc::new(Mutex::new(state));
    let ecs = ECSManager::with_registry(EntityShards::new(2)?, registry);
    spawn_agents_raw(&ecs, agent_id, &agents)?;
    let scheduler = make_scheduler(Arc::clone(&state), agent_id, use_gpu_metabolism);
    Ok((ecs, scheduler, state, agent_id))
}

#[cfg(feature = "gpu")]
pub fn raw_agents(ecs: &ECSManager) -> ECSResult<Vec<SugarAgent>> {
    collect_agents(ecs.world_ref())
}

#[cfg(feature = "model")]
pub fn build_model(
    config: SugarscapeConfig,
    use_gpu_metabolism: bool,
) -> (
    abm_framework::model::Model,
    Arc<Mutex<SugarscapeState>>,
    abm_framework::ComponentID,
) {
    let (registry, agent_id) = register_components();
    let mut state = SugarscapeState::new(config);
    let agents = initial_agents(&mut state);
    let state = Arc::new(Mutex::new(state));

    let mut builder = ModelBuilder::new()
        .with_component_registry(Arc::clone(&registry))
        .with_shards(EntityShards::new(2).unwrap());
    let grow_done = builder
        .register_environment::<u32>("sugarscape_grow_done", 0)
        .unwrap()
        .channel_id();
    let move_done = builder
        .register_environment::<u32>("sugarscape_move_done", 0)
        .unwrap()
        .channel_id();
    let metabolism_done = builder
        .register_environment::<u32>("sugarscape_metabolism_done", 0)
        .unwrap()
        .channel_id();

    let mut builder = builder
        .with_agent_template(
            AgentTemplate::builder("sugar_agent")
                .with_component::<SugarAgent>(agent_id)
                .unwrap()
                .build(),
        )
        .unwrap()
        .with_system(GrowbackSystem::new(Arc::clone(&state), grow_done))
        .with_system(MoveHarvestAgeSystem::new(
            Arc::clone(&state),
            agent_id,
            grow_done,
            move_done,
        ));

    #[cfg(feature = "gpu")]
    {
        if use_gpu_metabolism {
            builder = builder.with_system(GpuMetabolismSystem::new(
                agent_id,
                move_done,
                metabolism_done,
            ));
        } else {
            builder = builder.with_system(CpuMetabolismSystem::new(
                agent_id,
                move_done,
                metabolism_done,
            ));
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = use_gpu_metabolism;
        builder = builder.with_system(CpuMetabolismSystem::new(
            agent_id,
            move_done,
            metabolism_done,
        ));
    }

    let model = builder
        .with_system(ReplacementSystem::new(
            Arc::clone(&state),
            agent_id,
            metabolism_done,
        ))
        .build()
        .unwrap();

    let world = model.ecs().world_ref();
    let template = model.agents().get("sugar_agent").unwrap();
    for agent in &agents {
        template
            .spawner()
            .set(agent_id, *agent)
            .unwrap()
            .spawn(world)
            .unwrap();
    }
    model.ecs().apply_deferred_commands().unwrap();
    (model, state, agent_id)
}

#[cfg(feature = "model")]
pub fn model_agents(model: &abm_framework::model::Model) -> ECSResult<Vec<SugarAgent>> {
    collect_agents(model.ecs().world_ref())
}
