//! ECS system scheduling and execution.
//!
//! The scheduler compiles declared system access into an execution plan. The
//! plan is built from a dependency graph so producer/consumer channels and
//! explicit ordering constraints determine execution order, while component
//! conflicts are serialized deterministically without forcing a command
//! boundary.

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use rayon::prelude::*;

use crate::engine::activation::{with_activation_context, ActivationContext, ActivationOrder};
use crate::engine::boundary::BoundaryChannelProfile;
use crate::engine::commands::CommandEvents;
use crate::engine::component::or_signature_in_place;
use crate::engine::error::{ECSError, ECSResult, ExecutionError};
use crate::engine::manager::ECSReference;
use crate::engine::systems::{AccessSets, System, SystemBackend};
use crate::engine::types::{ChannelID, SystemID};
use crate::profiling::profiler;

#[cfg(feature = "gpu")]
use crate::engine::types::GPUResourceID;

#[cfg(feature = "gpu")]
use crate::gpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageType {
    Boundary,
    Cpu,
    Gpu,
}

impl Default for StageType {
    fn default() -> Self {
        StageType::Cpu
    }
}

/// A logical execution stage used by [`Scheduler`] during planning.
///
/// Stores indices into the scheduler's system list. Boundary stages carry
/// channels that should be finalised at that synchronization point.
#[derive(Clone, Debug, Default)]
pub struct Stage {
    stage_type: StageType,
    /// Indices of systems that execute in this stage.
    pub(crate) system_indices: Vec<usize>,
    /// Aggregate access of all systems in this stage.
    aggregate_access: AccessSets,

    #[cfg(feature = "gpu")]
    gpu_write_resources: Vec<GPUResourceID>,

    /// Channels finalised at this boundary.
    pub(crate) finalised_channels: Vec<ChannelID>,
    /// Backend profile for each finalised channel.
    pub(crate) finalised_channel_profiles: Vec<BoundaryChannelProfile>,
}

impl Stage {
    fn boundary() -> Self {
        Self {
            stage_type: StageType::Boundary,
            system_indices: Vec::new(),
            aggregate_access: AccessSets::default(),
            #[cfg(feature = "gpu")]
            gpu_write_resources: Vec::new(),
            finalised_channels: Vec::new(),
            finalised_channel_profiles: Vec::new(),
        }
    }

    fn cpu() -> Self {
        Self {
            stage_type: StageType::Cpu,
            system_indices: Vec::new(),
            aggregate_access: AccessSets::default(),
            #[cfg(feature = "gpu")]
            gpu_write_resources: Vec::new(),
            finalised_channels: Vec::new(),
            finalised_channel_profiles: Vec::new(),
        }
    }

    fn gpu() -> Self {
        Self {
            stage_type: StageType::Gpu,
            system_indices: Vec::new(),
            aggregate_access: AccessSets::default(),
            #[cfg(feature = "gpu")]
            gpu_write_resources: Vec::new(),
            finalised_channels: Vec::new(),
            finalised_channel_profiles: Vec::new(),
        }
    }

    /// Returns true if `access` does not conflict with this CPU stage.
    #[inline]
    pub fn can_accept(&self, access: &AccessSets) -> bool {
        debug_assert_eq!(self.stage_type, StageType::Cpu);
        !access.conflicts_with(&self.aggregate_access)
    }

    /// Adds a system index to this CPU stage and merges its access.
    #[inline]
    pub fn push_cpu(&mut self, index: usize, access: &AccessSets) {
        debug_assert_eq!(self.stage_type, StageType::Cpu);
        self.system_indices.push(index);
        self.merge_access(access);
    }

    /// Adds a system index to this GPU stage.
    #[inline]
    pub fn push_gpu(&mut self, index: usize, access: &AccessSets) {
        debug_assert_eq!(self.stage_type, StageType::Gpu);
        self.system_indices.push(index);
        self.merge_access(access);
    }

    #[inline]
    fn merge_access(&mut self, access: &AccessSets) {
        or_signature_in_place(&mut self.aggregate_access.read, &access.read);
        or_signature_in_place(&mut self.aggregate_access.write, &access.write);
        self.aggregate_access.produces.or_in_place(&access.produces);
        self.aggregate_access.consumes.or_in_place(&access.consumes);
    }

    /// Returns true if this stage is a boundary marker.
    #[inline]
    pub fn is_boundary(&self) -> bool {
        self.stage_type == StageType::Boundary
    }

    /// Returns member system indices.
    #[inline]
    pub fn system_indices(&self) -> &[usize] {
        &self.system_indices
    }

    /// Returns channels finalised at this boundary.
    #[inline]
    pub fn finalised_channels(&self) -> &[ChannelID] {
        &self.finalised_channels
    }
}

/// An explicit stage-ordering constraint between two systems.
#[derive(Clone, Copy, Debug)]
struct OrderingEdge {
    dependency: SystemID,
    dependent: SystemID,
}

#[derive(Clone, Copy, Debug)]
struct DependencyEdge {
    to: usize,
}

struct DependencyGraph {
    sorted_indices: Vec<usize>,
    edges: Vec<Vec<DependencyEdge>>,
    in_degree: Vec<usize>,
    boundary_predecessors: Vec<Vec<usize>>,
}

/// Stores systems, compiles them into conflict-free execution stages, and
/// executes stages with Rayon parallelism.
pub struct Scheduler {
    systems: Vec<Box<dyn System>>,
    /// Compiled stage plan. Rebuilt lazily when `dirty`.
    plan: Vec<Stage>,
    /// Whether `plan` needs rebuilding.
    dirty: bool,
    /// Compile error set by `rebuild` if a cycle is detected.
    cycle_error: Option<ECSError>,
    /// Registration-time validation error surfaced by `run`.
    validation_error: Option<ECSError>,
    /// Per-system activation orders.
    activation_orders: Vec<(SystemID, ActivationOrder)>,
    /// Global RNG seed for shuffle activation orders.
    seed: u64,
    /// Explicit ordering edges.
    ordering_edges: Vec<OrderingEdge>,
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
            cycle_error: None,
            validation_error: None,
            activation_orders: Vec::new(),
            seed: 0,
            ordering_edges: Vec::new(),
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
        self.cycle_error = None;
        self.validation_error = None;
        self.activation_orders.clear();
        self.ordering_edges.clear();
    }

    /// Registers a boxed system.
    #[inline]
    pub fn add_boxed(&mut self, system: Box<dyn System>) {
        if self.validation_error.is_none() {
            let system_id = system.id();
            if self
                .systems
                .iter()
                .any(|existing| existing.id() == system_id)
            {
                self.validation_error = Some(ECSError::from(ExecutionError::DuplicateSystemId {
                    system_id,
                }));
            } else if let Err(e) = system.access().validate() {
                self.validation_error = Some(e);
            }
        }
        self.systems.push(system);
        self.dirty = true;
    }

    /// Registers a concrete system.
    #[inline]
    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.add_boxed(Box::new(system));
    }

    /// Sets the activation order for a specific system.
    pub fn set_activation_order(&mut self, system_id: SystemID, order: ActivationOrder) {
        if let Some(entry) = self
            .activation_orders
            .iter_mut()
            .find(|(id, _)| *id == system_id)
        {
            entry.1 = order;
        } else {
            self.activation_orders.push((system_id, order));
        }
    }

    /// Returns the activation order for a system.
    pub fn activation_order(&self, system_id: SystemID) -> ActivationOrder {
        self.activation_orders
            .iter()
            .find(|(id, _)| *id == system_id)
            .map(|(_, o)| *o)
            .unwrap_or_default()
    }

    /// Sets the global RNG seed used by shuffle activation orders.
    pub fn seed(&mut self, global_seed: u64) {
        self.seed = global_seed;
    }

    /// Adds an explicit ordering constraint.
    pub fn add_ordering(&mut self, dependent: SystemID, dependency: SystemID) {
        self.ordering_edges.push(OrderingEdge {
            dependency,
            dependent,
        });
        self.dirty = true;
    }

    /// Ensures the execution plan is up to date.
    pub fn rebuild(&mut self) {
        if !self.dirty {
            return;
        }

        self.cycle_error = None;
        self.plan.clear();

        if self.validation_error.is_some() {
            self.dirty = false;
            return;
        }

        let mut indices: Vec<usize> = (0..self.systems.len()).collect();
        indices.sort_by_key(|&i| self.systems[i].id());

        if let Err(e) = self.validate_unique_system_ids(&indices) {
            self.validation_error = Some(e);
            self.dirty = false;
            return;
        }

        match self.build_dependency_graph(indices) {
            Ok(graph) => {
                if let Err(e) = self.pack_graph(graph) {
                    self.cycle_error = Some(e);
                }
            }
            Err(e) => {
                self.validation_error = Some(e);
            }
        }

        self.dirty = false;
    }

    /// Rebuilds the execution plan and returns any validation or graph error.
    pub fn try_rebuild(&mut self) -> ECSResult<()> {
        self.rebuild();
        if let Some(ref e) = self.validation_error {
            return Err(e.clone());
        }
        if let Some(ref e) = self.cycle_error {
            return Err(e.clone());
        }
        Ok(())
    }

    fn validate_unique_system_ids(&self, indices: &[usize]) -> ECSResult<()> {
        let mut last: Option<SystemID> = None;
        for &idx in indices {
            let id = self.systems[idx].id();
            if last == Some(id) {
                return Err(ECSError::from(ExecutionError::DuplicateSystemId {
                    system_id: id,
                }));
            }
            last = Some(id);
        }
        Ok(())
    }

    fn build_dependency_graph(&self, sorted_indices: Vec<usize>) -> ECSResult<DependencyGraph> {
        let n = sorted_indices.len();
        let mut edges = vec![Vec::new(); n];
        let mut in_degree = vec![0usize; n];
        let mut boundary_predecessors = vec![Vec::new(); n];

        let id_to_pos: HashMap<SystemID, usize> = sorted_indices
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (self.systems[idx].id(), pos))
            .collect();
        let access: Vec<&AccessSets> = sorted_indices
            .iter()
            .map(|&idx| self.systems[idx].access())
            .collect();

        for a in 0..n {
            for b in (a + 1)..n {
                if access[a].produces.intersects(&access[b].consumes) {
                    Self::add_edge(
                        &mut edges,
                        &mut in_degree,
                        &mut boundary_predecessors,
                        a,
                        b,
                    );
                }
                if access[b].produces.intersects(&access[a].consumes) {
                    Self::add_edge(
                        &mut edges,
                        &mut in_degree,
                        &mut boundary_predecessors,
                        b,
                        a,
                    );
                }
            }
        }

        for edge in &self.ordering_edges {
            if edge.dependency == edge.dependent {
                return Err(ECSError::from(ExecutionError::SelfSystemOrdering {
                    system_id: edge.dependency,
                }));
            }

            let dependency = id_to_pos.get(&edge.dependency).copied().ok_or_else(|| {
                ECSError::from(ExecutionError::UnknownSystemId {
                    system_id: edge.dependency,
                })
            })?;
            let dependent = id_to_pos.get(&edge.dependent).copied().ok_or_else(|| {
                ECSError::from(ExecutionError::UnknownSystemId {
                    system_id: edge.dependent,
                })
            })?;

            Self::add_edge(
                &mut edges,
                &mut in_degree,
                &mut boundary_predecessors,
                dependency,
                dependent,
            );
        }

        Ok(DependencyGraph {
            sorted_indices,
            edges,
            in_degree,
            boundary_predecessors,
        })
    }

    fn add_edge(
        edges: &mut [Vec<DependencyEdge>],
        in_degree: &mut [usize],
        boundary_predecessors: &mut [Vec<usize>],
        from: usize,
        to: usize,
    ) {
        if from == to {
            return;
        }

        if edges[from].iter().any(|edge| edge.to == to) {
            return;
        }

        edges[from].push(DependencyEdge { to });
        in_degree[to] += 1;
        boundary_predecessors[to].push(from);
    }

    fn pack_graph(&mut self, graph: DependencyGraph) -> ECSResult<()> {
        let n = graph.sorted_indices.len();
        let mut in_degree = graph.in_degree.clone();
        let mut scheduled = vec![false; n];
        let mut scheduled_stage: Vec<Option<usize>> = vec![None; n];
        let mut scheduled_count = 0usize;
        let mut last_boundary: Option<usize> = None;

        while scheduled_count < n {
            let ready: Vec<usize> = (0..n)
                .filter(|&pos| !scheduled[pos] && in_degree[pos] == 0)
                .collect();

            if ready.is_empty() {
                return Err(ECSError::from(ExecutionError::SchedulerCycle));
            }

            let eligible: Vec<usize> = ready
                .iter()
                .copied()
                .filter(|&pos| {
                    Self::boundary_predecessors_satisfied(
                        pos,
                        &graph.boundary_predecessors,
                        &scheduled_stage,
                        last_boundary,
                    )
                })
                .collect();

            if eligible.is_empty() {
                if matches!(self.plan.last(), Some(stage) if stage.is_boundary()) {
                    return Err(ECSError::from(ExecutionError::SchedulerInvariantViolation));
                }
                self.plan.push(Stage::boundary());
                last_boundary = Some(self.plan.len() - 1);
                continue;
            }

            let selected = self.select_ready_stage(&eligible, &graph.sorted_indices);
            if selected.is_empty() {
                return Err(ECSError::from(ExecutionError::SchedulerInvariantViolation));
            }

            let backend = self.systems[graph.sorted_indices[selected[0]]].backend();
            match backend {
                SystemBackend::CPU => {
                    let mut stage = Stage::cpu();
                    for &pos in &selected {
                        let idx = graph.sorted_indices[pos];
                        stage.push_cpu(idx, self.systems[idx].access());
                    }
                    self.plan.push(stage);
                    let stage_idx = self.plan.len() - 1;
                    Self::mark_selected_scheduled(
                        &selected,
                        stage_idx,
                        &graph,
                        &mut in_degree,
                        &mut scheduled,
                        &mut scheduled_stage,
                        &mut scheduled_count,
                    );
                }
                SystemBackend::GPU => {
                    if !matches!(self.plan.last(), Some(stage) if stage.is_boundary()) {
                        self.plan.push(Stage::boundary());
                    }

                    let mut stage = Stage::gpu();
                    for &pos in &selected {
                        let idx = graph.sorted_indices[pos];
                        stage.push_gpu(idx, self.systems[idx].access());
                    }
                    self.plan.push(stage);
                    let stage_idx = self.plan.len() - 1;
                    Self::mark_selected_scheduled(
                        &selected,
                        stage_idx,
                        &graph,
                        &mut in_degree,
                        &mut scheduled,
                        &mut scheduled_stage,
                        &mut scheduled_count,
                    );

                    let boundary = self.boundary_after_gpu_stage(stage_idx);
                    self.plan.push(boundary);
                    last_boundary = Some(self.plan.len() - 1);
                }
            }
        }

        if !matches!(self.plan.last(), Some(stage) if stage.is_boundary()) {
            self.plan.push(Stage::boundary());
        }

        self.compute_finalised_channels();
        Ok(())
    }

    fn boundary_predecessors_satisfied(
        pos: usize,
        boundary_predecessors: &[Vec<usize>],
        scheduled_stage: &[Option<usize>],
        last_boundary: Option<usize>,
    ) -> bool {
        let mut max_predecessor_stage: Option<usize> = None;
        for &pred in &boundary_predecessors[pos] {
            let Some(stage_idx) = scheduled_stage[pred] else {
                return false;
            };
            max_predecessor_stage =
                Some(max_predecessor_stage.map_or(stage_idx, |current| current.max(stage_idx)));
        }

        match max_predecessor_stage {
            None => true,
            Some(max_stage) => last_boundary.map_or(false, |boundary| boundary > max_stage),
        }
    }

    fn select_ready_stage(&self, ready: &[usize], sorted_indices: &[usize]) -> Vec<usize> {
        let first_idx = sorted_indices[ready[0]];
        match self.systems[first_idx].backend() {
            SystemBackend::CPU => {
                let mut selected = Vec::new();
                let mut stage = Stage::cpu();
                for &pos in ready {
                    let idx = sorted_indices[pos];
                    if self.systems[idx].backend() != SystemBackend::CPU {
                        continue;
                    }
                    let access = self.systems[idx].access();
                    if stage.can_accept(access) {
                        stage.push_cpu(idx, access);
                        selected.push(pos);
                    }
                }
                selected
            }
            SystemBackend::GPU => ready
                .iter()
                .copied()
                .filter(|&pos| self.systems[sorted_indices[pos]].backend() == SystemBackend::GPU)
                .collect(),
        }
    }

    fn mark_selected_scheduled(
        selected: &[usize],
        stage_idx: usize,
        graph: &DependencyGraph,
        in_degree: &mut [usize],
        scheduled: &mut [bool],
        scheduled_stage: &mut [Option<usize>],
        scheduled_count: &mut usize,
    ) {
        for &pos in selected {
            scheduled[pos] = true;
            scheduled_stage[pos] = Some(stage_idx);
            *scheduled_count += 1;

            for edge in &graph.edges[pos] {
                in_degree[edge.to] -= 1;
            }
        }
    }

    fn compute_finalised_channels(&mut self) {
        for stage in &mut self.plan {
            stage.finalised_channels.clear();
            stage.finalised_channel_profiles.clear();
        }

        let produced_channels: BTreeSet<ChannelID> = self
            .systems
            .iter()
            .flat_map(|system| system.access().produces.iter())
            .collect();

        for channel in produced_channels {
            let mut last_producer_stage: Option<usize> = None;
            let mut first_consumer_stage: Option<usize> = None;

            for (stage_idx, stage) in self.plan.iter().enumerate() {
                if stage.is_boundary() {
                    continue;
                }

                if self.stage_produces(stage, channel) {
                    last_producer_stage = Some(stage_idx);
                }
                if first_consumer_stage.is_none() && self.stage_consumes(stage, channel) {
                    first_consumer_stage = Some(stage_idx);
                }
            }

            let Some(producer_stage) = last_producer_stage else {
                continue;
            };
            let boundary_idx = match first_consumer_stage {
                Some(consumer_stage) => {
                    let Some(boundary_idx) = self.next_boundary_after(producer_stage) else {
                        continue;
                    };
                    if boundary_idx >= consumer_stage {
                        continue;
                    }
                    boundary_idx
                }
                None => {
                    let Some(boundary_idx) = self.trailing_boundary() else {
                        continue;
                    };
                    boundary_idx
                }
            };

            let channels = &mut self.plan[boundary_idx].finalised_channels;
            if !channels.contains(&channel) {
                channels.push(channel);
            }
        }

        for stage_idx in 0..self.plan.len() {
            if self.plan[stage_idx].finalised_channels.is_empty() {
                continue;
            }
            let profiles: Vec<_> = self.plan[stage_idx]
                .finalised_channels
                .iter()
                .copied()
                .map(|channel| self.channel_backend_profile(channel))
                .collect();
            self.plan[stage_idx].finalised_channel_profiles = profiles;
        }
    }

    fn channel_backend_profile(&self, channel_id: ChannelID) -> BoundaryChannelProfile {
        let mut profile = BoundaryChannelProfile {
            channel_id,
            ..BoundaryChannelProfile::default()
        };

        for system in &self.systems {
            let access = system.access();
            if access.produces.contains(channel_id) {
                match system.backend() {
                    SystemBackend::CPU => profile.cpu_producer = true,
                    SystemBackend::GPU => profile.gpu_producer = true,
                }
            }
            if access.consumes.contains(channel_id) {
                match system.backend() {
                    SystemBackend::CPU => profile.cpu_consumer = true,
                    SystemBackend::GPU => profile.gpu_consumer = true,
                }
            }
        }

        profile
    }

    fn stage_produces(&self, stage: &Stage, channel: ChannelID) -> bool {
        stage
            .system_indices
            .iter()
            .any(|&idx| self.systems[idx].access().produces.contains(channel))
    }

    fn stage_consumes(&self, stage: &Stage, channel: ChannelID) -> bool {
        stage
            .system_indices
            .iter()
            .any(|&idx| self.systems[idx].access().consumes.contains(channel))
    }

    fn next_boundary_after(&self, stage_idx: usize) -> Option<usize> {
        self.plan
            .iter()
            .enumerate()
            .skip(stage_idx + 1)
            .find_map(|(idx, stage)| stage.is_boundary().then_some(idx))
    }

    fn trailing_boundary(&self) -> Option<usize> {
        self.plan
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, stage)| stage.is_boundary().then_some(idx))
    }

    #[cfg(feature = "gpu")]
    fn boundary_after_gpu_stage(&self, stage_idx: usize) -> Stage {
        let mut boundary = Stage::boundary();
        boundary.gpu_write_resources = self.collect_gpu_write_resources(stage_idx);
        boundary
    }

    #[cfg(not(feature = "gpu"))]
    fn boundary_after_gpu_stage(&self, _stage_idx: usize) -> Stage {
        Stage::boundary()
    }

    #[cfg(feature = "gpu")]
    fn collect_gpu_write_resources(&self, stage_idx: usize) -> Vec<GPUResourceID> {
        let Some(stage) = self.plan.get(stage_idx) else {
            return Vec::new();
        };
        let mut resources = Vec::new();
        for &idx in &stage.system_indices {
            if let Some(gpu_cap) = self.systems[idx].gpu() {
                for &rid in gpu_cap.writes_resources() {
                    if !resources.contains(&rid) {
                        resources.push(rid);
                    }
                }
            }
        }
        resources
    }

    /// Runs the schedule once.
    #[allow(clippy::duplicated_code)]
    pub fn run(&mut self, ecs: ECSReference<'_>) -> ECSResult<()> {
        self.run_with_lifecycle_events(ecs, |_| Ok(()))
    }

    /// Runs the schedule once and reports lifecycle events after each command drain.
    #[allow(clippy::duplicated_code)]
    pub fn run_with_lifecycle_events(
        &mut self,
        ecs: ECSReference<'_>,
        mut on_lifecycle_events: impl FnMut(&CommandEvents) -> ECSResult<()>,
    ) -> ECSResult<()> {
        let _g = profiler::span("Scheduler::run");

        self.rebuild();

        if let Some(ref e) = self.validation_error {
            return Err(e.clone());
        }
        if let Some(ref e) = self.cycle_error {
            return Err(e.clone());
        }

        static BACKEND_CPU: &str = "CPU";
        #[cfg(feature = "gpu")]
        static BACKEND_GPU: &str = "GPU";

        for stage in &self.plan {
            match stage.stage_type {
                StageType::Boundary => {
                    let _s = profiler::span("Stage::Boundary");

                    ecs.clear_borrows();

                    #[cfg(feature = "gpu")]
                    {
                        let _g0 = profiler::span("GPU::sync_pending_to_cpu");
                        gpu::sync_pending_to_cpu(ecs, &stage.gpu_write_resources)?;
                    }

                    {
                        let _g1 = profiler::span("ECS::apply_deferred_commands");
                        match ecs.apply_deferred_commands_with_events() {
                            Ok(events) => on_lifecycle_events(&events)?,
                            Err(failure) => {
                                on_lifecycle_events(&failure.events)?;
                                return Err(failure.error);
                            }
                        }
                    }

                    if !stage.finalised_channels.is_empty() {
                        ecs.finalise_boundaries_with_profiles(
                            &stage.finalised_channels,
                            &stage.finalised_channel_profiles,
                        )?;
                    }
                }

                StageType::Cpu => {
                    let _s = profiler::span("Stage::Cpu").arg(
                        "systems",
                        profiler::Arg::U64(stage.system_indices.len() as u64),
                    );

                    stage.system_indices.par_iter().try_for_each(|&i| {
                        let sys = &self.systems[i];
                        profiler::next_arg("system_id", profiler::Arg::U64(sys.id() as u64));
                        profiler::next_arg("backend", profiler::Arg::Str(BACKEND_CPU.to_string()));
                        let _sg = profiler::span_fmt(format_args!("System::{}", sys.name()));
                        let activation = ActivationContext {
                            order: self.activation_order(sys.id()),
                            seed: self.seed,
                            system_id: sys.id(),
                        };
                        let result = with_activation_context(activation, || sys.run(ecs));
                        profiler::flush_thread();
                        result
                    })?;

                    ecs.clear_borrows();
                }

                StageType::Gpu => {
                    let _s = profiler::span("Stage::Gpu").arg(
                        "systems",
                        profiler::Arg::U64(stage.system_indices.len() as u64),
                    );

                    #[cfg(not(feature = "gpu"))]
                    {
                        let _ = &stage;
                        return Err(ECSError::from(ExecutionError::GpuNotEnabled));
                    }

                    #[cfg(feature = "gpu")]
                    {
                        for &idx in &stage.system_indices {
                            let system = &self.systems[idx];
                            profiler::next_arg("system_id", profiler::Arg::U64(system.id() as u64));
                            profiler::next_arg(
                                "backend",
                                profiler::Arg::Str(BACKEND_GPU.to_string()),
                            );
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

    /// Formats the compiled execution plan as a human-readable text table.
    pub fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, stage) in self.plan.iter().enumerate() {
            match stage.stage_type {
                StageType::Boundary => {
                    write!(f, "Stage {i} [Boundary]")?;
                    if !stage.finalised_channels.is_empty() {
                        let chs: Vec<String> = stage
                            .finalised_channels
                            .iter()
                            .map(|c| format!("ch:{c}"))
                            .collect();
                        write!(f, "    finalises: [{}]", chs.join(", "))?;
                    }
                    writeln!(f)?;
                }
                StageType::Cpu => {
                    writeln!(f, "Stage {i} [CPU]")?;
                    for &si in &stage.system_indices {
                        let sys = &self.systems[si];
                        let access = sys.access();
                        write!(f, "  system {:4}  {:?}", sys.id(), sys.name())?;
                        if !access.produces.is_empty() {
                            let ps: Vec<String> =
                                access.produces.iter().map(|c| format!("ch:{c}")).collect();
                            write!(f, "  produces: [{}]", ps.join(", "))?;
                        }
                        if !access.consumes.is_empty() {
                            let cs: Vec<String> =
                                access.consumes.iter().map(|c| format!("ch:{c}")).collect();
                            write!(f, "  consumes: [{}]", cs.join(", "))?;
                        }
                        writeln!(f)?;
                    }
                }
                StageType::Gpu => {
                    writeln!(f, "Stage {i} [GPU]")?;
                    for &si in &stage.system_indices {
                        writeln!(
                            f,
                            "  system {:4}  {:?}",
                            self.systems[si].id(),
                            self.systems[si].name()
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Formats the compiled execution plan as a Graphviz DOT graph.
    pub fn fmt_dot(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "digraph execution_plan {{")?;
        writeln!(f, "  rankdir=TB;")?;
        writeln!(f, "  node [fontname=\"monospace\"];")?;

        let mut prev_node: Option<String> = None;

        for (i, stage) in self.plan.iter().enumerate() {
            let node_name = format!("stage_{i}");

            match stage.stage_type {
                StageType::Boundary => {
                    let label = if stage.finalised_channels.is_empty() {
                        format!("Boundary {i}")
                    } else {
                        let chs: Vec<String> = stage
                            .finalised_channels
                            .iter()
                            .map(|c| format!("ch:{c}"))
                            .collect();
                        format!("Boundary {i}\\nfinalises: {}", chs.join(", "))
                    };
                    writeln!(
                        f,
                        "  {node_name} [shape=diamond, label=\"{}\"];",
                        dot_escape(&label)
                    )?;
                }
                StageType::Cpu | StageType::Gpu => {
                    let kind = if stage.stage_type == StageType::Cpu {
                        "CPU"
                    } else {
                        "GPU"
                    };
                    writeln!(f, "  subgraph cluster_{i} {{")?;
                    writeln!(
                        f,
                        "    label=\"{}\";",
                        dot_escape(&format!("Stage {i} [{kind}]"))
                    )?;
                    writeln!(f, "    style=filled; fillcolor=lightgrey;")?;
                    for &si in &stage.system_indices {
                        let sys = &self.systems[si];
                        writeln!(
                            f,
                            "    {}_{} [label=\"{}\", shape=box];",
                            node_name,
                            si,
                            dot_escape(&format!("id:{} {}", sys.id(), sys.name()))
                        )?;
                    }
                    writeln!(f, "  }}")?;
                    writeln!(f, "  {node_name} [shape=point, style=invis];")?;
                }
            }

            if let Some(prev) = &prev_node {
                writeln!(f, "  {prev} -> {node_name};")?;
            }
            prev_node = Some(node_name);
        }

        writeln!(f, "}}")
    }

    /// Returns a reference to the compiled execution plan stages.
    pub fn plan(&self) -> &[Stage] {
        &self.plan
    }

    /// Iterates registered systems in insertion order.
    pub fn systems_iter(&self) -> impl Iterator<Item = &dyn System> {
        self.systems.iter().map(|s| s.as_ref())
    }

    /// Returns the union of all produced channels declared by registered systems.
    pub fn aggregate_produces(&self) -> crate::engine::systems::ChannelSet {
        let mut out = crate::engine::systems::ChannelSet::new();
        for system in &self.systems {
            out.or_in_place(&system.access().produces);
        }
        out
    }

    /// Returns the union of all consumed channels declared by registered systems.
    pub fn aggregate_consumes(&self) -> crate::engine::systems::ChannelSet {
        let mut out = crate::engine::systems::ChannelSet::new();
        for system in &self.systems {
            out.or_in_place(&system.access().consumes);
        }
        out
    }

    /// Returns true if any registered system uses the GPU backend.
    pub fn has_gpu_systems(&self) -> bool {
        self.systems
            .iter()
            .any(|system| system.backend() == SystemBackend::GPU)
    }
}

fn dot_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}
