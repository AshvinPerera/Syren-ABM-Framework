//! ECS system scheduling and execution.
//!
//! This module is responsible for:
//! * grouping systems into execution stages based on access compatibility,
//! * running compatible systems in parallel using Rayon,
//! * enforcing structural synchronisation points between stages,
//! * channel-aware stage packing and cycle detection,
//! * per-system activation order and explicit ordering constraints,
//! * human-readable and Graphviz plan export.
//!
//! ## Scheduling model
//!
//! Systems are assigned to **stages** such that:
//! * systems within the same stage do **not** conflict on component access,
//! * systems within the same stage do **not** have a producer/consumer
//!   channel relationship,
//! * all systems in a stage may run in parallel,
//! * stages are executed sequentially.
//!
//! ## Channel finalisation
//!
//! After packing, each `Stage` records which channels are "finalised" by that
//! stage — meaning the stage contains the last producer in the schedule and a
//! `Boundary` follows before any consumer. At each `Boundary` stage, the
//! scheduler calls `ECSManager::finalise_boundaries` with that set of channels
//! so boundary resources (e.g., `MessageBufferSet`) can merge thread-local
//! buffers and build acceleration indices.
//!
//! ## Cycle detection
//!
//! After packing, a Kahn-style topological sort is run over the combined
//! (component-conflict + channel-ordering) partial order. If a cycle is
//! detected, `rebuild` stores the error and `run` returns it immediately.

use std::collections::VecDeque;
use std::fmt;

use rayon::prelude::*;

use crate::engine::activation::ActivationOrder;
use crate::engine::manager::ECSReference;
use crate::engine::systems::{AccessSets, System, SystemBackend};
use crate::engine::component::or_signature_in_place;
use crate::engine::types::{ChannelID, SystemID};
use crate::engine::error::{ECSResult, ECSError, ExecutionError};
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
    fn default() -> Self { StageType::Cpu }
}

/// A logical execution stage used by [`Scheduler`] during planning.
///
/// Stores *indices* into the scheduler's system list. Boundary stages carry
/// the list of channels whose last producer lives in the preceding CPU/GPU
/// stage; the scheduler finalises those channels at this boundary.
#[derive(Clone, Debug, Default)]
pub struct Stage {
    stage_type: StageType,
    /// Indices of systems that execute in this stage.
    pub(crate) system_indices: Vec<usize>,
    /// Aggregate access of all systems in this stage (for conflict detection).
    aggregate_access: AccessSets,

    #[cfg(feature = "gpu")]
    gpu_write_resources: Vec<GPUResourceID>,

    /// Channels whose last producer is in the CPU/GPU stage immediately
    /// preceding this boundary. Populated during `rebuild`; empty for
    /// non-boundary stages and boundary stages with no finalised channels.
    pub(crate) finalised_channels: Vec<ChannelID>,
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
        }
    }

    /// Returns true if `access` does NOT conflict with anything already in this stage.
    ///
    /// Checks both component conflicts and channel ordering constraints.
    #[inline]
    pub fn can_accept(&self, access: &AccessSets) -> bool {
        debug_assert_eq!(self.stage_type, StageType::Cpu);

        // Component conflicts (existing rule).
        if access.conflicts_with(&self.aggregate_access) {
            return false;
        }
        // Channel ordering: if the incoming system produces a channel already
        // consumed in this stage — or consumes a channel already produced —
        // they must be in different stages.
        if access.produces.intersects(&self.aggregate_access.consumes) {
            return false;
        }
        if access.consumes.intersects(&self.aggregate_access.produces) {
            return false;
        }
        true
    }

    /// Adds a system index to this stage and merges its access (CPU only).
    #[inline]
    pub fn push_cpu(&mut self, index: usize, access: &AccessSets) {
        debug_assert_eq!(self.stage_type, StageType::Cpu);
        self.system_indices.push(index);
        or_signature_in_place(&mut self.aggregate_access.read,  &access.read);
        or_signature_in_place(&mut self.aggregate_access.write, &access.write);
        self.aggregate_access.produces.or_in_place(&access.produces);
        self.aggregate_access.consumes.or_in_place(&access.consumes);
    }

    /// Adds a system index to this stage (GPU only; executed sequentially).
    #[inline]
    pub fn push_gpu(&mut self, index: usize) {
        debug_assert_eq!(self.stage_type, StageType::Gpu);
        self.system_indices.push(index);
    }

    /// Returns true if this stage is a boundary marker.
    #[inline]
    pub fn is_boundary(&self) -> bool {
        self.stage_type == StageType::Boundary
    }

    /// Returns the member system indices (positions into the scheduler's
    /// owning `systems: Vec<Box<dyn System>>`).
    ///
    /// Exposed as a read-only slice so external callers inspecting a
    /// compiled plan cannot mutate the scheduler's internal state.
    #[inline]
    pub fn system_indices(&self) -> &[usize] {
        &self.system_indices
    }

    /// Returns the channels finalised at this boundary, or an empty slice
    /// for non-boundary stages.
    #[inline]
    pub fn finalised_channels(&self) -> &[ChannelID] {
        &self.finalised_channels
    }
}

/// An explicit stage-ordering constraint between two systems.
///
/// `dependent` must be placed in a strictly later stage than `dependency`.
/// Used for cases not expressible via access sets (e.g., "always run the
/// logger last").
#[derive(Clone, Copy, Debug)]
struct OrderingEdge {
    dependency: SystemID,
    dependent:  SystemID,
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
    /// Cleared at the start of each rebuild and recomputed.
    cycle_error: Option<ECSError>,
    /// Registration-time error set by `add_boxed` when a system's
    /// `AccessSets::validate()` fails. **Persists across rebuilds** — only
    /// cleared by `clear()`. Surfaced by `run()` alongside `cycle_error`.
    validation_error: Option<ECSError>,
    /// Per-system activation orders (keyed by SystemID).
    activation_orders: Vec<(SystemID, ActivationOrder)>,
    /// Global RNG seed for shuffle activation orders.
    seed: u64,
    /// Explicit ordering edges (dependency must precede dependent).
    ordering_edges: Vec<OrderingEdge>,
}

impl Default for Scheduler {
    fn default() -> Self { Self::new() }
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
    pub fn len(&self) -> usize { self.systems.len() }

    /// Returns `true` if no systems are registered.
    #[inline]
    pub fn is_empty(&self) -> bool { self.systems.is_empty() }

    /// Removes all systems and stages.
    /// Removes all systems and stages.
    ///
    /// Resets all registration-time state: registered systems, compiled
    /// stages, explicit ordering edges, activation-order overrides, and
    /// both the validation and cycle errors. Does **not** reset the
    /// global RNG seed set by [`Scheduler::seed`]; call `.seed(0)`
    /// explicitly if a seed reset is required.
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
    ///
    /// The system's declared [`AccessSets`] is validated at registration
    /// time. If validation fails (e.g., the system declares the same
    /// channel in both `produces` and `consumes`, or the same component in
    /// both `read` and `write`), the error is stored in `validation_error`
    /// and will be returned by the next call to [`Scheduler::run`]. This
    /// matches the existing deferred-error pattern for scheduler cycles.
    ///
    /// Only the first validation error is retained — subsequent bad
    /// systems are still enqueued (so that `clear()` + re-registration
    /// works predictably) but the recorded error is not overwritten.
    #[inline]
    pub fn add_boxed(&mut self, system: Box<dyn System>) {
        if self.validation_error.is_none() {
            if let Err(e) = system.access().validate() {
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

    /// Sets the activation (iteration) order for a specific system.
    ///
    /// The default is [`ActivationOrder::Sequential`]. The new order takes
    /// effect on the next `run` call (no rebuild needed — activation orders
    /// are applied at iteration time, not during stage packing).
    pub fn set_activation_order(&mut self, system_id: SystemID, order: ActivationOrder) {
        if let Some(entry) = self.activation_orders.iter_mut()
            .find(|(id, _)| *id == system_id)
        {
            entry.1 = order;
        } else {
            self.activation_orders.push((system_id, order));
        }
    }

    /// Returns the activation order for a system, or [`ActivationOrder::Sequential`]
    /// if none has been explicitly set.
    pub fn activation_order(&self, system_id: SystemID) -> ActivationOrder {
        self.activation_orders.iter()
            .find(|(id, _)| *id == system_id)
            .map(|(_, o)| *o)
            .unwrap_or_default()
    }

    /// Sets the global RNG seed used by shuffle activation orders.
    ///
    /// A seed of 0 (the default) still produces deterministic shuffles; use
    /// different seeds to get different reproducible orderings across runs.
    pub fn seed(&mut self, global_seed: u64) {
        self.seed = global_seed;
    }

    /// Adds an explicit ordering constraint: `dependent` must run in a strictly
    /// later stage than `dependency`.
    ///
    /// Use for cases not expressible via `AccessSets` (e.g., "always run the
    /// audit logger last"). The constraint is incorporated into `rebuild`.
    pub fn add_ordering(&mut self, dependent: SystemID, dependency: SystemID) {
        self.ordering_edges.push(OrderingEdge { dependency, dependent });
        self.dirty = true;
    }

    /// Ensures the execution plan is up to date.
    ///
    /// If the plan is already current (`!dirty`), returns immediately.
    /// Otherwise, packs systems into stages, computes finalised channels for
    /// each boundary, and runs Kahn-style cycle detection. If a cycle is
    /// found, the error is stored and returned by the next `run` call.
    pub fn rebuild(&mut self) {
        if !self.dirty {
            return;
        }

        self.cycle_error = None;
        self.plan.clear();

        // Sort by SystemID for deterministic ordering.
        let mut indices: Vec<usize> = (0..self.systems.len()).collect();
        indices.sort_by_key(|&i| self.systems[i].id());

        let mut in_gpu_run = false;

        for index in &indices {
            let index = *index;
            let backend = self.systems[index].backend();
            let access = self.systems[index].access().clone();

            match backend {
                SystemBackend::CPU => {
                    if in_gpu_run {
                        self.close_gpu_run();
                        in_gpu_run = false;
                    }

                    // Try to slot into the most recent non-boundary CPU stage.
                    let mut placed = false;
                    for stage in self.plan.iter_mut().rev() {
                        if stage.stage_type == StageType::Boundary { break; }
                        if stage.stage_type != StageType::Cpu      { continue; }
                        if stage.can_accept(&access) {
                            stage.push_cpu(index, &access);
                            placed = true;
                            break;
                        }
                    }

                    if !placed {
                        let mut stage = Stage::cpu();
                        stage.push_cpu(index, &access);
                        self.plan.push(stage);
                    }
                }

                SystemBackend::GPU => {
                    if !in_gpu_run {
                        self.plan.push(Stage::boundary());
                        self.plan.push(Stage::gpu());
                        in_gpu_run = true;
                    }
                    let last = self.plan.last_mut()
                        .expect("plan must have a GPU stage");
                    debug_assert_eq!(last.stage_type, StageType::Gpu);
                    last.push_gpu(index);
                }
            }
        }

        if in_gpu_run {
            self.close_gpu_run();
        }

        // Apply explicit ordering edges: if `dependent` was placed before
        // `dependency`, open a new CPU stage after any stage containing
        // `dependency` and re-slot `dependent` there. Applied to a fixed
        // point so chained ordering constraints all settle correctly.
        self.apply_explicit_ordering(&indices);

        // Ensure the plan ends on a Boundary. Channels produced only in the
        // final CPU/GPU stage would otherwise never be finalised for the
        // tick (compute_finalised_channels looks for a *following* boundary
        // to annotate, and finds none for a trailing producer stage).
        //
        // A trailing boundary is also harmless for GPU-terminated plans
        // because `gpu_write_resources` on the inserted boundary is empty,
        // making `gpu::sync_pending_to_cpu` a no-op.
        if !matches!(self.plan.last(), Some(s) if s.is_boundary()) {
            self.plan.push(Stage::boundary());
        }

        // Compute `finalised_channels` for each stage.
        self.compute_finalised_channels();

        // Run Kahn cycle detection over the raw AccessSets-derived graph.
        // This MUST run after explicit ordering but its result is independent
        // of stage packing.
        if let Err(e) = self.detect_cycles(&indices) {
            self.cycle_error = Some(e);
        }

        self.dirty = false;
    }

    /// For each CPU/GPU stage, compute which channels are "finalised" by that
    /// stage: channels produced there whose no subsequent stage also produces them.
    /// Those channels are placed on the immediately following Boundary stage.
    fn compute_finalised_channels(&mut self) {
        let n = self.plan.len();

        for i in 0..n {
            if self.plan[i].is_boundary() {
                continue;
            }

            // Collect channels produced in stage i.
            let produced: Vec<ChannelID> = self.plan[i]
                .aggregate_access
                .produces
                .iter()
                .collect();

            for ch in produced {
                // A channel is finalised in stage i if no stage j > i also produces it.
                let is_last_producer = !self.plan[i+1..].iter()
                    .filter(|s| !s.is_boundary())
                    .any(|s| s.aggregate_access.produces.contains(ch));

                if is_last_producer {
                    // Find the next boundary stage after i and annotate it.
                    if let Some(boundary) = self.plan[i+1..].iter_mut()
                        .find(|s| s.is_boundary())
                    {
                        if !boundary.finalised_channels.contains(&ch) {
                            boundary.finalised_channels.push(ch);
                        }
                    }
                }
            }
        }
    }

    /// Kahn-style topological sort over the inherent ordering graph to detect cycles.
    ///
    /// **Important**: edges are derived from the *input* `AccessSets` and
    /// explicit ordering edges — NOT from the already-packed `plan`. Packing
    /// produces a linear stage order by construction, so rediscovering cycles
    /// from it is impossible. The genuine cycle shape we need to detect —
    /// system A produces channel X & consumes channel Y, system B produces
    /// channel Y & consumes channel X — is visible in the raw access sets and
    /// invisible in the stage assignment (the stage packer silently separates
    /// A and B into different stages regardless of ordering direction).
    ///
    /// Edges added:
    /// - For each ordered pair `(a, b)`: if `a.produces ∩ b.consumes ≠ ∅`,
    ///   add edge `a → b`. If `b.produces ∩ a.consumes ≠ ∅`, add edge
    ///   `b → a`. When both hold, the graph has a 2-cycle between a and b
    ///   and Kahn's algorithm will report it.
    /// - For each explicit ordering edge `dependency → dependent`, add
    ///   `dep → dependent`.
    ///
    /// Component conflicts are intentionally NOT modelled here: they have no
    /// inherent direction, so they cannot contribute to a real ordering
    /// cycle, and the stage packer's conflict check already prevents
    /// component-aliasing pairs from being co-scheduled.
    ///
    /// A cycle exists iff the topological sort cannot process all nodes.
    fn detect_cycles(&self, indices: &[usize]) -> ECSResult<()> {
        let n = self.systems.len();
        if n == 0 { return Ok(()); }

        // Node space: the "list position" under `indices`. This keeps the
        // identity stable across calls even if `self.systems` is indexed
        // differently elsewhere.
        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree: Vec<usize> = vec![0; n];

        // Convenience: list-position ↔ SystemID and list-position → AccessSets.
        let id_to_pos: std::collections::HashMap<SystemID, usize> = indices
            .iter()
            .enumerate()
            .map(|(pos, &si)| (self.systems[si].id(), pos))
            .collect();

        // Cache access sets by list position to avoid repeated virtual calls.
        let access: Vec<&AccessSets> = indices
            .iter()
            .map(|&si| self.systems[si].access())
            .collect();

        // Channel-direction edges. Iterate ordered pairs (a < b) once and add
        // the two possible directed edges independently.
        for a in 0..n {
            for b in (a + 1)..n {
                // a → b if a produces a channel that b consumes.
                if access[a].produces.intersects(&access[b].consumes) {
                    edges[a].push(b);
                    in_degree[b] += 1;
                }
                // b → a if b produces a channel that a consumes.
                if access[b].produces.intersects(&access[a].consumes) {
                    edges[b].push(a);
                    in_degree[a] += 1;
                }
            }
        }

        // Explicit ordering edges: `dependency` must precede `dependent`.
        for edge in &self.ordering_edges {
            if let (Some(&dep_pos), Some(&ant_pos)) = (
                id_to_pos.get(&edge.dependency),
                id_to_pos.get(&edge.dependent),
            ) {
                if !edges[dep_pos].contains(&ant_pos) {
                    edges[dep_pos].push(ant_pos);
                    in_degree[ant_pos] += 1;
                }
            }
        }

        // Kahn BFS.
        let mut queue: VecDeque<usize> = (0..n)
            .filter(|&i| in_degree[i] == 0)
            .collect();
        let mut visited = 0usize;

        while let Some(node) = queue.pop_front() {
            visited += 1;
            for &next in &edges[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if visited < n {
            Err(ECSError::from(ExecutionError::SchedulerCycle))
        } else {
            Ok(())
        }
    }

    /// Apply explicit ordering edges post-packing, iterating to a fixed point.
    ///
    /// Each edge `(dependency, dependent)` asserts that `dependent` must run
    /// in a strictly later stage than `dependency` — interpreted here as
    /// "separated by at least one `Boundary` stage", so that deferred
    /// commands emitted by `dependency` are visible to `dependent`.
    ///
    /// ## Fix-point iteration
    ///
    /// If edges chain (`A → B` and `B → C`) and are processed in an order
    /// that doesn't match dependency order, the first pass may settle one
    /// edge but invalidate another. We therefore repeat until a pass makes
    /// no changes, bounded by `ordering_edges.len() + 1` iterations. If the
    /// bound is reached the edges form a cycle; we mark the plan as cyclic
    /// and let `detect_cycles` surface the error.
    ///
    /// ## Boundary insertion
    ///
    /// When relocating `dependent` after `dependency`'s stage, we guarantee
    /// a `Boundary` stage sits between them (inserting one if missing). This
    /// makes `add_ordering`'s semantics unambiguous: state changes from
    /// `dependency` — including deferred commands — are fully applied before
    /// `dependent` runs.
    fn apply_explicit_ordering(&mut self, indices: &[usize]) {
        let max_iters = self.ordering_edges.len().saturating_add(1);
        let edges_snapshot = self.ordering_edges.clone();

        for _iter in 0..max_iters {
            let mut changed = false;

            for edge in &edges_snapshot {
                let dep_stage = self.find_stage_of(edge.dependency, indices);
                let ant_stage = self.find_stage_of(edge.dependent,  indices);

                let (Some(dep_stage_idx), Some(ant_stage_idx)) = (dep_stage, ant_stage) else {
                    continue; // unknown system IDs — silently skip
                };

                // The constraint is already satisfied if `dependent` is in
                // a stage strictly after `dependency` AND a boundary lies
                // between them.
                if ant_stage_idx > dep_stage_idx {
                    let has_boundary_between = self.plan
                        [dep_stage_idx + 1 .. ant_stage_idx]
                        .iter()
                        .any(|s| s.is_boundary());
                    if has_boundary_between {
                        continue;
                    }
                }

                // Need to relocate `dependent`.
                let Some(si) = indices.iter()
                    .find(|&&si| self.systems[si].id() == edge.dependent)
                    .copied()
                else { continue; };

                let access = self.systems[si].access().clone();
                self.plan[ant_stage_idx].system_indices.retain(|&x| x != si);
                self.recompute_aggregate(ant_stage_idx);

                // Target position: immediately after `dep_stage_idx`. If the
                // next slot is already a boundary, land after it; otherwise
                // insert a fresh boundary and land after that. This
                // guarantees the required "≥ 1 boundary between" invariant.
                let mut target = dep_stage_idx + 1;
                if target < self.plan.len() && self.plan[target].is_boundary() {
                    target += 1;
                } else {
                    self.plan.insert(target, Stage::boundary());
                    target += 1;
                }

                if target < self.plan.len()
                    && self.plan[target].stage_type == StageType::Cpu
                    && self.plan[target].can_accept(&access)
                {
                    self.plan[target].push_cpu(si, &access);
                } else {
                    let mut new_stage = Stage::cpu();
                    new_stage.push_cpu(si, &access);
                    self.plan.insert(target, new_stage);
                }

                changed = true;
            }

            if !changed {
                return;
            }
        }

        // Fix-point not reached within the iteration bound — the edges must
        // contain a cycle. Record it; detect_cycles will also flag this
        // structurally, but set the error defensively in case the cycle is
        // confined to explicit edges.
        self.cycle_error = Some(ECSError::from(ExecutionError::SchedulerCycle));
    }

    fn find_stage_of(&self, system_id: SystemID, indices: &[usize]) -> Option<usize> {
        let si = *indices.iter().find(|&&si| self.systems[si].id() == system_id)?;
        self.plan.iter().position(|s| s.system_indices.contains(&si))
    }

    fn recompute_aggregate(&mut self, stage_idx: usize) {
        let sys_indices = self.plan[stage_idx].system_indices.clone();
        let mut agg = AccessSets::default();
        for si in sys_indices {
            let access = self.systems[si].access();
            or_signature_in_place(&mut agg.read,  &access.read);
            or_signature_in_place(&mut agg.write, &access.write);
            agg.produces.or_in_place(&access.produces);
            agg.consumes.or_in_place(&access.consumes);
        }
        self.plan[stage_idx].aggregate_access = agg;
    }

    fn close_gpu_run(&mut self) {
        #[cfg(feature = "gpu")]
        let gpu_writes = self.collect_gpu_write_resources();

        #[allow(unused_mut)]
        let mut boundary = Stage::boundary();
        #[cfg(feature = "gpu")]
        { boundary.gpu_write_resources = gpu_writes; }
        self.plan.push(boundary);
    }

    #[cfg(feature = "gpu")]
    fn collect_gpu_write_resources(&self) -> Vec<GPUResourceID> {
        let gpu_stage = self.plan.iter().rev()
            .find(|s| s.stage_type == StageType::Gpu);
        let Some(stage) = gpu_stage else { return Vec::new(); };
        let mut resources = Vec::new();
        for &idx in &stage.system_indices {
            if let Some(gpu_cap) = self.systems[idx].gpu() {
                for &rid in gpu_cap.writes_resources() {
                    if !resources.contains(&rid) { resources.push(rid); }
                }
            }
        }
        resources
    }

    /// Runs the schedule once.
    ///
    /// 1. Rebuilds the plan if dirty.
    /// 2. Returns any cycle error stored during rebuild.
    /// 3. Executes each stage sequentially; systems within a CPU stage run in
    ///    parallel via Rayon.
    /// 4. At each Boundary stage: clears borrows, syncs GPU (if enabled),
    ///    applies deferred commands, and finalises boundary channels.
    #[allow(clippy::duplicated_code)]
    pub fn run(&mut self, ecs: ECSReference<'_>) -> ECSResult<()> {
        let _g = profiler::span("Scheduler::run");

        self.rebuild();

        // Surface any deferred build error. Validation errors (malformed
        // system access sets recorded at `add_boxed` time) take precedence
        // over cycle errors: if a system is structurally invalid, running
        // cycle analysis over it is meaningless.
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
                        ecs.apply_deferred_commands()?;
                    }

                    if !stage.finalised_channels.is_empty() {
                        ecs.finalise_boundaries(&stage.finalised_channels)?;
                    }
                }

                StageType::Cpu => {
                    let _s = profiler::span("Stage::Cpu")
                        .arg("systems", profiler::Arg::U64(stage.system_indices.len() as u64));

                    stage.system_indices
                        .par_iter()
                        .try_for_each(|&i| {
                            let sys = &self.systems[i];
                            profiler::next_arg("system_id", profiler::Arg::U64(sys.id() as u64));
                            profiler::next_arg("backend", profiler::Arg::Str(BACKEND_CPU.to_string()));
                            let _sg = profiler::span_fmt(format_args!("System::{}", sys.name()));
                            sys.run(ecs)
                        })?;

                    ecs.clear_borrows();
                }

                StageType::Gpu => {
                    let _s = profiler::span("Stage::Gpu")
                        .arg("systems", profiler::Arg::U64(stage.system_indices.len() as u64));

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
                            profiler::next_arg("backend", profiler::Arg::Str(BACKEND_GPU.to_string()));
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
    ///
    /// Called by [`PlanDisplay`](crate::engine::plan_display::PlanDisplay).
    pub fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, stage) in self.plan.iter().enumerate() {
            match stage.stage_type {
                StageType::Boundary => {
                    write!(f, "Stage {i} [Boundary]")?;
                    if !stage.finalised_channels.is_empty() {
                        let chs: Vec<String> = stage.finalised_channels
                            .iter().map(|c| format!("ch:{c}")).collect();
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
                            let ps: Vec<String> = access.produces.iter()
                                .map(|c| format!("ch:{c}")).collect();
                            write!(f, "  produces: [{}]", ps.join(", "))?;
                        }
                        if !access.consumes.is_empty() {
                            let cs: Vec<String> = access.consumes.iter()
                                .map(|c| format!("ch:{c}")).collect();
                            write!(f, "  consumes: [{}]", cs.join(", "))?;
                        }
                        writeln!(f)?;
                    }
                }
                StageType::Gpu => {
                    writeln!(f, "Stage {i} [GPU]")?;
                    for &si in &stage.system_indices {
                        writeln!(f, "  system {:4}  {:?}", self.systems[si].id(),
                                 self.systems[si].name())?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Formats the compiled execution plan as a Graphviz DOT graph.
    ///
    /// Called by [`DotExport`](crate::engine::dot_export::DotExport).
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
                        let chs: Vec<String> = stage.finalised_channels
                            .iter().map(|c| format!("ch:{c}")).collect();
                        format!("Boundary {i}\\nfinalises: {}", chs.join(", "))
                    };
                    writeln!(f,
                             "  {node_name} [shape=diamond, label=\"{label}\"];")?;
                }
                StageType::Cpu | StageType::Gpu => {
                    let kind = if stage.stage_type == StageType::Cpu { "CPU" } else { "GPU" };
                    writeln!(f, "  subgraph cluster_{i} {{")?;
                    writeln!(f, "    label=\"Stage {i} [{kind}]\";")?;
                    writeln!(f, "    style=filled; fillcolor=lightgrey;")?;
                    for &si in &stage.system_indices {
                        let sys = &self.systems[si];
                        writeln!(f,
                                 "    {}_{} [label=\"id:{} {}\", shape=box];",
                                 node_name, si, sys.id(), sys.name()
                        )?;
                    }
                    writeln!(f, "  }}")?;
                    // Invisible anchor node for edge routing.
                    writeln!(f,
                             "  {node_name} [shape=point, style=invis];")?;
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
    ///
    /// The plan may be stale if systems have been added since the last
    /// `rebuild` or `run` call.
    pub fn plan(&self) -> &[Stage] {
        &self.plan
    }
}
