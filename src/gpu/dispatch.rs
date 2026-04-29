//! # GPU Dispatch Runtime
//!
//! This module defines the **GPU execution bridge** between the ECS scheduler
//! and the GPU backend.
//!
//! ## Purpose
//!
//! The dispatch runtime is responsible for:
//! * coordinating GPU execution of ECS systems implementing [`GpuSystem`],
//! * mirroring ECS archetype component data to GPU buffers,
//! * dispatching compute workloads per archetype,
//! * synchronizing GPU execution,
//! * and copying mutated component data back into ECS storage.
//!
//! This module is the **only location** where ECS state, GPU pipelines, and
//! command submission intersect.
//!
//! ## High-level execution flow
//!
//! For each GPU-capable system invocation:
//!
//! 1. Acquire **exclusive ECS access** (`with_exclusive`).
//! 2. Compute the union of read/write component signatures.
//! 3. Upload matching component columns to GPU buffers.
//! 4. Dispatch the GPU compute pipeline **per matching archetype**.
//! 5. Submit GPU work; the following scheduler boundary performs the blocking
//!    synchronization/readback when CPU state is needed.
//! 6. Download mutated component columns back into ECS storage at that boundary.
//!
//! ## Design philosophy
//!
//! * **Global device runtime, world-local data**
//!   - GPU initialization and pipeline caches are shared globally.
//!   - Component mirrors, pending downloads, and params buffers are owned by
//!     each ECS world.
//! * **Archetype-granular dispatch**
//!   - Each archetype is dispatched independently for predictable memory layout.
//! * **Explicit data movement**
//!   - All CPU to GPU transfers are explicit and phase-controlled.
//! * **Strict ECS invariants**
//!   - Structural mutation and parallel iteration are forbidden during GPU execution.
//!
//! ## Concurrency and safety model
//!
//! * GPU execution occurs inside `ECSReference::with_exclusive`.
//! * No ECS iteration may be active during GPU dispatch.
//! * Component borrows are enforced before upload and after download.
//! * GPU synchronization is explicit and concentrated at sync/readback
//!   boundaries via `device.poll`.

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::engine::archetype::Archetype;
use crate::engine::component::{or_signature_in_place, Signature};
use crate::engine::error::{ECSError, ECSResult, ExecutionError};
use crate::engine::manager::ECSReference;
use crate::engine::systems::{GpuSystem, System};
use crate::engine::types::{GPUAccessMode, GPUResourceID};

use crate::gpu::mirror::Mirror;
use crate::gpu::pipeline::PipelineCache;
use crate::gpu::GPUContext;
use crate::gpu::GPUResourceRegistry;

#[cfg(feature = "messaging_gpu")]
use crate::gpu::pipeline::hash_str;
#[cfg(feature = "messaging_gpu")]
use crate::gpu::GPUBindingDesc;

struct DeviceRuntime {
    context: GPUContext,
    pipelines: PipelineCache,
    #[cfg(feature = "messaging_gpu")]
    framework_pipelines: FrameworkPipelineCache,
}

/// GPU state owned by one ECS world.
pub(crate) struct GpuWorldState {
    pub(crate) mirror: Mirror,
    pub(crate) pending_download: Signature,
    pub(crate) params_buffers: Vec<wgpu::Buffer>,
    params_generation: u64,
    bind_groups: GpuBindGroupCache,
}

impl GpuWorldState {
    /// Creates empty world-local GPU state.
    pub(crate) fn new() -> Self {
        Self {
            mirror: Mirror::new(),
            pending_download: Signature::default(),
            params_buffers: Vec::new(),
            params_generation: 0,
            bind_groups: GpuBindGroupCache::new(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ComponentBindGroupKey {
    system_id: crate::engine::types::SystemID,
    archetype_id: crate::engine::types::ArchetypeID,
    reads: Vec<crate::engine::types::ComponentID>,
    writes: Vec<crate::engine::types::ComponentID>,
    params_index: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ResourceBindGroupKey {
    system_id: crate::engine::types::SystemID,
    resource_ids: Vec<GPUResourceID>,
    layout_keys: Vec<u8>,
}

struct GpuBindGroupCache {
    component_groups: HashMap<ComponentBindGroupKey, Arc<wgpu::BindGroup>>,
    resource_groups: HashMap<ResourceBindGroupKey, Arc<wgpu::BindGroup>>,
    mirror_generation: u64,
    params_generation: u64,
    resource_generation: u64,
}

impl GpuBindGroupCache {
    fn new() -> Self {
        Self {
            component_groups: HashMap::new(),
            resource_groups: HashMap::new(),
            mirror_generation: 0,
            params_generation: 0,
            resource_generation: 0,
        }
    }

    fn sync_component_generations(&mut self, mirror_generation: u64, params_generation: u64) {
        if self.mirror_generation != mirror_generation
            || self.params_generation != params_generation
        {
            self.component_groups.clear();
            self.mirror_generation = mirror_generation;
            self.params_generation = params_generation;
        }
    }

    fn sync_resource_generation(&mut self, resource_generation: u64) {
        if self.resource_generation != resource_generation {
            self.resource_groups.clear();
            self.resource_generation = resource_generation;
        }
    }
}

static DEVICE_RUNTIME: OnceLock<ECSResult<Mutex<DeviceRuntime>>> = OnceLock::new();

fn device_runtime() -> ECSResult<MutexGuard<'static, DeviceRuntime>> {
    let cell: &ECSResult<Mutex<DeviceRuntime>> = DEVICE_RUNTIME.get_or_init(|| {
        let run_time = DeviceRuntime {
            context: GPUContext::new()?,
            pipelines: PipelineCache::new(),
            #[cfg(feature = "messaging_gpu")]
            framework_pipelines: FrameworkPipelineCache::new(),
        };
        Ok(Mutex::new(run_time))
    });

    let mutex: &Mutex<DeviceRuntime> = match cell {
        Ok(matched_runtime) => matched_runtime,
        Err(e) => {
            return Err(ECSError::from(ExecutionError::GpuInitFailed {
                message: format!("{e:?}").into(),
            }));
        }
    };

    mutex.lock().map_err(|_| {
        ECSError::from(ExecutionError::LockPoisoned {
            what: "gpu device runtime",
        })
    })
}

#[derive(Debug, Default)]
#[cfg(feature = "messaging_gpu")]
struct FrameworkPipelineCache {
    map: HashMap<(u64, u64, u64), (wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
}

#[cfg(feature = "messaging_gpu")]
impl FrameworkPipelineCache {
    fn new() -> Self {
        Self::default()
    }

    fn get_or_create(
        &mut self,
        context: &GPUContext,
        label: &'static str,
        shader_wgsl: &'static str,
        entry_point: &'static str,
        bindings: &[GPUBindingDesc],
    ) -> ECSResult<(&wgpu::ComputePipeline, &wgpu::BindGroupLayout)> {
        let key = (
            hash_str(label) ^ hash_str(shader_wgsl),
            hash_str(entry_point),
            hash_framework_layout(bindings),
        );
        self.map.entry(key).or_insert_with(|| {
            let mut entries = Vec::with_capacity(bindings.len());
            for (binding, desc) in bindings.iter().enumerate() {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: desc.read_only,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }

            let bgl = context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries: &entries,
                });
            let layout = context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                });
            let module = context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(shader_wgsl.into()),
                });
            let pipeline =
                context
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: Some(&layout),
                        module: &module,
                        entry_point: Some(entry_point),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });
            (pipeline, bgl)
        });

        let (pipeline, bgl) = self.map.get(&key).unwrap();
        Ok((pipeline, bgl))
    }
}

#[inline]
#[cfg(feature = "messaging_gpu")]
fn hash_framework_layout(bindings: &[GPUBindingDesc]) -> u64 {
    let mut hash: u64 = 1469598103934665603;
    for desc in bindings {
        hash ^= desc.key() as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

/// Description of one framework-owned compute dispatch.
#[cfg(feature = "messaging_gpu")]
pub(crate) struct BoundaryKernelDesc<'a> {
    /// Diagnostic label and pipeline-cache discriminator.
    pub label: &'static str,
    /// WGSL shader source.
    pub shader: &'static str,
    /// Compute entry point.
    pub entry_point: &'static str,
    /// Storage binding layout for group(0).
    pub bindings: &'a [GPUBindingDesc],
    /// Bind group entries for group(0).
    pub entries: &'a [wgpu::BindGroupEntry<'a>],
    /// Number of workgroups in x.
    pub workgroups_x: u32,
    /// Number of workgroups in y.
    pub workgroups_y: u32,
    /// Number of workgroups in z.
    pub workgroups_z: u32,
}

/// Narrow dispatch facade for framework-owned boundary compute work.
#[cfg(feature = "messaging_gpu")]
pub(crate) struct BoundaryGpuDispatch<'a> {
    runtime: &'a mut DeviceRuntime,
}

#[cfg(feature = "messaging_gpu")]
impl BoundaryGpuDispatch<'_> {
    /// Accesses the centralized GPU context for framework-owned buffer IO.
    #[inline]
    pub(crate) fn context(&self) -> &GPUContext {
        &self.runtime.context
    }

    /// Dispatches a framework-owned compute kernel through the shared runtime.
    pub(crate) fn dispatch(&mut self, desc: BoundaryKernelDesc<'_>) -> ECSResult<()> {
        let (pipeline, bgl) = self.runtime.framework_pipelines.get_or_create(
            &self.runtime.context,
            desc.label,
            desc.shader,
            desc.entry_point,
            desc.bindings,
        )?;

        let bind_group =
            self.runtime
                .context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(desc.label),
                    layout: bgl,
                    entries: desc.entries,
                });

        let mut encoder =
            self.runtime
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(desc.label),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(desc.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                desc.workgroups_x.max(1),
                desc.workgroups_y.max(1),
                desc.workgroups_z.max(1),
            );
        }

        self.runtime.context.queue.submit(Some(encoder.finish()));
        poll_context(&self.runtime.context)
    }
}

/// Runs framework-owned boundary GPU work through the centralized runtime.
#[cfg(feature = "messaging_gpu")]
pub(crate) fn with_boundary_dispatch<R>(
    gpu_resources: &mut GPUResourceRegistry,
    f: impl FnOnce(&mut BoundaryGpuDispatch<'_>, &mut GPUResourceRegistry) -> ECSResult<R>,
) -> ECSResult<R> {
    let mut run_time = device_runtime()?;
    gpu_resources.ensure_created(&run_time.context)?;
    gpu_resources.upload_dirty(&run_time.context)?;
    let mut dispatch = BoundaryGpuDispatch {
        runtime: &mut run_time,
    };
    f(&mut dispatch, gpu_resources)
}

/// Synchronize pending GPU downloads into ECS storage.
pub fn sync_pending_to_cpu(
    ecs: ECSReference<'_>,
    affected_resources: &[GPUResourceID],
) -> ECSResult<()> {
    ecs.with_exclusive(|data| {
        let run_time = device_runtime()?;

        data.gpu_resources_mut().ensure_created(&run_time.context)?;

        if affected_resources.is_empty() {
            data.gpu_resources_mut()
                .download_pending(&run_time.context)?;
        } else {
            data.gpu_resources_mut()
                .download_pending_filtered(&run_time.context, affected_resources)?;
        }

        let pending = {
            let p = data.gpu_world_state().pending_download;
            if is_signature_empty(&p) {
                return Ok(());
            }
            p
        };

        let registry_arc = data.registry().clone();
        let registry = registry_arc.read().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "component registry",
            })
        })?;

        let (archetypes_mut, world_state) = data.gpu_download_parts();
        world_state.mirror.download_signature(
            &run_time.context,
            archetypes_mut,
            &pending,
            &registry,
        )?;

        world_state.pending_download = Signature::default();

        Ok(())
    })
}

/// Executes a single GPU-backed ECS system with exclusive world access.
pub fn execute_gpu_system(
    ecs: ECSReference<'_>,
    system: &dyn System,
    gpu: &dyn GpuSystem,
) -> ECSResult<()> {
    ecs.with_exclusive(|data| {
        let mut access = system.access().clone();
        normalize_access_sets(&mut access);

        #[cfg(debug_assertions)]
        {
            for (r, w) in access
                .read
                .components
                .iter()
                .zip(access.write.components.iter())
            {
                debug_assert_eq!(r & w, 0, "AccessSets overlap after normalization");
            }
        }

        let read_signature = &access.read;
        let write_signature = &access.write;

        let union = union_signatures(read_signature, write_signature);

        let mut run_time = device_runtime()?;

        data.gpu_resources_mut().ensure_created(&run_time.context)?;
        data.gpu_resources_mut().upload_dirty(&run_time.context)?;

        let registry_arc = data.registry().clone();
        let registry = registry_arc.read().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "component registry",
            })
        })?;

        {
            let (archetypes, dirty_chunks, world_state, gpu_resources) = data.gpu_execution_parts();
            world_state.mirror.upload_signature_dirty_chunks(
                &run_time.context,
                archetypes,
                &union,
                dirty_chunks,
                &registry,
            )?;
            dispatch_over_archetypes(
                &mut run_time,
                world_state,
                system.id(),
                gpu,
                archetypes,
                &access,
                gpu_resources,
            )?;
        }

        or_signature_in_place(
            &mut data.gpu_world_state_mut().pending_download,
            write_signature,
        );

        let writes = gpu.writes_resources();
        if !writes.is_empty() {
            for &resource_id in writes {
                data.gpu_resources_mut()
                    .mark_pending_download(resource_id)?;
            }
        } else {
            for &resource_id in gpu.uses_resources() {
                data.gpu_resources_mut()
                    .mark_pending_download(resource_id)?;
            }
        }

        Ok(())
    })
}

fn dispatch_over_archetypes(
    run_time: &mut DeviceRuntime,
    world_state: &mut GpuWorldState,
    system_id: crate::engine::types::SystemID,
    gpu: &dyn GpuSystem,
    archetypes: &[Archetype],
    access: &crate::engine::systems::AccessSets,
    gpu_resources: &GPUResourceRegistry,
) -> ECSResult<()> {
    // Resolve GPU resources

    let mut resource_ids = Vec::new();
    for &resource_id in gpu.uses_resources() {
        if !resource_ids.contains(&resource_id) {
            resource_ids.push(resource_id);
        }
    }
    let resource_layout = gpu_resources.flattened_binding_descs(&resource_ids);
    let resource_layout_keys: Vec<u8> = resource_layout.iter().map(|desc| desc.key()).collect();
    let resource_generation = gpu_resources.binding_generation();

    // Resolve component access

    let mut reads: Vec<_> =
        crate::engine::component::iter_bits_from_words(&access.read.components).collect();
    let mut writes: Vec<_> =
        crate::engine::component::iter_bits_from_words(&access.write.components).collect();

    writes.sort_unstable();
    reads.retain(|cid| writes.binary_search(cid).is_err());
    reads.sort_unstable();

    let read_count = reads.len();
    let write_count = writes.len();

    // Pipeline

    let (pipeline, bgl0, bgl1_opt) = run_time.pipelines.get_or_create(
        &run_time.context,
        system_id,
        gpu.shader(),
        gpu.entry_point(),
        read_count,
        write_count,
        &resource_layout,
    )?;

    // Destructure run_time so we can borrow fields independently inside the loop.
    let DeviceRuntime { context, .. } = run_time;
    let GpuWorldState {
        mirror,
        params_buffers,
        params_generation,
        bind_groups,
        ..
    } = world_state;
    bind_groups.sync_component_generations(mirror.binding_generation(), *params_generation);
    bind_groups.sync_resource_generation(resource_generation);

    // Bind group 1 (GPU resources) is stable across every archetype for this
    // dispatch as long as resource buffers and layouts are unchanged.
    let bind_group1 = if let Some(bgl1) = bgl1_opt {
        let key = ResourceBindGroupKey {
            system_id,
            resource_ids: resource_ids.clone(),
            layout_keys: resource_layout_keys,
        };

        if let Some(cached) = bind_groups.resource_groups.get(&key) {
            Some(Arc::clone(cached))
        } else {
            let mut entries1 = Vec::with_capacity(resource_layout.len());
            gpu_resources.append_bind_group_entries(&resource_ids, 0, &mut entries1)?;
            let bind_group = Arc::new(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("abm_bind_group_group1"),
                    layout: bgl1,
                    entries: &entries1,
                },
            ));
            bind_groups
                .resource_groups
                .insert(key, Arc::clone(&bind_group));
            Some(bind_group)
        }
    } else {
        None
    };
    let mut params_index = 0usize;

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("abm_compute_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("abm_compute_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);

        // Dispatch per archetype

        let mut archetype_base = 0u32;
        for archetype in archetypes {
            if !archetype.signature().contains_all(&access.read)
                || !archetype.signature().contains_all(&access.write)
            {
                continue;
            }

            let entity_len = archetype.length()? as u32;
            if entity_len == 0 {
                continue;
            }

            // Params buffer

            #[repr(C, align(16))]
            #[derive(Clone, Copy)]
            struct Params {
                entity_len: u32,
                archetype_base: u32,
                _pad1: u32,
                _pad2: u32,
            }

            unsafe impl bytemuck::Pod for Params {}
            unsafe impl bytemuck::Zeroable for Params {}

            let params = Params {
                entity_len,
                archetype_base,
                _pad1: 0,
                _pad2: 0,
            };

            let size = size_of::<Params>() as u64;
            let current_params_index = params_index;
            if params_buffers.len() <= params_index {
                params_buffers.push(context.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("abm_params"),
                    size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                *params_generation = params_generation.wrapping_add(1);
            } else if params_buffers[params_index].size() < size {
                params_buffers[params_index] =
                    context.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("abm_params"),
                        size,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                *params_generation = params_generation.wrapping_add(1);
            }
            bind_groups.sync_component_generations(mirror.binding_generation(), *params_generation);

            let params_buf = &params_buffers[params_index];
            params_index += 1;
            context
                .queue
                .write_buffer(params_buf, 0, bytemuck::bytes_of(&params));

            // Bind group 0 (components + params) is stable while the component
            // mirror buffers and params buffer set keep the same generation.
            let component_key = ComponentBindGroupKey {
                system_id,
                archetype_id: archetype.archetype_id(),
                reads: reads.clone(),
                writes: writes.clone(),
                params_index: current_params_index,
            };

            let bind_group0 = if let Some(cached) = bind_groups.component_groups.get(&component_key)
            {
                Arc::clone(cached)
            } else {
                let mut entries0 = Vec::with_capacity(read_count + write_count + 1);

                for (i, &component_id) in reads.iter().enumerate() {
                    let entry = resolve_buffer_entry(
                        mirror,
                        archetype,
                        component_id,
                        i as u32,
                        GPUAccessMode::Read,
                    )?;
                    entries0.push(entry);
                }

                let base = reads.len();
                for (j, &component_id) in writes.iter().enumerate() {
                    let entry = resolve_buffer_entry(
                        mirror,
                        archetype,
                        component_id,
                        (base + j) as u32,
                        GPUAccessMode::Write,
                    )?;
                    entries0.push(entry);
                }

                entries0.push(wgpu::BindGroupEntry {
                    binding: (read_count + write_count) as u32,
                    resource: params_buf.as_entire_binding(),
                });

                let bind_group = Arc::new(context.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("abm_bind_group_group0"),
                        layout: bgl0,
                        entries: &entries0,
                    },
                ));
                bind_groups
                    .component_groups
                    .insert(component_key, Arc::clone(&bind_group));
                bind_group
            };

            // Dispatch

            pass.set_bind_group(0, bind_group0.as_ref(), &[]);
            if let Some(bg1) = &bind_group1 {
                pass.set_bind_group(1, bg1.as_ref(), &[]);
            }

            let workgroup = gpu.workgroup_size().max(1);
            let groups = entity_len.div_ceil(workgroup);
            pass.dispatch_workgroups(groups, 1, 1);
            archetype_base = archetype_base.saturating_add(entity_len);
        }
    }

    // Submit without blocking. The scheduler inserts a boundary after GPU
    // stages, and `sync_pending_to_cpu` performs the required blocking poll
    // before CPU-visible reads observe GPU-written state.
    context.queue.submit(Some(encoder.finish()));

    Ok(())
}

#[cfg(feature = "messaging_gpu")]
fn poll_context(context: &GPUContext) -> ECSResult<()> {
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| {
            ECSError::from(ExecutionError::GpuDispatchFailed {
                message: format!("wgpu device poll failed: {e:?}").into(),
            })
        })?;
    Ok(())
}

/// Looks up the mirror buffer for `component_id` within `archetype` and wraps
/// it in a [`wgpu::BindGroupEntry`] at the given `binding` slot.
fn resolve_buffer_entry<'a>(
    mirror: &'a Mirror,
    archetype: &Archetype,
    component_id: crate::engine::types::ComponentID,
    binding: u32,
    access: GPUAccessMode,
) -> ECSResult<wgpu::BindGroupEntry<'a>> {
    let buffer = mirror
        .buffer_for(archetype.archetype_id(), component_id)
        .ok_or_else(|| {
            ECSError::from(ExecutionError::GpuMissingBuffer {
                archetype_id: archetype.archetype_id(),
                component_id,
                access,
            })
        })?;

    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    })
}

fn union_signatures(a: &Signature, b: &Signature) -> Signature {
    let mut out = Signature::default();
    for ((o, av), bv) in out
        .components
        .iter_mut()
        .zip(a.components.iter())
        .zip(b.components.iter())
    {
        *o = *av | *bv;
    }
    out
}

#[inline]
fn is_signature_empty(sig: &Signature) -> bool {
    sig.components.iter().all(|&w| w == 0)
}

#[inline]
fn normalize_access_sets(access: &mut crate::engine::systems::AccessSets) {
    for (r, w) in access
        .read
        .components
        .iter_mut()
        .zip(access.write.components.iter())
    {
        *r &= !*w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_world_state_keeps_pending_and_mirror_state_independent() {
        let mut world_a = GpuWorldState::new();
        let world_b = GpuWorldState::new();

        world_a.pending_download.set(0);

        assert!(world_a.pending_download.has(0));
        assert!(!world_b.pending_download.has(0));
        assert_ne!(
            &world_a.mirror as *const Mirror,
            &world_b.mirror as *const Mirror
        );
    }
}
