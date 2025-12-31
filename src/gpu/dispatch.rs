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
//! 5. Synchronize GPU execution.
//! 6. Download mutated component columns back into ECS storage.
//!
//! ## Design philosophy
//!
//! * **Single global GPU runtime**
//!   - GPU initialization and pipeline caches are shared globally.
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
//! * GPU synchronization is explicit via `device.poll`.

#![cfg(feature = "gpu")]

use std::sync::{OnceLock, Mutex, MutexGuard};

use crate::engine::archetype::Archetype;
use crate::engine::component::Signature;
use crate::engine::error::{ECSResult, ECSError, ExecutionError};
use crate::engine::manager::ECSReference;
use crate::engine::systems::{System, GpuSystem};
use crate::engine::types::{GPUAccessMode};

use crate::gpu::context::GPUContext;
use crate::gpu::mirror::Mirror;
use crate::gpu::pipeline::PipelineCache;


/// Internal GPU runtime state.
///
/// ## Role
/// Bundles all long-lived GPU backend state required to execute ECS GPU systems:
/// * the GPU device and queue,
/// * the ECS GPU buffer mirror,
/// * and the compute pipeline cache.
///
/// ## Lifetime
/// Constructed lazily on first GPU system execution and stored globally.
///
/// ## Synchronization
/// Access to the runtime is serialized via a global `Mutex`. This ensures:
/// * pipelines are created deterministically,
/// * GPU buffer reuse is safe,
/// * device usage is ordered.

struct Runtime {
    context: GPUContext,
    mirror: Mirror,
    pipelines: PipelineCache,
    pending_download: Signature,
}

impl Runtime {
    #[inline]
    fn download_pending_into(
        &mut self,
        archetypes_mut: &mut [Archetype],
        pending: &Signature,
    ) -> ECSResult<()> {
        let ctx = &self.context;
        self.mirror.download_signature(ctx, archetypes_mut, pending)
    }
}

static RUNTIME: OnceLock<ECSResult<Mutex<Runtime>>> = OnceLock::new();

/// Lazily initializes and returns the global GPU runtime.
///
/// ## Semantics
/// * On first access, initializes the GPU device, buffer mirror, and pipeline cache.
/// * Subsequent calls reuse the same runtime instance.
///
/// ## Error handling
/// * GPU initialization errors are cached and rethrown on every access.
/// * Mutex poisoning is reported as an ECS execution error.
///
/// ## Thread safety
/// Serialized via `Mutex`; safe for use across scheduler threads.

fn runtime() -> ECSResult<MutexGuard<'static, Runtime>> {
    let cell: &ECSResult<Mutex<Runtime>> = RUNTIME.get_or_init(|| {
        let run_time = Runtime {
            context: GPUContext::new()?,
            mirror: Mirror::new(),
            pipelines: PipelineCache::new(),
            pending_download: Signature::default(),
        };
        Ok(Mutex::new(run_time))
    });

    let mutex: &Mutex<Runtime> = match cell {
        Ok(matched_runtime) => matched_runtime,
        Err(e) => {
            return Err(ECSError::from(ExecutionError::GpuInitFailed {
                message: format!("{e:?}").into(),
            }));
        }
    };

    mutex.lock().map_err(|_| {
        ECSError::from(ExecutionError::LockPoisoned { what: "gpu runtime" })
    })
}

/// Barrier that makes CPU storage consistent by downloading all pending GPU writes.
pub fn sync_pending_to_cpu(ecs: ECSReference<'_>) -> ECSResult<()> {
    ecs.with_exclusive(|data| {
        let mut run_time = runtime()?;

        // Extract pending signature
        let pending = {
            let p = run_time.pending_download.clone();
            if is_signature_empty(&p) {
                return Ok(());
            }
            p
        };

        // Download pending GPU writes
        let archetypes_mut: &mut [Archetype] = data.archetypes_mut();
        run_time.download_pending_into(archetypes_mut, &pending)?;

        // Clear pending set
        run_time.pending_download = Signature::default();

        Ok(())
    })
}



/// Executes a GPU-backed ECS system.
///
/// ## Role
/// This function is the **scheduler-facing entry point** for GPU systems.
/// It bridges ECS execution semantics with GPU compute dispatch.
///
/// ## Execution guarantees
/// * Runs with exclusive ECS access.
/// * Prevents structural mutation and parallel iteration.
/// * Ensures GPU-visible data is consistent with ECS storage.
///
/// ## Steps
/// 1. Compute read/write access signatures.
/// 2. Upload required components to GPU buffers.
/// 3. Dispatch compute workloads per matching archetype.
/// 4. Download mutated components back into ECS storage.
///
/// ## Errors
/// Returns an error if:
/// * GPU initialization fails,
/// * required component buffers are missing,
/// * pipeline compilation fails,
/// * GPU execution or synchronization fails.

pub fn execute_gpu_system(
    ecs: ECSReference<'_>,
    system: &dyn System,
    gpu: &dyn GpuSystem,
) -> ECSResult<()> {
    ecs.with_exclusive(|data| {
        let access = system.access();
        let read_signature = &access.read;
        let write_signature = &access.write;

        let union = union_signatures(read_signature, write_signature);

        let mut run_time = runtime()?;

        // Upload only dirty chunks, then dispatch.
        {
            let archetypes: &[Archetype] = data.archetypes();

            {
                let Runtime { context, mirror, .. } = &mut *run_time;
                mirror.upload_signature_dirty_chunks(
                    &*context,
                    archetypes,
                    &union,
                    data.gpu_dirty_chunks(),
                )?;
            }

            dispatch_over_archetypes(&mut *run_time, system.id(), gpu, archetypes, &access)?;
        }

        or_signature_in_place(&mut run_time.pending_download, write_signature);

        Ok(())
    })
}

/// Dispatches a GPU system over all matching archetypes.
///
/// ## Semantics
/// * Filters archetypes by required read/write signatures.
/// * Builds bind groups dynamically per archetype.
/// * Dispatches compute workloads sized to entity count.
///
/// ## Binding layout
/// Bindings are assigned in the following order:
/// 1. Read component buffers
/// 2. Write component buffers
/// 3. Uniform parameter buffer (`entity_len`)
///
/// ## Workgroup sizing
/// * Workgroup size is defined by the `GpuSystem`.
/// * Dispatch count is computed as `ceil(entity_len / workgroup_size)`.
///
/// ## Synchronization
/// GPU execution is synchronized after each archetype dispatch to ensure
/// correctness before subsequent uploads/downloads.

fn dispatch_over_archetypes(
    run_time: &mut Runtime,
    system_id: crate::engine::types::SystemID,
    gpu: &dyn GpuSystem,
    archetypes: &[Archetype],
    access: &crate::engine::systems::AccessSets,
) -> ECSResult<()> {
    use wgpu::util::DeviceExt;

    let mut reads: Vec<crate::engine::types::ComponentID> =
        crate::engine::component::iter_bits_from_words(&access.read.components).collect();
    let mut writes: Vec<crate::engine::types::ComponentID> =
        crate::engine::component::iter_bits_from_words(&access.write.components).collect();
    reads.sort_unstable();
    writes.sort_unstable();

    let read_count = reads.len();
    let write_count = writes.len();
    let binding_count = read_count + write_count + 1;

    let (pipeline, bind_group_layout) = run_time.pipelines.get_or_create(
        &run_time.context,
        system_id,
        gpu.shader(),
        gpu.entry_point(),
        read_count,
        write_count,
    )?;

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

        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(binding_count);

        for (i, &component_id) in reads.iter().enumerate() {
            let archetype_id = archetype.archetype_id();

            let buffer = run_time
                .mirror
                .buffer_for(archetype_id, component_id)
                .ok_or_else(|| {
                    ECSError::from(ExecutionError::GpuMissingBuffer {
                        archetype_id,
                        component_id,
                        access: GPUAccessMode::Read,
                    })
                })?;

            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            });
        }

        let base = reads.len();
        for (j, &component_id) in writes.iter().enumerate() {
            let archetype_id = archetype.archetype_id();

            let buffer = run_time
                .mirror
                .buffer_for(archetype_id, component_id)
                .ok_or_else(|| {
                    ECSError::from(ExecutionError::GpuMissingBuffer {
                        archetype_id,
                        component_id,
                        access: GPUAccessMode::Write,
                    })
                })?;

            entries.push(wgpu::BindGroupEntry {
                binding: (base + j) as u32,
                resource: buffer.as_entire_binding(),
            });
        }

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Params {
            entity_len: u32,
            _p0: u32,
            _p1: u32,
            _p2: u32,
        }
        
        unsafe impl bytemuck::Pod for Params {}
        unsafe impl bytemuck::Zeroable for Params {}

        let params = Params {
            entity_len,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        };

        let parameter_buffer = run_time.context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("abm_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        entries.push(wgpu::BindGroupEntry {
            binding: (binding_count - 1) as u32,
            resource: parameter_buffer.as_entire_binding(),
        });

        let bind_group = run_time.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("abm_bind_group"),
            layout: bind_group_layout,
            entries: &entries,
        });

        let mut encoder = run_time.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("abm_compute_encoder"),
            },
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("abm_compute_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let work_group_size = gpu.workgroup_size().max(1);
            let groups = (entity_len + work_group_size - 1) / work_group_size;
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let submission = run_time.context.queue.submit(Some(encoder.finish()));
        run_time
            .context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|e| {
                ECSError::from(ExecutionError::GpuDispatchFailed {
                    message: format!("wgpu device poll failed: {e:?}").into(),
                })
            })?;
    }

    Ok(())
}

/// Computes the union of two component signatures.
///
/// ## Purpose
/// Used to determine which components must be uploaded to the GPU when a system
/// reads and writes different component sets.

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
fn or_signature_in_place(dst: &mut Signature, src: &Signature) {
    for (d, s) in dst.components.iter_mut().zip(src.components.iter()) {
        *d |= *s;
    }
}

#[inline]
fn is_signature_empty(sig: &Signature) -> bool {
    sig.components.iter().all(|&w| w == 0)
}
