//! # GPU Compute Pipeline Cache
//!
//! This module provides a **compute pipeline cache** for the GPU execution backend.
//! It is responsible for creating, storing, and reusing `wgpu::ComputePipeline`
//! objects and their associated `BindGroupLayout`s.
//!
//! ## Purpose
//!
//! * each ECS system is compiled into a GPU pipeline **at most once per binding layout**,
//! * pipelines are reused across frames and dispatches,
//! * bind group layouts remain stable and compatible with system access signatures.
//!
//! The cache is indexed by `(SystemID, binding_count)`, where `binding_count`
//! reflects the number of component buffers (reads + writes) plus a uniform
//! parameter buffer.
//!
//! ---
//!
//! ## Binding model
//!
//! Pipelines created by this module follow a strict binding convention:
//!
//! * Bindings `0..N-1` - storage buffers for component columns
//! * Binding `N` - uniform buffer containing per-dispatch parameters
//!
//! This layout is compatible with archetype-based dispatch, where each archetype
//! instance binds a different set of buffers while reusing the same pipeline.
//!
//! ---
//!
//! ## Safety and invariants
//!
//! * All pipelines are created for **compute-only** execution.
//! * Shader source is assumed to be valid WGSL.
//! * Binding layouts must exactly match the shader's declared bindings.
//! * Pipeline creation errors are surfaced as ECS execution errors.
//!
//! ---
//!
//! ## Usage
//!
//! The cache is owned by the GPU runtime and accessed during dispatch via
//! [`PipelineCache::get_or_create`]. Callers must ensure that:
//!
//! * `binding_count` matches the system's access signature,
//! * `shader_wgsl` and `entry_point` are consistent for a given `SystemID`.
//!

#![cfg(feature = "gpu")]

use std::collections::HashMap;

use crate::engine::error::{ECSResult, ECSError, ExecutionError};
use crate::engine::types::SystemID;

use crate::gpu::GPUContext;
use crate::gpu::GPUBindingDesc;


#[inline]
pub(crate) fn hash_str(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

#[inline]
fn hash_resource_layout(descriptions: &[GPUBindingDesc]) -> u64 {
    let mut hash: u64 = 1469598103934665603;
    for description in descriptions {
        hash ^= description.key() as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

/// Cache of GPU compute pipelines and their bind group layouts.
///
/// ## Role
/// Stores `wgpu::ComputePipeline` objects keyed by `(SystemID, binding_count)`,
/// allowing systems to reuse pipelines across dispatches.
///
/// ## Design
/// * One pipeline per system per binding layout
/// * Pipelines are created lazily on first use
/// * Layouts are stored alongside pipelines to guarantee compatibility
///
/// ## Thread safety
/// This type is not thread-safe by itself and must be externally synchronized
/// by the GPU runtime.

#[derive(Debug)]
pub struct PipelineCache {
    map: HashMap<
    (SystemID, u64, u64, usize, usize, usize, u64),
    (wgpu::ComputePipeline, wgpu::BindGroupLayout, Option<wgpu::BindGroupLayout>),
    >,
}

impl PipelineCache {
    /// Creates an empty pipeline cache.
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    /// Retrieves an existing compute pipeline or creates a new one.
    ///
    /// ## Parameters
    /// * `context` - GPU device context
    /// * `system_id` - ECS system identifier
    /// * `shader_wgsl` - WGSL compute shader source
    /// * `entry_point` - shader entry point function
    /// * `binding_count` - number of bind group entries
    ///
    /// ## Semantics
    /// * If a pipeline for `(system_id, binding_count)` exists, it is reused.
    /// * Otherwise, a new pipeline and bind group layout are created and cached.
    ///
    /// ## Errors
    /// Returns an error if pipeline creation fails, typically due to:
    /// * invalid WGSL source
    /// * binding layout mismatch
    /// * GPU driver or device errors

    pub fn get_or_create(
        &mut self,
        context: &GPUContext,
        system_id: SystemID,
        shader_wgsl: &'static str,
        entry_point: &'static str,
        read_count: usize,
        write_count: usize,
        resource_layout: &[GPUBindingDesc],
    ) -> ECSResult<(
        &wgpu::ComputePipeline,
        &wgpu::BindGroupLayout,
        Option<&wgpu::BindGroupLayout>,
    )> {
        let shader_hash = hash_str(shader_wgsl);
        let entry_hash  = hash_str(entry_point);

        let group1_len = resource_layout.len();
        let group1_sig = hash_resource_layout(resource_layout);

        let key = (
            system_id,
            shader_hash,
            entry_hash,
            read_count,
            write_count,
            group1_len,
            group1_sig,
        );

        if !self.map.contains_key(&key) {
            let (pipeline, bgl0, bgl1) = create_pipeline(
                context,
                shader_wgsl,
                entry_point,
                read_count,
                write_count,
                resource_layout,
            )
            .map_err(|e| ECSError::from(ExecutionError::GpuDispatchFailed {
                message: e.into(),
            }))?;

            self.map.insert(key, (pipeline, bgl0, bgl1));
        }

        let (pipeline, bgl0, bgl1) = self.map.get(&key).unwrap();
        Ok((pipeline, bgl0, bgl1.as_ref()))
    }
}

/// Creates a compute pipeline and its bind group layout.
///
/// ## Binding layout
/// * Storage buffers: `0..binding_count-1`
/// * Uniform buffer: `binding_count-1`
///
/// ## Parameters
/// * `context` - GPU device context
/// * `shader_wgsl` - WGSL shader source
/// * `entry_point` - compute entry point
/// * `binding_count` - total number of bindings
///
/// ## Errors
/// Returns an error string if pipeline creation fails.

fn create_pipeline(
    context: &GPUContext,
    shader_wgsl: &'static str,
    entry_point: &'static str,
    read_count: usize,
    write_count: usize,
    resource_layout: &[GPUBindingDesc],
) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout, Option<wgpu::BindGroupLayout>), String> {
    let resource_count = resource_layout.len();

    // group(0): reads + writes + params
    let mut entries0 = Vec::with_capacity(read_count + write_count + 1);

    // read-only storage
    for i in 0..read_count {
        entries0.push(wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    // read-write storage
    for j in 0..write_count {
        let binding = (read_count + j) as u32;
        entries0.push(wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    // uniform
    let params_binding = (read_count + write_count) as u32;
    entries0.push(wgpu::BindGroupLayoutEntry {
        binding: params_binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });

    let bgl0 = context.device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("abm_bgl_group0"),
            entries: &entries0,
        }
    );

    // group(1): resources
    let bgl1 = if resource_count > 0 {
        let mut entries1 = Vec::with_capacity(resource_count);
        for (k, desc) in resource_layout.iter().enumerate() {
            entries1.push(wgpu::BindGroupLayoutEntry {
                binding: k as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: desc.read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        Some(context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("abm_bgl_group1"),
            entries: &entries1,
        }))
    } else {
        None
    };

    let mut layouts: Vec<&wgpu::BindGroupLayout> = vec![&bgl0];
    if let Some(ref b) = bgl1 {
        layouts.push(b);
    }

    let pl = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("abm_pipeline_layout"),
        bind_group_layouts: &layouts,
        push_constant_ranges: &[],
    });

    let module = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("abm_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_wgsl.into()),
    });

    let pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("abm_compute_pipeline"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    Ok((pipeline, bgl0, bgl1))
}
