//! GPU resource ownership, registration, and synchronization for the ECS world.
//!
//! This module provides the [`GPUResource`] trait and [`GPUResourceRegistry`], which together
//! manage the full lifecycle of persistent GPU-side state (buffers, bindable storage) owned
//! by the world rather than by individual components or systems.
//!
//! # Design
//!
//! Resources are registered once and assigned a stable [`GPUResourceID`]. The registry tracks
//! three independent dirty flags per resource:
//!
//! - **`created`** - GPU buffers have been allocated via [`GPUResource::create_gpu`].
//! - **`cpu_dirty`** - CPU-side data has changed and needs to be uploaded to the GPU.
//! - **`pending_download`** - GPU-side data has been written and needs to be read back to the CPU.
//!
//! Synchronization is **explicit**: callers must call [`GPUResourceRegistry::mark_cpu_dirty`] or
//! [`GPUResourceRegistry::mark_pending_download`] to schedule transfers, then drive them with
//! [`GPUResourceRegistry::upload_dirty`] and [`GPUResourceRegistry::download_pending`] at the
//! appropriate points in the frame.
//!
//! # Bind Group Integration
//!
//! Each resource declares its binding layout via [`GPUResource::bindings`] and writes
//! [`wgpu::BindGroupEntry`] values via [`GPUResource::encode_bind_group_entries`]. The registry
//! can flatten these across multiple resources with [`GPUResourceRegistry::append_bind_group_entries`],
//! making it straightforward to build bind groups that span several world resources.
//!
//! # Feature Flag
//!
//! The entire module is gated behind the `gpu` feature flag.

#![cfg(feature = "gpu")]

use std::any::Any;
use std::fmt;

use crate::engine::error::{ECSError, ECSResult, ExecutionError};
use crate::engine::types::GPUResourceID;

use crate::gpu::GPUContext;

/// Describes one bindable GPU buffer entry for a resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GPUBindingDesc {
    /// Whether the binding is read-only (storage read) or read-write (storage read_write).
    pub read_only: bool,
}

impl GPUBindingDesc {
    /// Compact key for pipeline-cache hashing.
    #[inline]
    pub fn key(self) -> u8 {
        if self.read_only {
            1
        } else {
            2
        }
    }
}

/// Trait implemented by world-owned GPU resources.
///
/// Semantics:
/// - `create_gpu` is called once per world (lazy-initialized) when GPU runtime exists.
/// - `upload` is called when CPU-side data changed (explicitly marked dirty).
/// - `download` is called when CPU needs GPU-written data (explicitly marked pending).
pub trait GPUResource: Send + Sync {
    /// Human-readable name for diagnostics.
    ///
    /// Returning a borrow from `&self` is permitted, so implementations that
    /// need dynamic names can store an owned `String` and return `&self.name`
    /// without leaking memory.
    fn name(&self) -> &str;

    /// Called once when the GPU runtime is initialized for this world.
    fn create_gpu(&mut self, ctx: &GPUContext) -> ECSResult<()>;

    /// Called when CPU-side data changed.
    fn upload(&mut self, ctx: &GPUContext) -> ECSResult<()>;

    /// Called when GPU-side data must be read back for CPU usage.
    fn download(&mut self, ctx: &GPUContext) -> ECSResult<()>;

    /// Whether the generic GPU synchronization step should automatically call
    /// [`GPUResource::download`] when this resource is marked pending.
    ///
    /// Most resources use the default. Some framework-owned resources, such as
    /// GPU message buffers, need boundary-specific synchronization so they can
    /// avoid unnecessary GPU-to-CPU transfers for GPU-only consumers.
    fn automatic_download(&self) -> bool {
        true
    }

    /// Returns the binding contract for this resource.
    ///
    /// The number of entries here must match the number of buffers bound by
    /// `encode_bind_group_entries`.
    fn bindings(&self) -> &[GPUBindingDesc];

    /// Appends bind group entries for this resource at `base_binding`.
    ///
    /// This avoids returning references with complex lifetimes from trait objects.
    fn encode_bind_group_entries<'a>(
        &'a self,
        base: u32,
        out: &mut Vec<wgpu::BindGroupEntry<'a>>,
    ) -> ECSResult<()>;

    /// Downcasts to `Any`.
    fn as_any(&self) -> &dyn Any;

    /// Downcasts to mutable `Any`.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

#[cfg(feature = "gpu")]
struct GPUResourceEntry {
    resource: Box<dyn GPUResource>,
    created: bool,
    cpu_dirty: bool,
    pending_download: bool,
}

/// World-owned registry of GPU resources.
///
/// ## Responsibilities
/// * Owns the lifetime of all GPU resources.
/// * Assigns stable `GPUResourceID`s.
/// * Tracks CPU to GPU dirtiness and GPU to CPU pending downloads.
/// * Enforces explicit synchronization.

#[cfg(feature = "gpu")]
#[derive(Default)]
pub struct GPUResourceRegistry {
    entries: Vec<GPUResourceEntry>,
    next_id: GPUResourceID,
}

#[cfg(feature = "gpu")]
impl GPUResourceRegistry {
    #[inline]
    /// Creates an empty GPU resource registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new world-owned GPU resource and returns its ID.
    pub fn register<R: GPUResource + 'static>(&mut self, r: R) -> ECSResult<GPUResourceID> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| {
            ECSError::from(ExecutionError::GpuDispatchFailed {
                message: "GPU resource id space exhausted".into(),
            })
        })?;

        self.entries.push(GPUResourceEntry {
            resource: Box::new(r),
            created: false,
            cpu_dirty: true,
            pending_download: false,
        });

        Ok(id)
    }

    /// Marks a resource as modified on the CPU.
    #[inline]
    pub fn mark_cpu_dirty(&mut self, id: GPUResourceID) -> ECSResult<()> {
        let e = self.entries.get_mut(id as usize).ok_or_else(|| {
            ECSError::from(ExecutionError::GpuDispatchFailed {
                message: format!("missing gpu resource id {id}").into(),
            })
        })?;
        e.cpu_dirty = true;
        Ok(())
    }

    /// Marks a resource as requiring GPU to CPU synchronization.
    #[inline]
    pub fn mark_pending_download(&mut self, id: GPUResourceID) -> ECSResult<()> {
        let e = self.entries.get_mut(id as usize).ok_or_else(|| {
            ECSError::from(ExecutionError::GpuDispatchFailed {
                message: format!("missing gpu resource id {id}").into(),
            })
        })?;
        e.pending_download = true;
        Ok(())
    }

    /// Ensures GPU buffers exist for every registered resource.
    pub fn ensure_created(&mut self, context: &GPUContext) -> ECSResult<()> {
        for e in &mut self.entries {
            if !e.created {
                e.resource.create_gpu(context)?;
                e.created = true;
            }
        }
        Ok(())
    }

    /// Uploads all dirty resources.
    pub fn upload_dirty(&mut self, context: &GPUContext) -> ECSResult<()> {
        for e in &mut self.entries {
            if e.cpu_dirty {
                e.resource.upload(context)?;
                e.cpu_dirty = false;
            }
        }
        Ok(())
    }

    /// Downloads all pending resources.
    pub fn download_pending(&mut self, context: &GPUContext) -> ECSResult<()> {
        for e in &mut self.entries {
            if e.pending_download {
                if e.resource.automatic_download() {
                    e.resource.download(context)?;
                }
                e.pending_download = false;
            }
        }
        Ok(())
    }

    /// Downloads only the specified resources that are pending.
    pub fn download_pending_filtered(
        &mut self,
        context: &GPUContext,
        ids: &[GPUResourceID],
    ) -> ECSResult<()> {
        for &id in ids {
            if let Some(e) = self.entries.get_mut(id as usize) {
                if e.pending_download {
                    if e.resource.automatic_download() {
                        e.resource.download(context)?;
                    }
                    e.pending_download = false;
                }
            }
        }
        Ok(())
    }

    /// Returns flattened binding descriptions for a set of resource IDs.
    pub fn flattened_binding_descs(&self, ids: &[GPUResourceID]) -> Vec<GPUBindingDesc> {
        let mut out = Vec::new();
        for &id in ids {
            if let Some(e) = self.entries.get(id as usize) {
                out.extend_from_slice(e.resource.bindings());
            }
        }
        out
    }

    /// Appends bind group entries for the given resources at `base_binding`.
    pub fn append_bind_group_entries<'a>(
        &'a self,
        ids: &[GPUResourceID],
        base_binding: u32,
        out: &mut Vec<wgpu::BindGroupEntry<'a>>,
    ) -> ECSResult<u32> {
        let mut cursor = base_binding;
        for &id in ids {
            let e = self.entries.get(id as usize).ok_or_else(|| {
                ECSError::from(ExecutionError::GpuDispatchFailed {
                    message: format!("missing gpu resource id {id}").into(),
                })
            })?;
            e.resource.encode_bind_group_entries(cursor, out)?;
            cursor += e.resource.bindings().len() as u32;
        }
        Ok(cursor)
    }

    /// Returns a mutable reference to a typed GPU resource by ID.
    pub fn get_mut_typed<R: 'static>(&mut self, id: GPUResourceID) -> Option<&mut R> {
        self.entries
            .get_mut(id as usize)
            .and_then(|e| e.resource.as_any_mut().downcast_mut::<R>())
    }

    /// Returns an immutable reference to a typed GPU resource by ID.
    pub fn get_typed<R: 'static>(&self, id: GPUResourceID) -> Option<&R> {
        self.entries
            .get(id as usize)
            .and_then(|e| e.resource.as_any().downcast_ref::<R>())
    }
}

#[cfg(feature = "gpu")]
impl fmt::Debug for GPUResourceRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GPUResourceRegistry")
            .field("entry_count", &self.entries.len())
            .field("next_id", &self.next_id)
            .finish()
    }
}
