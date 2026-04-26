//! GPU uniform buffer backed by a subset of [`Environment`] parameters.
//!
//! ## Design
//!
//! [`EnvUniformBuffer`] packs a declared subset of environment parameters into a
//! single `wgpu` uniform buffer that can be bound to compute shaders. Parameters
//! included must implement [`bytemuck::Pod`] so they can be transmuted to raw
//! bytes without any intermediate serialization.
//!
//! ### Dirty detection
//!
//! The buffer maintains an internal `cpu_dirty` flag that is independent of the
//! environment's dirty channel set. The flag is set to `true` by
//! [`mark_cpu_dirty`](EnvUniformBuffer::mark_cpu_dirty), which is called by
//! [`EnvironmentBoundary::finalise`](super::boundary::EnvironmentBoundary) when
//! it detects that at least one channel owned by this buffer appears in the
//! environment's dirty set.
//!
//! On every [`GPUResource::upload`] call, the buffer checks `cpu_dirty`. If the
//! flag is set, the CPU buffer is repacked from current environment values and
//! written to the GPU buffer, then the flag is cleared.
//!
//! The decoupling between the environment dirty set and the `cpu_dirty` flag
//! ensures that [`EnvironmentBoundary::finalise`] can clear the environment's
//! dirty channels immediately (preventing double-marking) without racing with
//! the GPU upload path, which may execute later in the same tick.
//!
//! ### Channel ownership
//!
//! Each included key is assigned a [`ChannelID`] at
//! [`EnvUniformBufferBuilder::include`] time by querying
//! [`Environment::channel_of`]. [`EnvUniformBuffer::owns_channel`] performs a
//! linear scan over the included packers to answer ownership queries from
//! [`EnvironmentBoundary`](super::boundary::EnvironmentBoundary).
//!
//! ### Struct layout
//!
//! Fields are packed in **declaration order** (the order that keys were passed
//! to [`EnvUniformBufferBuilder::include`]). The WGSL `struct` on the shader
//! side must match this layout exactly, including any padding required by WGSL's
//! alignment rules. Callers are responsible for ensuring this correspondence.
//!
//! ### Alignment
//!
//! [`repack`](EnvUniformBuffer::repack) concatenates field bytes with **no
//! padding**. WGSL uniform structs may insert implicit padding between fields
//! depending on their alignment requirements. Callers must either declare fields
//! in an order that produces matching layout (e.g. largest-alignment fields
//! first), or use [`EnvUniformBufferBuilder::expect_byte_size`] to catch
//! mismatches at build time. For most scalar-only environments (`f32`, `u32`),
//! natural 4-byte alignment matches and no padding is needed.
//!
//! ### WGSL type support
//!
//! Not all [`EnvPod`] types have WGSL equivalents. In particular, `f64`, `u64`,
//! and `i64` are not universally supported in WGSL. Callers should verify that
//! the types they pack into the uniform buffer are supported by their target
//! GPU and shader profile.
//!
//! ## Feature flag
//!
//! This entire module is gated behind the `gpu` feature.

#![cfg(feature = "gpu")]

use std::any::Any;
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::engine::error::{ECSError, ECSResult, ExecutionError};
use crate::engine::types::ChannelID;
use crate::gpu::{GPUBindingDesc, GPUContext, GPUResource};

use super::store::Environment;

// ─────────────────────────────────────────────────────────────────────────────
// GPUPod marker
// ─────────────────────────────────────────────────────────────────────────────

/// Marker trait for environment parameter types that are safe to transmute into
/// GPU uniform bytes.
///
/// # Safety
///
/// Implementors must be `bytemuck::Pod` — no padding bytes, no interior
/// mutability, no pointers. Implementing this for a type that is not `Pod`
/// is undefined behaviour.
///
/// The crate provides blanket implementations for `f32`, `f64`, `u32`, `i32`,
/// `u64`, `i64`, and arrays thereof.
///
/// **Note:** Not all of these types have WGSL equivalents. `f64`, `u64`, and
/// `i64` in particular are not universally supported in WGSL. Verify that the
/// types you include in [`EnvUniformBuffer`] are supported by your target GPU.
pub unsafe trait EnvPod: Any + Clone + Send + Sync + bytemuck::Pod {}

// Blanket implementations for common scalar types.
unsafe impl EnvPod for f32 {}
unsafe impl EnvPod for f64 {}
unsafe impl EnvPod for u32 {}
unsafe impl EnvPod for i32 {}
unsafe impl EnvPod for u64 {}
unsafe impl EnvPod for i64 {}
unsafe impl EnvPod for u16 {}
unsafe impl EnvPod for i16 {}
unsafe impl EnvPod for u8 {}
unsafe impl EnvPod for i8 {}

// ─────────────────────────────────────────────────────────────────────────────
// Erased packer — per-key closure that knows how to pack bytes from Environment
// ─────────────────────────────────────────────────────────────────────────────

/// A type-erased closure that reads one environment key and appends its raw
/// bytes to a `Vec<u8>`.
struct Packer {
    key: String,
    channel_id: ChannelID,
    byte_size: usize,
    pack: Box<dyn Fn(&Environment, &mut Vec<u8>) -> Result<(), String> + Send + Sync>,
}

impl Packer {
    fn new<T: EnvPod>(key: impl Into<String>, channel_id: ChannelID) -> Self {
        let key = key.into();
        let key2 = key.clone();
        Self {
            byte_size: std::mem::size_of::<T>(),
            channel_id,
            pack: Box::new(move |env, buf| {
                let v: T = env.get::<T>(&key2).map_err(|e| e.to_string())?;
                let bytes: &[u8] = bytemuck::bytes_of(&v);
                buf.extend_from_slice(bytes);
                Ok(())
            }),
            key,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EnvUniformBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// Packs a declared subset of environment parameters into a `wgpu` uniform
/// buffer.
///
/// Parameters included must implement [`EnvPod`] (i.e., `bytemuck::Pod`).
/// The buffer is marked dirty by
/// [`mark_cpu_dirty`](Self::mark_cpu_dirty), which is driven by
/// [`EnvironmentBoundary`](super::boundary::EnvironmentBoundary) whenever it
/// detects that one of the channels owned by this buffer has been written in
/// the current tick.
///
/// # Usage
///
/// ```ignore
/// let buf = EnvUniformBuffer::builder(Arc::clone(&env))
///     .include::<f32>("interest_rate")
///     .include::<u32>("world_width")
///     .expect_byte_size(8)  // optional: catch layout mismatches early
///     .build();
/// ```
pub struct EnvUniformBuffer {
    env: Arc<Environment>,
    /// Ordered packers — one per tracked key, in declaration order.
    packers: Vec<Packer>,
    /// Packed CPU-side buffer (matches WGSL uniform struct layout).
    cpu_buf: Vec<u8>,
    /// GPU buffer (created lazily by `create_gpu`).
    gpu_buf: Option<wgpu::Buffer>,
    /// Set to `true` by [`mark_cpu_dirty`](Self::mark_cpu_dirty); cleared
    /// to `false` after a successful [`GPUResource::upload`] or
    /// [`GPUResource::create_gpu`].
    cpu_dirty: bool,
}

impl EnvUniformBuffer {
    /// Returns a builder for constructing an [`EnvUniformBuffer`].
    pub fn builder(env: Arc<Environment>) -> EnvUniformBufferBuilder {
        EnvUniformBufferBuilder {
            env,
            packers: Vec::new(),
        }
    }

    /// Keys tracked by this buffer, in declaration order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.packers.iter().map(|p| p.key.as_str())
    }

    /// Returns the total byte size of the CPU buffer.
    pub fn byte_size(&self) -> usize {
        self.packers.iter().map(|p| p.byte_size).sum()
    }

    /// Marks the CPU buffer as dirty, indicating that a re-pack and GPU upload
    /// are required on the next [`GPUResource::upload`] call.
    ///
    /// Called by
    /// [`EnvironmentBoundary::finalise`](super::boundary::EnvironmentBoundary)
    /// when it detects that at least one channel owned by this buffer appears
    /// in the environment's dirty channel set. This decouples the environment's
    /// dirty-tracking lifecycle (cleared per-tick by the boundary) from the
    /// GPU upload lifecycle (cleared only after a successful upload).
    pub fn mark_cpu_dirty(&mut self) {
        self.cpu_dirty = true;
    }

    /// Returns `true` if this buffer owns the given [`ChannelID`].
    ///
    /// Used by
    /// [`EnvironmentBoundary::finalise`](super::boundary::EnvironmentBoundary)
    /// to decide which uniform buffers need to be marked dirty after detecting
    /// channel writes in the environment's dirty set.
    ///
    /// Performs a linear scan over the included packers; this is acceptable
    /// since the number of keys per uniform buffer is typically small.
    pub fn owns_channel(&self, id: ChannelID) -> bool {
        self.packers.iter().any(|p| p.channel_id == id)
    }

    /// Repacks `cpu_buf` from current environment values.
    ///
    /// Fields are concatenated in declaration order with no padding. The
    /// WGSL `struct` on the shader side must match this layout exactly.
    fn repack(&mut self) -> Result<(), ECSError> {
        self.cpu_buf.clear();
        for p in &self.packers {
            (p.pack)(&self.env, &mut self.cpu_buf).map_err(|e| {
                ECSError::from(ExecutionError::GpuDispatchFailed {
                    message: format!("EnvUniformBuffer pack error: {e}").into(),
                })
            })?;
        }
        Ok(())
    }
}

impl GPUResource for EnvUniformBuffer {
    fn name(&self) -> &str {
        "EnvUniformBuffer"
    }

    /// Allocates the GPU uniform buffer and performs an initial upload.
    ///
    /// After the initial upload the environment's dirty channels for all
    /// included keys are cleared and `cpu_dirty` is reset, so that no
    /// redundant upload occurs on the first call to
    /// [`GPUResource::upload`] in the first tick.
    ///
    /// # Errors
    ///
    /// Returns an error if any tracked key cannot be read from the environment
    /// (e.g. type mismatch — should not happen if the buffer was built correctly).
    fn create_gpu(&mut self, ctx: &GPUContext) -> ECSResult<()> {
        self.repack()?;
        let buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("EnvUniformBuffer"),
                contents: &self.cpu_buf,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        self.gpu_buf = Some(buf);
        // Clear dirty tracking for all channels owned by this buffer so that
        // any marks created before GPU initialisation don't trigger a
        // redundant upload on the first tick.
        let owned: Vec<ChannelID> = self.packers.iter().map(|p| p.channel_id).collect();
        self.env.clear_dirty_for_channels(&owned)?;
        self.cpu_dirty = false;
        Ok(())
    }

    /// Uploads the CPU buffer to the GPU if the `cpu_dirty` flag is set.
    ///
    /// The flag is cleared after a successful upload. If the flag is not set
    /// this call is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The GPU buffer has not been created yet (call [`create_gpu`] first).
    /// - A tracked key cannot be read from the environment.
    fn upload(&mut self, ctx: &GPUContext) -> ECSResult<()> {
        if !self.cpu_dirty {
            return Ok(());
        }
        self.repack()?;
        let buf = self.gpu_buf.as_ref().ok_or_else(|| {
            ECSError::from(ExecutionError::GpuDispatchFailed {
                message: "EnvUniformBuffer::upload called before create_gpu".into(),
            })
        })?;
        ctx.queue.write_buffer(buf, 0, &self.cpu_buf);
        self.cpu_dirty = false;
        Ok(())
    }

    /// Environment uniform buffers are GPU-write-only from the ECS perspective;
    /// download is a no-op.
    fn download(&mut self, _ctx: &GPUContext) -> ECSResult<()> {
        Ok(())
    }

    fn bindings(&self) -> &[GPUBindingDesc] {
        // One read-only uniform binding.
        static B: [GPUBindingDesc; 1] = [GPUBindingDesc { read_only: true }];
        &B
    }

    /// # Errors
    ///
    /// Returns an error if the GPU buffer has not been created yet.
    fn encode_bind_group_entries<'a>(
        &'a self,
        base: u32,
        out: &mut Vec<wgpu::BindGroupEntry<'a>>,
    ) -> ECSResult<()> {
        let buf = self.gpu_buf.as_ref().ok_or_else(|| {
            ECSError::from(ExecutionError::GpuDispatchFailed {
                message: "EnvUniformBuffer not yet created on GPU".into(),
            })
        })?;
        out.push(wgpu::BindGroupEntry {
            binding: base,
            resource: buf.as_entire_binding(),
        });
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl EnvUniformBuffer {
    /// Returns `true` if [`mark_cpu_dirty`](Self::mark_cpu_dirty) has been
    /// called since the last upload.
    ///
    /// The [`GPUResourceRegistry`](crate::gpu::GPUResourceRegistry) tracks its
    /// own per-resource dirty flag. The owning layer must query this method and
    /// call [`GPUResourceRegistry::mark_cpu_dirty`] to bridge the two when
    /// integrating the uniform buffer into a registry-based upload pipeline:
    ///
    /// ```ignore
    /// if env_uniform.is_cpu_dirty() {
    ///     gpu_registry.mark_cpu_dirty(env_uniform_resource_id);
    /// }
    /// ```
    pub fn is_cpu_dirty(&self) -> bool {
        self.cpu_dirty
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for [`EnvUniformBuffer`].
///
/// Keys are packed into the uniform buffer in the order they are
/// [`include`](Self::include)d. The WGSL `struct` on the shader side must
/// match this order and layout exactly.
pub struct EnvUniformBufferBuilder {
    env: Arc<Environment>,
    packers: Vec<Packer>,
}

impl EnvUniformBufferBuilder {
    /// Includes a typed key in the uniform buffer.
    ///
    /// Keys are packed in the order they are included here; the WGSL `struct`
    /// must match. The key's [`ChannelID`] is resolved from the environment at
    /// this point and stored in the packer so that
    /// [`EnvUniformBuffer::owns_channel`] can answer ownership queries without
    /// a string comparison.
    ///
    /// # Panics
    ///
    /// Panics if `key` is not registered in the environment. This is a build-
    /// time misconfiguration and is always checked regardless of the `debug`
    /// profile.
    pub fn include<T: EnvPod>(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        let channel_id = self.env.channel_of(&key).unwrap_or_else(|| {
            panic!("EnvUniformBuffer: key '{key}' is not registered in the environment")
        });
        self.packers.push(Packer::new::<T>(key, channel_id));
        self
    }

    /// Asserts that the total packed byte size matches the expected WGSL
    /// struct size.
    ///
    /// Call after all [`include`](Self::include) calls to catch alignment or
    /// padding mismatches between the CPU-side packed layout and the GPU-side
    /// WGSL struct at build time, rather than producing silent data corruption
    /// at runtime.
    ///
    /// # Panics
    ///
    /// Panics if the computed byte size does not match `expected`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let buf = EnvUniformBuffer::builder(env)
    ///     .include::<f32>("rate")   // 4 bytes
    ///     .include::<u32>("width")  // 4 bytes
    ///     .expect_byte_size(8)      // catches mismatches early
    ///     .build();
    /// ```
    pub fn expect_byte_size(self, expected: usize) -> Self {
        let actual: usize = self.packers.iter().map(|p| p.byte_size).sum();
        assert_eq!(
            actual, expected,
            "EnvUniformBuffer: packed {actual} bytes but expected {expected}. \
             Check WGSL struct alignment and field order."
        );
        self
    }

    /// Finalises and returns an [`EnvUniformBuffer`].
    ///
    /// The GPU buffer is **not** created here; call
    /// [`GPUResource::create_gpu`] when the GPU context is available.
    pub fn build(self) -> EnvUniformBuffer {
        let byte_size: usize = self.packers.iter().map(|p| p.byte_size).sum();
        EnvUniformBuffer {
            env: self.env,
            packers: self.packers,
            cpu_buf: Vec::with_capacity(byte_size),
            gpu_buf: None,
            cpu_dirty: false,
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::environment::builder::EnvironmentBuilder;

    fn make_env() -> Arc<Environment> {
        EnvironmentBuilder::new()
            .register::<f32>("rate", 0.05f32)
            .unwrap()
            .register::<u32>("size", 100u32)
            .unwrap()
            .build()
            .unwrap()
    }

    #[test]
    fn builder_tracks_correct_keys() {
        let env = make_env();
        let buf = EnvUniformBuffer::builder(Arc::clone(&env))
            .include::<f32>("rate")
            .include::<u32>("size")
            .build();
        let keys: Vec<&str> = buf.keys().collect();
        assert_eq!(keys, ["rate", "size"]);
    }

    #[test]
    fn byte_size_matches_expected() {
        let env = make_env();
        let buf = EnvUniformBuffer::builder(Arc::clone(&env))
            .include::<f32>("rate") // 4 bytes
            .include::<u32>("size") // 4 bytes
            .build();
        assert_eq!(buf.byte_size(), 8);
    }

    #[test]
    fn expect_byte_size_passes_on_match() {
        let env = make_env();
        let buf = EnvUniformBuffer::builder(Arc::clone(&env))
            .include::<f32>("rate")
            .include::<u32>("size")
            .expect_byte_size(8)
            .build();
        assert_eq!(buf.byte_size(), 8);
    }

    #[test]
    #[should_panic(expected = "packed 8 bytes but expected 16")]
    fn expect_byte_size_panics_on_mismatch() {
        let env = make_env();
        EnvUniformBuffer::builder(Arc::clone(&env))
            .include::<f32>("rate")
            .include::<u32>("size")
            .expect_byte_size(16);
    }

    #[test]
    fn is_cpu_dirty_starts_false() {
        let env = make_env();
        let buf = EnvUniformBuffer::builder(Arc::clone(&env))
            .include::<f32>("rate")
            .build();
        assert!(!buf.is_cpu_dirty());
    }

    #[test]
    fn mark_cpu_dirty_sets_flag() {
        let env = make_env();
        let mut buf = EnvUniformBuffer::builder(Arc::clone(&env))
            .include::<f32>("rate")
            .build();
        assert!(!buf.is_cpu_dirty());
        buf.mark_cpu_dirty();
        assert!(buf.is_cpu_dirty());
    }

    #[test]
    fn owns_channel_tracks_included_keys() {
        let env = make_env();
        let id_rate = env.channel_of("rate").unwrap();
        let id_size = env.channel_of("size").unwrap();

        let buf = EnvUniformBuffer::builder(Arc::clone(&env))
            .include::<f32>("rate")
            .build();

        assert!(buf.owns_channel(id_rate));
        assert!(!buf.owns_channel(id_size));
        // An arbitrary ID that was never registered.
        assert!(!buf.owns_channel(999));
    }
}
