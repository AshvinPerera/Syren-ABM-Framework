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
//! The buffer checks [`Environment::has_any_dirty`] on every
//! [`GPUResource::upload`] call via [`EnvUniformBuffer::is_cpu_dirty`]. If any
//! of the tracked keys have been mutated since the last upload, the CPU buffer
//! is repacked from the current environment values and written to the GPU buffer.
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
//! ## Integration with `GPUResourceRegistry`
//!
//! The [`GPUResourceRegistry`](crate::gpu::GPUResourceRegistry) tracks its own
//! `cpu_dirty` flag per resource and only calls [`GPUResource::upload`] when
//! that flag is set. [`Environment::set`] marks keys in the environment's
//! internal dirty set but does **not** notify the registry. The owning layer
//! (typically `Model`) must bridge these two systems:
//!
//! ```ignore
//! // Before each GPU boundary stage:
//! if env_uniform.is_cpu_dirty() {
//!     gpu_registry.mark_cpu_dirty(env_uniform_resource_id);
//! }
//! ```
//!
//! Failing to do this will cause GPU shaders to read stale uniform values.
//!
//! ## Feature flag
//!
//! This entire module is gated behind the `gpu` feature.

#![cfg(feature = "gpu")]

use std::any::Any;
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::engine::error::{ECSResult, ECSError, ExecutionError};
use crate::gpu::{GPUResource, GPUBindingDesc, GPUContext};

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
    byte_size: usize,
    pack: Box<dyn Fn(&Environment, &mut Vec<u8>) -> Result<(), String> + Send + Sync>,
}

impl Packer {
    fn new<T: EnvPod>(key: impl Into<String>) -> Self {
        let key = key.into();
        let key2 = key.clone();
        Self {
            byte_size: std::mem::size_of::<T>(),
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
/// The buffer reports itself as dirty when any included key changes via
/// [`Environment::set`], allowing the owning layer to trigger a re-upload.
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

    /// Returns `true` if any tracked key appears in the environment's dirty
    /// set. Uses [`Environment::has_any_dirty`] for a zero-allocation check.
    fn any_tracked_dirty(&self) -> bool {
        self.env.has_any_dirty(self.packers.iter().map(|p| p.key.as_str()))
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
    fn name(&self) -> &'static str {
        "EnvUniformBuffer"
    }

    /// Allocates the GPU uniform buffer and performs an initial upload.
    ///
    /// # Errors
    ///
    /// Returns an error if any tracked key cannot be read from the environment
    /// (e.g. type mismatch — should not happen if the buffer was built correctly).
    fn create_gpu(&mut self, ctx: &GPUContext) -> ECSResult<()> {
        self.repack()?;
        let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("EnvUniformBuffer"),
            contents: &self.cpu_buf,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        self.gpu_buf = Some(buf);
        self.env.clear_dirty_keys(self.packers.iter().map(|p| p.key.as_str()));
        Ok(())
    }

    /// Uploads the CPU buffer to the GPU if any tracked key is dirty.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The GPU buffer has not been created yet (call [`create_gpu`] first).
    /// - A tracked key cannot be read from the environment.
    fn upload(&mut self, ctx: &GPUContext) -> ECSResult<()> {
        if !self.is_cpu_dirty() {
            return Ok(());
        }
        self.repack()?;
        let buf = self.gpu_buf.as_ref().ok_or_else(|| {
            ECSError::from(ExecutionError::GpuDispatchFailed {
                message: "EnvUniformBuffer::upload called before create_gpu".into(),
            })
        })?;
        ctx.queue.write_buffer(buf, 0, &self.cpu_buf);
        self.env.clear_dirty_keys(self.packers.iter().map(|p| p.key.as_str()));
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
    /// Returns `true` if any tracked environment key has changed since the
    /// last upload.
    ///
    /// The [`GPUResourceRegistry`](crate::gpu::GPUResourceRegistry) tracks its
    /// own per-resource dirty flag from the environment's internal
    /// dirty set. The owning layer must query this method and call
    /// [`GPUResourceRegistry::mark_cpu_dirty`] to bridge the two:
    ///
    /// ```ignore
    /// if env_uniform.is_cpu_dirty() {
    ///     gpu_registry.mark_cpu_dirty(env_uniform_resource_id);
    /// }
    /// ```
    pub fn is_cpu_dirty(&self) -> bool {
        self.any_tracked_dirty()
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
    /// must match.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `key` is not registered in the environment.
    pub fn include<T: EnvPod>(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        debug_assert!(
            self.env.contains_key(&key),
            "EnvUniformBuffer: key '{key}' is not registered in the environment"
        );
        self.packers.push(Packer::new::<T>(key));
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
            .register::<u32>("size", 100u32)
            .build()
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
            .include::<f32>("rate")  // 4 bytes
            .include::<u32>("size")  // 4 bytes
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
    fn dirty_detection_after_set() {
        let env = make_env();
        let buf = EnvUniformBuffer::builder(Arc::clone(&env))
            .include::<f32>("rate")
            .build();
        // env is clean initially.
        assert!(!buf.is_cpu_dirty());
        env.set::<f32>("rate", 0.10).unwrap();
        assert!(buf.is_cpu_dirty());
    }

    #[test]
    fn untracked_key_does_not_trigger_dirty() {
        let env = make_env();
        let buf = EnvUniformBuffer::builder(Arc::clone(&env))
            .include::<f32>("rate")
            .build();
        // Mutate a key NOT tracked by this buffer.
        env.set::<u32>("size", 200).unwrap();
        assert!(!buf.is_cpu_dirty());
    }
}
