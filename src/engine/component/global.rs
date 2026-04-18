//! Global convenience API for the component registry.
//!
//! Provides a set of free functions that delegate to a single, process-wide
//! [`ComponentRegistry`] instance, making it straightforward to register
//! components, query IDs, and allocate storage without managing registry
//! lifetimes manually.
//!
//! # Intended use
//! This module is designed for **single-world applications** where one shared
//! component namespace is sufficient.
//!
//! **Multi-world applications** that need isolated registries should construct
//! and own a [`ComponentRegistry`] directly, and pass it by reference to
//! archetype-creation paths (e.g. `Archetype::new`) rather than relying on
//! these global convenience functions.
//!
//! # Lifecycle
//! 1. **Registration** — call [`register_component`] (or [`register_gpu_component`]
//!    with the `gpu` feature) for every component type before simulation begins.
//! 2. **Freeze** — call [`freeze_components`] once all components are registered.
//!    After this point the registry is immutable: component IDs and storage
//!    factories are stable and safe to use for archetype construction.
//! 3. **Query / allocate** — use [`component_id_of`], [`component_description_by_component_id`],
//!    and [`make_empty_component`] freely throughout the rest of the program.
//!
//! # GPU support
//! When the `gpu` feature is enabled, [`register_gpu_component`] marks a
//! component as GPU-safe via the [`GPUPod`] contract, allowing it to be
//! mirrored into GPU storage buffers.
//!
//! # Thread safety
//! The global registry is protected by an [`RwLock`]. Concurrent reads are
//! supported; writes (registration, freezing) require exclusive access. All
//! functions return [`RegistryError::PoisonedLock`] if the lock has been
//! poisoned by a panicking writer.

use std::any::TypeId;
use std::mem::size_of;
use std::sync::{OnceLock, RwLock};

use crate::engine::types::{ComponentID};
use crate::engine::error::{ECSResult, RegistryError};

use super::registry::{ComponentRegistry};

// ---------------------------------------------------------------------------
// Global convenience API (delegates to a shared global registry)
// ---------------------------------------------------------------------------

/// Global registry backing the convenience free functions.
static GLOBAL_REGISTRY: OnceLock<RwLock<ComponentRegistry>> = OnceLock::new();

/// Returns the global component registry.
///
/// For single-world use cases where an instance-owned registry is not needed.
/// Multi-world applications should construct and hold their own
/// [`ComponentRegistry`] instances and pass them explicitly to archetype
/// creation paths instead of using this global.
fn global_registry() -> &'static RwLock<ComponentRegistry> {
    GLOBAL_REGISTRY.get_or_init(|| RwLock::new(ComponentRegistry::new()))
}

/// Registers component type `T` in the global registry and returns its `ComponentID`.
///
/// ## Purpose
/// Convenience wrapper around the global `ComponentRegistry`.
///
/// For multi-world applications, prefer calling [`ComponentRegistry::register`]
/// directly on the registry instance that will be passed to `Archetype::new`.
///
/// ## Errors
/// Returns an error if:
/// - the registry is frozen,
/// - `COMPONENT_CAP` is exceeded,
/// - the component is zero-sized,
/// - the registry lock is poisoned.

pub fn register_component<T: 'static + Send + Sync>() -> ECSResult<ComponentID> {
    let registry = global_registry();
    let mut registry = registry
        .write()
        .map_err(|_| RegistryError::PoisonedLock)?;

    if size_of::<T>() == 0 {
        return Err(RegistryError::ZeroSizedComponent { type_id: TypeId::of::<T>() }.into());
    }

    Ok(registry.register::<T>()?)
}

/// Marker trait for component types that are safe to transfer to and from the GPU.
///
/// ## Purpose
/// `GPUPod` marks a component as **plain-old-data (POD)** suitable for:
/// * direct byte-wise copying into GPU buffers,
/// * use inside GPU storage or uniform buffers,
/// * round-tripping between CPU and GPU without transformation.
///
/// ## Safety
/// This trait is **unsafe** because incorrect implementations can cause
/// undefined behaviour on the GPU or silent data corruption.
///
/// Implementors **must guarantee**:
/// * The type has **no padding with uninitialized bytes**.
/// * The memory layout is stable and identical on CPU and GPU.
/// * The type contains **no pointers, references, or heap allocations**.
/// * The type is trivially copyable (`Copy`) and has no drop glue.
/// * The alignment is compatible with GPU storage buffer rules.
///
/// ## Example
/// ```
/// # #[cfg(feature = "gpu")]
/// # {
/// use abm_framework::GPUPod;
///
/// #[repr(C)]
/// #[derive(Copy, Clone)]
/// struct Position {
///     x: f32,
///     y: f32,
/// }
///
/// unsafe impl GPUPod for Position {}
/// # }
/// ```

#[cfg(feature = "gpu")]
pub unsafe trait GPUPod: Copy + Send + Sync + 'static {}

/// Registers a component type as GPU-safe and eligible for GPU execution.
///
/// ## Purpose
/// This function is a GPU-aware extension of [`register_component`] that
/// explicitly marks the component as safe to:
/// * mirror into GPU buffers,
/// * be bound as a storage buffer in compute shaders,
///
/// Internally, this delegates to [`ComponentRegistry::register_gpu`], which
/// sets the `gpu_usage` flag on the component's [`ComponentDesc`].
///
/// ## Requirements
/// * The component type must implement [`GPUPod`].
/// * The component must already satisfy all normal ECS component constraints
///   (non-zero-sized, `'static`, `Send`, `Sync`).
///
/// ## Safety model
/// This function is safe to call, but relies on the **unsafe contract**
/// of [`GPUPod`] being upheld by the caller.
///
/// ## Freezing behaviour
/// This function must be called **before** [`freeze_components`].
/// Calling it after the registry is frozen will return an error.
///
/// ## Returns
/// The assigned [`ComponentID`] for the registered component.
///
/// ## Errors
/// Returns an error if:
/// * the component registry is frozen,
/// * the registry lock is poisoned,
/// * the component violates ECS registration constraints.

#[cfg(feature = "gpu")]
pub fn register_gpu_component<T: GPUPod + 'static + Send + Sync>() -> ECSResult<ComponentID> {
    let registry = global_registry();
    let mut registry = registry
        .write()
        .map_err(|_| RegistryError::PoisonedLock)?;

    Ok(registry.register_gpu::<T>()?)
}

/// Freezes the global component registry.
///
/// ## Purpose
/// Prevents any further component registration, making component IDs and storage
/// factories stable for archetype construction.
///
/// ## Errors
/// Returns `RegistryError::PoisonedLock` if the registry lock is poisoned.

pub fn freeze_components() -> ECSResult<()> {
    let registry = global_registry();
    let mut registry = registry
        .write()
        .map_err(|_| RegistryError::PoisonedLock)?;
    registry.freeze();
    Ok(())
}


/// Returns the registered `ComponentID` for type `T`.
///
/// ## Errors
/// Returns `RegistryError::NotRegistered` if `T` is not registered.
/// Returns `RegistryError::PoisonedLock` if the registry lock is poisoned.

pub fn component_id_of<T: 'static>() -> ECSResult<ComponentID> {
    let registry = global_registry();
    let registry = registry
        .read()
        .map_err(|_| RegistryError::PoisonedLock)?;
    Ok(registry.require_id_of::<T>()?)
}
