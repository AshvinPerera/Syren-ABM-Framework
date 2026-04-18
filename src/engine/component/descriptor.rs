//! Metadata descriptor for ECS component types.
//!
//! This module defines [`ComponentDesc`], a lightweight, copyable struct that captures
//! static type information about a registered component — including its runtime [`TypeId`],
//! Rust type name, memory layout (size and alignment), assigned [`ComponentID`], and
//! whether it is marked as GPU-safe.
//!
//! ## Usage
//!
//! Descriptors are typically constructed via [`ComponentDesc::of::<T>()`] and finalized
//! with a registry-assigned ID using [`ComponentDesc::with_id`]:
//!
//! ```
//! use abm_framework::ComponentDesc;
//!
//! struct Health(f32);
//!
//! let desc = ComponentDesc::of::<Health>()
//!     .use_gpu(false)
//!     .with_id(42);
//!
//! assert_eq!(desc.component_id, Some(42));
//! ```
//!
//! ## Design Notes
//!
//! - [`ComponentDesc`] is `Copy`, making it cheap to pass around for diagnostics,
//!   validation, and tooling without lifetime concerns.
//! - The `component_id` field is `None` until the descriptor is registered via
//!   [`ComponentDesc::with_id`]. Consumers must handle the `None` case explicitly
//!   rather than relying on a sentinel value.
//! - GPU safety is opt-in and carries no automatic enforcement — it is a hint for
//!   systems that need to distinguish GPU-uploadable components.

use std::any::{TypeId, type_name};
use std::mem::{size_of, align_of};

use crate::engine::types::ComponentID;

/// Describes a registered component type.
///
/// ## Purpose
/// Provides metadata about a component type for debugging, validation, and tooling.
///
/// ## Fields
/// - `component_id`: The runtime identifier assigned by the registry, or `None` if
///   the descriptor has not yet been registered.
/// - `name`: The Rust type name (`type_name::<T>()`).
/// - `type_id`: The runtime `TypeId` for the component.
/// - `size`: `size_of::<T>()` in bytes.
/// - `align`: `align_of::<T>()` in bytes.
///
/// ## Notes
/// `ComponentDesc` is `Copy` and safe to clone freely for reporting and diagnostics.

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ComponentDesc {
    /// Runtime identifier assigned to this component type by the registry.
    /// `None` if the descriptor has not yet been registered.
    pub component_id: Option<ComponentID>,

    /// Rust type name for diagnostics.
    pub name: &'static str,

    /// Runtime `TypeId` of the component.
    pub type_id: TypeId,

    /// Size of the component type in bytes.
    pub size: usize,

    /// Alignment of the component type in bytes.
    pub align: usize,

    /// True if this component is explicitly marked as GPU-safe.
    pub gpu_usage: bool,
}

impl ComponentDesc {

    /// Creates a descriptor from explicit metadata.
    #[inline]
    pub fn new(
        component_id: Option<ComponentID>,
        name: &'static str,
        type_id: TypeId,
        size: usize,
        align: usize,
        gpu_usage: bool
    ) -> Self {
        Self { component_id, name, type_id, size, align, gpu_usage }
    }

    /// Constructs a descriptor for type `T` using its `TypeId`, name, size, and alignment.
    ///
    /// ## Notes
    /// The returned descriptor has `component_id: None` and must be finalized via
    /// [`ComponentDesc::with_id`] before use in a registry context.

    #[inline]
    pub fn of<T: 'static>() -> Self {
        Self {
            component_id: None,
            name: type_name::<T>(),
            type_id: TypeId::of::<T>(),
            size: size_of::<T>(),
            align: align_of::<T>(),
            gpu_usage: false,
        }
    }

    /// Marks this component descriptor as GPU-safe.
    #[inline]
    pub fn use_gpu(mut self, gpu_usage: bool) -> Self {
        self.gpu_usage = gpu_usage;
        self
    }

    /// Returns `true` if this descriptor refers to type `T`.
    #[inline]
    pub fn matches_type<T: 'static>(&self) -> bool {
        self.type_id == TypeId::of::<T>()
    }

    /// Returns a copy of this descriptor with `component_id` set to `Some(component_id)`.
    #[inline]
    pub fn with_id(mut self, component_id: ComponentID) -> Self {
        self.component_id = Some(component_id);
        self
    }
}

impl std::fmt::Display for ComponentDesc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id = match self.component_id {
            Some(id) => id.to_string(),
            None => "unassigned".to_string(),
        };
        write!(
            f,
            "ComponentDesc {{ id: {}, name: {}, size: {}, align: {}, uses gpu: {} }}",
            id, self.name, self.size, self.align, self.gpu_usage
        )
    }
}
