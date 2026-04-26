//! # Component Registry
//!
//! This module provides a component registry that assigns stable `ComponentID`
//! values to Rust component types and exposes type-erased storage factories for
//! archetype column allocation.
//!
//! ## Purpose
//! The registry decouples component type information (`TypeId`, name, size, alignment)
//! from runtime storage, enabling archetypes to store heterogeneous component columns
//! behind `TypeErasedAttribute`.
//!
//! ## Design
//! - Components are registered once and assigned a compact `ComponentID` in `[0, COMPONENT_CAP]`.
//! - A per-component factory function is stored for constructing empty column storage.
//! - The registry can be `freeze()`d to prevent further registrations after world setup.
//!
//! ## Instance-owned registry
//!
//! `ComponentRegistry` is a standalone, instance-owned struct that holds the
//! component table, the factory table, and the next-ID counter together.  This
//! enables multiple independent ECS worlds in the same process.
//!
//! A global `static REGISTRY` and `static COMPONENT_FACTORIES` is
//! maintained behind convenience functions for simple, single-world use cases.
//! These delegate to a shared global `ComponentRegistry` instance.
//!
//! ## Invariants
//! - `ComponentID` values are unique and stable for the lifetime of the registry.
//! - A registered component must have a corresponding storage factory.
//! - When frozen, registration is disallowed.
//!
//! ## Concurrency
//! The global registry is protected by `RwLock` for concurrent reads and
//! serialized writes.  Instance-owned registries are wrapped in
//! `Arc<RwLock<ComponentRegistry>>` by `ECSData`.

mod bundle;
mod descriptor;
mod global;
mod registry;
mod signature;

// -- Signature & helpers --
pub(crate) use signature::or_signature_in_place;
pub use signature::{iter_bits_from_words, Signature};

// -- Component descriptor --
pub use descriptor::ComponentDesc;

// -- Bundle --
pub use bundle::{Bundle, DynamicBundle};

// -- Instance-owned registry --
pub use registry::ComponentRegistry;

// -- Global convenience API --
// These remain available for internal use and single-world convenience,
// but are no longer re-exported from the crate root.
#[allow(unused)]
pub(crate) use global::{component_id_of, freeze_components, register_component};

#[cfg(feature = "gpu")]
pub use global::{register_gpu_component, GPUPod};
