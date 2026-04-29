//! Instance-owned registry mapping Rust component types to compact runtime identifiers.
//!
//! # Overview
//!
//! [`ComponentRegistry`] is the authoritative source of truth for component identity
//! within an ECS world. It assigns each registered Rust type a stable [`ComponentID`]
//! and stores the metadata and storage factories needed to allocate archetype columns
//! at runtime.
//!
//! # Responsibilities
//!
//! - **Identity assignment** - maps [`TypeId`] -> [`ComponentID`] via a monotonically
//!   increasing counter bounded by [`COMPONENT_CAP`].
//! - **Metadata storage** - holds a [`ComponentDesc`] per registered component for use
//!   in diagnostics and validation.
//! - **Factory registration** - stores a [`FactoryFn`] per component so that
//!   [`Attribute`] columns can be constructed without knowing the concrete type at the
//!   call site.
//! - **Lifecycle enforcement** - can be [`freeze`](ComponentRegistry::freeze)'d once the
//!   world is initialized, after which further registration is rejected, guaranteeing
//!   that component IDs and storage layouts remain stable.
//!
//! # Instance ownership
//!
//! All state lives inside the [`ComponentRegistry`] value itself rather than in
//! process-wide statics. This means multiple independent ECS worlds can coexist in the
//! same process, each with its own isolated component namespace.
//!
//! # GPU support
//!
//! When the `gpu` feature is enabled, [`ComponentRegistry::register_gpu`] registers a
//! component and marks it as GPU-safe in a single step.  Alternatively,
//! [`ComponentRegistry::mark_gpu_safe`] can retroactively flag an already-registered
//! component.  Both methods require the type to implement [`GPUPod`].
//!
//! # Invariants
//!
//! - Every entry in `by_type` has a matching `by_id[id]` and `factories[id]`.
//! - All assigned IDs satisfy `id < COMPONENT_CAP`.
//! - No new registrations are accepted after [`ComponentRegistry::freeze`] is called.

use std::any::TypeId;
use std::collections::HashMap;
use std::fmt;
use std::mem::size_of;

use crate::engine::error::{RegistryError, RegistryResult};
use crate::engine::storage::{Attribute, TypeErasedAttribute};
use crate::engine::types::{ComponentID, COMPONENT_CAP};

use super::descriptor::ComponentDesc;

#[cfg(feature = "gpu")]
use super::global::GPUPod;

/// Factory function for constructing an empty type-erased component attribute column.
pub type FactoryFn = fn() -> Box<dyn TypeErasedAttribute>;

/// Constructs an empty attribute storage column for component type `T`.
///
/// ## Purpose
/// Used as the registered factory for a component ID.
fn new_attribute_storage<T: 'static + Send + Sync>() -> Box<dyn TypeErasedAttribute> {
    Box::new(Attribute::<T>::default())
}

// ---------------------------------------------------------------------------
// Instance-owned ComponentRegistry
// ---------------------------------------------------------------------------

/// Mapping between Rust component types and compact `ComponentID` values.
///
/// ## Purpose
/// Assigns stable runtime identifiers (`ComponentID`) to component types (`TypeId`)
/// and stores component metadata (`ComponentDesc`) used for diagnostics and validation.
///
/// ## Instance ownership
///
/// `ComponentRegistry` is a self-contained struct that holds *both* the
/// component descriptor table and the storage factory table. This allows multiple
/// independent registries (and therefore multiple independent ECS worlds) in the
/// same process.
///
/// ## Design
/// - `by_type` maps `TypeId -> ComponentID`.
/// - `by_id` stores `ComponentDesc` indexed by `ComponentID`.
/// - `factories` stores `FactoryFn` indexed by `ComponentID`.
/// - `next_id` assigns new IDs sequentially until `COMPONENT_CAP`.
/// - `frozen` prevents further registration once the ECS is initialized.
///
/// ## Invariants
/// - Every entry in `by_type` has a matching `by_id[id]` and `factories[id]`.
/// - IDs are always in bounds of `COMPONENT_CAP`.
/// - When a component is registered, its storage factory is installed.
pub struct ComponentRegistry {
    next_id: ComponentID,
    by_type: HashMap<TypeId, ComponentID>,
    pub(crate) by_id: Vec<Option<ComponentDesc>>,
    /// Factory table is co-located with the registry.
    pub(crate) factories: Vec<Option<FactoryFn>>,
    frozen: bool,
}

impl ComponentRegistry {
    /// Creates a new, empty `ComponentRegistry`.
    pub fn new() -> Self {
        Self {
            next_id: 0,
            by_type: HashMap::new(),
            by_id: vec![None; COMPONENT_CAP],
            factories: vec![None; COMPONENT_CAP],
            frozen: false,
        }
    }

    /// Allocates a new `ComponentID`.
    ///
    /// ## Errors
    /// Returns `RegistryError::CapacityExceeded` if `COMPONENT_CAP` is reached.
    fn alloc_id(&mut self) -> Result<ComponentID, RegistryError> {
        let component_id = self.next_id;
        if (component_id as usize) >= COMPONENT_CAP {
            return Err(RegistryError::CapacityExceeded { cap: COMPONENT_CAP });
        }
        self.next_id = component_id.wrapping_add(1);
        Ok(component_id)
    }

    /// Freezes the registry, preventing further component registrations.
    ///
    /// ## Purpose
    /// Locks component identity and storage layout so archetypes can assume IDs
    /// are complete and stable.
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Returns `true` if the registry has been frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Returns the `ComponentID` associated with a `TypeId`, if registered.
    pub fn component_id_of_type_id(&self, type_id: TypeId) -> Option<ComponentID> {
        self.by_type.get(&type_id).copied()
    }

    /// Returns the component descriptor for a `ComponentID`, if registered.
    pub fn description_by_component_id(&self, component_id: ComponentID) -> Option<&ComponentDesc> {
        self.by_id
            .get(component_id as usize)
            .and_then(|o| o.as_ref())
    }

    /// Validates that a component ID is in range and registered in this registry.
    pub fn require_component_id(&self, component_id: ComponentID) -> Result<(), RegistryError> {
        let index = component_id as usize;
        if index >= COMPONENT_CAP {
            return Err(RegistryError::InvalidComponentId {
                component_id,
                cap: COMPONENT_CAP,
            });
        }
        if self.by_id[index].is_none() {
            return Err(RegistryError::ComponentIdNotRegistered { component_id });
        }
        Ok(())
    }

    /// Returns the storage factory for `component_id`, if registered.
    ///
    /// Factory lookup is a method on the registry.
    pub fn get_factory(&self, component_id: ComponentID) -> Option<FactoryFn> {
        self.factories.get(component_id as usize).and_then(|o| *o)
    }

    /// Creates an empty type-erased storage column for `component_id`.
    ///
    /// Convenience method on the registry instance.
    pub fn make_empty_component(
        &self,
        component_id: ComponentID,
    ) -> RegistryResult<Box<dyn TypeErasedAttribute>> {
        let factory = self
            .get_factory(component_id)
            .ok_or(RegistryError::MissingFactory { component_id })?;
        Ok(factory())
    }

    /// Registers component type `T` and returns its assigned `ComponentID`.
    ///
    /// ## Purpose
    /// Associates a Rust type with a stable runtime identifier and installs the
    /// storage factory used to allocate archetype columns for this type.
    ///
    /// ## Behaviour
    /// - If `T` is already registered, returns the existing ID.
    /// - Otherwise allocates a new ID, stores a `ComponentDesc`, and registers
    ///   a corresponding `TypeErasedAttribute` factory.
    ///
    /// ## Errors
    /// - Returns `RegistryError::Frozen` if the registry is frozen.
    /// - Returns `RegistryError::CapacityExceeded` if `COMPONENT_CAP` is exceeded.
    pub fn register<T: 'static + Send + Sync>(&mut self) -> Result<ComponentID, RegistryError> {
        let type_id = TypeId::of::<T>();
        if let Some(&existing) = self.by_type.get(&type_id) {
            return Ok(existing);
        }

        if size_of::<T>() == 0 {
            return Err(RegistryError::ZeroSizedComponent { type_id });
        }

        if self.frozen {
            return Err(RegistryError::Frozen);
        }

        let id = self.alloc_id()?;
        self.by_type.insert(type_id, id);
        self.by_id[id as usize] = Some(ComponentDesc::of::<T>().with_id(id));
        // Factory is stored in the registry.
        self.factories[id as usize] = Some(new_attribute_storage::<T>);

        Ok(id)
    }

    /// Registers a component type as GPU-safe and returns its assigned `ComponentID`.
    ///
    /// ## Purpose
    /// GPU-aware extension of [`ComponentRegistry::register`] that registers the
    /// component *and* sets the `gpu_usage` flag on its [`ComponentDesc`] in a
    /// single step.  This is the instance-method counterpart of the global
    /// [`register_gpu_component`](super::global::register_gpu_component) function.
    ///
    /// ## Requirements
    /// * The component type must implement [`GPUPod`], guaranteeing that its
    ///   memory layout is safe for direct CPU<->GPU transfers.
    /// * The component must satisfy all normal ECS component constraints
    ///   (non-zero-sized, `'static`, `Send`, `Sync`).
    ///
    /// ## Behaviour
    /// - If `T` is already registered, the existing ID is returned **and** the
    ///   `gpu_usage` flag is set to `true` (idempotent upgrade).
    /// - Otherwise allocates a new ID with `gpu_usage = true`.
    ///
    /// ## Freezing behaviour
    /// This method must be called **before** [`ComponentRegistry::freeze`].
    /// Calling it after the registry is frozen will return an error (unless `T`
    /// is already registered, in which case the flag is still upgraded).
    ///
    /// ## Errors
    /// - Returns `RegistryError::Frozen` if the registry is frozen and `T` is
    ///   not yet registered.
    /// - Returns `RegistryError::CapacityExceeded` if `COMPONENT_CAP` is exceeded.
    #[cfg(feature = "gpu")]
    pub fn register_gpu<T: GPUPod + 'static + Send + Sync>(
        &mut self,
    ) -> Result<ComponentID, RegistryError> {
        let id = self.register::<T>()?;
        if let Some(desc) = self.by_id[id as usize].as_mut() {
            desc.gpu_usage = true;
        }
        Ok(id)
    }

    /// Marks an already-registered component as GPU-safe.
    ///
    /// ## Purpose
    /// Retroactively sets the `gpu_usage` flag on a component that was
    /// previously registered via [`ComponentRegistry::register`].  This is
    /// useful when GPU-safety is determined after initial registration, for
    /// example in test harnesses or plugin-style architectures.
    ///
    /// ## Errors
    /// - Returns `RegistryError::NotRegistered` if `component_id` does not
    ///   correspond to a registered component.
    #[cfg(feature = "gpu")]
    pub fn mark_gpu_safe(&mut self, component_id: ComponentID) -> Result<(), RegistryError> {
        match self.by_id.get_mut(component_id as usize) {
            Some(Some(desc)) => {
                desc.gpu_usage = true;
                Ok(())
            }
            _ => Err(RegistryError::NotRegistered {
                type_id: TypeId::of::<()>(),
            }),
        }
    }

    /// Returns the `ComponentID` for `T`, if registered.
    pub fn id_of<T: 'static>(&self) -> Option<ComponentID> {
        self.component_id_of_type_id(TypeId::of::<T>())
    }

    /// Returns the `ComponentID` for `T`, or an error if it is not registered.
    ///
    /// ## Errors
    /// Returns `RegistryError::NotRegistered` if `T` was not registered.
    pub fn require_id_of<T: 'static>(&self) -> Result<ComponentID, RegistryError> {
        self.id_of::<T>().ok_or(RegistryError::NotRegistered {
            type_id: TypeId::of::<T>(),
        })
    }
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ComponentRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComponentRegistry")
            .field("next_id", &self.next_id)
            .field("frozen", &self.frozen)
            .field("registered_count", &self.by_type.len())
            .finish()
    }
}
