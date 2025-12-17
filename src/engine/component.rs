//! # Component Registry
//!
//! This module provides a global registry that assigns stable `ComponentID` values
//! to Rust component types and exposes type-erased storage factories for archetype
//! column allocation.
//!
//! ## Purpose
//! The registry decouples component type information (`TypeId`, name, size, alignment)
//! from runtime storage, enabling archetypes to store heterogeneous component columns
//! behind `TypeErasedAttribute`.
//!
//! ## Design
//! - Components are registered once and assigned a compact `ComponentID` in `[0, COMPONENT_CAP)`.
//! - A per-component factory function is stored for constructing empty column storage.
//! - The registry can be `freeze()`d to prevent further registrations after world setup.
//!
//! ## Invariants
//! - `ComponentID` values are unique and stable for the lifetime of the process.
//! - A registered component must have a corresponding storage factory.
//! - When frozen, registration is disallowed.
//!
//! ## Concurrency
//! The registry is protected by `RwLock` for concurrent reads and serialized writes.
//! Factories are stored separately behind an `RwLock` and are expected to be set
//! during registration only.

use std::{
    any::{TypeId, type_name},
    mem::{size_of, align_of},
    sync::{OnceLock, RwLock},
    collections::HashMap,    
};

use crate::engine::storage::{Attribute, TypeErasedAttribute};
use crate::engine::types::{ComponentID, COMPONENT_CAP};


/// Factory function for constructing an empty type-erased component attribute column.
type FactoryFn = fn() -> Box<dyn TypeErasedAttribute>;

/// Global table of component storage factories indexed by `ComponentID`.
///
/// ## Invariants
/// - `factories[id]` is `Some` if and only if component `id` is registered.
/// - The table length is always `COMPONENT_CAP`.

static COMPONENT_FACTORIES: OnceLock<RwLock<Vec<Option<FactoryFn>>>> = OnceLock::new();

/// Returns the global factory table used to allocate empty component storage columns.
///
/// ## Purpose
/// Provides archetype construction with a way to create column storage from a `ComponentID`.
///
/// ## Invariants
/// The returned table always has length `COMPONENT_CAP`.

fn component_factories() -> &'static RwLock<Vec<Option<FactoryFn>>> {
    COMPONENT_FACTORIES.get_or_init(|| RwLock::new(vec![None; COMPONENT_CAP]))
}

/// Constructs an empty attribute storage column for component type `T`.
///
/// ## Purpose
/// Used as the registered factory for a component ID.

fn new_attribute_storage<T: 'static + Send + Sync>() -> Box<dyn TypeErasedAttribute> {
    Box::new(Attribute::<T>::default())
}

/// Global mapping between Rust component types and compact `ComponentID` values.
///
/// ## Purpose
/// Assigns stable runtime identifiers (`ComponentID`) to component types (`TypeId`)
/// and stores component metadata (`ComponentDesc`) used for diagnostics and validation.
///
/// ## Design
/// - `by_type` maps `TypeId -> ComponentID`.
/// - `by_id` stores `ComponentDesc` indexed by `ComponentID`.
/// - `next_id` assigns new IDs sequentially until `COMPONENT_CAP`.
/// - `frozen` prevents further registration once the ECS is initialized.
///
/// ## Invariants
/// - Every entry in `by_type` has a matching `by_id[id]`.
/// - IDs are always in bounds of `COMPONENT_CAP`.
/// - When a component is registered, its storage factory is installed.

pub struct ComponentRegistry {
    next_id: ComponentID,
    by_type: HashMap<TypeId, ComponentID>,
    by_id: Vec<Option<ComponentDesc>>,
    frozen: bool,
}

static REGISTRY: OnceLock<RwLock<ComponentRegistry>> = OnceLock::new();

fn component_registry() -> &'static RwLock<ComponentRegistry> {
    REGISTRY.get_or_init(|| {
        RwLock::new(ComponentRegistry {
            next_id: 0 as ComponentID,
            by_type: HashMap::new(),
            by_id: vec![None; COMPONENT_CAP],
            frozen: false,
        })
    })
}

impl ComponentRegistry {

    /// Allocates a new `ComponentID`.
    ///
    /// ## Panics
    /// Panics if `COMPONENT_CAP` is exceeded.

    fn alloc_id(&mut self) -> ComponentID {
        let component_id = self.next_id;
        assert!((component_id as usize) < COMPONENT_CAP, "Exceeded configured component capacity.");
        self.next_id = component_id.wrapping_add(1);
        component_id
    }

    /// Freezes the registry, preventing further component registrations.
    ///
    /// ## Purpose
    /// Locks component identity and storage layout so archetypes can assume IDs
    /// are complete and stable.

    pub fn freeze(&mut self) { self.frozen = true; }

    /// Returns `true` if the registry has been frozen.    
    pub fn is_frozen(&self) -> bool { self.frozen }

    /// Returns the `ComponentID` associated with a `TypeId`, if registered.    
    pub fn component_id_of_type_id(&self, type_id: TypeId) -> Option<ComponentID> {
        self.by_type.get(&type_id).copied()
    }

    /// Returns the component descriptor for a `ComponentID`, if registered.
    pub fn description_by_component_id(&self, component_id: ComponentID) -> Option<&ComponentDesc> {
        self.by_id.get(component_id as usize).and_then(|o| o.as_ref())
    }
}

impl ComponentRegistry {

    /// Registers component type `T` and returns its assigned `ComponentID`.
    ///
    /// ## Purpose
    /// Associates a Rust type with a stable runtime identifier and installs the
    /// storage factory used to allocate archetype columns for this type.
    ///
    /// ## Behavior
    /// - If `T` is already registered, returns the existing ID.
    /// - Otherwise allocates a new ID, stores a `ComponentDesc`, and registers
    ///   a corresponding `TypeErasedAttribute` factory.
    ///
    /// ## Panics
    /// - Panics if the registry is frozen.
    /// - Panics if `COMPONENT_CAP` is exceeded.    

    pub fn register<T: 'static + Send + Sync>(&mut self) -> ComponentID {
        let type_id = TypeId::of::<T>();
        if let Some(&existing) = self.by_type.get(&type_id) { 
            return existing; 
        }
        
        assert!(!self.frozen, "Registry frozen");
        let id = self.alloc_id();
        self.by_type.insert(type_id, id);
        self.by_id[id as usize] = Some(ComponentDesc::of::<T>().with_id(id));
        
        component_factories().write().unwrap()[id as usize] = Some(new_attribute_storage::<T>);
        id
    }

    /// Returns the `ComponentID` for `T`, if registered.    
    pub fn id_of<T: 'static>(&self) -> Option<ComponentID> {
        self.component_id_of_type_id(TypeId::of::<T>())
    }

    /// Returns the `ComponentID` for `T`, panicking if it is not registered.
    ///
    /// ## Panics
    /// Panics if `T` was not registered.

    pub fn require_id_of<T: 'static>(&self) -> ComponentID {
        self.id_of::<T>().expect("component not registered.")
    }
}

/// Registers component type `T` in the global registry and returns its `ComponentID`.
///
/// ## Purpose
/// Convenience wrapper around the global `ComponentRegistry`.
///
/// ## Panics
/// Panics if the registry is frozen or capacity is exceeded.

pub fn register_component<T: 'static + Send + Sync>() -> ComponentID {
    let registry = component_registry();
    let mut registry = registry.write().unwrap();
    registry.register::<T>()
}

/// Freezes the global component registry.
///
/// ## Purpose
/// Prevents any further component registration, making component IDs and storage
/// factories stable for archetype construction.
///
/// ## Panics
/// Panics if the registry lock is poisoned.

pub fn freeze_components() {
    let registry = component_registry();
    let mut registry = registry.write().unwrap();
    registry.freeze();
}

/// Returns the registered `ComponentID` for type `T`.
///
/// ## Panics
/// Panics if `T` is not registered.

pub fn component_id_of<T: 'static>() -> ComponentID {
    let registry = component_registry();
    let registry = registry.read().unwrap();
    registry.require_id_of::<T>()
}

/// Returns the `ComponentID` associated with a runtime `TypeId`, if registered.
pub fn component_id_of_type_id(type_id: TypeId) -> Option<ComponentID> {
    let registry = component_registry();
    let registry = registry.read().unwrap();
    registry.component_id_of_type_id(type_id)
}

/// Returns a copy of the descriptor for `component_id`, if registered.
pub fn component_description_by_component_id(component_id: ComponentID) -> Option<ComponentDesc> {
    let registry = component_registry();
    let registry = registry.read().unwrap();
    registry.description_by_component_id(component_id).cloned()
}

/// Describes a registered component type.
///
/// ## Purpose
/// Provides metadata about a component type for debugging, validation, and tooling.
///
/// ## Fields
/// - `component_id`: The runtime identifier assigned by the registry.
/// - `name`: The Rust type name (`type_name::<T>()`).
/// - `type_id`: The runtime `TypeId` for the component.
/// - `size`: `size_of::<T>()` in bytes.
/// - `align`: `align_of::<T>()` in bytes.
///
/// ## Notes
/// `ComponentDesc` is `Copy` and safe to clone freely for reporting and diagnostics.

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ComponentDesc {
    /// Runtime identifier assigned to this component type.
    pub component_id: ComponentID,

    /// Rust type name for diagnostics.
    pub name: &'static str,

    /// Runtime `TypeId` of the component.
    pub type_id: TypeId,

    /// Size of the component type in bytes.
    pub size: usize,

    /// Alignment of the component type in bytes.
    pub align: usize
}

impl ComponentDesc {

    /// Creates a descriptor from explicit metadata.    
    #[inline]
    pub fn new(component_id: ComponentID, name: &'static str, type_id: TypeId, size: usize, align: usize) -> Self {
        Self { component_id, name, type_id, size, align }
    }

    /// Constructs a descriptor for type `T` using its `TypeId`, name, size, and alignment.
    ///
    /// ## Notes
    /// The returned descriptor uses `component_id = 0` and should be finalized via `with_id`.
   
    #[inline]
    pub fn of<T: 'static>() -> Self {
        Self {
            component_id: 0,
            name: type_name::<T>(),
            type_id: TypeId::of::<T>(),
            size: size_of::<T>(),
            align: align_of::<T>(),
        }
    }

    /// Returns `true` if this descriptor refers to type `T`.
    #[inline]
    pub fn matches_type<T: 'static>(&self) -> bool {
        self.type_id == TypeId::of::<T>()
    }

    /// Returns a copy of this descriptor with `component_id` set to the provided value.    
    #[inline]
    pub fn with_id(mut self, component_id: ComponentID) -> Self {
        self.component_id = component_id;
        self
    }   
}

impl std::fmt::Display for ComponentDesc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ComponentDesc {{ id: {}, name: {}, size: {}, align: {} }}",
            self.component_id, self.name, self.size, self.align
        )
    }
}

/// Returns the storage factory function for the given `component_id`.
///
/// ## Purpose
/// Used by archetype construction to allocate an empty attribute column for a component.
///
/// ## Panics
/// Panics if no factory was registered for this component ID.

pub fn get_component_storage_factory(component_id: ComponentID) -> FactoryFn {
    component_factories()
        .read().unwrap()[component_id as usize]
        .expect("no factory registered for this component id")
}

/// Creates an empty type-erased storage column for `component_id`.
///
/// ## Purpose
/// Convenience wrapper around `get_component_storage_factory`.
///
/// ## Panics
/// Panics if no factory exists for the provided ID.

pub fn make_empty_component(component_id: ComponentID) -> Box<dyn TypeErasedAttribute> {
    get_component_storage_factory(component_id)()
}
