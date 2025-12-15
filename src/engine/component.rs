use std::{
    any::{TypeId, type_name},
    mem::{size_of, align_of},
    sync::{OnceLock, RwLock},
    collections::HashMap,    
};

use crate::storage::{Attribute, TypeErasedAttribute};
use crate::types::{ComponentID, COMPONENT_CAP};


type FactoryFn = fn() -> Box<dyn TypeErasedAttribute>;

static COMPONENT_FACTORIES: OnceLock<RwLock<Vec<Option<FactoryFn>>>> = OnceLock::new();
fn component_factories() -> &'static RwLock<Vec<Option<FactoryFn>>> {
    COMPONENT_FACTORIES.get_or_init(|| RwLock::new(vec![None; COMPONENT_CAP]))
}

fn new_attribute_storage<T: 'static + Send + Sync>() -> Box<dyn TypeErasedAttribute> {
    Box::new(Attribute::<T>::default())
}

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
    fn alloc_id(&mut self) -> ComponentID {
        let component_id = self.next_id;
        assert!((component_id as usize) < COMPONENT_CAP, "Exceeded configured component capacity.");
        self.next_id = component_id.wrapping_add(1);
        component_id
    }

    pub fn freeze(&mut self) { self.frozen = true; }
    pub fn is_frozen(&self) -> bool { self.frozen }

    pub fn component_id_of_type_id(&self, type_id: TypeId) -> Option<ComponentID> {
        self.by_type.get(&type_id).copied()
    }

    pub fn description_by_component_id(&self, component_id: ComponentID) -> Option<&ComponentDesc> {
        self.by_id.get(component_id as usize).and_then(|o| o.as_ref())
    }
}

impl ComponentRegistry {
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

    pub fn id_of<T: 'static>(&self) -> Option<ComponentID> {
        self.component_id_of_type_id(TypeId::of::<T>())
    }

    pub fn require_id_of<T: 'static>(&self) -> ComponentID {
        self.id_of::<T>().expect("component not registered.")
    }
}

pub fn register_component<T: 'static + Send + Sync>() -> ComponentID {
    let registry = component_registry();
    let mut registry = registry.write().unwrap();
    registry.register::<T>()
}
pub fn freeze_components() {
    let registry = component_registry();
    let mut registry = registry.write().unwrap();
    registry.freeze();
}
pub fn component_id_of<T: 'static>() -> ComponentID {
    let registry = component_registry();
    let registry = registry.read().unwrap();
    registry.require_id_of::<T>()
}
pub fn component_id_of_type_id(type_id: TypeId) -> Option<ComponentID> {
    let registry = component_registry();
    let registry = registry.read().unwrap();
    registry.component_id_of_type_id(type_id)
}
pub fn component_description_by_component_id(component_id: ComponentID) -> Option<ComponentDesc> {
    let registry = component_registry();
    let registry = registry.read().unwrap();
    registry.description_by_component_id(component_id).cloned()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ComponentDesc {
    pub component_id: ComponentID,
    pub name: &'static str,
    pub type_id: TypeId,
    pub size: usize,
    pub align: usize
}

impl ComponentDesc {
    #[inline]
    pub fn new(component_id: ComponentID, name: &'static str, type_id: TypeId, size: usize, align: usize) -> Self {
        Self { component_id, name, type_id, size, align }
    }

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

    #[inline]
    pub fn matches_type<T: 'static>(&self) -> bool {
        self.type_id == TypeId::of::<T>()
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

pub fn get_component_storage_factory(component_id: ComponentID) -> FactoryFn {
    component_factories()
        .read().unwrap()[component_id as usize]
        .expect("no factory registered for this component id")
}

pub fn make_empty_component(component_id: ComponentID) -> Box<dyn TypeErasedAttribute> {
    get_component_storage_factory(component_id)()
}
