//! Archetype component management operations.
//!
//! This module implements the component-level mutation methods on [`Archetype`],
//! covering the full lifecycle of component attributes within an archetype:
//!
//! - **Insertion** – [`Archetype::ensure_component`] and [`Archetype::insert_empty_component`]
//!   add new component columns, update the archetype signature, and maintain the
//!   sorted invariant of the internal component list.
//!
//! - **Removal** – [`Archetype::remove_component`] extracts and returns a component
//!   column, guarded by a precondition that the archetype must be empty to preserve
//!   row alignment.
//!
//! - **Construction** – [`Archetype::from_components`] builds a fully-signed, empty
//!   archetype from an iterator of [`std::any::TypeId`]s, resolving each to a
//!   registered [`ComponentID`] via the provided [`ComponentRegistry`].
//!
//! ## Invariants
//!
//! All methods in this module uphold the following structural guarantees:
//!
//! - The `components` vec remains sorted by [`ComponentID`] at all times, enabling
//!   binary-search access.
//! - The archetype [`Signature`] always reflects the exact set of component columns
//!   present — no more, no less.
//! - Component attributes are never added to or removed from a populated archetype,
//!   as doing so would break row-index alignment across columns.

use crate::engine::types::{ArchetypeID, ComponentID, COMPONENT_CAP};

use crate::engine::storage::{LockedAttribute, TypeErasedAttribute};

use crate::engine::component::{ComponentRegistry, Signature};

use crate::engine::error::{ECSError, ECSResult, InternalViolation, RegistryError, SpawnError};

use super::core::Archetype;

impl Archetype {
    /// Guarantees that a component attribute exists for the given `component_id`.
    ///
    /// ## Behaviour
    /// - Allocates a new column using the provided factory if not already present.
    /// - Marks the component bit in the signature.
    /// - Maintains sorted invariant of `components`.
    ///
    /// ## Invariants
    /// Attribute allocation and signature must remain consistent.

    #[inline]
    pub fn ensure_component(
        &mut self,
        component_id: ComponentID,
        factory: impl FnOnce() -> Result<Box<dyn TypeErasedAttribute>, RegistryError>,
    ) -> Result<(), SpawnError> {
        let index = component_id as usize;
        if index >= COMPONENT_CAP {
            return Err(SpawnError::InvalidComponentId);
        }

        // Binary search on sorted vec.
        match self
            .components
            .binary_search_by_key(&component_id, |(cid, _)| *cid)
        {
            Ok(_) => { /* already present */ }
            Err(insert_pos) => {
                let col = factory()?;
                self.components
                    .insert(insert_pos, (component_id, LockedAttribute::new(col)));
                self.signature.set(component_id);
            }
        }

        Ok(())
    }

    /// Inserts an empty component attribute into the archetype.
    ///
    /// ## Purpose
    /// Used when constructing archetypes from predefined type lists.
    ///
    /// ## Behaviour
    /// - Fails if the index exceeds `COMPONENT_CAP`.
    /// - Assumes the attribute did not previously exist.
    /// - Sets the signature bit for the component.
    /// - Maintains sorted invariant of `components`.
    ///
    /// ## Invariants
    /// Component attributes must be added only before entities are inserted.

    pub fn insert_empty_component(
        &mut self,
        component_id: ComponentID,
        component: Box<dyn TypeErasedAttribute>,
    ) -> ECSResult<()> {
        let index = component_id as usize;
        if index >= COMPONENT_CAP {
            return Err(SpawnError::InvalidComponentId.into());
        }

        // Binary search for insertion.
        match self
            .components
            .binary_search_by_key(&component_id, |(cid, _)| *cid)
        {
            Ok(_) => {
                return Err(InternalViolation::ComponentAlreadyPresent.into());
            }
            Err(insert_pos) => {
                self.components
                    .insert(insert_pos, (component_id, LockedAttribute::new(component)));
                self.signature.set(component_id);
            }
        }

        Ok(())
    }

    /// Removes a component attribute from an empty archetype.
    ///
    /// ## Invariants
    /// Removing attributes in a populated archetype would break row alignment.

    pub fn remove_component(
        &mut self,
        component_id: ComponentID,
    ) -> ECSResult<Option<Box<dyn TypeErasedAttribute>>> {
        if self.length()? > 0 {
            return Err(SpawnError::ArchetypeNotEmpty.into());
        }

        let index = component_id as usize;
        if index >= COMPONENT_CAP {
            return Err(SpawnError::InvalidComponentId.into());
        }

        // Binary search for removal.
        match self
            .components
            .binary_search_by_key(&component_id, |(cid, _)| *cid)
        {
            Ok(pos) => {
                let (_, locked) = self.components.remove(pos);
                self.signature.clear(component_id);
                Ok(Some(
                    locked
                        .into_inner()
                        .map_err(|e| SpawnError::StoragePushFailedWith(e))?,
                ))
            }
            Err(_) => Ok(None),
        }
    }

    /// Constructs a new archetype and inserts empty attributes for the provided
    /// component types.
    ///
    /// ## Behaviour
    /// - Each type ID must correspond to a component registered in `registry`.
    /// - Component attributes are allocated via the storage factory.
    /// - No entities are created.
    ///
    /// ## Invariants
    /// The resulting archetype is empty but has a fully defined signature.

    pub fn from_components<T: IntoIterator<Item = std::any::TypeId>>(
        archetype_id: ArchetypeID,
        types: T,
        registry: &ComponentRegistry,
    ) -> ECSResult<Self> {
        let mut signature = Signature::default();
        let mut component_ids = Vec::new();

        for type_id in types {
            let component_id = registry
                .component_id_of_type_id(type_id)
                .ok_or(ECSError::from(
                    InternalViolation::ComponentTypeNotRegistered,
                ))?;
            signature.set(component_id);
            component_ids.push(component_id);
        }

        let archetype = Self::new(archetype_id, signature, registry)?;

        if !component_ids.iter().all(|&id| archetype.has(id)) {
            return Err(InternalViolation::SignatureStorageMismatch.into());
        }

        Ok(archetype)
    }
}
