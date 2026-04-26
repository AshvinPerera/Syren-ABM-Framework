//! Type-erased component bundles for dynamic entity composition.
//!
//! This module provides [`Bundle`], a runtime-typed container that maps
//! [`ComponentID`]s to heap-allocated component values. Bundles are the
//! primary vehicle for inserting or transferring components without
//! compile-time type information, making them useful for scripting layers,
//! serialization, and editor tooling.
//!
//! # Core abstractions
//!
//! - [`DynamicBundle`] — a trait for any type that can yield component values
//!   by ID, enabling interoperability between different bundle representations.
//! - [`Bundle`] — the standard implementation, backed by a [`Signature`]
//!   bitset for O(1) membership checks and a sparse `Vec` for value storage.
//!
//! # Example
//!
//! ```
//! use abm_framework::{Bundle, Signature};
//!
//! # #[derive(Clone)] struct Position { x: f32, y: f32 }
//! # #[derive(Clone)] struct Velocity { dx: f32, dy: f32 }
//! # let position_id: abm_framework::ComponentID = 0;
//! # let velocity_id: abm_framework::ComponentID = 1;
//! # let mut archetype_requirements = Signature::default();
//! # archetype_requirements.set(position_id);
//! # archetype_requirements.set(velocity_id);
//! let mut bundle = Bundle::new();
//! bundle.insert(position_id, Position { x: 1.0, y: 2.0 });
//! bundle.insert(velocity_id, Velocity { dx: 0.5, dy: 0.0 });
//!
//! assert!(bundle.is_complete_for(&archetype_requirements));
//! ```

use std::any::Any;

use crate::engine::error::RegistryResult;
use crate::engine::types::ComponentID;

use super::signature::Signature;

/// Type-erased container for component values.
pub trait DynamicBundle {
    /// Removes and returns the value for `component_id`, if present.
    fn take(&mut self, component_id: ComponentID) -> Option<Box<dyn Any + Send>>;
}

/// Concrete implementation of a dynamic component bundle.
pub struct Bundle {
    /// Component presence signature
    signature: Signature,
    /// Sparse storage of component values
    values: Vec<(ComponentID, Box<dyn Any + Send>)>,
}

impl Bundle {
    /// Creates an empty bundle.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears all stored component values.
    #[inline]
    pub fn clear(&mut self) {
        self.signature = Signature::default();
        self.values.clear();
    }

    /// Inserts a component value into the bundle.
    #[inline]
    pub fn insert<T: Any + Send>(&mut self, component_id: ComponentID, value: T) {
        self.signature.set(component_id);
        if let Some((_, slot)) = self.values.iter_mut().find(|(cid, _)| *cid == component_id) {
            *slot = Box::new(value);
        } else {
            self.values.push((component_id, Box::new(value)));
        }
    }

    /// Inserts a component value, returning an error if `component_id` is out of range.
    #[inline]
    pub fn try_insert<T: Any + Send>(
        &mut self,
        component_id: ComponentID,
        value: T,
    ) -> RegistryResult<()> {
        self.signature.try_set(component_id)?;
        if let Some((_, slot)) = self.values.iter_mut().find(|(cid, _)| *cid == component_id) {
            *slot = Box::new(value);
        } else {
            self.values.push((component_id, Box::new(value)));
        }
        Ok(())
    }

    /// Inserts multiple component values from an iterator.
    #[inline]
    pub fn extend_from_iter<T: Any + Send, I: IntoIterator<Item = (ComponentID, T)>>(
        &mut self,
        iter: I,
    ) {
        for (component_id, value) in iter {
            self.insert(component_id, value);
        }
    }

    /// Returns `true` if all components required by `required` are present in this bundle.
    #[inline]
    pub fn is_complete_for(&self, required: &Signature) -> bool {
        required
            .iterate_over_components()
            .all(|cid| self.signature.has(cid))
    }

    /// Builds a signature representing the components present in this bundle.
    #[inline]
    pub fn signature(&self) -> Signature {
        self.signature
    }

    /// Inserts a pre-boxed component value directly.
    pub fn insert_boxed(&mut self, id: ComponentID, value: Box<dyn Any + Send>) {
        self.signature.set(id);
        if let Some((_, slot)) = self.values.iter_mut().find(|(cid, _)| *cid == id) {
            *slot = value;
        } else {
            self.values.push((id, value));
        }
    }

    /// Inserts a boxed component value, returning an error if `id` is out of range.
    pub fn try_insert_boxed(
        &mut self,
        id: ComponentID,
        value: Box<dyn Any + Send>,
    ) -> RegistryResult<()> {
        self.signature.try_set(id)?;
        if let Some((_, slot)) = self.values.iter_mut().find(|(cid, _)| *cid == id) {
            *slot = value;
        } else {
            self.values.push((id, value));
        }
        Ok(())
    }
}

impl Default for Bundle {
    fn default() -> Self {
        Self {
            signature: Signature::default(),
            values: Vec::new(),
        }
    }
}

impl DynamicBundle for Bundle {
    #[inline]
    fn take(&mut self, component_id: ComponentID) -> Option<Box<dyn Any + Send>> {
        let index = self
            .values
            .iter()
            .position(|(cid, _)| *cid == component_id)?;

        let (_, value) = self.values.swap_remove(index);
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::error::RegistryError;
    use crate::engine::types::COMPONENT_CAP;

    #[test]
    fn fallible_insert_helpers_reject_out_of_range_component_id() {
        let invalid = COMPONENT_CAP as ComponentID;
        let mut bundle = Bundle::new();

        assert!(matches!(
            bundle.try_insert(invalid, 1_u32),
            Err(RegistryError::InvalidComponentId { .. })
        ));
        assert!(matches!(
            bundle.try_insert_boxed(invalid, Box::new(2_u32)),
            Err(RegistryError::InvalidComponentId { .. })
        ));
    }
}
