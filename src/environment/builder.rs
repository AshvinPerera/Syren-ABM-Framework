//! Fluent builder for [`Environment`].
//!
//! ## Usage
//!
//! ```ignore
//! use std::sync::Arc;
//! use abm_framework::environment::EnvironmentBuilder;
//!
//! let env: Arc<Environment> = EnvironmentBuilder::new()
//!     .register::<f32>("interest_rate", 0.05)
//!     .register::<u32>("world_width",    100)
//!     .build();
//! ```
//!
//! ## Design
//!
//! The builder collects schema entries as a `Vec` in declaration order.
//! If the same key is registered more than once with the same type, the last
//! default value wins. Re-registering a key with a different type panics
//! immediately — this catches configuration bugs at setup time rather than
//! producing silent runtime type-mismatch errors later.
//!
//! [`build`](EnvironmentBuilder::build) consumes `self` and returns an
//! `Arc<Environment>` so that the store can be shared across systems,
//! the GPU uniform buffer, and the scheduler with zero cloning of values.

use std::any::{Any, TypeId};
use std::sync::Arc;

use super::store::Environment;

/// Fluent builder for [`Environment`].
///
/// Call [`register`](Self::register) for each simulation parameter, then
/// [`build`](Self::build) to freeze the schema and obtain a shared
/// [`Environment`] handle.
pub struct EnvironmentBuilder {
    /// Ordered entries: `(key, initial_value, TypeId, type_name)`.
    ///
    /// Stored as a `Vec` (not a `HashMap`) to preserve declaration order,
    /// which matters for deterministic unit tests and for GPU struct layout.
    schema: Vec<(String, Box<dyn Any + Send + Sync>, TypeId, &'static str)>,
}

impl EnvironmentBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self { schema: Vec::new() }
    }

    /// Registers a typed key with a default value.
    ///
    /// If the same key is registered more than once **with the same type**,
    /// the last default value wins. This is useful in generated or conditional
    /// setup code where a parameter may be overridden.
    ///
    /// Must be called before [`build`](Self::build).
    ///
    /// # Type parameters
    ///
    /// `T` must be `Any + Clone + Send + Sync + 'static`.
    ///
    /// # Panics
    ///
    /// - Panics if the same key was previously registered with a **different
    ///   type**. Changing a key's type during the build phase is almost
    ///   certainly a bug and is caught eagerly here rather than at runtime.
    /// - Panics in debug mode if `key` is empty.
    pub fn register<T>(mut self, key: impl Into<String>, default: T) -> Self
    where
        T: Any + Clone + Send + Sync + 'static,
    {
        let key = key.into();
        debug_assert!(!key.is_empty(), "environment key must not be empty");

        let new_type_id = TypeId::of::<T>();

        if let Some(pos) = self.schema.iter().position(|(k, _, _, _)| k == &key) {
            let (_, _, existing_type_id, existing_type_name) = &self.schema[pos];
            assert_eq!(
                *existing_type_id, new_type_id,
                "EnvironmentBuilder: key '{key}' was previously registered as \
                 {existing_type_name}, cannot re-register as {}",
                std::any::type_name::<T>()
            );
            // Same type, updated default value.
            self.schema[pos].1 = Box::new(default);
        } else {
            self.schema.push((
                key,
                Box::new(default),
                new_type_id,
                std::any::type_name::<T>(),
            ));
        }
        self
    }

    /// Freezes the schema and returns a shared [`Environment`].
    ///
    /// After this call, no new keys can be added. The returned `Arc` can be
    /// cloned and shared across systems, uniform buffers, and the scheduler.
    pub fn build(self) -> Arc<Environment> {
        Arc::new(Environment::from_schema(self.schema))
    }
}

impl Default for EnvironmentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::error::EnvironmentError;

    #[test]
    fn register_and_build() {
        let env = EnvironmentBuilder::new()
            .register::<f32>("rate", 0.1)
            .register::<u32>("size", 50)
            .build();

        assert_eq!(env.len(), 2);
        assert!((env.get::<f32>("rate").unwrap() - 0.1).abs() < f32::EPSILON);
        assert_eq!(env.get::<u32>("size").unwrap(), 50u32);
    }

    #[test]
    fn same_type_re_register_updates_value() {
        let env = EnvironmentBuilder::new()
            .register::<f32>("rate", 0.1)
            .register::<f32>("rate", 0.9)
            .build();

        // Last registration takes precedence; only one entry.
        assert_eq!(env.len(), 1);
        assert!((env.get::<f32>("rate").unwrap() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    #[should_panic(expected = "cannot re-register")]
    fn type_change_panics() {
        EnvironmentBuilder::new()
            .register::<f32>("rate", 0.05)
            .register::<u32>("rate", 5);
    }

    #[test]
    fn empty_build() {
        let env = EnvironmentBuilder::new().build();
        assert!(env.is_empty());
    }

    #[test]
    fn get_unknown_key_after_build() {
        let env = EnvironmentBuilder::new()
            .register::<f32>("rate", 0.1)
            .build();
        let err = env.get::<f32>("unknown").unwrap_err();
        assert!(matches!(err, EnvironmentError::KeyNotFound(_)));
    }

    #[test]
    fn arc_is_shared_between_clones() {
        let env = EnvironmentBuilder::new()
            .register::<i32>("counter", 0)
            .build();
        let env2 = Arc::clone(&env);
        env.set::<i32>("counter", 42).unwrap();
        assert_eq!(env2.get::<i32>("counter").unwrap(), 42);
    }
}
