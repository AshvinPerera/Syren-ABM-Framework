//! Fluent builder for [`Environment`].
//!
//! ## Usage
//!
//! ```ignore
//! use std::sync::Arc;
//! use abm_framework::environment::EnvironmentBuilder;
//!
//! let env: Arc<Environment> = EnvironmentBuilder::new()
//!     .register::<f32>("interest_rate", 0.05)?
//!     .register::<u32>("world_width",    100)?
//!     .build()?;
//! ```
//!
//! When constructing an environment as part of a larger model that already
//! owns a [`ChannelAllocator`], use [`build_with_allocator`] instead so that
//! the environment's channel IDs do not collide with those assigned to other
//! subsystems:
//!
//! ```ignore
//! let mut alloc = ChannelAllocator::new();
//! let env = EnvironmentBuilder::new()
//!     .register::<f32>("interest_rate", 0.05)?
//!     .build_with_allocator(&mut alloc)?;
//! // alloc.peek_next() is now past the IDs consumed by the environment.
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
//! Each key is assigned a monotonically increasing [`ChannelID`] during
//! [`build`] / [`build_with_allocator`]. The IDs are stored inside the
//! environment's entries and are used by the scheduler to order writers before
//! readers of the same key.
//!
//! [`build`](EnvironmentBuilder::build) consumes `self` and returns an
//! `Arc<Environment>` so that the store can be shared across systems,
//! the GPU uniform buffer, and the scheduler with zero cloning of values.

use std::any::{Any, TypeId};
use std::sync::Arc;

use crate::engine::channel_allocator::ChannelAllocator;
use crate::engine::error::ECSResult;
use crate::engine::types::ChannelID;

use super::error::{EnvironmentError, EnvironmentResult};
use super::store::Environment;

/// Fluent builder for [`Environment`].
///
/// Call [`register`](Self::register) for each simulation parameter, then
/// [`build`](Self::build) (or [`build_with_allocator`](Self::build_with_allocator))
/// to freeze the schema and obtain a shared [`Environment`] handle.
pub struct EnvironmentBuilder {
    /// Ordered entries: `(key, initial_value, TypeId, type_name, preassigned_channel)`.
    ///
    /// Stored as a `Vec` (not a `HashMap`) to preserve declaration order,
    /// which matters for deterministic unit tests and for GPU struct layout.
    schema: Vec<(
        String,
        Box<dyn Any + Send + Sync>,
        TypeId,
        &'static str,
        Option<ChannelID>,
    )>,
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
    /// Must be called before [`build`](Self::build) or
    /// [`build_with_allocator`](Self::build_with_allocator).
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
    pub fn register<T>(mut self, key: impl Into<String>, default: T) -> EnvironmentResult<Self>
    where
        T: Any + Clone + Send + Sync + 'static,
    {
        let key = key.into();
        if key.is_empty() {
            return Err(EnvironmentError::EmptyKey);
        }

        let new_type_id = TypeId::of::<T>();

        if let Some(pos) = self.schema.iter().position(|(k, _, _, _, _)| k == &key) {
            let (_, _, existing_type_id, existing_type_name, _) = &self.schema[pos];
            if *existing_type_id != new_type_id {
                return Err(EnvironmentError::TypeMismatch {
                    key,
                    expected: existing_type_name,
                    actual: std::any::type_name::<T>(),
                });
            }
            // Same type, updated default value.
            self.schema[pos].1 = Box::new(default);
        } else {
            self.schema.push((
                key,
                Box::new(default),
                new_type_id,
                std::any::type_name::<T>(),
                None,
            ));
        }
        Ok(self)
    }

    /// Registers a typed key with a preassigned scheduler channel.
    #[allow(dead_code)]
    pub(crate) fn register_with_channel<T>(
        mut self,
        key: impl Into<String>,
        default: T,
        channel_id: ChannelID,
    ) -> EnvironmentResult<Self>
    where
        T: Any + Clone + Send + Sync + 'static,
    {
        let key = key.into();
        if key.is_empty() {
            return Err(EnvironmentError::EmptyKey);
        }
        let new_type_id = TypeId::of::<T>();

        if let Some(pos) = self.schema.iter().position(|(k, _, _, _, _)| k == &key) {
            {
                let (_, _, existing_type_id, existing_type_name, _) = &self.schema[pos];
                if *existing_type_id != new_type_id {
                    return Err(EnvironmentError::TypeMismatch {
                        key,
                        expected: existing_type_name,
                        actual: std::any::type_name::<T>(),
                    });
                }
            }
            self.schema[pos].1 = Box::new(default);
            self.schema[pos].4 = Some(channel_id);
        } else {
            self.schema.push((
                key,
                Box::new(default),
                new_type_id,
                std::any::type_name::<T>(),
                Some(channel_id),
            ));
        }
        Ok(self)
    }

    /// Freezes the schema and returns a shared [`Environment`].
    ///
    /// Each registered key is assigned a [`ChannelID`](crate::engine::types::ChannelID)
    /// from a freshly created [`ChannelAllocator`] starting at `0`. This is
    /// suitable for tests and standalone environments that do not share a
    /// channel namespace with other subsystems.
    ///
    /// When the environment must coexist with other channel-bearing resources
    /// in a shared model, use [`build_with_allocator`](Self::build_with_allocator)
    /// so the IDs are drawn from a shared allocator and cannot overlap.
    ///
    /// After this call, no new keys can be added. The returned `Arc` can be
    /// cloned and shared across systems, uniform buffers, and the scheduler.
    pub fn build(self) -> ECSResult<Arc<Environment>> {
        let mut alloc = ChannelAllocator::new();
        self.build_with_allocator(&mut alloc)
    }

    /// Freezes the schema and returns a shared [`Environment`], drawing
    /// [`ChannelID`](crate::engine::types::ChannelID)s from a caller-supplied
    /// [`ChannelAllocator`].
    ///
    /// Each registered key consumes exactly one ID from `allocator` in
    /// declaration order. After this call, `allocator.peek_next()` is advanced
    /// past all IDs assigned to the environment. Subsequent calls to
    /// `alloc` on the same allocator will produce IDs that do not collide with
    /// any environment key.
    ///
    /// # Panics
    ///
    /// Does not panic. If the allocator's counter would overflow `u32::MAX`
    /// the allocator itself will panic (see
    /// [`ChannelAllocator::alloc`](crate::engine::channel_allocator::ChannelAllocator::alloc)).
    pub fn build_with_allocator(
        self,
        allocator: &mut ChannelAllocator,
    ) -> ECSResult<Arc<Environment>> {
        let mut channel_ids = Vec::with_capacity(self.schema.len());
        let mut schema = Vec::with_capacity(self.schema.len());

        for (key, value, type_id, type_name, preassigned) in self.schema {
            channel_ids.push(match preassigned {
                Some(channel_id) => channel_id,
                None => allocator.alloc()?,
            });
            schema.push((key, value, type_id, type_name));
        }

        Ok(Arc::new(Environment::from_schema(schema, channel_ids)))
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
            .unwrap()
            .register::<u32>("size", 50)
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(env.len(), 2);
        assert!((env.get::<f32>("rate").unwrap() - 0.1).abs() < f32::EPSILON);
        assert_eq!(env.get::<u32>("size").unwrap(), 50u32);
    }

    #[test]
    fn same_type_re_register_updates_value() {
        let env = EnvironmentBuilder::new()
            .register::<f32>("rate", 0.1)
            .unwrap()
            .register::<f32>("rate", 0.9)
            .unwrap()
            .build()
            .unwrap();

        // Last registration takes precedence; only one entry.
        assert_eq!(env.len(), 1);
        assert!((env.get::<f32>("rate").unwrap() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn type_change_returns_structured_error() {
        let err = match EnvironmentBuilder::new()
            .register::<f32>("rate", 0.05)
            .unwrap()
            .register::<u32>("rate", 5)
        {
            Ok(_) => panic!("expected type mismatch"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            EnvironmentError::TypeMismatch { key, .. } if key == "rate"
        ));
    }

    #[test]
    fn empty_build() {
        let env = EnvironmentBuilder::new().build().unwrap();
        assert!(env.is_empty());
    }

    #[test]
    fn get_unknown_key_after_build() {
        let env = EnvironmentBuilder::new()
            .register::<f32>("rate", 0.1)
            .unwrap()
            .build()
            .unwrap();
        let err = env.get::<f32>("unknown").unwrap_err();
        assert!(matches!(err, EnvironmentError::KeyNotFound(_)));
    }

    #[test]
    fn arc_is_shared_between_clones() {
        let env = EnvironmentBuilder::new()
            .register::<i32>("counter", 0)
            .unwrap()
            .build()
            .unwrap();
        let env2 = Arc::clone(&env);
        env.set::<i32>("counter", 42).unwrap();
        assert_eq!(env2.get::<i32>("counter").unwrap(), 42);
    }

    #[test]
    fn build_assigns_channel_ids_in_order() {
        let env = EnvironmentBuilder::new()
            .register::<f32>("a", 0.0)
            .unwrap()
            .register::<u32>("b", 0)
            .unwrap()
            .register::<bool>("c", false)
            .unwrap()
            .build()
            .unwrap();

        let id_a = env.channel_of("a").unwrap();
        let id_b = env.channel_of("b").unwrap();
        let id_c = env.channel_of("c").unwrap();

        // build() starts from 0; IDs must be sequential.
        assert_eq!(id_a, 0);
        assert_eq!(id_b, 1);
        assert_eq!(id_c, 2);
    }

    #[test]
    fn build_with_allocator_continues_from_offset() {
        let mut alloc = ChannelAllocator::new();
        // Consume the first two IDs for some other subsystem.
        let _ = alloc.alloc().unwrap();
        let _ = alloc.alloc().unwrap();

        let env = EnvironmentBuilder::new()
            .register::<f32>("rate", 0.05)
            .unwrap()
            .register::<u32>("width", 100)
            .unwrap()
            .build_with_allocator(&mut alloc)
            .unwrap();

        // Environment IDs should start at 2.
        assert_eq!(env.channel_of("rate").unwrap(), 2);
        assert_eq!(env.channel_of("width").unwrap(), 3);

        // Allocator should now be past all four IDs.
        assert_eq!(alloc.peek_next(), 4);
    }

    #[test]
    fn build_with_allocator_does_not_overlap_with_subsequent_allocs() {
        let mut alloc = ChannelAllocator::new();
        let env = EnvironmentBuilder::new()
            .register::<f32>("rate", 0.05)
            .unwrap()
            .build_with_allocator(&mut alloc)
            .unwrap();

        let env_id = env.channel_of("rate").unwrap();
        let other_id = alloc.alloc().unwrap();

        assert_ne!(env_id, other_id);
    }
}
