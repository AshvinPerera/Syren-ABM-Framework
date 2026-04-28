//! Typed handle to a registered environment key.
//!
//! An [`EnvKey<T>`] is a lightweight, copyable token produced by
//! [`Environment::env_key`](crate::environment::Environment::env_key) after
//! the environment is built. It bundles two pieces of information that systems
//! need when interacting with a specific environment key:
//!
//! - The key's name (`&'static str`), for passing to
//!   [`Environment::get`](crate::environment::Environment::get) and
//!   [`Environment::set`](crate::environment::Environment::set).
//! - The key's [`ChannelID`], for inserting into
//!   [`AccessSets::produces`](crate::engine::systems::AccessSets::produces) or
//!   [`AccessSets::consumes`](crate::engine::systems::AccessSets::consumes) so
//!   the scheduler can order env writers before env readers of the same key.
//!
//! ## Acquiring a handle
//!
//! ```text
//! let env = EnvironmentBuilder::new()
//!     .register::<f32>("interest_rate", 0.05)
//!     .build();
//!
//! let rate_key: EnvKey<f32> = env
//!     .env_key::<f32>("interest_rate")
//!     .expect("interest_rate not registered");
//! ```
//!
//! ## Using the handle in a system
//!
//! ```text
//! struct RateWriterSystem {
//!     rate_key: EnvKey<f32>,
//!     access:   AccessSets,
//! }
//!
//! // During construction - declare the scheduling dependency:
//! let mut access = AccessSets::default();
//! access.produces.insert(rate_key.channel_id());
//!
//! // Inside System::run():
//! fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
//!     // ... read env via Arc<Environment> captured in the system ...
//!     env.set(self.rate_key.name(), new_rate)?;
//!     Ok(())
//! }
//! ```
//!
//! ## Key name requirement
//!
//! [`Environment::env_key`](crate::environment::Environment::env_key) requires
//! a `&'static str`. Environment keys intended for use with typed handles must
//! therefore be registered with string literal names (e.g. `"interest_rate"`).
//! Keys registered with a runtime [`String`] cannot produce handles unless the
//! string happens to alias a `'static` reference.

use std::marker::PhantomData;

use crate::engine::types::ChannelID;

/// A typed, copyable handle to a single registered environment key.
///
/// Systems store one `EnvKey<T>` per environment key they interact with. The
/// handle is used in two ways:
///
/// 1. **Name access** - [`name`](Self::name) returns the `&'static str` to
///    pass to [`Environment::get`](crate::environment::Environment::get) /
///    [`Environment::set`](crate::environment::Environment::set).
///
/// 2. **Scheduling** - [`channel_id`](Self::channel_id) returns the
///    [`ChannelID`] to insert into
///    [`AccessSets::produces`](crate::AccessSets::produces) (for writing
///    systems) or [`AccessSets::consumes`](crate::AccessSets::consumes) (for
///    reading systems), so the scheduler places every writer of this key in an
///    earlier stage than every reader.
///
/// `EnvKey<T>` is [`Copy`] - duplicate it freely. The type parameter `T`
/// is a compile-time marker only; no `T` is stored or accessed at runtime.
///
/// ## Example
///
/// ```text
/// let key: EnvKey<f32> = env.env_key::<f32>("tax_rate").unwrap();
///
/// // Writer system access declaration:
/// access.produces.insert(key.channel_id());
///
/// // Reader system access declaration:
/// access.consumes.insert(key.channel_id());
///
/// // Runtime value access:
/// env.set(key.name(), 0.25_f32)?;
/// let v: f32 = env.get(key.name())?;
/// ```
#[derive(Clone, Copy)]
pub struct EnvKey<T> {
    pub(crate) name: &'static str,
    pub(crate) channel_id: ChannelID,
    // `fn() -> T` makes PhantomData covariant in T while keeping EnvKey
    // unconditionally Send + Sync - no T is stored at runtime.
    _marker: PhantomData<fn() -> T>,
}

impl<T> EnvKey<T> {
    /// Constructs a new handle.
    ///
    /// Called exclusively by
    /// [`Environment::env_key`](crate::environment::Environment::env_key).
    #[inline]
    pub(crate) fn new(name: &'static str, channel_id: ChannelID) -> Self {
        Self {
            name,
            channel_id,
            _marker: PhantomData,
        }
    }

    /// Returns the [`ChannelID`] assigned to this key at environment build time.
    ///
    /// Insert this into
    /// [`AccessSets::produces`](crate::AccessSets::produces) on any system that
    /// calls [`Environment::set`](crate::environment::Environment::set) for
    /// this key, and into
    /// [`AccessSets::consumes`](crate::AccessSets::consumes) on any system that
    /// calls [`Environment::get`](crate::environment::Environment::get) for it.
    /// This enables the scheduler to enforce writer-before-reader ordering.
    #[inline]
    pub fn channel_id(self) -> ChannelID {
        self.channel_id
    }

    /// Returns the key name as a `&'static str`.
    ///
    /// Pass this directly to
    /// [`Environment::get`](crate::environment::Environment::get) and
    /// [`Environment::set`](crate::environment::Environment::set).
    #[inline]
    pub fn name(self) -> &'static str {
        self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_id_roundtrip() {
        let key: EnvKey<f32> = EnvKey::new("rate", 42);
        assert_eq!(key.channel_id(), 42);
    }

    #[test]
    fn name_roundtrip() {
        let key: EnvKey<u32> = EnvKey::new("world_width", 7);
        assert_eq!(key.name(), "world_width");
    }

    #[test]
    fn is_copy() {
        let key: EnvKey<f32> = EnvKey::new("x", 1);
        let key2 = key; // copy
        let _ = key; // original still accessible
        assert_eq!(key2.channel_id(), 1);
    }

    #[test]
    fn distinct_type_params_compile() {
        let _f: EnvKey<f32> = EnvKey::new("a", 0);
        let _u: EnvKey<u32> = EnvKey::new("b", 1);
        let _b: EnvKey<bool> = EnvKey::new("c", 2);
    }
}
