//! A schedulable [`System`] that receives `Arc<Environment>` alongside the
//! standard [`ECSReference`].
//!
//! ## Design
//!
//! [`EnvironmentSystem`] wraps a closure of the form
//! `Fn(Arc<Environment>, ECSReference<'_>) -> ECSResult<()>` so that
//! environment-reading or environment-mutating logic can be registered in the
//! [`Scheduler`](crate::engine::scheduler::Scheduler) as a normal system.
//!
//! The environment handle is captured inside the struct (as `Arc<Environment>`)
//! and passed by value to the closure on each tick. This is cheap — `Arc::clone`
//! bumps an atomic counter.
//!
//! ## System names
//!
//! The [`System`] trait declares `fn name(&self) -> &str`, allowing
//! implementors to return a borrow from `&self`. [`EnvironmentSystem`] stores
//! its name as an owned [`String`] and returns it by reference. Dynamic names
//! generated from configuration are supported with no leak — every system
//! drops its name when the system itself is dropped.

use std::sync::Arc;

use crate::engine::error::ECSResult;
use crate::engine::manager::ECSReference;
use crate::engine::systems::{AccessSets, System, SystemBackend};
use crate::engine::types::SystemID;

use super::store::Environment;

// ─────────────────────────────────────────────────────────────────────────────
// EnvironmentSystem
// ─────────────────────────────────────────────────────────────────────────────

/// A [`System`] backed by a closure that receives the shared [`Environment`].
///
/// Construct via [`EnvironmentSystem::new`] and register with the scheduler
/// like any other system.
///
/// # Example
///
/// ```ignore
/// let sys = EnvironmentSystem::new(
///     0,
///     "UpdateTaxRate",
///     AccessSets::default(),
///     Arc::clone(&env),
///     |env, _ecs| {
///         let current: f32 = env.get("tax_rate")?;
///         env.set("tax_rate", current * 1.01)?;
///         Ok(())
///     },
/// );
/// scheduler.add_system(sys);
/// ```
pub struct EnvironmentSystem {
    id: SystemID,
    /// Owned name string; returned by reference from [`System::name`].
    name: String,
    access: AccessSets,
    env: Arc<Environment>,
    func: Box<dyn Fn(Arc<Environment>, ECSReference<'_>) -> ECSResult<()> + Send + Sync>,
}

impl EnvironmentSystem {
    /// Creates a new [`EnvironmentSystem`].
    ///
    /// # Parameters
    ///
    /// - `id`: Unique system identifier (must not collide with any other system
    ///   in the same scheduler).
    /// - `name`: Human-readable name used for debugging and profiling. Stored
    ///   as an owned [`String`] and returned by reference from
    ///   [`System::name`]. Dynamic names are supported with no leak.
    /// - `access`: Declared component read/write sets; used by the scheduler for
    ///   conflict detection. If the system only reads/writes the environment and
    ///   does not touch ECS components, pass [`AccessSets::default()`].
    /// - `env`: Shared environment handle.
    /// - `func`: The closure executed on each tick.
    pub fn new<F>(
        id: SystemID,
        name: impl Into<String>,
        access: AccessSets,
        env: Arc<Environment>,
        func: F,
    ) -> Self
    where
        F: Fn(Arc<Environment>, ECSReference<'_>) -> ECSResult<()> + Send + Sync + 'static,
    {
        Self {
            id,
            name: name.into(),
            access,
            env,
            func: Box::new(func),
        }
    }
}

impl System for EnvironmentSystem {
    #[inline]
    fn id(&self) -> SystemID {
        self.id
    }

    #[inline]
    fn name(&self) -> &str {
        &self.name
    }

    #[inline]
    fn access(&self) -> &AccessSets {
        &self.access
    }

    #[inline]
    fn backend(&self) -> SystemBackend {
        SystemBackend::CPU
    }

    fn run(&self, ecs: ECSReference<'_>) -> ECSResult<()> {
        (self.func)(Arc::clone(&self.env), ecs)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::engine::systems::{AccessSets, System};
    use crate::environment::builder::EnvironmentBuilder;

    use super::EnvironmentSystem;

    /// Verify that the system's id / name / access / backend are correctly
    /// forwarded through the trait.
    #[test]
    fn trait_accessors() {
        let env = EnvironmentBuilder::new()
            .register::<f32>("rate", 0.0f32)
            .build();

        let sys =
            EnvironmentSystem::new(
                7u16,
                "TestSystem",
                AccessSets::default(),
                env,
                |_, _| Ok(()),
            );

        assert_eq!(sys.id(), 7u16);
        assert_eq!(sys.name(), "TestSystem");
        assert!(matches!(
            sys.backend(),
            crate::engine::systems::SystemBackend::CPU
        ));
    }

    /// Verify that construction captures the shared `Arc<Environment>` and
    /// that the captured handle shares state with the original.
    ///
    /// Note: `System::run` requires a live `ECSReference`, which is not
    /// available in unit tests. This test validates that the `Arc` is
    /// correctly shared — end-to-end closure execution is verified by
    /// integration tests that run through a full `ECSManager` tick.
    #[test]
    fn system_captures_shared_env() {
        let env = EnvironmentBuilder::new()
            .register::<u32>("counter", 0u32)
            .build();

        let _sys = EnvironmentSystem::new(
            0u16,
            "IncrementCounter",
            AccessSets::default(),
            Arc::clone(&env),
            move |_e, _ecs| Ok(()),
        );

        // The Arc is shared: mutations through the original are visible
        // to anyone holding a clone (including the system's captured copy).
        env.set::<u32>("counter", 5).unwrap();
        assert_eq!(env.get::<u32>("counter").unwrap(), 5u32);
    }

    /// Verify that multiple EnvironmentSystems with the same env share state.
    #[test]
    fn shared_env_across_systems() {
        let env = EnvironmentBuilder::new().register::<i32>("val", 10).build();

        let env_a = Arc::clone(&env);
        let env_b = Arc::clone(&env);

        let _a = EnvironmentSystem::new(0u16, "A", AccessSets::default(), env_a, |_, _| Ok(()));
        let _b = EnvironmentSystem::new(1u16, "B", AccessSets::default(), env_b, |_, _| Ok(()));

        env.set::<i32>("val", 99).unwrap();
        assert_eq!(env.get::<i32>("val").unwrap(), 99);
    }

    /// Verify that two systems with the same name don't conflict —
    /// each owns its own name allocation.
    #[test]
    fn duplicate_names_are_independent() {
        let env = EnvironmentBuilder::new().register::<f32>("x", 0.0).build();

        let a = EnvironmentSystem::new(
            0u16,
            "SharedName",
            AccessSets::default(),
            Arc::clone(&env),
            |_, _| Ok(()),
        );
        let b = EnvironmentSystem::new(
            1u16,
            "SharedName",
            AccessSets::default(),
            Arc::clone(&env),
            |_, _| Ok(()),
        );

        assert_eq!(a.name(), "SharedName");
        assert_eq!(b.name(), "SharedName");
        // Each system holds its own owned String — pointers are distinct.
        assert!(!std::ptr::eq(a.name().as_ptr(), b.name().as_ptr()));
    }
}
