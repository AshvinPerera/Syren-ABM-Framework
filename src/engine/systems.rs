//! ECS System Abstractions
//!
//! This module defines the core *system execution model* used by the engine.
//!
//! A **system** is a unit of logic that operates over the ECS world. Systems:
//! - declare which components they read and write,
//! - are scheduled based on access conflicts,
//! - may be executed sequentially or in parallel,
//! - operate through a controlled [`ECSReference`] rather than direct world access.
//!
//! ## Design Goals
//!
//! The system abstraction is designed to:
//!
//! - **Enable safe parallelism**
//!   by statically declaring component access (`read` / `write`) via [`AccessSets`].
//!
//! - **Decouple logic from storage**
//!   so systems operate on *views* of the world rather than concrete data layouts.
//!
//! - **Support lightweight system definitions**
//!   through function-backed systems (`FnSystem`) without requiring boilerplate
//!   types for every system.
//!
//! ## Scheduling Model
//!
//! Systems are scheduled by the engine using their declared access sets:
//!
//! - Systems with *non-conflicting* access may run in parallel.
//! - Systems with conflicting writes are serialized.
//! - Ordering is stabilized using system IDs.
//!
//! The scheduler is free to group systems into execution stages based on this
//! information.
//!
//! ## System Trait
//!
//! The [`System`] trait defines the minimal interface required for execution:
//!
//! - [`System::id`] provides a stable identifier.
//! - [`System::access`] declares component read/write requirements.
//! - [`System::run`] executes the system logic.
//!
//! All systems must be `Send + Sync` to allow execution on worker threads.
//!
//! ## Function-backed Systems
//!
//! [`FnSystem`] provides a convenient way to define systems using closures or
//! functions. This is the preferred mechanism for most gameplay and simulation
//! logic, as it avoids unnecessary type definitions while remaining fully
//! schedulable and parallel-safe.
//!
//! ## Thread Safety
//!
//! Systems do **not** receive direct mutable access to the world. Instead, they
//! operate through [`ECSReference`], which enforces the access guarantees declared
//! by the system. This prevents data races even under parallel execution.
//!
//! ## Intended Usage
//!
//! This module is intended to be used in conjunction with:
//! - the scheduler (`scheduler` module),
//! - query construction (`query` module),
//! - and deferred command processing (`manager` module).
//!
//! Together, these components form the execution layer of the ECS.

use crate::engine::types::{SystemID, AccessSets};
use crate::engine::manager::ECSReference;


/// A unit of executable logic operating on the ECS world.
///
/// A `System` represents a scheduled computation that:
/// - declares which components it reads and writes,
/// - can be ordered and parallelized based on access conflicts,
/// - is executed with a shared reference to the ECS world.
///
/// Systems must be `Send + Sync` so they can be scheduled and executed
/// in parallel across threads.

pub trait System: Send + Sync {
    /// Returns the unique identifier of this system.
    fn id(&self) -> SystemID;

    /// Returns the component access sets required by this system.
    fn access(&self) -> AccessSets;

    /// Executes the system logic against the ECS world.
    fn run(&self, world: ECSReference<'_>);
}

/// A concrete [`System`] backed by a function or closure.
///
/// `FnSystem` allows systems to be defined inline using a function or
/// closure, without requiring a custom system type.
///
/// It stores:
/// - a system ID,
/// - a human-readable name,
/// - declared component access,
/// - and the executable function itself.

pub struct FnSystem<F>
where
    F: Fn(ECSReference<'_>) + Send + Sync + 'static,
{
    id: SystemID,
    name: &'static str,
    access: AccessSets,
    f: F,
}

impl<F> FnSystem<F>
where
    F: Fn(ECSReference<'_>) + Send + Sync + 'static,
{
    /// Creates a new function-backed system.
    ///
    /// # Parameters
    /// - `id`: Unique identifier for the system.
    /// - `name`: Human-readable name, useful for debugging and profiling.
    /// - `access`: Declared component access used for scheduling.
    /// - `f`: The function or closure executed when the system runs.
    pub fn new(
        id: SystemID,
        name: &'static str,
        access: AccessSets,
        f: F,
    ) -> Self {
        Self { id, name, access, f }
    }

    /// Returns the human-readable name of this system.
    pub fn name(&self) -> &'static str {
        self.name
    }
}

impl<F> System for FnSystem<F>
where
    F: Fn(ECSReference<'_>) + Send + Sync + 'static,
{
    fn id(&self) -> SystemID {
        self.id
    }

    fn access(&self) -> AccessSets {
        self.access.clone()
    }

    fn run(&self, world: ECSReference<'_>) {
        (self.f)(world)
    }
}
