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
//! - **Enable parallel scheduling**
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
//! - Systems with conflicting writes are serialized relative to one another.
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
//! operate through [`ECSReference`], which provides controlled entry points into
//! ECS execution phases.
//!
//! Correctness is enforced at runtime via borrow tracking and execution-phase discipline; 
//! the scheduler optimizes parallelism.
//!
//! ## Intended Usage
//!
//! This module is intended to be used in conjunction with:
//! - the scheduler (`scheduler` module),
//! - query construction (`query` module),
//! - and deferred command processing (`manager` module).
//!
//! Together, these components form the execution layer of the ECS.

#[cfg(feature = "gpu")]
use crate::engine::types::GPUResourceID;
use crate::engine::types::{SystemID};
use crate::engine::component::{Signature};
use crate::engine::manager::ECSReference;
use crate::engine::error::{ECSResult};


/// Declares the component access set of a system.
#[derive(Clone, Debug, Default)]
pub struct AccessSets {
    /// Components read by the system.
    pub read: Signature,
    /// Components written by the system.
    pub write: Signature,
}

impl AccessSets {
    /// Returns `true` if this access set conflicts with another.
    #[inline]
    pub fn conflicts_with(&self, other: &AccessSets) -> bool {
        // Conflicts if: (W ∩ W) or (W ∩ R) or (R ∩ W)
        let mut w_and_w = false;
        let mut w_and_r = false;
        let mut r_and_w = false;

        for ((a_w, a_r), (b_w, b_r)) in self.write.components.iter().zip(self.read.components.iter())
            .zip(other.write.components.iter().zip(other.read.components.iter()))
        {
            if (a_w & b_w) != 0 { w_and_w = true; }
            if (a_w & b_r) != 0 { w_and_r = true; }
            if (a_r & b_w) != 0 { r_and_w = true; }
            if w_and_w || w_and_r || r_and_w { return true; }
        }
        false
    }
}

/// Execution backend for a system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SystemBackend {
    /// Standard Rust / Rayon execution on CPU.
    CPU,
    /// GPU dispatch.
    GPU
}

/// GPU capability trait (feature-gated).
/// A GPU system is still a `System`, but additionally provides WGSL.

#[cfg(feature = "gpu")]
pub trait GpuSystem {
    /// WGSL source
    fn shader(&self) -> &'static str;

    /// Entry point name (default "main")
    fn entry_point(&self) -> &'static str { "main" }

    /// Workgroup size (default 256)
    fn workgroup_size(&self) -> u32 { 256 }

    /// GPU resources read or used by this kernel.
    fn uses_resources(&self) -> &[GPUResourceID] { &[] }

    /// GPU resources that this kernel may write.
    fn writes_resources(&self) -> &[GPUResourceID] { &[] }
}

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
    /// Human-readable name (used for debugging/profiling).
    #[inline]
    fn name(&self) -> &'static str {
        std::any::type_name_of_val(self)
    }

    /// Returns the unique identifier of this system.
    fn id(&self) -> SystemID;

    /// Returns the component access sets required by this system.
    fn access(&self) -> AccessSets;

    /// Returns which backend this system should run on.
    /// Defaults to [`SystemBackend::Cpu`].
    #[inline]
    fn backend(&self) -> SystemBackend {
        SystemBackend::CPU
    }    

    /// Executes the system logic against the ECS world.
    fn run(&self, world: ECSReference<'_>) -> ECSResult<()>;

    /// GPU capability hook.
    /// A GPU system should override this to return `Some(self)` (as `&dyn GpuSystem`)
    /// and also return `SystemBackend::GPU` from `backend()`.
    #[cfg(feature = "gpu")]
    #[inline]
    fn gpu(&self) -> Option<&dyn GpuSystem> {
        None
    }
    
}

/// A concrete [`System`] backed by a function or closure.
///
/// `FnSystem` allows systems to be defined inline using a function or
/// closure, without requiring a custom system type.
///
/// The function must return `ECSResult<()>` so that
/// execution failures can be propagated through the scheduler.
pub struct FnSystem<F>
where
    F: Fn(ECSReference<'_>) -> ECSResult<()>
        + Send
        + Sync
        + 'static,
{
    id: SystemID,
    name: &'static str,
    access: AccessSets,
    f: F,
}

impl<F> FnSystem<F>
where
    F: Fn(ECSReference<'_>) -> ECSResult<()>
        + Send
        + Sync
        + 'static,
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
    F: Fn(ECSReference<'_>) -> ECSResult<()>
        + Send
        + Sync
        + 'static,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn id(&self) -> SystemID {
        self.id
    }

    fn access(&self) -> AccessSets {
        self.access.clone()
    }

    fn run(&self, world: ECSReference<'_>) -> ECSResult<()> {
        (self.f)(world)
    }
}
