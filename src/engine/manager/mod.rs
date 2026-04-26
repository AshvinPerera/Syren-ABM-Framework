//! ECS world management and execution layer.
//!
//! This module defines the central orchestration layer of the ECS. It is
//! responsible for:
//!
//! * owning archetypes and component storage,
//! * coordinating entity movement between archetypes,
//! * managing deferred structural mutations,
//! * enforcing safe parallel iteration at runtime,
//! * providing controlled access to ECS state.
//!
//! ## Concurrency model
//!
//! The ECS uses **interior mutability** (`UnsafeCell`) to allow highly parallel
//! iteration while avoiding fine-grained locks in hot paths.
//!
//! Safety is enforced by **runtime mechanisms**:
//!
//! * **Phase discipline**
//!   - A global read/write phase lock prevents structural mutation during iteration.
//! * **IterationScope**
//!   - A global iteration counter forbids structural changes while iteration is active.
//! * **BorrowTracker**
//!   - Per-component runtime borrow tracking enforces Rust-like read/write rules.
//!
//! ## Structural mutation
//!
//! Structural changes (spawn, despawn, add/remove component) must be:
//!
//! * deferred via [`ECSReference::defer`], or
//! * performed inside an exclusive phase using [`ECSReference::with_exclusive`].
//!
//! Applying deferred commands is a global synchronization point and is
//! guaranteed not to overlap with iteration.
//!
//! ## Safety
//!
//! This module contains unsafe code for:
//!
//! * interior mutability (`UnsafeCell`),
//! * raw pointer component access,
//! * parallel execution using Rayon.
//!
//! Each unsafe block documents the invariants it relies on.
//! Violating these invariants results in undefined behaviour.

mod data;
mod ecs_manager;
mod ecs_reference;
mod iteration;
mod phase;
mod query_executor;
mod query_param;

pub use data::ECSData;
pub use ecs_manager::ECSManager;
pub use ecs_reference::ECSReference;
pub use query_param::{Read, Write};
