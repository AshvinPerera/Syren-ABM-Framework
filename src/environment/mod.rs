//! Simulation-wide typed parameter store.
//!
//! The `environment` module provides [`Environment`]: a named, typed key-value
//! store for parameters that are shared across all agents and systems in a
//! simulation. It is the idiomatic equivalent of FLAME GPU's `Environment`
//! object.
//!
//! ## Typical uses
//!
//! - Interest rates, tax rates, reproduction probabilities
//! - World dimensions, grid resolution, time-step size
//! - Any scalar or small struct that every system needs to read
//!
//! ## Enabling
//!
//! Add `--features environment` to your `cargo` invocation. The module is a
//! strict no-op (zero compile cost) when the feature is absent.
//!
//! ## Quick-start
//!
//! ```text
//! use std::sync::Arc;
//! use abm_framework::environment::{Environment, EnvironmentBuilder, EnvironmentSystem};
//! use abm_framework::{AccessSets, Scheduler, Stage};
//!
//! // 1. Declare parameters.
//! let env: Arc<Environment> = EnvironmentBuilder::new()
//!     .register::<f32>("interest_rate", 0.05)
//!     .register::<u32>("world_width", 100)
//!     .build();
//!
//! // 2. Optionally wrap mutation logic as a schedulable system.
//! let interest_sys = EnvironmentSystem::new(
//!     42u16,
//!     "CompoundInterest",
//!     AccessSets::default(),
//!     Arc::clone(&env),
//!     |e, _ecs| {
//!         let r: f32 = e.get("interest_rate")?;
//!         e.set("interest_rate", r * 1.001)?;
//!         Ok(())
//!     },
//! );
//!
//! // 3. Register with the scheduler.
//! // scheduler.add_system(Stage::Update, Box::new(interest_sys));
//! ```
//!
//! ## GPU uniform buffers
//!
//! When `--features gpu` is also active, any subset of `Pod` parameters can be
//! packed into a `wgpu` uniform buffer:
//!
//! ```text
//! use abm_framework::environment::uniform::EnvUniformBuffer;
//!
//! let ubuf = EnvUniformBuffer::builder(Arc::clone(&env))
//!     .include::<f32>("interest_rate")?
//!     .include::<u32>("world_width")?
//!     .build();
//! ```
//!
//! ## Design notes
//!
//! * Schema is **frozen** after `EnvironmentBuilder::build()` - no new keys.
//! * Values can be updated at any point via `Environment::set`.
//! * Thread-safe: each key has its own `RwLock`.
//! * Dirty tracking powers GPU upload without full buffer re-scan.

pub mod boundary;
pub mod builder;
pub mod error;
pub mod handle;
pub mod store;
pub mod system;

#[cfg(feature = "gpu")]
pub mod uniform;

pub use boundary::EnvironmentBoundary;
pub use builder::EnvironmentBuilder;
pub use error::{EnvironmentError, EnvironmentResult};
pub use handle::EnvKey;
pub use store::Environment;
pub use system::EnvironmentSystem;

#[cfg(feature = "gpu")]
pub use uniform::{EnvPod, EnvUniformBuffer};
