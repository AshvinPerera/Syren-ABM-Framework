//! Domain-facing agent API layered over the raw ECS.
//!
//! Enable with `--features agents`.
//!
//! ## What this module provides
//!
//! | Type | Role |
//! |------|------|
//! | [`AgentTemplate`] | Named, typed agent class definition |
//! | [`AgentTemplateBuilder`] | Fluent builder for templates |
//! | [`AgentSpawner`] | Deferred spawn builder with per-field overrides |
//! | [`AgentHandle`] | Single-entity typed component access (CPU lookups only) |
//! | [`AgentRegistry`] | World-scoped template store owned by the model |
//! | [`AgentError`] / [`AgentResult`] | Module error type and result alias |
//!
//! ## Error boundary
//!
//! * Inside this module, all fallible functions return `Result<T, AgentError>`.
//! * At scheduler/manager boundaries, convert with `map_err(ECSError::from)?`.

pub mod error;
pub mod handle;
pub mod hooks;
pub mod registry;
pub mod spawner;
pub mod template;

pub use error::{AgentError, AgentResult};
pub use handle::{AgentHandle, AgentHandleError};
pub use registry::AgentRegistry;
pub use spawner::AgentSpawner;
pub use template::{AgentTemplate, AgentTemplateBuilder, DefaultFactory};
