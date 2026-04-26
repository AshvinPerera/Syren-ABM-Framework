//! Top-level model composition API.

pub mod builder;
pub mod error;
pub mod model;
pub mod sub_scheduler;

pub use builder::ModelBuilder;
pub use error::ModelError;
pub use model::Model;
pub use sub_scheduler::SubScheduler;
