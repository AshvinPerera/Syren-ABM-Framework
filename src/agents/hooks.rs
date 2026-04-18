//! Lifecycle hooks for agent spawn and despawn events.
//!
//! ## Design
//!
//! Hooks are Boxed closures stored inside [`AgentTemplate`]. They are
//! **never** called inside `System::run`. Instead, they are invoked by the
//! model orchestration layer (e.g. `Model::tick`) after
//! `apply_deferred_commands` completes and the resolved entity handles are
//! available.
//!
//! Hook invocation is wired through
//! [`AgentRegistry::flush_spawn_hooks`](super::registry::AgentRegistry::flush_spawn_hooks),
//! which the model calls at the existing `apply_deferred_commands` sync
//! boundary.
//!
//! ## Constraints
//!
//! * Hooks **must not** be called while a query iteration is live.
//! * Hooks receive an [`ECSReference`] and an [`Entity`]; they may enqueue
//!   further [`Command`]s via [`ECSReference::defer`] but must not call
//!   [`ECSReference::with_exclusive`].
//! * Both hook types are `Send + Sync` so that [`AgentTemplate`] remains
//!   `Send + Sync`.

use crate::engine::entity::Entity;
use crate::engine::manager::ECSReference;

/// Called after an agent entity is resolved from a `Command::Spawn`.
///
/// Receives a shared [`ECSReference`] (for deferred commands or reads) and
/// the freshly resolved [`Entity`] handle.
///
/// # Timing
///
/// Invoked by `AgentRegistry::flush_spawn_hooks` **after**
/// `apply_deferred_commands` returns.  Never called inside a running system.
pub type SpawnHook = Box<dyn Fn(ECSReference<'_>, Entity) + Send + Sync>;

/// Called before a `Command::Despawn` is applied.
///
/// Receives an [`ECSReference`] and the entity about to be destroyed. The
/// hook may enqueue additional commands via [`ECSReference::defer`] but must
/// not perform immediate structural mutations.
///
/// # Timing
///
/// Invoked by `AgentRegistry::flush_despawn_hooks` before the despawn command
/// is applied.  Never called inside a running system.
pub type DespawnHook = Box<dyn Fn(ECSReference<'_>, Entity) + Send + Sync>;
