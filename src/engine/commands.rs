//! # Commands
//!
//! This module defines deferred commands used to mutate the ECS world.
//!
//! ## Purpose
//! Commands provide an explicit, ordered representation of structural world
//! mutations such as entity creation, destruction, and component addition,
//! removal, or in-place overwrite.
//!
//! Rather than mutating archetypes directly during system execution, systems
//! emit `Command` values that are applied later at a synchronisation point.
//! This enables safe parallel system execution and deterministic world updates.
//!
//! ## Design
//! - Commands are plain data describing *what* change should occur, not *how*.
//! - Execution is handled by a centralised command processor.
//! - Commands may cause archetype transitions, including component row moves.
//!
//! ## Invariants
//! - Commands must be executed in the order they are recorded.
//! - Target entities must exist at execution time unless being spawned.
//! - Component identifiers and values must be valid and registered.
//!
//! ## Safety
//! Command execution may involve structural changes such as archetype moves
//! and swap-removes. Correctness depends on commands being applied atomically
//! and outside active system borrows.

use std::any::Any;

use crate::engine::entity::Entity;
use crate::engine::types::ComponentID;
use crate::engine::component::Bundle;


/// Represents a deferred ecs mutation command.
///
/// ## Purpose
/// `Command` values describe structural changes to the ECS that are
/// recorded and executed at a later synchronisation point.
///
/// This decouples system execution from direct ecs mutation, allowing
/// safe parallelism and deterministic ordering of entity and component changes.
///
/// ## Design
/// Commands are typically produced by systems and consumed by a command
/// processor that applies them to archetypes and entity storage.
///
/// ## Invariants
/// - Commands must be applied in the order they are issued.
/// - Entity and component identifiers must be valid at execution time.
/// - Component values must match the registered component type.

pub enum Command {
    /// Spawns a new entity into a specific archetype.
    Spawn {
        /// Data bundle for the new entity.
        bundle: Bundle
    },

    /// Despawns an existing entity.
    ///
    /// ## Behaviour
    /// - Removes the entity from its archetype.
    /// - Releases the entity handle.
    /// - Performs swap-remove on component storage as needed
    Despawn {
        /// Entity to be removed from the world.
        entity: Entity
    },

    /// Adds a component to an existing entity.
    ///
    /// ## Behaviour
    /// - Moves the entity to a new archetype that includes the added component.
    /// - The provided value is inserted into the destination archetype.    
    Add {
        /// Target entity receiving the component.        
        entity: Entity,
        /// Identifier of the component type to add.
        component_id: ComponentID,
        /// Component value to insert.
        ///
        /// Must match the registered component type for `component_id`.
        value: Box<dyn Any + Send>
    },

    /// Removes a component from an existing entity.
    ///
    /// ## Behaviour
    /// - Moves the entity to a new archetype that excludes the component.
    /// - The removed component value is dropped.    
    Remove {
        /// Target entity losing the component.
        entity: Entity,
        /// Identifier of the component type to remove.
        component_id: ComponentID
    },

    /// Overwrites a component value on a specific entity **in place**.
    ///
    /// Unlike [`Add`](Command::Add), `Set` does **not** perform an archetype
    /// transition. The entity's archetype must already contain `component_id`;
    /// if it does not, the command is rejected with an error at apply time.
    ///
    /// ## Behaviour
    /// - No archetype transition. Fails if the entity's current archetype does
    ///   not contain the component.
    /// - The concrete type of `value` is checked against the component registry
    ///   at apply time; a mismatch returns an error.
    /// - Applied during `apply_deferred_commands` under the exclusive phase lock.
    ///
    /// ## Visibility
    /// A `Set` emitted in stage N is only visible to systems in stage N+1 or
    /// later — identical to the visibility rules for `Add` / `Remove`. There
    /// is no mid-stage visibility.
    ///
    /// ## Use case
    /// Rare, low-frequency targeted writes (bankruptcy declarations,
    /// administrative fiat, one-off resets) where emitting a message would be
    /// overkill. **Not** a substitute for messaging in hot paths — the boxing
    /// cost dominates above a few thousand calls per tick.
    Set {
        /// Target entity whose component value is being overwritten.
        entity: Entity,
        /// Identifier of the component type to overwrite.
        component_id: ComponentID,
        /// New component value.
        ///
        /// The dynamic type must exactly match the registered storage type for
        /// `component_id`. A mismatch is detected at apply time and returns
        /// `ExecutionError::InternalExecutionError`.
        value: Box<dyn Any + Send>,
    },
}
