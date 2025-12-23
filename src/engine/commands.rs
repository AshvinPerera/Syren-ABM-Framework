//! # Commands
//!
//! This module defines deferred commands used to mutate the ECS world.
//!
//! ## Purpose
//! Commands provide an explicit, ordered representation of structural world
//! mutations such as entity creation, destruction, and component addition or
//! removal.
//!
//! Rather than mutating archetypes directly during system execution, systems
//! emit `Command` values that are applied later at a synchronization point.
//! This enables safe parallel system execution and deterministic world updates.
//!
//! ## Design
//! - Commands are plain data describing *what* change should occur, not *how*.
//! - Execution is handled by a centralized command processor.
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
/// recorded and executed at a later synchronization point.
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
    /// ## Behavior
    /// - Removes the entity from its archetype.
    /// - Releases the entity handle.
    /// - Performs swap-remove on component storage as needed
    Despawn { 
        /// Entity to be removed from the world.
        entity: Entity 
    },

    /// Adds a component to an existing entity.
    ///
    /// ## Behavior
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
    /// ## Behavior
    /// - Moves the entity to a new archetype that excludes the component.
    /// - The removed component value is dropped.    
    Remove { 
        /// Target entity losing the component.
        entity: Entity, 
        /// Identifier of the component type to remove.
        component_id: ComponentID
    },
}
