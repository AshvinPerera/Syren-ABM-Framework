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

use crate::engine::component::Bundle;
use crate::engine::component::Signature;
use crate::engine::entity::Entity;
use crate::engine::types::{AgentTemplateId, ComponentID};

/// One component column in a dynamically-typed spawn batch.
pub struct BatchColumn {
    /// Component identifier for this column.
    pub component_id: ComponentID,
    /// One value per entity in the batch.
    pub values: Vec<Box<dyn Any + Send>>,
}

/// Dynamically-typed batch payload for spawning many entities with one signature.
pub struct SpawnBatch {
    /// Number of entities to spawn.
    pub count: usize,
    /// Component signature shared by all spawned entities.
    pub signature: Signature,
    /// Component columns. Each column must contain `count` values.
    pub columns: Vec<BatchColumn>,
}

/// Entity created by a deferred spawn command, plus optional producer tag.
///
/// Untagged spawns are ordinary ECS entity creation. Tagged spawns are used by
/// higher-level model code that needs post-spawn lifecycle hooks once the
/// concrete [`Entity`] handle is known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpawnEvent {
    /// Newly-created entity.
    pub entity: Entity,
    /// Optional logical producer tag.
    pub tag: Option<String>,
    /// Optional compact agent template identifier.
    pub template_id: Option<AgentTemplateId>,
}

impl SpawnEvent {
    /// Creates an untagged spawn event.
    #[inline]
    pub fn untagged(entity: Entity) -> Self {
        Self {
            entity,
            tag: None,
            template_id: None,
        }
    }

    /// Creates a tagged spawn event.
    #[inline]
    pub fn tagged(entity: Entity, tag: String) -> Self {
        Self {
            entity,
            tag: Some(tag),
            template_id: None,
        }
    }

    /// Creates a template-id tagged spawn event.
    #[inline]
    pub fn template_tagged(entity: Entity, template_id: AgentTemplateId) -> Self {
        Self {
            entity,
            tag: None,
            template_id: Some(template_id),
        }
    }
}

/// Entity removed by a deferred despawn command, plus optional producer tag.
///
/// Tagged despawns allow model-owned lifecycle registries to run cleanup hooks
/// after the ECS has accepted and applied the structural removal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DespawnEvent {
    /// Removed entity.
    pub entity: Entity,
    /// Optional logical producer tag.
    pub tag: Option<String>,
    /// Optional compact agent template identifier.
    pub template_id: Option<AgentTemplateId>,
}

impl DespawnEvent {
    /// Creates an untagged despawn event.
    #[inline]
    pub fn untagged(entity: Entity) -> Self {
        Self {
            entity,
            tag: None,
            template_id: None,
        }
    }

    /// Creates a tagged despawn event.
    #[inline]
    pub fn tagged(entity: Entity, tag: String) -> Self {
        Self {
            entity,
            tag: Some(tag),
            template_id: None,
        }
    }

    /// Creates a template-id tagged despawn event.
    #[inline]
    pub fn template_tagged(entity: Entity, template_id: AgentTemplateId) -> Self {
        Self {
            entity,
            tag: None,
            template_id: Some(template_id),
        }
    }
}

/// Group of lifecycle events produced by one template-id tagged batch command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemplateLifecycleBatch {
    /// Agent template that produced the lifecycle events.
    pub template_id: AgentTemplateId,
    /// Entities produced or removed by the command.
    pub entities: Vec<Entity>,
}

/// Lifecycle events produced by one deferred command drain.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommandEvents {
    /// Entities created during the drain.
    pub spawned: Vec<SpawnEvent>,
    /// Entities removed during the drain.
    pub despawned: Vec<DespawnEvent>,
    /// Spawned entities grouped by compact template ID.
    pub spawned_batches: Vec<TemplateLifecycleBatch>,
    /// Despawned entities grouped by compact template ID.
    pub despawned_batches: Vec<TemplateLifecycleBatch>,
}

impl CommandEvents {
    /// Returns true if no lifecycle events were produced.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.spawned.is_empty()
            && self.despawned.is_empty()
            && self.spawned_batches.is_empty()
            && self.despawned_batches.is_empty()
    }
}

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
        bundle: Bundle,
    },

    /// Spawns a new entity and records a logical producer tag.
    ///
    /// The tag is returned in [`SpawnEvent`] after the command is applied. The
    /// core ECS does not interpret it; model-level code may use it to
    /// dispatch lifecycle hooks.
    SpawnTagged {
        /// Data bundle for the new entity.
        bundle: Bundle,
        /// Producer tag associated with this spawn.
        tag: String,
    },

    /// Spawns many entities with one shared signature and records a template ID.
    SpawnBatchTagged {
        /// Batch payload for the new entities.
        batch: SpawnBatch,
        /// Agent template associated with this batch.
        template_id: AgentTemplateId,
    },

    /// Despawns an existing entity.
    ///
    /// ## Behaviour
    /// - Removes the entity from its archetype.
    /// - Releases the entity handle.
    /// - Performs swap-remove on component storage as needed
    Despawn {
        /// Entity to be removed from the world.
        entity: Entity,
    },

    /// Despawns an existing entity and records a logical producer tag.
    ///
    /// The tag is returned in [`DespawnEvent`] after the command is applied.
    /// The core ECS does not interpret it; model-level code may use it to
    /// dispatch lifecycle hooks.
    DespawnTagged {
        /// Entity to be removed from the world.
        entity: Entity,
        /// Producer tag associated with this despawn.
        tag: String,
    },

    /// Despawns many entities associated with the same agent template.
    DespawnBatchTagged {
        /// Entities to remove from the world.
        entities: Vec<Entity>,
        /// Agent template associated with this batch.
        template_id: AgentTemplateId,
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
        value: Box<dyn Any + Send>,
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
        component_id: ComponentID,
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
    /// - Applied during `apply_deferred_commands` while the world is held exclusively.
    ///
    /// ## Visibility
    /// A `Set` emitted in stage N is only visible to systems in stage N+1 or
    /// later - identical to the visibility rules for `Add` / `Remove`. There
    /// is no mid-stage visibility.
    ///
    /// ## Use case
    /// Rare, low-frequency targeted writes (bankruptcy declarations,
    /// administrative fiat, one-off resets) where emitting a message would be
    /// overkill. **Not** a substitute for messaging in hot paths - the boxing
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
