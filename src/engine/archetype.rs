use crate::types::{
    ArchetypeID, 
    ShardID,
    ChunkID,
    RowID, 
    CHUNK_CAP, 
    ComponentID, 
    COMPONENT_CAP,
    Signature,
    DynamicBundle
};
use crate::storage::{
    TypeErasedAttribute,
    Attribute
};
use crate::entity::{
    Entity, 
    EntityLocation, 
    EntityShards
};
use crate::component::{ 
    component_id_of_type_id,
    get_component_storage_factory
};
use crate::error::{
    SpawnError
};

//! # Archetype
//!
//! The `Archetype` type represents a storage container for all entities that
//! share an identical component signature. Each archetype owns a set of
//! component attributes, arranged in a column-major layout, where every attribute
//! stores values of a single component type.
//!
//! ## Operations
//!
//! Archetypes support:
//!
//! - Spawning entities into a new row across all component attribute.
//! - Removing entities and maintaining compactness through `swap_remove`.
//! - Moving entities between archetypes when their component signatures change.
//! - Borrowing chunk views for system execution with component-level read/write access.
//!
//! These operations maintain the required alignment invariants and ensure that
//! component data remains consistent with entity metadata.
//!
//! ## Invariants
//!
//! - All component attributes in an archetype share identical row counts.
//! - Component presence is determined solely by the archetype's `Signature`.
//! - Any row movement (via `push_from`, `swap_remove`, or despawn) must update
//!   `entity_positions` to remain consistent with component storage.
//!
//! Violating these invariants results in undefined entity/component alignment.
//!
//! ## Error Conditions
//!
//! Archetype operations produce `SpawnError` values when invariants cannot be
//! upheld, such as:
//!
//! - Missing components in a bundle during spawn.
//! - Storage push failures.
//! - Misaligned writes during spawn or move.
//! - Despawning stale or untracked entities.
//!
//! ## Summary
//!
//! `Archetype` provides the core high-performance storage representation for
//! entities in the ECS. Its chunked, columnar design ensures locality, fast
//! traversal, predictable memory behavior, and compatibility with CPU/GPU
//! parallelization strategies.

pub struct ChunkBorrow<'a> {
    pub length: usize,
    pub reads: Vec<&'a [u8]>,
    pub writes: Vec<*mut u8>,
    pub _marker: std::marker::PhantomData<&'a mut u8>
}

#[derive(Debug)]
pub struct Archetype {
    archetype_id: ArchetypeID,
    components: Vec<Option<Box<dyn TypeErasedAttribute>>>,
    signature: Signature,
    length: usize,
    entity_positions: Vec<Vec<Option<Entity>>>
}

impl Archetype {

    /// Creates a new empty `Archetype` with the given identifier.
    ///
    /// ## Purpose
    /// Initializes component column storage, the signature bitset, entity tracking
    /// buffers, and internal counters.
    ///
    /// ## Behavior
    /// - Allocates `COMPONENT_CAP` component slots, all initially empty.
    /// - Initializes an empty `Signature`.
    /// - No component columns are allocated until explicitly inserted.
    ///
    /// ## Invariants
    /// The archetype contains no entities upon creation.

    pub fn new(archetype_id: ArchetypeID) -> Self {
            Self {
                archetype_id,
                components: vec![None; COMPONENT_CAP], // fixed-size component attribute slots
                signature: Signature::default(),
                length: 0,
                entity_positions: Vec::new(), // grows chunk-by-chunk on demand
            }
        }

    /// Returns the number of active entities stored in the archetype.
    ///
    /// ## Notes
    /// This reflects logical count only; physical chunk storage may contain unused rows.

    pub fn length(&self) -> usize {
        self.length
    }

    /// Returns the `ArchetypeID` associated with this archetype.
    ///
    /// ## Notes
    /// This value is stable for the lifetime of the archetype.

    pub fn archetype_id(&self) -> ArchetypeID {
        self.archetype_id 
    }

    /// Ensures that `entity_positions` contains at least `chunk_count` chunks.
    ///
    /// ## Purpose
    /// Expands chunk metadata storage to match component column allocations.
    ///
    /// ## Invariants
    /// - Each added chunk contains exactly `CHUNK_CAP` rows.
    /// - Does not allocate component data; only entity metadata.

    fn ensure_capacity(&mut self, chunk_count: usize) {
        // Ensure entity_positions always has a slot for every allocated chunk.
        while self.entity_positions.len() < chunk_count {
            self.entity_positions.push(vec![None; CHUNK_CAP]);
        }
    }

    /// Returns mutable references to matching component columns in this archetype
    /// and another archetype.
    ///
    /// ## Purpose
    /// Used when transferring a row between archetypes.
    ///
    /// ## Behavior
    /// Returns `None` if either archetype is missing the requested component.
    ///
    /// ## Invariants
    /// Both archetypes must share the component for a row transfer to be valid.

    #[inline]
    fn get_component_pair_mut<'a>(
        &'a mut self,
        other: &'a mut Archetype,
        component_id: ComponentID
    ) -> Option<(&'a mut Box<dyn TypeErasedAttribute>, &'a mut Box<dyn TypeErasedAttribute>)> {
        // Returns matching component columns in both archetypes; used for row movement.
        if self.archetype_id == other.archetype_id { return None; }
        let component_a = self.components[component_id as usize].as_mut()?;
        let component_b = other.components[component_id as usize].as_mut()?;
        Some((component_a, component_b))
    }

    /// Guarantees that a component attribute exists for the given `component_id`.
    ///
    /// ## Behavior
    /// - Allocates a new column using the provided factory if not already present.
    /// - Marks the component bit in the signature.
    ///
    /// ## Invariants
    /// Attribute allocation and signature must remain consistent.

    #[inline]
    pub fn ensure_component(&mut self, component_id: ComponentID, factory: impl FnOnce() -> Box<dyn TypeErasedAttribute>) -> Result<(), SpawnError>{
        // Lazily creates the column for a component type.
        let index = component_id as usize;
        if index >= COMPONENT_CAP { return Err(SpawnError::InvalidComponentId); }

        if self.components[index].is_none() {
            self.components[index] = Some(factory());
            self.signature.set(component_id);
        }
        Ok(())
    }

    /// Returns `true` if the archetype contains the specified component.
    ///
    /// ## Notes
    /// This checks the signature only; does not inspect the attribute buffer.

    #[inline]
    pub fn has(&self, component_id: ComponentID) -> bool {
        self.signature.has(component_id)
    }

    /// Returns an immutable reference to the component attribute for `component_id`,
    /// if present.
    ///
    /// ## Failure
    /// Returns `None` when the component is not part of the signature.

    #[inline]
    pub fn component(&self, component_id: ComponentID) -> Option<&dyn TypeErasedAttribute> {
        self.components.get(component_id as usize).and_then(|o| o.as_deref())
    }

    /// Returns a mutable reference to the component attribute for `component_id`,
    /// if present.
    ///
    /// ## Failure
    /// Returns `None` when the component is not part of the signature.
    ///
    /// ## Safety
    /// Caller must ensure no aliasing occurs with simultaneous borrows.

    #[inline]
    pub fn component_mut(&mut self, component_id: ComponentID) -> Option<&mut dyn TypeErasedAttribute> {
        self.components.get_mut(component_id as usize).and_then(|o| o.as_deref_mut())
    } 

    /// Computes how many chunks are required to store all active rows.
    ///
    /// ## Behavior
    /// - Returns `0` if no entities exist.
    /// - Otherwise computes `(length - 1) / CHUNK_CAP + 1`.

    pub fn chunk_count(&self) -> usize {
        if self.length == 0 {
            0
        } else {
            ((self.length - 1) / CHUNK_CAP) + 1 // last chunk may be partially full
        }
    }

    /// Returns the number of valid rows in the specified chunk.
    ///
    /// ## Behavior
    /// - Returns `0` if the chunk is unused.
    /// - Returns `CHUNK_CAP` for fully populated chunks.
    /// - Returns remaining entity count for the final partial chunk.
    ///
    /// ## Invariants
    /// Must reflect row count across all component attributes.

    pub fn chunk_valid_length(&self, chunk_index: usize) -> usize {
        // Returns how many rows in a chunk contain valid entities.
        if self.length == 0 || chunk_index > (self.length - 1) / CHUNK_CAP {
            0
        } else if chunk_index < (self.length - 1) / CHUNK_CAP {
            CHUNK_CAP
        } else {
            let used = self.length % CHUNK_CAP;
            if used == 0 { CHUNK_CAP } else { used } // possibly partial last chunk
        }
    }

    /// Inserts an empty component attribute into the archetype.
    ///
    /// ## Purpose
    /// Used when constructing archetypes from predefined type lists.
    ///
    /// ## Behavior
    /// - Fails if the index exceeds `COMPONENT_CAP`.
    /// - Assumes the attribute did not previously exist.
    /// - Sets the signature bit for the component.
    ///
    /// ## Invariants
    /// Component attributes must be added only before entities are inserted.

    pub fn insert_empty_component(&mut self, component_id: ComponentID, component: Box<dyn TypeErasedAttribute>) {
        let index = component_id as usize;
        if index >= COMPONENT_CAP {
            panic!("component_id out of range for COMPONENT_CAP.");
        }

        // Only safe to insert into an empty slot; archetype signature must match storage layout.
        let slot = &mut self.components[index];
        debug_assert!(slot.is_none(), "the component is already present.");
        *slot = Some(component);
        self.signature.set(component_id);
    }

    /// Removes a component attribute from an empty archetype.
    ///
    /// ## Behavior
    /// - Panics if the archetype still contains entities.
    /// - Clears the signature bit for the component.
    /// - Returns the removed attribute if present.
    ///
    /// ## Invariants
    /// Removing attributes in a populated archetype would break row alignment.

    pub fn remove_component(&mut self, component_id: ComponentID) -> Result<Option<Box<dyn TypeErasedAttribute>>, SpawnError> {
        // Components cannot be removed while entities exist—would break row alignment.
        if self.length > 0 {
            return Err(SpawnError::ArchetypeNotEmpty);
        } 

        let index = component_id as usize;
        let taken = self.components.get_mut(index)?.take();
        if taken.is_some() {
            self.signature.clear(component_id); // signature always matches stored columns
        }
        Ok(taken)
    }

    /// Moves an entity's component row from this archetype to another.
    ///
    /// ## Purpose
    /// Used when an entity changes signatures (adding or removing components).
    ///
    /// ## Behavior
    /// - Copies shared components to the destination.
    /// - Handles optional newly added component insertion.
    /// - Updates both archetypes’ `entity_positions`.
    /// - Applies `swap_remove` in the source to maintain compactness.
    ///
    /// ## Invariants
    /// - Destination must contain all components being moved.
    /// - All component attributes must agree on the row index of the moved entity.
    /// - Row alignment must be preserved across all participating components.
    ///
    /// ## Failure
    /// Panics internally if invariants are violated (storage inconsistency).

    pub fn move_row_to_archetype(
        &mut self,
        destination: &mut Archetype,
        shards: &EntityShards,
        entity: Entity,
        source_chunk: ChunkID,
        source_row: RowID,
        added_component: Option<(ComponentID, Box<dyn std::any::Any>)>,
    ) -> (ChunkID, RowID) {
        // For the first moved component, record where the row landed.
        let mut first_move_destination: Option<(ChunkID, RowID)> = None;
        // For swap_remove behavior: record if any component swapped from the last element.
        let mut first_swap_information: Option<(ChunkID, RowID)> = None;

        for component_id in self.signature.iterate_over_components() {
            if !destination.signature.has(component_id) {
                continue; // skip components not in destination archetype
            }

            // both archetypes must contain this component column.
            let (destination_component, source_component) = {
                let source_component = self.components[component_id as usize]
                    .as_mut()
                    .expect("source archetype must have this component");
                let destination_component = destination.components[component_id as usize]
                    .as_mut()
                    .expect("destination archetype must have this component");
                (destination_component, source_component)
            };

            let ((destination_chunk, destination_row), moved_from_last) =
                destination_component.push_from(source_component, source_chunk, source_row)
                    .expect("push_from failed to move component");;
            
            // Record the storage location where this new row lives.
            if first_move_destination.is_none() {
                first_move_destination = Some((destination_chunk, destination_row));
            }
            // Record swap_remove metadata; all components must agree.
            if first_swap_information.is_none() {
                first_swap_information = moved_from_last;
            }
        }

        // Handle new component added during move (component insertion operation).
        if let Some((added_component_id, value)) = added_component {
            let destination_component = destination.components[added_component_id as usize]
                .as_mut()
                .expect("destination must have the newly added component");

            let (added_chunk, added_row) = destination_component.push_dyn(value);

            // All columns must share identical row placement for the entity.
            if let Some((destination_chunk, destination_row)) = first_move_destination {
                debug_assert_eq!(
                    (added_chunk, added_row),
                    (destination_chunk, destination_row),
                    "added component storage row must match existing moved row"
                );
            } else {
                // If no other component existed, this defines the row.
                first_move_destination = Some((added_chunk, added_row));
            }
        }

        if first_move_destination.is_none() && added_component.is_none() {
            // handle empty move safely
            let destination_chunk = (destination.length / CHUNK_CAP) as ChunkID;
            let destination_row = (destination.length % CHUNK_CAP) as RowID;
            destination.ensure_capacity(destination_chunk as usize + 1);
            destination.entity_positions[destination_chunk as usize][destination_row as usize] = Some(entity);
            shards.set_location(entity, EntityLocation { archetype: destination.archetype_id, chunk: destination_chunk, row: destination_row });
            destination.length += 1;
            first_move_destination = Some((destination_chunk, destination_row));
        }

        // Safe: At least one component must have been created or moved.        
        let (destination_chunk, destination_row) = first_move_destination.unwrap();

        // Ensure entity_positions can store this row.
        destination.ensure_capacity((destination_chunk as usize) + 1);
        
        // Set entity at destination.
        destination.entity_positions[destination_chunk as usize][destination_row as usize] = Some(entity);
        shards.set_location(entity, EntityLocation { archetype: destination.archetype_id, chunk: destination_chunk, row: destination_row });

        // If swap_remove moved something into the vacated slot, update entity_positions accordingly.
        if let Some((source_last_chunk, source_last_row)) = first_swap_information {

            self.ensure_capacity((source_last_chunk as usize) + 1);

            let moved_entity = self.entity_positions[source_last_chunk as usize][source_last_row as usize]
                .expect("entity must exist in swapped slot");
            
            // Fill hole created by swap_remove.
            self.entity_positions[source_chunk as usize][source_row as usize] = Some(moved_entity);
            shards.set_location(moved_entity, EntityLocation { archetype: self.archetype_id, chunk: source_chunk, row: source_row });
            // Clear swapped source slot.
            self.entity_positions[source_last_chunk as usize][source_last_row as usize] = None;
        } else {
            // No swapping occurred; simply clear the source slot.
            self.entity_positions[source_chunk as usize][source_row as usize] = None;
        }

        // If a component was removed during relocation, clean up the removed component row.
        if added_component.is_none() {
            if let Some(removed_id) = self.signature.iterate_over_components()
                                       .find(|&component_id| !destination.signature.has(component_id)) {
                if let Some(source_component) = self.components[removed_id as usize].as_mut() {
                    let _ = source_component.swap_remove(source_chunk, source_row);
                }
            }
        }

        self.length -= 1; // entity removed from source
        destination.length += 1; // entity added to destination

        (destination_chunk, destination_row)
    }

    /// Spawns a new entity into this archetype using the provided component bundle.
    ///
    /// ## Purpose
    /// Writes a full row of component values and allocates an entity handle.
    ///
    /// ## Behavior
    /// - Each component in the archetype’s signature must be supplied by the bundle.
    /// - All component attributes must write to the same `(chunk, row)` location.
    /// - On failure, all partial writes are rolled back.
    ///
    /// ## Errors
    /// - `MissingComponent` when the bundle does not contain a required value.
    /// - `StoragePushFailedWith` on backend storage errors.
    /// - `MisalignedStorage` when attributes disagree on row placement.
    /// - `EmptyArchetype` if no components exist.
    ///
    /// ## Invariants
    /// Attribute alignment and entity position mappings must remain consistent.

    pub fn spawn_on(&mut self, shards: &mut EntityShards, shard_id: ShardID, mut bundle: impl DynamicBundle) -> Result<Entity, SpawnError> {
        // Keep track of columns already written so that roll back is possible on error.
        let mut written_index: Vec<usize> = Vec::new();
        let mut reference_position: Option<(ChunkID, RowID)> = None;

        for (index, component_option) in self.components.iter_mut().enumerate() {
            let Some(component) = component_option.as_mut() else { continue };

            let component_id = index as ComponentID;

            // Identify the expected type so error messages are meaningful.
            let type_id = component.element_type_id();
            let name = component.element_type_name();

            // The bundle must contain every component required by signature.
            let Some(value) = bundle.take(component_id) else {
                // Roll back already-written components.
                if let Some((c, r)) = reference_position {
                    for &j in &written_index {
                        if let Some(s) = self.components[j].as_mut() {
                            let _ = s.swap_remove(c, r);
                        }
                    }
                }
                return Err(SpawnError::MissingComponent { type_id, name });
            };

            let position = match component.push_dyn(value) {
                Ok(p) => p,
                Err(e) => {
                    // Rollback on storage failure.
                    if let Some((c, r)) = reference_position {
                        for &j in &written_index {
                            if let Some(s) = self.components[j].as_mut() {
                                let _ = s.swap_remove(c, r);
                            }
                        }
                    }
                    return Err(SpawnError::StoragePushFailedWith(e));
                }
            };

            // Ensure all components push to identical coordinates.
            if let Some(rp) = reference_position {
                debug_assert_eq!(position, rp, "attributes must stay aligned per row.");
                if position != rp {
                    // Roll back if misalignment is detected.
                    for &j in &written_index {
                        if let Some(s) = self.components[j].as_mut() {
                            let _ = s.swap_remove(rp.0, rp.1);
                        }
                    }
                    return Err(SpawnError::MisalignedStorage { expected: rp, got: position });
                }
            } else {
                // First component defines row location for this entity.
                reference_position = Some(position);
            }

            written_index.push(index);
        }

        let Some((chunk, row)) = reference_position else {
            return Err(SpawnError::EmptyArchetype); // No components existed; invalid archetype
        };

        self.ensure_capacity(chunk as usize + 1);

        debug_assert!(
            self.entity_positions[chunk as usize][row as usize].is_none(),
            "spawn_on_bundle: target entity slot is already occupied."
        );

        let location = EntityLocation { archetype: self.archetype_id, chunk, row };

        // Allocate the actual entity handle.
        let entity = shards.spawn_on(shard_id, location).map_err(|e| {
            // Roll back component writes.
            for &j in &written_index {
                if let Some(s) = self.components[j].as_mut() {
                    let _ = s.swap_remove(chunk, row);
                }
            }
            e
        })?;

        self.entity_positions[chunk as usize][row as usize] = Some(entity);
        self.length += 1;

        Ok(entity)
    }

    /// Removes an entity from this archetype and maintains row compactness.
    ///
    /// ## Purpose
    /// Ensures component attributes remain dense by using `swap_remove`.
    ///
    /// ## Behavior
    /// - Updates the entity tracker to reflect despawn.
    /// - All component attributes must agree on the swapped row, if any.
    /// - Updates `entity_positions` for any entity moved via swap.
    ///
    /// ## Errors
    /// - `StaleEntity` when the entity does not exist.
    ///
    /// ## Invariants
    /// Component storage and entity metadata must remain synchronized.

    pub fn despawn_on(&mut self, shards: &mut EntityShards, entity: Entity) -> Result<(), SpawnError> {
        let Some(location) = shards.get_location(entity) else {
            return Err(SpawnError::StaleEntity);
        };

        debug_assert_eq!(
            location.archetype, self.archetype_id,
            "the entity is not in this archetype."
        );

        let entity_chunk = location.chunk;
        let entity_row = location.row;

        let ok = shards.despawn(entity);
        if !ok { return Err(SpawnError::StaleEntity); }

        // Track whether swap_remove relocated another entity.
        let mut moved_from: Option<(ChunkID, RowID)> = None;

        for component in self.components.iter_mut().filter_map(|c| c.as_mut()) {
            // swap_remove keeps columns compact; all components must agree on moved row.
            let position = component.swap_remove(entity_chunk, entity_row)
                .expect("swap_remove failed in despawn");
            if let Some(expected) = moved_from {
                debug_assert_eq!(position, Some(expected), "all components must move the same row");
            } else {
                moved_from = position;
            }
        }

        if let Some((moved_chunk, moved_row)) = moved_from {
            // Update entity_positions to reflect the swap.
            let moved_entity = self.entity_positions[moved_chunk as usize][moved_row as usize]
                .expect("moved slot should hold an entity; storage and positions out of sync.");

            // Fill hole
            self.entity_positions[entity_chunk as usize][entity_row as usize] = Some(moved_entity);
            
            shards.set_location(moved_entity, EntityLocation {
                archetype: self.archetype_id, chunk: entity_chunk, row: entity_row
            });
            
            // Clear old swapped-from slot
            self.entity_positions[moved_chunk as usize][moved_row as usize] = None;
        } else {
            // No swap occurred; simply clear the slot.
            self.entity_positions[entity_chunk as usize][entity_row as usize] = None;
        }
        
        self.length -= 1;
        if self.length == 0 {
            self.entity_positions.clear();
        }
        Ok(())
    }

    /// Borrows the specified chunk for system execution, providing read and write
    /// access to component buffers.
    ///
    /// ## Purpose
    /// Allows systems to operate on tightly packed slices of component data.
    ///
    /// ## Behavior
    /// - `read_ids` produce immutable byte slices.
    /// - `write_ids` produce raw pointers for write access.
    /// - Caller must ensure all borrow and aliasing rules are upheld.
    ///
    /// ## Invariants
    /// Returned slices must correspond to valid rows of the requested chunk.

    pub fn borrow_chunk_for(
        &self,
        chunk: ChunkID,
        read_ids: &[ComponentID],
        write_ids: &[ComponentID],
    ) -> ChunkBorrow<'_> {
        // Borrow a chunk view for system execution; rows must be contiguous and type-aligned.
        let length = self.chunk_valid_length(chunk);
        let mut reads = Vec::with_capacity(read_ids.len());
        let mut writes = Vec::with_capacity(write_ids.len());

        for &component_id in read_ids {
            let component = self.components[component_id as usize].as_ref().expect("missing read component");
            reads.push(component.chunk_bytes_ref(chunk, length));
        }

        for &component_id in write_ids {
            let component = self.components[component_id as usize].as_ref().expect("missing write component");
            let bytes = component.chunk_bytes_mut(chunk, length);
            writes.push(bytes.as_mut_ptr());
        }

        ChunkBorrow { length, reads, writes, _marker: std::marker::PhantomData }
    }

    /// Constructs a new archetype and inserts empty attributes for the provided
    /// component types.
    ///
    /// ## Behavior
    /// - Each type ID must correspond to a registered component.
    /// - Component attributes are allocated via the storage factory.
    /// - No entities are created.
    ///
    /// ## Invariants
    /// The resulting archetype is empty but has a fully defined signature.

    pub fn from_components<T: IntoIterator<Item = std::any::TypeId>>(archetype_id: ArchetypeID, types: T) -> Self {
        let mut me = Self::new(archetype_id);

        // Create empty component columns for a predefined signature.
        for type_id in types {
            let component_id = component_id_of_type_id(type_id)
                .expect("component type must be registered before creating archetypes.");
            let component: Box<dyn TypeErasedAttribute> = make_empty_component_for(component_id);
            me.insert_empty_component(component_id, component);
        }
        me
    }
}

impl Archetype {

    /// Returns a reference to the archetype's signature.
    ///
    /// ## Notes
    /// Used by query and filtering logic.

    pub fn signature(&self) -> &Signature { &self.signature }

    /// Returns `true` if this archetype contains all components described in `need`.
    ///
    /// ## Notes
    /// This performs a subset check using signature bits.

    pub fn matches_all(&self, need: &Signature) -> bool {
        self.signature.contains_all(need)
    }
}

pub struct ArchetypeMatch {
    pub archetype_id: ArchetypeID,
    pub chunks: usize,
}

fn make_empty_component_for(component_id: ComponentID) -> Box<dyn TypeErasedAttribute> {
    get_component_storage_factory(component_id)()
}
