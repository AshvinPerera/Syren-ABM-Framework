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
//! - Any row movement (via `push_from`, `swap_remove`, or de-spawn) must update
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
//! - De-spawning stale or untracked entities.
//!
//! ## Summary
//!
//! `Archetype` provides the core high-performance storage representation for
//! entities in the ECS. Its chunked, columnar design ensures locality, fast
//! traversal, predictable memory behavior, and compatibility with CPU/GPU
//! parallelization strategies.

use std::any::Any;

use crate::engine::types::{
    ArchetypeID, 
    ShardID,
    ChunkID,
    RowID, 
    CHUNK_CAP, 
    ComponentID, 
    COMPONENT_CAP,
    SIGNATURE_SIZE,
    Signature,
    DynamicBundle,
    iter_bits_from_words,
};

use crate::engine::storage::{
    TypeErasedAttribute
};

use crate::engine::entity::{
    Entity, 
    EntityLocation, 
    EntityShards
};

use crate::engine::component::{ 
    component_id_of_type_id,
    get_component_storage_factory
};

use crate::engine::error::{
    SpawnError,
    MoveError
};


/// Represents a temporary borrow of a single archetype chunk for system execution.
///
/// ## Purpose
/// Provides systems with contiguous read and/or write access to component data
/// for a specific chunk, enabling tight iteration over component arrays.
///
/// ## Structure
/// - `length` specifies the number of valid rows in the chunk.
/// - `reads` contains immutable byte-slice views of component data.
/// - `writes` contains raw mutable pointers for writable component data.
///
/// ## Safety
/// This type does not enforce Rust borrowing rules at the type level.
/// Callers must ensure:
/// - No aliasing between read and write components.
/// - No simultaneous mutable borrows of the same component.
/// - The borrowed data does not outlive the archetype mutation phase.

pub struct ChunkBorrow {
     /// Number of valid rows in the borrowed chunk.
    pub length: usize, 
    /// Immutable component views as `(ptr, byte_len)` pairs.
    pub reads: Vec<(*const u8, usize)>,
    /// Mutable component data pointers for write access.
    pub writes: Vec<*mut u8>,
}

/// Stores entities that share an identical component signature.
///
/// ## Purpose
/// An `Archetype` owns columnar component storage for a fixed set of component
/// types and maintains dense, chunked layouts for fast iteration and mutation.
///
/// ## Design
/// - Component data is stored column-major by component type.
/// - Entities are densely packed using swap-remove semantics.
/// - Entity locations are tracked explicitly for fast lookup.
///
/// ## Invariants
/// - All component columns have identical row counts.
/// - `entity_positions` is kept consistent with component storage.
/// - Signature bits exactly reflect allocated component attributes.

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
                components: (0..COMPONENT_CAP).map(|_| None).collect(), // fixed-size component attribute slots
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

    /// Returns mutable references to multiple distinct component attributes at once.
    ///
    /// ## Purpose
    /// Enables safe batch access to multiple component columns without violating
    /// Rust aliasing rules.
    ///
    /// ## Behavior
    /// - Component IDs must be unique.
    /// - Missing components yield `None` in the output array.
    ///
    /// ## Safety
    /// Internally uses raw pointers; correctness relies on the uniqueness of IDs.

    pub fn components_many_mut<const N: usize>(
        &mut self,
        ids: [ComponentID; N],
    ) -> [Option<&mut dyn TypeErasedAttribute>; N] {
        debug_assert!({
            let mut tmp = ids;
            tmp.sort();
            tmp.windows(2).all(|w| w[0] != w[1])
        });

        let mut out: [Option<&mut dyn TypeErasedAttribute>; N] =
            std::array::from_fn(|_| None);

        let components = self.components.as_mut_ptr();

        for (i, &id) in ids.iter().enumerate() {
            let idx = id as usize;

            unsafe {
                let slot = &mut *components.add(idx);
                if let Some(c) = slot.as_deref_mut() {
                    out[i] = Some(c);
                }
            }
        }

        out
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
        let max_chunk = self.chunk_count().saturating_sub(1);

        if chunk_index > max_chunk {
            return 0;
        }     
        if chunk_index < max_chunk {
            CHUNK_CAP
        } else {
            let used = self.length % CHUNK_CAP;
            if used == 0 { CHUNK_CAP } else { used }
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
        // Components cannot be removed while entities exist�would break row alignment.
        if self.length > 0 {
            return Err(SpawnError::ArchetypeNotEmpty);
        } 

        let index = component_id as usize;
        if index >= COMPONENT_CAP { 
            return Err(SpawnError::InvalidComponentId); 
        }

        let taken = self.components[index].take();
        if taken.is_some() { 
            self.signature.clear(component_id); 
        }
        Ok(taken)
    }

    #[cfg(feature = "rollback")]
    pub fn move_row_across_shared_components(
        &mut self,
        destination: &mut Archetype,
        source_position: (ChunkID, RowID),
        shared_components: Vec<ComponentID>
    ) -> Result<((ChunkID, RowID), (ChunkID, RowID), Vec<(ComponentID, Box<dyn Any>)>), MoveError> 
    {
        let (source_chunk, source_row) = source_position;
        let mut destination_position: Option<(ChunkID, RowID)> = None;
        let mut swap_information: Option<(ChunkID, RowID)> = None;
        let mut rollback_sequence: Vec<(ComponentID, Box<dyn Any>)> = Vec::new();

        for component_id in shared_components {
            if !self.signature.has(component_id) || !destination.signature.has(component_id) {
                continue;
            }

            let source_component = match self.components[*component_id as usize].as_mut() {
                Some(c) => c,
                None => {
                    self.rollback_into(destination, rollback_sequence);
                    return Err(MoveError::InconsistentStorage);
                }
            };

            let destination_component = match destination.components[component_id as usize].as_mut() {
                Some(c) => c,
                None => {
                    self.rollback_into(destination, rollback_sequence);
                    return Err(MoveError::InconsistentStorage);
                }
            };

            let ((destination_chunk, destination_row), moved_from, rollback) = 
                match destination_component.push_from_dyn(source_component, source_chunk, source_row) {
                    Ok(result) => result,
                    Err(e) => {
                        self.rollback_into(destination, rollback_sequence);
                        return Err(MoveError::PushFromFailed { component_id, source_error: e });
                    }
                };

            match destination_position {
                Some(position) if position != (destination_chunk, destination_row) => {
                    self.rollback_into(destination, rollback_sequence);
                    return Err(
                        MoveError::RowMisalignment {
                            expected: position,
                            got: (destination_chunk, destination_row),
                            component_id
                        }
                    );
                }
                
                None => destination_position = Some((destination_chunk, destination_row)),
                _ => {}
            }

            if let Some(moved_from_information) = moved_from {
                match swap_information {
                    Some(existing) if existing != moved_from_information => {
                        self.rollback_into(destination, rollback_sequence);                        
                        return Err(MoveError::InconsistentSwapInfo);
                    }
                    None => {
                        swap_information = Some(moved_from_information);
                    }
                    _ => {}
                }
            }

            rollback_sequence.push((component_id, rollback));
        }

        let destination_position = destination_position.ok_or(MoveError::NoComponentsMoved)?;

        Ok((destination_position, swap_information, rollback_sequence))
    }

    /// Moves component data shared between source and destination archetypes.
    ///
    /// ## Purpose
    /// Transfers component rows that exist in both archetypes during an entity
    /// migration, preserving dense storage and row alignment.
    ///
    /// ## Behavior
    /// - Shared components are moved using `push_from_dyn`.
    /// - The first successful move determines the destination `(chunk, row)`.
    /// - All subsequent moves must resolve to the same location.
    /// - Swap-remove behavior is tracked to update entity metadata correctly.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if component columns are missing.
    /// - `PushFromFailed` if backend storage transfer fails.
    /// - `RowMisalignment` if components disagree on row placement.
    /// - `InconsistentSwapInfo` if swap metadata differs between columns.
    /// - `NoComponentsMoved` if no shared components exist.

    #[cfg(not(feature = "rollback"))]
    pub fn move_row_across_shared_components(
        &mut self,
        destination: &mut Archetype,
        source_position: (ChunkID, RowID),
        shared_components: Vec<ComponentID>
    ) -> Result<((ChunkID, RowID), Option<(ChunkID, RowID)>), MoveError> 
    {
        let (source_chunk, source_row) = source_position;
        let mut destination_position: Option<(ChunkID, RowID)> = None;
        let mut swap_information: Option<(ChunkID, RowID)> = None;

        for component_id in shared_components {
            if !self.signature.has(component_id) || !destination.signature.has(component_id) {
                continue;
            }


            let source_component = self.components[component_id as usize]
                .as_deref_mut()
                .ok_or(MoveError::InconsistentStorage)?;

            let destination_component = destination.components[component_id as usize]
                .as_deref_mut()
                .ok_or(MoveError::InconsistentStorage)?;

            let ((destination_chunk, destination_row), moved_from) = 
                match destination_component.push_from_dyn(source_component, source_chunk, source_row) {
                    Ok(result) => result,
                    Err(e) => {
                        return Err(MoveError::PushFromFailed { component_id, source_error: e });
                    }
                };

            match destination_position {
                Some(position) if position != (destination_chunk, destination_row) => {
                    return Err(
                        MoveError::RowMisalignment {
                            expected: position,
                            got: (destination_chunk, destination_row),
                            component_id
                        }
                    );
                }
                
                None => destination_position = Some((destination_chunk, destination_row)),
                _ => {}
            }

            if let Some(moved_from_information) = moved_from {
                match swap_information {
                    Some(existing) if existing != moved_from_information => {                   
                        return Err(MoveError::InconsistentSwapInfo);
                    }
                    None => {
                        swap_information = Some(moved_from_information);
                    }
                    _ => {}
                }
            }

        }

        let destination_position = destination_position.ok_or(MoveError::NoComponentsMoved)?;

        Ok((destination_position, swap_information))
    }

    /// Inserts newly added component values into the destination archetype at a fixed row.
    ///
    /// ## Purpose
    /// Completes entity migration by inserting component values that exist only
    /// in the destination archetype.
    ///
    /// ## Behavior
    /// - Each component value is inserted using `push_dyn`.
    /// - All inserts must resolve to the exact same `(chunk, row)` location.
    /// - Rollback data is collected for failure recovery.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if a required component column is missing.
    /// - `PushFailed` if backend storage insertion fails.
    /// - `RowMisalignment` if component columns disagree on row placement.

    #[cfg(feature = "rollback")]
    pub fn add_row_in_components_at_destination(
        &mut self,
        destination: &mut Archetype,
        destination_position: (ChunkID, RowID),
        added_components: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<Vec<(ComponentID, Box<dyn Any>)>, MoveError> 
    {
        let mut rollback_sequence: Vec<(ComponentID, Box<dyn Any>)> = Vec::new();
        let (destination_chunk, destination_row) = destination_position;

        for (component_id, value) in added_components {
            if !destination.signature.has(component_id) {
                continue;
            }

            let destination_component = match destination.components[component_id as usize].as_mut() {
                Some(c) => c,
                None => {
                    self.rollback_into(destination, rollback_sequence);
                    return Err(MoveError::InconsistentStorage);
                }
            };

            let ((chunk, row), rollback) = 
                match destination_component.push_dyn(value) {
                    Ok(result) => result,
                    Err(e) => {
                        self.rollback_into(destination, rollback_sequence);
                        return Err(MoveError::PushFailed { component_id, source_error: e });
                    }
                };

            if (chunk, row) != (destination_chunk, destination_row) {
                self.rollback_into(destination, rollback_sequence);
                return Err(MoveError::RowMisalignment {
                    expected: (destination_chunk, destination_row),
                    got: (chunk, row),
                    component_id,
                });
            }           

            rollback_sequence.push((component_id, rollback));
        }

        Ok(rollback_sequence)
    }

    /// Inserts newly added component values into the destination archetype at a fixed row.
    ///
    /// ## Purpose
    /// Completes entity migration by inserting component values that exist only
    /// in the destination archetype.
    ///
    /// ## Behavior
    /// - Each component value is inserted using `push_dyn`.
    /// - All inserts must resolve to the exact same `(chunk, row)` location.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if a required component column is missing.
    /// - `PushFailed` if backend storage insertion fails.
    /// - `RowMisalignment` if component columns disagree on row placement.

    #[cfg(not(feature = "rollback"))]
    pub fn add_row_in_components_at_destination(
        &mut self,
        destination: &mut Archetype,
        destination_position: (ChunkID, RowID),
        added_components: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<(), MoveError> 
    {
        let (destination_chunk, destination_row) = destination_position;

        for (component_id, value) in added_components {
            if !destination.signature.has(component_id) {
                continue;
            }

            let destination_component = match destination.components[component_id as usize].as_mut() {
                Some(c) => c,
                None => {
                    return Err(MoveError::InconsistentStorage);
                }
            };

            let (chunk, row) = 
                match destination_component.push_dyn(value) {
                    Ok(result) => result,
                    Err(e) => {
                        return Err(MoveError::PushFailed { component_id, source_error: e });
                    }
                };

            if (chunk, row) != (destination_chunk, destination_row) {
                return Err(MoveError::RowMisalignment {
                    expected: (destination_chunk, destination_row),
                    got: (chunk, row),
                    component_id,
                });
            }
        }

        Ok(())
    }

    /// Removes source-only component values from the archetype during entity migration.
    ///
    /// ## Purpose
    /// Discards component data that is not present in the destination archetype
    /// while preserving storage compactness.
    ///
    /// ## Behavior
    /// - Uses `swap_remove` to maintain dense component storage.
    /// - All removed components must report identical swap positions.
    /// - Rollback data is collected for failure recovery.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if a component column is missing.
    /// - `SwapRemoveError` if backend storage removal fails.
    /// - `InconsistentSwapInfo` if component columns disagree on swap behavior.

    #[cfg(feature = "rollback")]    
    pub fn remove_row_in_components_at_source(
        &mut self,
        source_position: (ChunkID, RowID),
        removed_components: &[ComponentID],
        source_swap_position: Option<(ChunkID, RowID)>
    ) -> Result<Vec<(ComponentID, Box<dyn Any>)>, MoveError> 
    {
        let (source_chunk, source_row) = source_position;
        let mut rollback_sequence: Vec<(ComponentID, Box<dyn Any>)> = Vec::new();       

        for component_id in removed_components {
            if !self.signature.has(component_id) {
                continue;
            }

            let source_component = match self.components[component_id as usize].as_mut() {
                Some(c) => c,
                None => {
                    self.rollback_self(rollback_sequence);    
                    return Err(MoveError::InconsistentStorage);
                }
            };

            let (moved_from, rollback) = 
                match source_component.swap_remove_dyn(source_chunk, source_row) {
                    Ok(result) => result,
                    Err(e) => {
                        self.rollback_self(rollback_sequence);     
                        return Err(MoveError::SwapRemoveError { component_id, source_error: e });
                    }
                };

            if let Some(moved_from) = moved_from {
                match source_swap_position {
                    Some(existing) if existing != moved_from => {
                        self.rollback_self(rollback_sequence);    
                        return Err(MoveError::InconsistentSwapInfo)
                    },
                    _ => {}
                }
            }

            rollback_sequence.push((component_id, rollback));
        }

        Ok(rollback_sequence)
    }

    /// Removes source-only component values from the archetype during entity migration.
    ///
    /// ## Purpose
    /// Deletes component data that does not exist in the destination archetype
    /// while keeping component columns densely packed.
    ///
    /// ## Behavior
    /// - Uses `swap_remove` for compact storage.
    /// - All components must report identical swap positions.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if a component column is missing.
    /// - `SwapRemoveError` if storage removal fails.
    /// - `InconsistentSwapInfo` if component columns disagree on swap behavior.

    #[cfg(not(feature = "rollback"))]  
    pub fn remove_row_in_components_at_source(
        &mut self,
        source_position: (ChunkID, RowID),
        removed_components: &[ComponentID],
        source_swap_position: Option<(ChunkID, RowID)>,
    ) -> Result<(), MoveError> 
    {
        let (source_chunk, source_row) = source_position;     

        for &component_id in removed_components {
            if !self.signature.has(component_id) {
                continue;
            }

            let source_component = match self.components[component_id as usize].as_mut() {
                Some(c) => c,
                None => {
                    return Err(MoveError::InconsistentStorage);
                }
            };

            let moved_from = 
                match source_component.swap_remove_dyn(source_chunk, source_row) {
                    Ok(result) => result,
                    Err(e) => {   
                        return Err(MoveError::SwapRemoveError { component_id, source_error: e });
                    }
                };

            if let Some(moved_from) = moved_from {
                match source_swap_position {
                    Some(existing) if existing != moved_from => {  
                        return Err(MoveError::InconsistentSwapInfo)
                    },
                    _ => {}
                }
            }
        }

        Ok(())
    }

    /// Updates entity metadata after a row is moved between archetypes.
    ///
    /// ## Purpose
    /// Synchronizes `entity_positions` and global entity location tracking
    /// after component data has been relocated.
    ///
    /// ## Behavior
    /// - Writes the entity ID into the destination archetype metadata.
    /// - Updates the entity's global location in `EntityShards`.
    /// - Fixes metadata for any entity relocated via swap-remove.
    ///
    /// ## Errors
    /// - `MetadataFailure` if internal entity tracking is inconsistent.

    pub fn update_entity_on_row_move(
        &mut self,
        destination: &mut Archetype,
        source_position: (ChunkID, RowID),
        destination_position: (ChunkID, RowID),
        source_swap_position: Option<(ChunkID, RowID)>,
        shards: &EntityShards,
        entity: Entity,
    ) -> Result<(), MoveError> 
    {
        let (destination_chunk, destination_row) = destination_position;
        let (source_chunk, source_row) = source_position;

        destination.ensure_capacity(destination_chunk as usize + 1);
        destination.entity_positions[destination_chunk as usize][destination_row as usize] = Some(entity);

        shards.set_location(
            entity,
            EntityLocation {
                archetype: destination.archetype_id,
                chunk: destination_chunk,
                row: destination_row,
            },
        );

        match source_swap_position {
            Some((last_chunk, last_row)) => {
                self.ensure_capacity(last_chunk as usize + 1);

                let swapped_entity = self.entity_positions[last_chunk as usize][last_row as usize]
                    .ok_or(MoveError::MetadataFailure)?;

                self.entity_positions[source_chunk as usize][source_row as usize] =
                    Some(swapped_entity);

                shards.set_location(
                    swapped_entity,
                    EntityLocation {
                        archetype: self.archetype_id,
                        chunk: source_chunk,
                        row: source_row,
                    },
                );

                self.entity_positions[last_chunk as usize][last_row as usize] = None;
            }
            None => {
                self.entity_positions[source_chunk as usize][source_row as usize] = None;
            }
        }

        Ok(())
    }

    #[cfg(feature = "rollback")]
    fn rollback_into(
        &mut self,
        destination: &mut Archetype,
        rollback_sequence: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<(), MoveError> {
        for (rolled_back_id, rollback_action) in rollback_sequence.into_iter().rev() {
            let rollback_destination_component = destination.components[rolled_back_id as usize]
                .as_mut()
                .ok_or(MoveError::InconsistentStorage)?;

            let rollback_source_component = self.components[rolled_back_id as usize]
                .as_mut()
                .ok_or(MoveError::InconsistentStorage)?;

            rollback_destination_component
                .rollback_dyn(rollback_action, Some(rollback_source_component))
                .map_err(|_| MoveError::RollbackFailed)?;
        }
        Ok(())
    }

    #[cfg(feature = "rollback")]
    fn rollback_self(
        &mut self,
        rollback_sequence: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<(), MoveError> {
        for (rolled_back_id, rollback_action) in rollback_sequence.into_iter().rev() {
            let component = self.components[rolled_back_id as usize]
                .as_mut()
                .ok_or(MoveError::InconsistentStorage)?;

            component
                .rollback_dyn(rollback_action, None)
                .map_err(|_| MoveError::RollbackFailed)?;
        }
        Ok(())
    }
    
    /// Moves an entity's component row from this archetype to another.
    ///
    /// # Purpose
    /// This operation is used when an entity transitions to a new archetype
    /// because its set of components has changed (added or removed).
    ///
    /// The function constructs a new row in the destination archetype containing
    /// exactly the components described by the destination's signature.
    ///
    /// # Behavior
    ///
    /// For each component type:
    ///
    /// - **If the component exists in both source and destination**  
    ///   The component value at `(source_chunk, source_row)` is moved into the
    ///   destination component column via `push_from`, preserving its internal
    ///   ordering guarantees (including swap-remove semantics).
    ///
    /// - **If the component exists in the destination but not in the source**  
    ///   A value for this component **must** be supplied in `added_components`.
    ///   That value is inserted using `push_dyn`.
    ///
    /// - **If the component exists in the source but not in the destination**  
    ///   The component value at `(source_chunk, source_row)` is discarded using
    ///   `swap_remove`, removing the row compactly from the source column.
    ///
    /// The first column to receive the moved or inserted value defines the
    /// destination `(chunk, row)` for this entity. All other component columns
    /// for the entity must place their data **at exactly the same location**.
    /// This preserves strict row alignment across all component arrays.
    ///
    /// After all component values are written:
    ///
    /// - The destination archetype's `entity_positions` entry for the final
    ///   `(chunk, row)` is updated to record the entity's ID.
    ///
    /// - The source archetype's row at `(source_chunk, source_row)` is cleared.
    ///
    /// - If any source component column performed a swap-remove, the function
    ///   updates `entity_positions` and the global shard registry so the moved
    ///   entity now references the correct new position.
    ///
    /// - Archetype `length` counters are updated in both source and destination.

    #[cfg(feature = "rollback")]    
    pub fn move_row_to_archetype(
        &mut self,
        destination: &mut Archetype,
        shards: &EntityShards,
        entity: Entity,
        source_position: (ChunkID, RowID),
        mut added_components: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<(ChunkID, RowID), MoveError> {
        let mut shared_words = [0u64; SIGNATURE_SIZE];
        let mut source_only_words = [0u64; SIGNATURE_SIZE];
        let mut destination_only_words = [0u64; SIGNATURE_SIZE];

        for i in 0..SIGNATURE_SIZE {
            let a = self.signature.components[i];
            let b = destination.signature.components[i];

            shared_words[i] = a & b;
            source_only_words[i] = a & !b;
            destination_only_words[i] = b & !a;
        }

        let shared_components: Vec<ComponentID> =
            iter_bits_from_words(&shared_words).collect();

        let source_only_components: Vec<ComponentID> =
            iter_bits_from_words(&source_only_words).collect();

        let destination_only_components: Vec<ComponentID> =
            iter_bits_from_words(&destination_only_words).collect();

        let mut destination_only_values: Vec<(ComponentID, Box<dyn Any>)> =
            Vec::with_capacity(destination_only_components.len());

        for &need_id in &destination_only_components {
            if let Some(pos) = added_components.iter().position(|(id, _)| *id == need_id) {
                let (_id, val) = added_components.swap_remove(pos);
                destination_only_values.push((need_id, val));
            } else {
                return Err(MoveError::InconsistentStorage);
            }
        }

        let (destination_position, source_swap_position, mut moved_rollbacks) =
            self.move_row_across_shared_components(destination, source_position, shared_components)?;

        let add_rollbacks = match self.add_row_in_components_at_destination(
            destination,
            destination_position,
            destination_only_values,
        ) {
            Ok(r) => r,
            Err(e) => {
                let _ = self.rollback_into(destination, moved_rollbacks);
                return Err(e);
            }
        };
        moved_rollbacks.extend(add_rollbacks);

        let remove_rollbacks = match self.remove_row_in_components_at_source(
            source_position,
            &source_only_components,
            source_swap_position
        ) {
            Ok(r) => r,
            Err(e) => {
                self.rollback_into(destination, moved_rollbacks);
                return Err(e);
            }
        };

        if let Err(e) = self.update_entity_on_row_move(
            destination,
            source_position,
            destination_position,
            source_swap_position,
            shards,
            entity
        ) {
            let _ = self.rollback_self(remove_rollbacks);
            let _ = self.rollback_into(destination, moved_rollbacks);
            return Err(e);
        }

        destination.length += 1;
        self.length = self.length.saturating_sub(1);
        if self.length == 0 {
            self.entity_positions.clear();
        }

        Ok(destination_position)
    }      

    /// Moves an entity's component row from this archetype to another.
    ///
    /// ## Purpose
    /// Transfers an entity between archetypes when its component signature changes,
    /// constructing a new row in the destination archetype that exactly matches
    /// the destination signature.
    ///
    /// This is the core operation used when components are added to or removed
    /// from an entity.
    ///
    /// ## Behavior
    ///
    /// The move is performed in four ordered phases:
    ///
    /// 1. **Signature Analysis**
    ///    - Computes the set of components shared between source and destination.
    ///    - Computes components present only in the source (to be removed).
    ///    - Computes components present only in the destination (to be added).
    ///
    /// 2. **Shared Component Transfer**
    ///    - For each shared component, the value at `source_position` is moved
    ///      into the destination archetype using `push_from_dyn`.
    ///    - The first successful transfer determines the destination `(chunk, row)`.
    ///    - All subsequent transfers must resolve to the same location.
    ///    - Any swap-remove performed during transfer is recorded.
    ///
    /// 3. **Destination-Only Component Insertion**
    ///    - Components that exist only in the destination archetype are inserted
    ///      using values supplied in `added_components`.
    ///    - All insertions must target the previously established destination row.
    ///
    /// 4. **Source-Only Component Removal**
    ///    - Components that exist only in the source archetype are removed using
    ///      `swap_remove`, preserving dense storage.
    ///    - All removals must agree on swap behavior.
    ///
    /// After component data movement:
    /// - Entity metadata is updated in both archetypes.
    /// - Any entity relocated via swap-remove has its location corrected.
    /// - Archetype entity counts are updated.
    ///
    /// ## Parameters
    /// - `destination`: Target archetype whose signature the entity will match.
    /// - `shards`: Global entity registry used to update entity locations.
    /// - `entity`: The entity being moved.
    /// - `source_position`: The `(chunk, row)` of the entity in the source archetype.
    /// - `added_components`: Component values required by the destination archetype
    ///   but not present in the source.
    ///
    /// ## Returns
    /// Returns the `(chunk, row)` of the entity in the destination archetype.
    ///
    /// ## Errors
    /// - `InconsistentStorage` if required component columns are missing or
    ///   `added_components` does not supply all required destination-only values.
    /// - `PushFromFailed` if transferring shared component data fails.
    /// - `PushFailed` if inserting destination-only components fails.
    /// - `SwapRemoveError` if removing source-only components fails.
    /// - `RowMisalignment` if component columns disagree on row placement.
    /// - `InconsistentSwapInfo` if swap-remove metadata differs between components.
    /// - `MetadataFailure` if entity location tracking becomes inconsistent.
    ///
    /// ## Invariants
    /// - All component columns remain row-aligned.
    /// - Source and destination archetypes remain densely packed.
    /// - Entity location metadata is always consistent with component storage.

    #[cfg(not(feature = "rollback"))]   
    pub fn move_row_to_archetype(
        &mut self,
        destination: &mut Archetype,
        shards: &EntityShards,
        entity: Entity,
        source_position: (ChunkID, RowID),
        mut added_components: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<(ChunkID, RowID), MoveError> {
        let mut shared_words = [0u64; SIGNATURE_SIZE];
        let mut source_only_words = [0u64; SIGNATURE_SIZE];
        let mut destination_only_words = [0u64; SIGNATURE_SIZE];

        for i in 0..SIGNATURE_SIZE {
            let a = self.signature.components[i];
            let b = destination.signature.components[i];

            shared_words[i] = a & b;
            source_only_words[i] = a & !b;
            destination_only_words[i] = b & !a;
        }

        let shared_components: Vec<ComponentID> =
            iter_bits_from_words(&shared_words).collect();

        let source_only_components: Vec<ComponentID> =
            iter_bits_from_words(&source_only_words).collect();

        let destination_only_components: Vec<ComponentID> =
            iter_bits_from_words(&destination_only_words).collect();

        let mut destination_only_values: Vec<(ComponentID, Box<dyn Any>)> =
            Vec::with_capacity(destination_only_components.len());

        for &need_id in &destination_only_components {
            if let Some(pos) = added_components.iter().position(|(id, _)| *id == need_id) {
                let (_id, val) = added_components.swap_remove(pos);
                destination_only_values.push((need_id, val));
            } else {
                return Err(MoveError::InconsistentStorage);
            }
        }

        let (destination_position, source_swap_position) =
            self.move_row_across_shared_components(destination, source_position, shared_components)?;

        self.add_row_in_components_at_destination(
            destination,
            destination_position,
            destination_only_values,
        )?;

        self.remove_row_in_components_at_source(
            source_position,
            &source_only_components,
            source_swap_position
        )?;

        self.update_entity_on_row_move(
            destination,
            source_position,
            destination_position,
            source_swap_position,
            shards,
            entity
        )?;

        destination.length += 1;
        self.length = self.length.saturating_sub(1);
        if self.length == 0 {
            self.entity_positions.clear();
        }

        Ok(destination_position)
    }  

    /// Spawns a new entity into this archetype using the provided component bundle.
    ///
    /// ## Purpose
    /// Writes a full row of component values and allocates an entity handle.
    ///
    /// ## Behavior
    /// - Each component in the archetype�s signature must be supplied by the bundle.
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

    pub fn spawn_on(
        &mut self, 
        shards: &mut EntityShards, 
        shard_id: ShardID, 
        mut bundle: impl DynamicBundle
    ) -> Result<Entity, SpawnError> {
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
                            let _ = s.swap_remove_dyn(c, r);
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
                                let _ = s.swap_remove_dyn(c, r);
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
                            let _ = s.swap_remove_dyn(rp.0, rp.1);
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
                    let _ = s.swap_remove_dyn(chunk, row);
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
            let position = component.swap_remove_dyn(entity_chunk, entity_row)
                .map_err(|e| SpawnError::StorageSwapRemoveFailed(e))?;
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

    /// Borrows a chunk of component data for system execution.
    ///
    /// ## Purpose
    /// Provides systems with efficient access to contiguous component data
    /// for iteration and mutation.
    ///
    /// ## Behavior
    /// - Read components are exposed as immutable byte slices.
    /// - Write components are exposed as raw mutable pointers.
    /// - Read and write component sets must not overlap.
    ///
    /// ## Panics
    /// Panics if a component appears in both read and write lists.
    ///
    /// ## Safety
    /// The caller must uphold aliasing and synchronization guarantees.

    pub fn borrow_chunk_for(
        &mut self,
        chunk: ChunkID,
        read_ids: &[ComponentID],
        write_ids: &[ComponentID],
    ) -> ChunkBorrow {
        let length = self.chunk_valid_length(chunk as usize);

        // Validate no overlap
        for &read_id in read_ids {
            if write_ids.contains(&read_id) {
                panic!("Component {} appears in both read and write lists", read_id);
            }
        }

        let mut reads = Vec::with_capacity(read_ids.len());
        let mut writes = Vec::with_capacity(write_ids.len());

        // Read views
        {
            let components = &self.components;
            for &component_id in read_ids {
                let component = components[component_id as usize]
                    .as_deref()
                    .expect("missing read component");

                let (ptr, bytes) = component
                    .chunk_bytes(chunk, length)
                    .expect("missing read component data");

                reads.push((ptr, bytes));
            }
        }

        // Write views
        {
            let components = &mut self.components;
            for &component_id in write_ids {
                let component = components[component_id as usize]
                    .as_deref_mut()
                    .expect("missing write component");

                let (ptr, _bytes) = component
                    .chunk_bytes_mut(chunk, length)
                    .expect("missing write component data");

                writes.push(ptr);
            }
        }

        ChunkBorrow { length, reads, writes }
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
        let mut archetype = Self::new(archetype_id);

        // Create empty component columns for a predefined signature.
        for type_id in types {
            let component_id = component_id_of_type_id(type_id)
                .expect("component type must be registered before creating archetypes.");
            let component: Box<dyn TypeErasedAttribute> = make_empty_component_for(component_id);
            archetype.insert_empty_component(component_id, component);
        }
        archetype
    }
}

/// Represents an archetype selected during query matching.
///
/// ## Purpose
/// Used by query systems to record which archetypes satisfy a component filter
/// and how many chunks they contain.

pub struct ArchetypeMatch {
    /// Identifier of the matched archetype.
    pub archetype_id: ArchetypeID,
    /// Number of chunks currently allocated in the archetype.
    pub chunks: usize,
}

fn make_empty_component_for(component_id: ComponentID) -> Box<dyn TypeErasedAttribute> {
    get_component_storage_factory(component_id)()
}
