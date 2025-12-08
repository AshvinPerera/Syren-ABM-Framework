use std::collections::HashMap;
use crate::types::{
    Signature, 
    ComponentID, 
    ArchetypeID, 
    ChunkID, 
    RowID,
    SIGNATURE_SIZE,
    COMPONENT_CAP,
    QuerySignature
};
use crate::query::QueryBuilder;
use crate::archetype::{
    Archetype,
    ArchetypeMatch
};
use crate::entity::{Entity, EntityShards, EntityLocation};
use crate::component::make_empty_component;


pub struct ECS {
    pub archetypes: Vec<Archetype>,
    signature_map: HashMap<[u64; SIGNATURE_SIZE], ArchetypeID>,
    shards: EntityShards,
}

impl ECS {
    pub fn new(shards: EntityShards) -> Self {
        Self { archetypes: Vec::new(), signature_map: HashMap::new(), shards }
    }

    fn get_or_create_archetype(&mut self, signature: &Signature) -> ArchetypeID {
        let key = signature.components;
        if let Some(&id) = self.signature_map.get(&key) { return id; }
        let id = self.archetypes.len() as ArchetypeID;
        self.signature_map.insert(key, id);
        self.archetypes.push(Archetype::new(id));
        id
    }

    #[inline]
    fn get_archetype_pair_mut(&mut self, archetype_a: ArchetypeID, archetype_b: ArchetypeID) -> (&mut Archetype, &mut Archetype) {
        assert!(archetype_a != archetype_b, "source and destination archetype must differ");
        let (left, right) = if archetype_a < archetype_b { (archetype_a, archetype_b) } else { (archetype_b, archetype_a) };

        let (head, tail) = self.archetypes.split_at_mut(right as usize);
        let left_reference = &mut head[left as usize];
        let right_reference = &mut tail[0];
        if archetype_a < archetype_b { (left_reference, right_reference) } else { (right_reference, left_reference) }
    }

    pub fn add_component(
        &mut self,
        entity: Entity,
        added_component_id: ComponentID,
        added_value: Box<dyn std::any::Any>
    ) {
        let Some(location) = self.shards.get_location(entity) else { return; };
        let source_id = location.archetype;
        let mut new_signature = self.archetypes[source_id as usize].signature().clone();
        new_signature.set(added_component_id);

        let destination_id = self.get_or_create_archetype(&new_signature);
        {
            let destination = &mut self.archetypes[destination_id as usize];
            destination.ensure_component(added_component_id, || make_empty_component(added_component_id));
            let source = &self.archetypes[source_id as usize];
            for component_id in source.signature().iterate_over_components() {
                if component_id == added_component_id { continue; }
                destination.ensure_component(component_id, || make_empty_component(component_id));
            }
        }

        let (source, destination) = self.get_archetype_pair_mut(source_id, destination_id);
        source.move_row_to_archetype(
            destination,
            &self.shards,
            entity,
            location.chunk,
            location.row,
            Some((added_component_id, added_value)),
        );
    }

    pub fn remove_component(
        &mut self,
        entity: Entity,
        removed_component_id: ComponentID,
    ) {
        let Some(location) = self.shards.get_location(entity) else { return; };
        let source_id = location.archetype;
        if !self.archetypes[source_id as usize].has(removed_component_id) {
            return;
        }

        let mut new_signature = self.archetypes[source_id as usize].signature().clone();
        new_signature.clear(removed_component_id);

        if new_signature.components.iter().all(|&bits| bits == 0) {
            let _ = self.archetypes[source_id as usize].despawn_on(&mut self.shards, entity);
            return;
        }

        let destination_id = self.get_or_create_archetype(&new_signature);
        let source = &self.archetypes[source_id as usize];
        let destination = &mut self.archetypes[destination_id as usize];
        for component_id in source.signature().iterate_over_components() {
            if component_id == removed_component_id {
                continue;
            }
            destination.ensure_component(component_id, || make_empty_component(component_id));
        }

        let (source_arch, dest_arch) = self.get_archetype_pair_mut(source_id, destination_id);
        source_arch.move_row_to_archetype(
            dest_arch,
            &self.shards,
            entity,
            location.chunk,
            location.row,
            None,
        );
    }
}

impl ECS {
    pub fn query(&mut self) -> QueryBuilder {
        QueryBuilder::new()
    }

    pub fn matching_archetypes(&self, query: &QuerySignature) -> Vec<ArchetypeMatch> {
        let mut out = Vec::new();
        for archetype in &self.archetypes {
            if query.requires_all(archetype.signature()) {
                out.push(ArchetypeMatch { archetype_id: archetype.archetype_id(), chunks: archetype.chunk_count() });
            }
        }
        out
    }
}
