use std::collections::HashMap;
use std::sync::{Mutex, Arc};
use rayon::prelude::*;
use rayon::ThreadPool;

use crate::commands::Command;
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


pub struct ECSManager {
    pub archetypes: Vec<Archetype>,
    signature_map: HashMap<[u64; SIGNATURE_SIZE], ArchetypeID>,
    shards: EntityShards,
    exclusive: Mutex<()>,
    deferred: Mutex<Vec<Command>>,
}

impl ECSManager {
    pub fn new(shards: EntityShards) -> Self {
        Self {
            archetypes: Vec::new(),
            by_signature: std::collections::HashMap::new(),
            shards,
            exclusive: Mutex::new(()),
            deferred: Mutex::new(Vec::new()),
        }
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

    pub fn with_exclusive<R>(&self, f: impl FnOnce(&mut ECSManager) -> R) -> R {
        let _g = self.exclusive.lock().unwrap();
        let ecs_manager_ptr = self as *const _ as *mut ECSManager;
        let m = unsafe { &mut *ecs_manager_ptr };
        f(m)
    }    

    pub fn defer(&self, command: Command) {
        self.deferred.lock().unwrap().push(command);
    }

    pub fn apply_deferred_commands(&mut self) {
        let mut commands = self.deferred.lock().unwrap();
        for command in commands.drain(..) {
            match command {
                Command::Spawn { shard, archetype } => {
                    // TODO
                }
                Command::Despawn { entity } => {
                    // TODO
                }
                Command::Add { entity, component_id, value } => {
                    // TODO
                }
                Command::Remove { entity, component_id } => {
                    self.remove_component(entity, component_id);
                }
            }
        }
    }

    pub fn par_for_each3<A: 'static + Send + Sync, B: 'static + Send + Sync, C: 'static + Send + Sync>(
        &mut self,
        reads: [crate::types::ComponentId; 2],
        writes: [crate::types::ComponentId; 1],
        mut f: impl Fn(&A, &B, &mut C) + Send + Sync,
    ) {
        for archetype in &mut self.archetypes {
            if !(archetype.has(reads[0]) && archetype.has(reads[1]) && archetype.has(writes[0])) { continue; }

            let chunks = archetype.chunk_count();
            (0..chunks).into_par_iter().for_each(|chunk| {
                let len = archetype.chunk_valid_len(chunk as _);
                if len == 0 { return; }

                let a_col = archetype.columns[reads[0] as usize].as_mut().unwrap();
                let b_col = archetype.columns[reads[1] as usize].as_mut().unwrap();
                let c_col = archetype.columns[writes[0] as usize].as_mut().unwrap();

                let a_slice = a_col.chunk_slice_ref::<A>(chunk as _, len).unwrap();
                let b_slice = b_col.chunk_slice_ref::<B>(chunk as _, len).unwrap();
                let c_slice = c_col.chunk_slice_mut::<C>(chunk as _, len).unwrap();

                let n = len.min(a_slice.len()).min(b_slice.len()).min(c_slice.len());
                for i in 0..n {
                    f(&a_slice[i], &b_slice[i], &mut c_slice[i]);
                }
            });
        }
    }
}

impl ECSManager {
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
