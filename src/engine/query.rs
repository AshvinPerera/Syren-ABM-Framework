use std::any::Any;

use crate::storage::{Attribute, TypeErasedAttribute};
use crate::types::{QuerySignature, ComponentID, ChunkID};
use crate::component::{set_read, set_write, set_without, component_id_of};
use crate::archetype::ArchetypeMatch;
use crate::ecs::ECS;

pub struct QueryBuilder {
    signature: QuerySignature,
    reads: Vec<ComponentID>,
    writes: Vec<ComponentID>,
}

impl QueryBuilder {
    pub fn new() -> Self { Self { signature: QuerySignature::default(), reads: vec![], writes: vec![] } }

    pub fn read<T: 'static + Send + Sync>(mut self) -> Self {
        set_read::<T>(&mut self.signature);
        self.reads.push(component_id_of::<T>());
        self
    }
    pub fn write<T: 'static + Send + Sync>(mut self) -> Self {
        set_write::<T>(&mut self.signature);
        self.writes.push(component_id_of::<T>());
        self
    }
    pub fn without<T: 'static + Send + Sync>(mut self) -> Self {
        set_without::<T>(&mut self.signature);
        self
    }

    pub fn for_each<F>(self, ecs_manager: &mut ECS, mut f: F)
    where
        F: FnMut( /* filled at call site by helper methods below */ ),
    {
        // This base method will be specialized by typed adapters below.
        unreachable!("use typed adapters: for_each3::<A,B,C>, etc");
    }
}

impl QueryBuilder {
    pub fn for_each3<A: 'static + Send + Sync, B: 'static + Send + Sync, C: 'static + Send + Sync, F>(
        self,
        ecs_manager: &mut ECS,
        mut f: F,
    )
    where
        F: FnMut(&A, &B, &mut C),
    {
        debug_assert_eq!(self.reads.len(), 2);
        debug_assert_eq!(self.writes.len(), 1);

        let matches = ecs.matching_archetypes(&self.signature);
        for archetype_match in matches {
            let archetype = &mut ecs.archetypes[archetype_match.archetype_id as usize];

            let a_attribute = archetype.component_mut(self.reads[0]).unwrap()
                .as_any_mut().downcast_mut::<Attribute<A>>().unwrap();
            let b_attribute = archetype.component_mut(self.reads[1]).unwrap()
                .as_any_mut().downcast_mut::<Attribute<B>>().unwrap();
            let c_attribute = archetype.component_mut(self.writes[0]).unwrap()
                .as_any_mut().downcast_mut::<Attribute<C>>().unwrap();

            for chunk in 0..archetype.chunk_count() {
                let length = archetype.chunk_valid_length(chunk);
                if length == 0 { continue; }

                let a_slice = a_attribute.chunk_slice_ref(chunk, length).unwrap();
                let b_slice = b_attribute.chunk_slice_ref(chunk, length).unwrap();
                let c_slice = c_attribute.chunk_slice_mut(chunk, length).unwrap();

                let n = length.min(a_slice.len()).min(b_slice.len()).min(c_slice.len());
                for i in 0..n {
                    let a = &a_slice[i];
                    let b = &b_slice[i];
                    let c = &mut c_slice[i];
                    f(a, b, c);
                }
            }
        }
    }
}
