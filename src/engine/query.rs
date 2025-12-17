//! Typed query construction and execution for the ECS.
//!
//! This module provides a *builder-style* API for constructing component queries
//! with explicit read/write access declarations, and for executing those queries
//! over matching archetypes.
//!
//! ## Design goals
//! * **Static intent:** Read/write/without component intent is encoded at build time.
//! * **Runtime efficiency:** Queries operate directly on archetype storage without
//!   intermediate allocations.
//! * **Safety by discipline:** The API enforces access correctness by construction,
//!   not by the borrow checker.
//!
//! ## Execution model
//! Queries:
//! 1. Construct a [`QuerySignature`] describing required, read, write, and excluded components.
//! 2. Resolve matching archetypes at execution time.
//! 3. Iterate chunk-by-chunk over component storage.
//! 4. Invoke user-provided closures on typed component references.
//!
//! ## Concurrency
//! This module itself does not perform parallel execution. It relies on the
//! caller (typically `ECSManager`) to ensure:
//! * exclusive access when mutating structure,
//! * non-overlapping write sets between concurrent queries.

use crate::engine::storage::{Attribute, TypeErasedAttribute};
use crate::engine::types::{QuerySignature, ComponentID, ChunkID, AccessSets, set_read, set_write, set_without};
use crate::engine::archetype::{ArchetypeMatch};
use crate::engine::component::{component_id_of};
use crate::engine::manager::ECSManager;


/// Builder for ECS component queries.
///
/// `QueryBuilder` incrementally constructs a [`QuerySignature`] describing:
/// * which components must be present,
/// * which components are read-only,
/// * which components are written,
/// * which components must be absent.
///
/// The builder is *consumed* when executing a query, ensuring that a query
/// definition cannot be reused incorrectly.
///
/// ## Typing model
/// The base `for_each` method is intentionally uncallable. Instead, users must
/// select a typed adapter (e.g. [`for_each3`]) matching the declared access
/// pattern.
///
/// ## Example
/// ```ignore
/// ecs.query()
///     .read::<Position>()
///     .read::<Velocity>()
///     .write::<Transform>()
///     .for_each3(&ecs_manager, |pos, vel, transform| {
///         transform.update(pos, vel);
///     });
/// ```

pub struct QueryBuilder {
    /// Structural and access-level query signature.
    signature: QuerySignature,

    /// Component IDs read by the query (in declaration order).
    reads: Vec<ComponentID>,

    /// Component IDs written by the query (in declaration order).
    writes: Vec<ComponentID>,
}

impl QueryBuilder {
    /// Creates a new, empty query builder.
    pub fn new() -> Self { Self { signature: QuerySignature::default(), reads: vec![], writes: vec![] } }

    /// Declares a read-only dependency on component `T`.
    ///
    /// ## Semantics
    /// * Adds `T` to the query’s required component set.
    /// * Marks `T` as read-only for access conflict analysis.
    ///
    /// ## Type constraints
    /// `T` must be `'static + Send + Sync` to allow safe storage access.
    
    pub fn read<T: 'static + Send + Sync>(mut self) -> Self {
        set_read::<T>(&mut self.signature);
        self.reads.push(component_id_of::<T>());
        self
    }

    /// Declares a mutable dependency on component `T`.
    ///
    /// ## Semantics
    /// * Adds `T` to the query’s required component set.
    /// * Marks `T` as write-access for conflict detection.
    ///
    /// ## Safety
    /// Only one query with write access to a given component may be executed
    /// at a time.

    pub fn write<T: 'static + Send + Sync>(mut self) -> Self {
        set_write::<T>(&mut self.signature);
        self.writes.push(component_id_of::<T>());
        self
    }


    /// Excludes component `T` from matching archetypes.  
    pub fn without<T: 'static + Send + Sync>(mut self) -> Self {
        set_without::<T>(&mut self.signature);
        self
    }

    /// Base execution method (intentionally unreachable).
    ///
    /// This exists only to provide a uniform API surface; execution must
    /// be performed via typed adapters such as [`for_each3`].
    ///
    /// ## Panics
    /// Always panics.    

    pub fn for_each<F>(self)
    where
        F: FnMut( /* filled at call site by helper methods below */ ),
    {
        // This base method will be specialized by typed adapters below.
        unreachable!("use typed adapters: for_each3::<A,B,C>, etc");
    }

    /// Returns the read/write access sets declared by this query.
    ///
    /// This is typically used by schedulers to detect conflicts between
    /// queries before execution.

    pub fn access_sets(&self) -> AccessSets {
        AccessSets { read: self.signature.read, write: self.signature.write }
    }
}

impl QueryBuilder {

    /// Resolves the query into matching archetypes.
    ///
    /// This performs no iteration and does not borrow component storage.
    pub fn resolve<'a>(
        &self,
        ecs: &'a ECSManager,
    ) -> Vec<ArchetypeMatch> {
        let world = ecs.world_ref();
        world.data().matching_archetypes(&self.signature)
    }    

    /// Executes a query over three components:
    /// * two read-only (`A`, `B`)
    /// * one mutable (`C`)
    ///
    /// ## Execution steps
    /// 1. Resolve archetypes matching the query signature.
    /// 2. For each archetype:
    ///    * identify valid chunks,
    ///    * downcast component storage to typed columns,
    ///    * iterate row-by-row within each chunk.
    /// 3. Invoke the user-provided closure for each entity.
    ///
    /// ## Safety and invariants
    /// * The query must declare exactly two reads and one write.
    /// * Component storage types must match `A`, `B`, and `C`.
    /// * Chunk slices must be correctly aligned and non-overlapping.
    ///
    /// Violating these invariants results in undefined behavior.
    ///
    /// ## Parameters
    /// * `ecs_manager` — ECS entry point providing access to world data.
    /// * `f` — Closure executed for each matching entity.
    ///
    /// ## Performance
    /// * Zero heap allocation per entity.
    /// * Iteration is cache-friendly and archetype-local.
    
    pub fn for_each3<
        A: 'static + Send + Sync,
        B: 'static + Send + Sync,
        C: 'static + Send + Sync,
        F,
    >(
        self,
        ecs_manager: &ECSManager,
        mut f: F,
    )
    where
        F: FnMut(&A, &B, &mut C),
    {
        debug_assert_eq!(self.reads.len(), 2);
        debug_assert_eq!(self.writes.len(), 1);

        // Acquire world capability
        let world = ecs_manager.world_ref();
        let data = world.data_mut();

        let matches = data.matching_archetypes(&self.signature);

        for m in matches {
            let archetype =
                &mut data.archetypes[m.archetype_id as usize];

            let chunk_count = archetype.chunk_count();
            let mut chunk_jobs = Vec::with_capacity(chunk_count);

            for chunk in 0..chunk_count {
                let len = archetype.chunk_valid_length(chunk);
                if len != 0 {
                    chunk_jobs.push((chunk as ChunkID, len));
                }
            }

            let [a_any, b_any, c_any] = archetype.components_many_mut([
                self.reads[0],
                self.reads[1],
                self.writes[0],
            ]);

            let a_attribute = a_any.unwrap()
                .as_any_mut()
                .downcast_mut::<Attribute<A>>()
                .unwrap();

            let b_attribute = b_any.unwrap()
                .as_any_mut()
                .downcast_mut::<Attribute<B>>()
                .unwrap();

            let c_attribute = c_any.unwrap()
                .as_any_mut()
                .downcast_mut::<Attribute<C>>()
                .unwrap();

            for (chunk, length) in chunk_jobs {
                let a_slice = a_attribute.chunk_slice(chunk, length).unwrap();
                let b_slice = b_attribute.chunk_slice(chunk, length).unwrap();
                let c_slice = c_attribute.chunk_slice_mut(chunk, length).unwrap();

                let n = length
                    .min(a_slice.len())
                    .min(b_slice.len())
                    .min(c_slice.len());

                for i in 0..n {
                    f(&a_slice[i], &b_slice[i], &mut c_slice[i]);
                }
            }
        }
    }
}
