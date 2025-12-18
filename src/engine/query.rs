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

use crate::engine::types::{QuerySignature, ComponentID, AccessSets, set_read, set_write, set_without};
use crate::engine::archetype::{ArchetypeMatch};
use crate::engine::component::{component_id_of};
use crate::engine::manager::ECSManager;


/// A fully constructed, immutable ECS query description.
///
/// ## Purpose
/// `BuiltQuery` represents the *resolved* form of a query after it has been
/// assembled by the query builder. It contains all information required to
/// execute the query over matching archetypes without further mutation.
///
/// Unlike `QueryBuilder`, this type:
/// * is immutable,
/// * is cheap to clone,
/// * can be safely passed across scheduling and execution layers.
///
/// ## Usage
/// `BuiltQuery` is typically produced by a query builder and then consumed by
/// execution backends (sequential or parallel) to perform component iteration.
///
/// ## Execution semantics
/// * `signature` determines which archetypes match
/// * `reads` specifies read-only component columns
/// * `writes` specifies mutable component columns
///
/// Correctness relies on higher-level scheduling to ensure:
/// * no overlapping mutable access,
/// * no structural mutation during execution.

#[derive(Clone)]
pub struct BuiltQuery {
    /// Structural query signature used to match archetypes.
    pub signature: QuerySignature,

    /// Component IDs accessed in read-only mode.
    pub reads: Vec<ComponentID>,

    /// Component IDs accessed mutably.
    pub writes: Vec<ComponentID>,
}

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

    pub fn build(self) -> BuiltQuery {
        BuiltQuery {
            signature: self.signature,
            reads: self.reads,
            writes: self.writes,
        }
    }

    /// Returns the read/write access sets declared by this query.
    ///
    /// This is typically used by schedulers to detect conflicts between
    /// queries before execution.

    pub fn access_sets(&self) -> AccessSets {
        AccessSets { read: self.signature.read, write: self.signature.write }
    }

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
}
