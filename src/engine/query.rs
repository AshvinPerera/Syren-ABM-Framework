//! ECS query description and construction.
//!
//! This module defines data structures and a builder-style API for
//! *describing* ECS queries: which components are required, which are read,
//! which are written, and which must be absent.
//!
//! `QueryBuilder` supports two construction modes:
//!
//! 1. **Global registry** (default) — `QueryBuilder::new()` resolves component
//!    IDs through the global convenience functions.
//!
//! 2. **Instance-owned registry** — `QueryBuilder::with_registry(registry)`
//!    accepts an `Arc<RwLock<ComponentRegistry>>` and resolves all component IDs
//!    through that instance.

use std::sync::{Arc, RwLock};

use crate::engine::component::{component_id_of, ComponentRegistry, Signature};
use crate::engine::error::{
    ECSError, ECSResult, ExecutionError, InvalidAccessReason, RegistryError,
};
use crate::engine::systems::AccessSets;
use crate::engine::types::ComponentID;

/// Component signature used for query matching.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuerySignature {
    /// Components read by the query.
    pub read: Signature,

    /// Components written by the query.
    pub write: Signature,

    /// Components explicitly excluded from the query.
    pub without: Signature,
}

impl QuerySignature {
    /// Returns `true` if an archetype satisfies this query.
    pub fn requires_all(&self, archetype_signature: &Signature) -> bool {
        archetype_signature.contains_all(&self.read)
            && archetype_signature.contains_all(&self.write)
            && archetype_signature
                .components
                .iter()
                .zip(self.without.components.iter())
                .all(|(arch_word, without_word)| (arch_word & without_word) == 0)
    }
}

/// An immutable, fully constructed ECS query description.
///
/// `BuiltQuery` is a **data-only representation** of a query after it has been
/// assembled by `QueryBuilder`.
///
/// It contains:
/// - a structural [`QuerySignature`] used for archetype matching,
/// - an ordered list of components accessed immutably,
/// - an ordered list of components accessed mutably.
///
/// # Important: Declaration Order Contract
///
/// The `reads` and `writes` vectors store component IDs in the order
/// they were declared via `QueryBuilder::read::<T>()` and
/// `QueryBuilder::write::<T>()`. The byte slice arrays passed to
/// iteration callbacks use this same ordering: `cols[0]` corresponds
/// to the first declared read component, `cols[1]` to the second, etc.

#[derive(Clone)]
pub struct BuiltQuery {
    /// Structural query signature used to match archetypes.
    pub signature: QuerySignature,

    /// Component IDs accessed in read-only mode.
    pub reads: Vec<ComponentID>,

    /// Component IDs accessed mutably.
    pub writes: Vec<ComponentID>,
}

// ---------------------------------------------------------------------------
// Registry source abstraction
// ---------------------------------------------------------------------------

/// Encapsulates the registry used to resolve `TypeId` → `ComponentID`.
///
/// `Global` delegates to the process-wide convenience functions (original
/// behaviour).  `Instance` holds an `Arc<RwLock<ComponentRegistry>>` for
/// per-world isolation.
enum RegistrySource {
    /// Use the global static registry (default, backward-compatible).
    Global,

    /// Instance-owned registry.
    Instance(Arc<RwLock<ComponentRegistry>>),
}

impl RegistrySource {
    /// Resolves the `ComponentID` for type `T`.
    fn resolve<T: 'static + Send + Sync>(&self) -> ECSResult<ComponentID> {
        match self {
            RegistrySource::Global => component_id_of::<T>(),
            RegistrySource::Instance(registry) => {
                let registry = registry.read().map_err(|_| RegistryError::PoisonedLock)?;
                Ok(registry.require_id_of::<T>()?)
            }
        }
    }
}

/// Builder for constructing ECS query descriptions.
///
/// `QueryBuilder` incrementally records:
/// - which components must be present,
/// - which components are read-only,
/// - which components are written,
/// - which components must be absent.
///
/// The builder follows a *builder-style* API and is typically consumed
/// by calling [`build`](Self::build) to produce a [`BuiltQuery`].
///
/// Dual construction modes
///
/// Use `QueryBuilder::new()` for the global registry (default), or
/// `QueryBuilder::with_registry(registry)` for an instance-owned registry.

pub struct QueryBuilder {
    /// Structural and access-level query signature.
    signature: QuerySignature,

    /// Component IDs read by the query (in declaration order).
    reads: Vec<ComponentID>,

    /// Component IDs written by the query (in declaration order).
    writes: Vec<ComponentID>,

    /// Where to resolve `TypeId` → `ComponentID`.
    registry_source: RegistrySource,
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryBuilder {
    /// Creates a new, empty query builder that resolves component IDs through
    /// the **global** component registry.
    pub fn new() -> Self {
        Self {
            signature: QuerySignature::default(),
            reads: vec![],
            writes: vec![],
            registry_source: RegistrySource::Global,
        }
    }

    /// Creates a new, empty query builder that resolves component IDs through
    /// the provided **instance-owned** registry.
    pub fn with_registry(registry: Arc<RwLock<ComponentRegistry>>) -> Self {
        Self {
            signature: QuerySignature::default(),
            reads: vec![],
            writes: vec![],
            registry_source: RegistrySource::Instance(registry),
        }
    }

    /// Declares read-only access to component `T`.
    ///
    /// This:
    /// - marks `T` as a required component in the query signature,
    /// - records `T` as read-access for conflict analysis,
    /// - appends `T`'s component ID to the read list.

    pub fn read<T: 'static + Send + Sync>(mut self) -> ECSResult<Self> {
        let id = self.registry_source.resolve::<T>()?;
        self.signature.read.set(id);
        self.reads.push(id);
        Ok(self)
    }

    /// Declares mutable access to component `T`.
    ///
    /// This:
    /// - marks `T` as a required component in the query signature,
    /// - records `T` as write-access for conflict analysis,
    /// - appends `T`'s component ID to the write list.

    pub fn write<T: 'static + Send + Sync>(mut self) -> ECSResult<Self> {
        let id = self.registry_source.resolve::<T>()?;
        self.signature.write.set(id);
        self.writes.push(id);
        Ok(self)
    }

    /// Excludes component `T` from matching archetypes.
    pub fn without<T: 'static + Send + Sync>(mut self) -> ECSResult<Self> {
        let id = self.registry_source.resolve::<T>()?;
        self.signature.without.set(id);
        Ok(self)
    }

    /// Finalizes the query description and returns an immutable [`BuiltQuery`].
    pub fn build(self) -> ECSResult<BuiltQuery> {
        let mut reads_sorted = self.reads.clone();
        let mut writes_sorted = self.writes.clone();

        reads_sorted.sort_unstable();
        writes_sorted.sort_unstable();

        // Detect duplicates
        let check_duplicates = |sorted: &[ComponentID]| -> ECSResult<()> {
            for w in sorted.windows(2) {
                if w[0] == w[1] {
                    return Err(ECSError::Execute(ExecutionError::InvalidQueryAccess {
                        component_id: w[0],
                        reason: InvalidAccessReason::DuplicateAccess,
                    }));
                }
            }
            Ok(())
        };

        check_duplicates(&reads_sorted)?;
        check_duplicates(&writes_sorted)?;

        reads_sorted.dedup();
        writes_sorted.dedup();

        // Disallow overlap: read & write same component
        for component_id in &reads_sorted {
            if writes_sorted.binary_search(component_id).is_ok() {
                return Err(ECSError::Execute(ExecutionError::InvalidQueryAccess {
                    component_id: *component_id,
                    reason: InvalidAccessReason::ReadAndWrite,
                }));
            }
        }

        // Disallow write & without overlap
        for (word_idx, (&w_word, &without_word)) in self
            .signature
            .write
            .components
            .iter()
            .zip(self.signature.without.components.iter())
            .enumerate()
        {
            let overlap = w_word & without_word;
            if overlap != 0 {
                let bit = overlap.trailing_zeros();
                let component_id = (word_idx as u32) * 64 + bit;
                return Err(ECSError::Execute(ExecutionError::InvalidQueryAccess {
                    component_id: component_id as ComponentID,
                    reason: InvalidAccessReason::WriteAndWithout,
                }));
            }
        }

        Ok(BuiltQuery {
            signature: self.signature,
            reads: self.reads,
            writes: self.writes,
        })
    }

    /// Returns the declared read/write access sets for this query.
    ///
    /// This is used by schedulers to detect conflicts between
    /// queries before execution.
    pub fn access_sets(&self) -> AccessSets {
        AccessSets {
            read: self.signature.read,
            write: self.signature.write,
            produces: Default::default(),
            consumes: Default::default(),
        }
    }
}
