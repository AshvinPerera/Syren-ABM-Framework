//! Chunk borrowing and column-level synchronization for archetype system access.
//!
//! This module provides [`ChunkBorrow`], a scoped borrow over a single archetype
//! chunk that enforces column-level read/write locking for the duration of system
//! execution. It exposes raw component data as typed byte views while ensuring
//! that concurrent access patterns remain sound.
//!
//! ## Overview
//!
//! During system execution, components are accessed in bulk across a contiguous
//! chunk of archetype storage. [`ChunkBorrow`] captures this access pattern by
//! holding [`RwLock`] guards for every requested component column — read guards
//! for shared access, write guards for exclusive access — and exposing the
//! underlying data as raw pointers for zero-cost iteration.
//!
//! Borrows are constructed via [`Archetype::borrow_chunk_for`], which validates
//! that the read and write sets are disjoint, then acquires all column locks in
//! ascending [`ComponentID`] order to satisfy the global lock ordering contract
//! and prevent deadlock.
//!
//! ## Safety Contract
//!
//! - Read and write component sets must not overlap; violation returns an error.
//! - All column locks are held for the lifetime `'a` of the borrow.
//! - Raw pointers derived from this borrow must not outlive it.
//! - Structural archetype mutations must not occur while any borrow is live.
//!
//! [`RwLock`]: std::sync::RwLock

use std::sync::{RwLockReadGuard, RwLockWriteGuard};

use smallvec::SmallVec;

use crate::engine::types::{ChunkID, ComponentID};

use crate::engine::storage::TypeErasedAttribute;

use crate::engine::error::{ECSError, ECSResult, ExecutionError};

use super::core::Archetype;

/// Represents a temporary borrow of a single archetype chunk for system execution.
///
/// ## Safety
/// This type *maintains* column-level synchronization by holding the underlying
/// `RwLock` guards for all accessed component columns for the duration of the borrow.
///
/// While a `ChunkBorrow` exists:
/// - Any component in `read_guards` is read-locked (shared).
/// - Any component in `write_guards` is write-locked (exclusive).
/// - Other systems may read the same components but may not write them.
/// - Structural archetype mutation must not occur concurrently.
///
/// This type does **not** prevent misuse such as:
/// - retaining raw pointers beyond the borrow lifetime,
/// - performing structural ECS mutations in parallel.
///
/// Violating these constraints may cause deadlock or panic.

pub struct ChunkBorrow<'a> {
    /// Number of valid rows in the borrowed chunk.
    pub length: usize,
    /// Immutable component views as `(ptr, byte_len)` pairs.
    pub reads: SmallVec<[(*const u8, usize); 8]>,
    /// Mutable component data pointers for write access.
    pub writes: SmallVec<[*mut u8; 8]>,

    /// Holds the read locks alive for the lifetime of this borrow.
    _read_guards: SmallVec<
        [(
            ComponentID,
            RwLockReadGuard<'a, Box<dyn TypeErasedAttribute>>,
        ); 8],
    >,
    /// Holds the write locks alive for the lifetime of this borrow.
    _write_guards: SmallVec<
        [(
            ComponentID,
            RwLockWriteGuard<'a, Box<dyn TypeErasedAttribute>>,
        ); 8],
    >,
}

impl Archetype {
    /// Borrows a chunk of component data for system execution.
    ///
    /// ## Purpose
    /// Provides systems with efficient access to contiguous component data
    /// for iteration and mutation.
    ///
    /// ## Behaviour
    /// - Read components are exposed as immutable byte slices.
    /// - Write components are exposed as raw mutable pointers.
    /// - Read and write component sets must not overlap.
    ///
    /// ## Panics
    /// Panics if a component appears in both read and write lists.
    ///
    /// ## Safety
    /// `ChunkBorrow` holds column-level locks for all accessed components.
    ///
    /// While a `ChunkBorrow` exists:
    /// - No other system may mutably access the *same component columns*.
    /// - Other systems may read the same components concurrently.
    /// - Writes to disjoint component sets are permitted.
    /// - Structural archetype mutation must not occur.
    ///
    /// These guarantees are enforced by column-level locking and execution
    /// phase discipline. Violations may cause deadlock or panic.
    ///
    /// ## Lock ordering
    /// All column locks are acquired in ascending `ComponentID` order,
    /// consistent with the global lock ordering contract.

    pub fn borrow_chunk_for<'a>(
        &'a self,
        chunk: ChunkID,
        read_ids: &[ComponentID],
        write_ids: &[ComponentID],
    ) -> ECSResult<ChunkBorrow<'a>> {
        let length = self
            .chunk_valid_length(chunk as usize)
            .map_err(|_| ExecutionError::InternalExecutionError)?;

        for &id in read_ids {
            if write_ids.contains(&id) {
                return Err(ECSError::Execute(ExecutionError::InvalidQueryAccess {
                    component_id: id,
                    reason: crate::engine::error::InvalidAccessReason::ReadAndWrite,
                }));
            }
        }

        // Lock ordering scratch buffer.
        let mut all: SmallVec<[(ComponentID, bool); 8]> =
            SmallVec::with_capacity(read_ids.len() + write_ids.len());
        for &cid in read_ids {
            all.push((cid, false));
        }
        for &cid in write_ids {
            all.push((cid, true));
        }

        // Lock in ascending ComponentID order (global lock ordering contract).
        all.sort_unstable_by_key(|(cid, _)| *cid);

        // Guard storage.
        let mut read_guards: SmallVec<
            [(
                ComponentID,
                RwLockReadGuard<'a, Box<dyn TypeErasedAttribute>>,
            ); 8],
        > = SmallVec::with_capacity(read_ids.len());
        let mut write_guards: SmallVec<
            [(
                ComponentID,
                RwLockWriteGuard<'a, Box<dyn TypeErasedAttribute>>,
            ); 8],
        > = SmallVec::with_capacity(write_ids.len());

        for (cid, is_write) in all {
            // Use find_component for sparse lookup.
            let attr = self
                .find_component(cid)
                .ok_or(ExecutionError::MissingComponent { component_id: cid })?;

            if is_write {
                let g = attr
                    .write()
                    .map_err(|_| ExecutionError::InternalExecutionError)?;
                write_guards.push((cid, g));
            } else {
                let g = attr
                    .read()
                    .map_err(|_| ExecutionError::InternalExecutionError)?;
                read_guards.push((cid, g));
            }
        }

        // Read/Write view buffers.
        let mut reads: SmallVec<[(*const u8, usize); 8]> = SmallVec::with_capacity(read_ids.len());
        for &cid in read_ids {
            let guard = read_guards
                .iter()
                .find(|(id, _)| *id == cid)
                .map(|(_, g)| g)
                .ok_or(ExecutionError::InternalExecutionError)?;

            let (ptr, bytes) = guard
                .as_ref()
                .chunk_bytes(chunk, length)
                .ok_or(ExecutionError::InternalExecutionError)?;

            reads.push((ptr, bytes));
        }

        let mut writes: SmallVec<[*mut u8; 8]> = SmallVec::with_capacity(write_ids.len());
        for &cid in write_ids {
            let guard = write_guards
                .iter_mut()
                .find(|(id, _)| *id == cid)
                .map(|(_, g)| g)
                .ok_or(ExecutionError::InternalExecutionError)?;

            let (ptr, _bytes) = guard
                .as_mut()
                .chunk_bytes_mut(chunk, length)
                .ok_or(ExecutionError::InternalExecutionError)?;

            writes.push(ptr);
        }

        Ok(ChunkBorrow {
            length,
            reads,
            writes,
            _read_guards: read_guards,
            _write_guards: write_guards,
        })
    }
}
