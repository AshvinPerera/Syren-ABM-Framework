//! Scoped borrow guard for coordinating component access across systems.
//!
//! This module provides [`BorrowGuard`], a RAII type that acquires and releases
//! read/write borrows on [`ComponentID`]s through a shared [`BorrowTracker`].
//! It is the primary mechanism by which the engine enforces Rust-like aliasing
//! rules at runtime — ensuring no component is accessed mutably while any other
//! system holds a reference to it.
//!
//! ## Acquisition protocol
//!
//! Borrows are acquired in a deterministic order to minimize deadlock risk:
//!
//! 1. Component IDs are **sorted** before acquisition.
//! 2. **Writes** are acquired before reads.
//! 3. If any acquisition fails, all previously acquired borrows are
//!    **rolled back** before the error is returned.
//!
//! ## Usage
//!
//! A guard is typically constructed by the scheduler immediately before a
//! system runs and is dropped when the system's closure returns:
//!
//! ```rust
//! let guard = BorrowGuard::new(&tracker, &[position_id], &[velocity_id])?;
//! // system executes here
//! // borrows released automatically when `guard` is dropped
//! ```
//!
//! Attempting to include the same [`ComponentID`] in both the read and write
//! sets, or acquiring a borrow that conflicts with one held by another system,
//! will return an [`ExecutionError`] without leaving the tracker in a dirty state.

use std::fmt;

use crate::engine::types::ComponentID;
use crate::engine::error::{ExecutionError, InvalidAccessReason};
use super::tracker::BorrowTracker;

/// RAII guard representing a system or query's full borrow lifetime.
///
/// When created, this guard:
/// - Acquires all requested write borrows
/// - Then acquires all requested read borrows
///
/// When dropped, all borrows are released automatically.
///
/// ## Ordering
///
/// Component IDs are sorted before acquisition to reduce deadlock risk.
/// Writes are always acquired before reads.

pub struct BorrowGuard<'a> {
    tracker: &'a BorrowTracker,
    reads: Vec<ComponentID>,
    writes: Vec<ComponentID>,
}

impl fmt::Debug for BorrowGuard<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowGuard")
            .field("reads", &self.reads)
            .field("writes", &self.writes)
            .finish()
    }
}

impl<'a> BorrowGuard<'a> {

    /// Creates a new `BorrowGuard` and acquires all requested borrows.
    ///
    /// ## Parameters
    ///
    /// - `tracker`: Shared borrow tracker
    /// - `reads`: Component types to borrow immutably
    /// - `writes`: Component types to borrow mutably
    ///
    /// ## Errors
    ///
    /// - Returns `ExecutionError::InvalidQueryAccess` if any component
    ///   appears in both `reads` and `writes`.
    /// - Returns `ExecutionError::BorrowConflict` if acquiring any borrow
    ///   exceeds the tracker's spin limit.
    ///
    /// ## Rollback on Failure
    ///
    /// If acquisition fails partway through, all borrows that were
    /// successfully acquired are released before the error is returned.

    pub fn new(
        tracker: &'a BorrowTracker,
        reads: &[ComponentID],
        writes: &[ComponentID],
    ) -> Result<Self, ExecutionError> {
        let mut r = reads.to_vec();
        let mut w = writes.to_vec();

        r.sort_unstable();
        w.sort_unstable();

        r.dedup();
        w.dedup();

        for component_id in &r {
            if w.binary_search(component_id).is_ok() {
                return Err(ExecutionError::InvalidQueryAccess {
                    component_id: *component_id,
                    reason: InvalidAccessReason::ReadAndWrite,
                });
            }
        }

        // Acquire writes first, then reads.
        // On failure, release everything acquired so far.
        let mut acquired_writes: Vec<ComponentID> = Vec::with_capacity(w.len());
        let mut acquired_reads: Vec<ComponentID> = Vec::with_capacity(r.len());

        let rollback = |aw: &[ComponentID], ar: &[ComponentID]| {
            for &id in ar.iter().rev() { tracker.release_read(id); }
            for &id in aw.iter().rev() { tracker.release_write(id); }
        };

        for &component_id in &w {
            if let Err(e) = tracker.acquire_write(component_id) {
                rollback(&acquired_writes, &acquired_reads);
                return Err(e);
            }
            acquired_writes.push(component_id);
        }

        for &component_id in &r {
            if let Err(e) = tracker.acquire_read(component_id) {
                rollback(&acquired_writes, &acquired_reads);
                return Err(e);
            }
            acquired_reads.push(component_id);
        }

        Ok(Self { tracker, reads: r, writes: w })
    }
}

impl Drop for BorrowGuard<'_> {
    /// Releases all acquired borrows in reverse order.
    fn drop(&mut self) {
        for &component_id in self.reads.iter().rev() { self.tracker.release_read(component_id); }
        for &component_id in self.writes.iter().rev() { self.tracker.release_write(component_id); }
    }
}
