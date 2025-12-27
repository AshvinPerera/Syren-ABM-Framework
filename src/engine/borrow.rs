//! # Borrow Tracking Module
//!
//! This module implements a **per-component read/write borrow tracker**
//!
//! ## Purpose
//!
//! The goal is to enforce Rust-like borrowing rules *at runtime* across
//! dynamically scheduled systems:
//!
//! - Multiple systems may **read** the same component type concurrently.
//! - Only one system may **write** to a component type at a time.
//! - No system may read a component type while another system writes it.
//!
//! This is achieved without OS locks by using **atomic state machines
//!
//! ## State Encoding
//!
//! Each component ID maps to one `AtomicUsize` with the following meaning:
//!
//! | State | Meaning |
//! |------:|--------|
//! | `0` | Unlocked |
//! | `1` | Write-locked (exclusive writer) |
//! | `>= 2` | Read-locked (`state - 1` active readers) |
//!
//! ## Synchronization Strategy
//!
//! - Uses atomic operations with acquire/release.
//! - The same thread re-acquiring the same component will deadlock.
//!
//! ## RAII Integration
//!
//! The [`BorrowGuard`] type provides RAII-style acquisition and release
//! of multiple component borrows for the full lifetime of a query or system.

use std::sync::atomic::{AtomicUsize, Ordering};
use crate::engine::types::{ComponentID, COMPONENT_CAP};
use crate::engine::error::{ExecutionError, InvalidAccessReason};


/// Tracks runtime read/write borrows for each component type.
///
/// Each component has an associated atomic state encoding whether it is:
/// - Unborrowed
/// - Borrowed mutably (write)
/// - Borrowed immutably by one or more readers

pub struct BorrowTracker {
    /// Per-component atomic borrow state.
    states: [AtomicUsize; COMPONENT_CAP],
}

impl BorrowTracker {
    /// Creates a new `BorrowTracker` with all components unlocked.
    pub fn new() -> Self {
        Self { states: std::array::from_fn(|_| AtomicUsize::new(0)) }
    }

    /// 
    #[inline]
    pub fn clear(&self) {
        for state in &self.states {
            state.store(0, Ordering::Release);
        }
    }

    /// Acquires a **shared (read) borrow** for the given component.
    ///
    /// This method spins until no writer is present, then atomically
    /// increments the reader count.
    ///
    /// ## Behavior
    ///
    /// - Blocks (spins) if the component is write-locked.
    /// - Allows multiple concurrent readers.
    ///
    /// ## State Transitions
    ///
    /// - `0 → 2` : first reader
    /// - `N → N+1` : additional reader
  
    pub fn acquire_read(&self, component_id: ComponentID) -> Result<(), ExecutionError> {
        let state = &self.states[component_id as usize];
        let mut spins = 0u32;

        loop {
            let current_state = state.load(Ordering::Acquire);

            if current_state == 1 {
                std::hint::spin_loop();
                continue;
            }

            let next = if current_state == 0 { 2 } else { current_state + 1 };

            if state.compare_exchange_weak(current_state, next, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                return Ok(());
            }

            spins += 1;
            if spins % 1024 == 0 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
    }


    /// Releases a previously acquired **shared (read) borrow**.
    ///
    /// ## Behavior
    ///
    /// - Decrements the reader count.
    /// - If this was the last reader, unlocks the component.
    ///
    /// ## Safety
    ///
    /// This method assumes a matching `acquire_read` call.

    pub fn release_read(&self, component_id: ComponentID) {
        let component_state = &self.states[component_id as usize];
        let previous_state = component_state.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous_state >= 2);
        if previous_state == 2 {
            component_state.store(0, Ordering::Release);
        }
    }

    /// Acquires an **exclusive (write) borrow** for the given component.
    ///
    /// This method spins until the component is fully unlocked.
    ///
    /// ## Behavior
    ///
    /// - Blocks if any readers or writers are present.
    /// - Only one writer may hold the lock at a time.
    ///
    /// ## State Transition
    ///
    /// - `0 → 1`

    pub fn acquire_write(&self, component_id: ComponentID) -> Result<(), ExecutionError> {
        let state = &self.states[component_id as usize];
        let mut spins = 0u32;

        loop {
            let current_state = state.load(Ordering::Acquire);

            if current_state != 0 {
                std::hint::spin_loop();
                continue;
            }

            if state.compare_exchange_weak(0, 1, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                return Ok(());
            }

            spins += 1;
            if spins % 1024 == 0 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
    }

    /// Releases a previously acquired **exclusive (write) borrow**.
    ///
    /// ## Safety
    ///
    /// This must only be called by the thread that successfully
    /// acquired the write lock.

    pub fn release_write(&self, component_id: ComponentID) {
        let component_state = &self.states[component_id as usize];
        let previous_state = component_state.swap(0, Ordering::AcqRel);
        debug_assert!(previous_state == 1);
    }
}

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

impl<'a> BorrowGuard<'a> {

    /// Creates a new `BorrowGuard` and acquires all requested borrows.
    ///
    /// ## Parameters
    ///
    /// - `tracker`: Shared borrow tracker
    /// - `reads`: Component types to borrow immutably
    /// - `writes`: Component types to borrow mutably
    
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

        for &component_id in &w { tracker.acquire_write(component_id)?; }
        for &component_id in &r { tracker.acquire_read(component_id)?; }

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
