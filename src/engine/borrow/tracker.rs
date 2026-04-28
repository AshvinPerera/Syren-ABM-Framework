//! Runtime borrow tracking for ECS component access.
//!
//! This module implements [`BorrowTracker`], a lock-free spinlock-based
//! structure that enforces Rust's aliasing rules across systems executing
//! in parallel during a single scheduler stage.
//!
//! # Borrow State Model
//!
//! Each component slot is represented by an [`AtomicUsize`] encoding one
//! of three states:
//!
//! | Value | Meaning                                      |
//! |-------|----------------------------------------------|
//! | `0`   | Unlocked - no active borrows                 |
//! | `1`   | Write-locked - one exclusive writer          |
//! | `>= 2` | Read-locked - `state - 1` concurrent readers |
//!
//! This encoding ensures that the transient value `1` is never produced
//! during read-lock transitions, keeping write-lock detection unambiguous.
//!
//! # Contention Handling
//!
//! Acquisition methods ([`BorrowTracker::acquire_read`],
//! [`BorrowTracker::acquire_write`]) spin using [`std::hint::spin_loop`],
//! yielding the thread every 1 024 iterations. If the configured
//! [`spin_limit`](BorrowTracker::with_spin_limit) is exceeded,
//! [`ExecutionError::BorrowConflict`] is returned - signalling a scheduling
//! bug rather than hanging indefinitely.
//!
//! # Stage Boundaries
//!
//! A dirty bitset ([`BorrowTracker::dirty`]) records every component touched
//! within a stage. [`BorrowTracker::clear`] uses this bitset to reset only
//! the affected slots at stage boundaries, avoiding a full `O(COMPONENT_CAP)`
//! sweep on every flush.
//!
//! # Allocation
//!
//! The per-component state array is heap-allocated to prevent a ~32 KB stack
//! frame during construction (see [`BorrowTracker::with_spin_limit`]).

use crate::engine::error::{AccessKind, ExecutionError};
use crate::engine::types::{ComponentID, COMPONENT_CAP};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Default maximum number of spin iterations before `acquire_read` or
/// `acquire_write` gives up and returns `Err(ExecutionError::BorrowConflict)`.
///
/// This value is high enough to tolerate normal contention bursts under
/// parallel Rayon dispatch, but low enough to surface scheduling bugs
/// within a reasonable wall-clock window rather than hanging forever.
pub const DEFAULT_SPIN_LIMIT: u32 = 100_000;

/// Number of `AtomicU64` words needed to cover `COMPONENT_CAP` bits.
const DIRTY_WORDS: usize = (COMPONENT_CAP + 63) / 64;

/// Tracks runtime read/write borrows for each component type.
///
/// Each component has an associated atomic state encoding whether it is:
/// - Unborrowed (`0`)
/// - Borrowed mutably / write-locked (`1`)
/// - Borrowed immutably by one or more readers (`>= 2`, with `state - 1` readers)
///
/// A configurable `spin_limit` controls how many iterations the acquire
/// methods will attempt before returning an error.

pub struct BorrowTracker {
    /// Per-component atomic borrow state.
    ///
    /// Heap-allocated to avoid 32 KB stack frame during construction.
    pub(super) states: Box<[AtomicUsize; COMPONENT_CAP]>,

    /// Bitset tracking which component IDs were touched
    /// (borrowed read or write) during the current stage.
    pub(crate) dirty: [AtomicU64; DIRTY_WORDS],

    /// Maximum spin iterations before acquisition fails with
    /// [`ExecutionError::BorrowConflict`].
    spin_limit: u32,
}

impl BorrowTracker {
    /// Creates a new `BorrowTracker` with all components unlocked and the
    /// default spin limit ([`DEFAULT_SPIN_LIMIT`]).

    pub fn new() -> Self {
        Self::with_spin_limit(DEFAULT_SPIN_LIMIT)
    }

    /// Creates a new `BorrowTracker` with a custom spin limit.
    ///
    /// ## Parameters
    ///
    /// - `spin_limit`: Maximum number of spin iterations before `acquire_read`
    ///   or `acquire_write` returns `Err(ExecutionError::BorrowConflict)`.
    ///   A value of `0` means the acquire methods will attempt exactly one
    ///   compare-exchange before failing.
    pub fn with_spin_limit(spin_limit: u32) -> Self {
        // Phase 2.3: Heap-allocate `states` to avoid 32 KB on the stack.
        //
        // SAFETY: The layout is valid (non-zero size - COMPONENT_CAP > 0 and
        // AtomicUsize is at least 1 word).  `alloc_zeroed` returns zeroed
        // memory, which is a valid bit pattern for `AtomicUsize` (represents 0,
        // the "unlocked" state).  `Box::from_raw` takes ownership and will
        // deallocate with the matching layout on drop.
        let states: Box<[AtomicUsize; COMPONENT_CAP]> = unsafe {
            let layout = std::alloc::Layout::new::<[AtomicUsize; COMPONENT_CAP]>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut [AtomicUsize; COMPONENT_CAP];
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr)
        };

        Self {
            states,
            dirty: std::array::from_fn(|_| AtomicU64::new(0)),
            spin_limit,
        }
    }

    /// Marks a component ID as touched in the dirty bitset.
    #[inline]
    fn mark_dirty(&self, component_id: ComponentID) {
        let word = component_id as usize / 64;
        let bit = component_id as usize % 64;
        self.dirty[word].fetch_or(1u64 << bit, Ordering::Relaxed);
    }

    /// Resets borrow states for all components that were touched during
    /// the current stage.
    ///
    /// ## Purpose
    ///
    /// Called at stage boundaries by the scheduler to ensure no stale borrow
    /// state leaks across execution phases.

    #[inline]
    pub fn clear(&self) {
        for word_idx in 0..DIRTY_WORDS {
            let bits = self.dirty[word_idx].swap(0, Ordering::Relaxed);
            if bits == 0 {
                continue;
            }
            let base = word_idx * 64;
            let mut remaining = bits;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let component_id = base + bit;
                if component_id < COMPONENT_CAP {
                    self.states[component_id].store(0, Ordering::Release);
                }
                remaining &= remaining - 1; // clear lowest set bit
            }
        }
    }

    /// Acquires a **shared (read) borrow** for the given component.
    ///
    /// Spins until no writer is present, then atomically increments the
    /// reader count. Returns `Err(ExecutionError::BorrowConflict)` if the
    /// spin limit is exceeded, which typically indicates a scheduling bug
    /// (e.g., two systems with conflicting access placed in the same stage).
    ///
    /// ## Behaviour
    ///
    /// - Spins while the component is write-locked (state == `1`).
    /// - Allows multiple concurrent readers.
    ///
    /// ## State Transitions
    ///
    /// - `0 -> 2` : first reader
    /// - `N -> N+1` : additional reader (where `N >= 2`)

    pub fn acquire_read(&self, component_id: ComponentID) -> Result<(), ExecutionError> {
        let state = &self.states[component_id as usize];
        let mut spins = 0u32;

        loop {
            let current = state.load(Ordering::Acquire);

            // State 1 means write-locked - spin or bail.
            if current == 1 {
                spins += 1;
                if spins > self.spin_limit {
                    return Err(ExecutionError::BorrowConflict {
                        component_id,
                        held: AccessKind::Write,
                        requested: AccessKind::Read,
                    });
                }
                if spins % 1024 == 0 {
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
                continue;
            }

            // 0 -> 2 (first reader) or N -> N+1 (additional reader).
            let next = if current == 0 { 2 } else { current + 1 };

            if state
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // Phase 3.1: record this component as touched.
                self.mark_dirty(component_id);
                return Ok(());
            }

            spins += 1;
            if spins > self.spin_limit {
                return Err(ExecutionError::BorrowConflict {
                    component_id,
                    held: AccessKind::Write,
                    requested: AccessKind::Read,
                });
            }
            if spins % 1024 == 0 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
    }

    /// Releases a previously acquired **shared (read) borrow**.
    ///
    /// ## Behaviour
    ///
    /// Uses a `compare_exchange` loop to atomically transition:
    /// - `2 -> 0` when this is the last reader (avoids the transient state `1`
    ///   which is indistinguishable from a write lock).
    /// - `N -> N-1` when other readers remain (where `N > 2`).
    ///
    /// ## Safety
    ///
    /// This method assumes a matching `acquire_read` call was made previously.
    /// Calling without a corresponding 'acquire' is a logic error and will
    /// trigger a `debug_assert` failure.

    pub fn release_read(&self, component_id: ComponentID) {
        let state = &self.states[component_id as usize];

        loop {
            let current = state.load(Ordering::Acquire);
            debug_assert!(
                current >= 2,
                "release_read called with state {} for component {}; expected >= 2",
                current,
                component_id
            );

            // If exactly one reader (state == 2), go straight to 0 (unlocked).
            // If multiple readers (state > 2), decrement by 1.
            // Both transitions are done atomically via compare_exchange to avoid
            // the race where fetch_sub(1) would produce transient state 1.
            let next = if current == 2 { 0 } else { current - 1 };

            if state
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }

            // Another reader released concurrently - retry with fresh state.
            std::hint::spin_loop();
        }
    }

    /// Acquires an **exclusive (write) borrow** for the given component.
    ///
    /// Spins until the component is fully unlocked (`0`), then atomically
    /// transitions to write-locked (`1`). Returns
    /// `Err(ExecutionError::BorrowConflict)` if the spin limit is exceeded.
    ///
    /// ## Behaviour
    ///
    /// - Spins while any readers or another writer are present.
    /// - Only one writer may hold the lock at a time.
    ///
    /// ## State Transition
    ///
    /// - `0 -> 1`

    pub fn acquire_write(&self, component_id: ComponentID) -> Result<(), ExecutionError> {
        let state = &self.states[component_id as usize];
        let mut spins = 0u32;

        loop {
            let current = state.load(Ordering::Acquire);

            if current != 0 {
                spins += 1;
                if spins > self.spin_limit {
                    let held = if current == 1 {
                        AccessKind::Write
                    } else {
                        AccessKind::Read
                    };
                    return Err(ExecutionError::BorrowConflict {
                        component_id,
                        held,
                        requested: AccessKind::Write,
                    });
                }
                if spins % 1024 == 0 {
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
                continue;
            }

            if state
                .compare_exchange_weak(0, 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // Phase 3.1: record this component as touched.
                self.mark_dirty(component_id);
                return Ok(());
            }

            spins += 1;
            if spins > self.spin_limit {
                return Err(ExecutionError::BorrowConflict {
                    component_id,
                    held: AccessKind::Read,
                    requested: AccessKind::Write,
                });
            }
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
        let state = &self.states[component_id as usize];
        let previous = state.swap(0, Ordering::AcqRel);
        debug_assert!(
            previous == 1,
            "release_write called with state {} for component {}; expected 1",
            previous,
            component_id
        );
    }
}
