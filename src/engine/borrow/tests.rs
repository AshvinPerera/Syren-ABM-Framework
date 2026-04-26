//! Unit tests for the [`BorrowTracker`] and [`BorrowGuard`] types.
//!
//! Coverage includes:
//!
//! - **BorrowTracker** — initial state, single and multiple readers, write
//!   acquisition and release, conflict detection (read-write, write-read,
//!   write-write), dirty-tracked [`BorrowTracker::clear`], and the atomicity
//!   guarantee that the last reader transitions directly `2 → 0` without
//!   passing through the transient write-lock state `1`.
//!
//! - **BorrowGuard** — RAII acquisition and release on drop, rejection of
//!   overlapping read/write component sets, deduplication of repeated
//!   component IDs, and rollback of partially-acquired locks on failure.
//!
//! - **Concurrency** — multithreaded readers holding locks simultaneously,
//!   and a stress test asserting that state `1` is never transiently
//!   observable during concurrent read-only access.

use super::*;
use crate::engine::error::{AccessKind, ExecutionError, InvalidAccessReason};
use crate::engine::types::ComponentID;
use std::sync::atomic::Ordering;

// Helpers

/// Asserts that `result` is a `BorrowConflict` error with the expected fields.
macro_rules! assert_borrow_conflict {
    ($result:expr, $id:expr, $held:expr, $requested:expr) => {
        match $result.unwrap_err() {
            ExecutionError::BorrowConflict {
                component_id,
                held,
                requested,
            } => {
                assert_eq!(component_id, $id);
                assert_eq!(held, $held);
                assert_eq!(requested, $requested);
            }
            other => panic!("expected BorrowConflict, got {:?}", other),
        }
    };
}

// BorrowTracker unit tests

#[test]
fn new_tracker_is_unlocked() {
    let tracker = BorrowTracker::new();
    // All states should be 0 (unlocked).
    for s in tracker.states.iter() {
        assert_eq!(s.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn single_read_acquire_and_release() {
    let tracker = BorrowTracker::new();
    let id: ComponentID = 7;

    tracker.acquire_read(id).unwrap();
    assert_eq!(tracker.states[id as usize].load(Ordering::Relaxed), 2);

    tracker.release_read(id);
    assert_eq!(tracker.states[id as usize].load(Ordering::Relaxed), 0);
}

#[test]
fn multiple_readers() {
    let tracker = BorrowTracker::new();
    let id: ComponentID = 3;

    tracker.acquire_read(id).unwrap();
    tracker.acquire_read(id).unwrap();
    tracker.acquire_read(id).unwrap();
    // 3 readers → state = 4
    assert_eq!(tracker.states[id as usize].load(Ordering::Relaxed), 4);

    tracker.release_read(id);
    // 2 readers → state = 3
    assert_eq!(tracker.states[id as usize].load(Ordering::Relaxed), 3);

    tracker.release_read(id);
    // 1 reader → state = 2
    assert_eq!(tracker.states[id as usize].load(Ordering::Relaxed), 2);

    tracker.release_read(id);
    // 0 readers → state = 0 (unlocked, not 1)
    assert_eq!(tracker.states[id as usize].load(Ordering::Relaxed), 0);
}

#[test]
fn write_acquire_and_release() {
    let tracker = BorrowTracker::new();
    let id: ComponentID = 5;

    tracker.acquire_write(id).unwrap();
    assert_eq!(tracker.states[id as usize].load(Ordering::Relaxed), 1);

    tracker.release_write(id);
    assert_eq!(tracker.states[id as usize].load(Ordering::Relaxed), 0);
}

#[test]
fn write_blocked_by_reader_returns_borrow_conflict() {
    let tracker = BorrowTracker::with_spin_limit(0);
    let id: ComponentID = 10;

    tracker.acquire_read(id).unwrap();

    let result = tracker.acquire_write(id);
    assert!(result.is_err());
    assert_borrow_conflict!(result, id, AccessKind::Read, AccessKind::Write);

    // Clean up.
    tracker.release_read(id);
}

#[test]
fn read_blocked_by_writer_returns_borrow_conflict() {
    let tracker = BorrowTracker::with_spin_limit(0);
    let id: ComponentID = 12;

    tracker.acquire_write(id).unwrap();

    let result = tracker.acquire_read(id);
    assert!(result.is_err());
    assert_borrow_conflict!(result, id, AccessKind::Write, AccessKind::Read);

    // Clean up.
    tracker.release_write(id);
}

#[test]
fn write_blocked_by_writer_returns_borrow_conflict() {
    let tracker = BorrowTracker::with_spin_limit(0);
    let id: ComponentID = 15;

    tracker.acquire_write(id).unwrap();

    let result = tracker.acquire_write(id);
    assert!(result.is_err());
    assert_borrow_conflict!(result, id, AccessKind::Write, AccessKind::Write);

    // Clean up.
    tracker.release_write(id);
}

#[test]
fn clear_resets_all_states() {
    let tracker = BorrowTracker::new();

    tracker.acquire_read(0).unwrap();
    tracker.acquire_read(1).unwrap();
    tracker.acquire_write(2).unwrap();

    tracker.clear();

    // Verify touched components are cleared.
    assert_eq!(tracker.states[0].load(Ordering::Relaxed), 0);
    assert_eq!(tracker.states[1].load(Ordering::Relaxed), 0);
    assert_eq!(tracker.states[2].load(Ordering::Relaxed), 0);

    // Verify dirty bitset is also cleared.
    for word in &tracker.dirty {
        assert_eq!(word.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn clear_only_touches_dirty_components() {
    // This test verifies that clear() uses dirty tracking:
    // components that were never borrowed should remain untouched.
    let tracker = BorrowTracker::new();

    // Borrow a few specific components.
    tracker.acquire_read(100).unwrap();
    tracker.acquire_write(200).unwrap();

    // Manually set a non-dirty component to a sentinel value to verify
    // clear() doesn't touch it. This is a white-box test.
    tracker.states[255].store(42, Ordering::Relaxed);

    tracker.clear();

    // Borrowed components should be cleared.
    assert_eq!(tracker.states[100].load(Ordering::Relaxed), 0);
    assert_eq!(tracker.states[200].load(Ordering::Relaxed), 0);

    // Non-dirty component should still have sentinel value.
    assert_eq!(tracker.states[255].load(Ordering::Relaxed), 42);

    // Clean up the sentinel so it doesn't affect other tests.
    tracker.states[255].store(0, Ordering::Relaxed);
}

#[test]
fn release_read_last_reader_goes_to_zero_not_one() {
    // This test verifies the fix for the race condition in the original
    // release_read. The last reader must transition 2 → 0 atomically
    // (via compare_exchange), never passing through the transient state
    // 1 which is indistinguishable from a write lock.
    let tracker = BorrowTracker::new();
    let id: ComponentID = 20;

    tracker.acquire_read(id).unwrap();
    assert_eq!(tracker.states[id as usize].load(Ordering::Relaxed), 2);

    tracker.release_read(id);
    // Must be 0, not 1.
    assert_eq!(tracker.states[id as usize].load(Ordering::Relaxed), 0);
}

#[test]
fn reacquire_after_release() {
    let tracker = BorrowTracker::new();
    let id: ComponentID = 0;

    tracker.acquire_write(id).unwrap();
    tracker.release_write(id);

    // Should be acquirable again.
    tracker.acquire_read(id).unwrap();
    tracker.release_read(id);

    tracker.acquire_write(id).unwrap();
    tracker.release_write(id);
}

#[test]
fn independent_components_do_not_interfere() {
    let tracker = BorrowTracker::new();

    tracker.acquire_write(0).unwrap();
    tracker.acquire_read(1).unwrap();
    tracker.acquire_write(2).unwrap();

    assert_eq!(tracker.states[0].load(Ordering::Relaxed), 1);
    assert_eq!(tracker.states[1].load(Ordering::Relaxed), 2);
    assert_eq!(tracker.states[2].load(Ordering::Relaxed), 1);

    tracker.release_write(0);
    tracker.release_read(1);
    tracker.release_write(2);
}

// BorrowGuard unit tests

#[test]
fn guard_acquires_and_releases_on_drop() {
    let tracker = BorrowTracker::new();

    {
        let _guard = BorrowGuard::new(&tracker, &[0, 1], &[2]).unwrap();
        assert_eq!(tracker.states[0].load(Ordering::Relaxed), 2); // read
        assert_eq!(tracker.states[1].load(Ordering::Relaxed), 2); // read
        assert_eq!(tracker.states[2].load(Ordering::Relaxed), 1); // write
    }

    // After guard is dropped, everything should be unlocked.
    assert_eq!(tracker.states[0].load(Ordering::Relaxed), 0);
    assert_eq!(tracker.states[1].load(Ordering::Relaxed), 0);
    assert_eq!(tracker.states[2].load(Ordering::Relaxed), 0);
}

#[test]
fn guard_rejects_read_write_overlap() {
    let tracker = BorrowTracker::new();
    let result = BorrowGuard::new(&tracker, &[5], &[5]);

    assert!(result.is_err());
    match result.unwrap_err() {
        ExecutionError::InvalidQueryAccess {
            component_id,
            reason,
        } => {
            assert_eq!(component_id, 5);
            assert_eq!(reason, InvalidAccessReason::ReadAndWrite);
        }
        other => panic!("expected InvalidQueryAccess, got {:?}", other),
    }
}

#[test]
fn guard_deduplicates_components() {
    let tracker = BorrowTracker::new();

    let _guard = BorrowGuard::new(&tracker, &[1, 1, 1], &[2, 2]).unwrap();
    // 1 reader (deduped), not 3.
    assert_eq!(tracker.states[1].load(Ordering::Relaxed), 2);
    assert_eq!(tracker.states[2].load(Ordering::Relaxed), 1);
}

#[test]
fn guard_rollback_on_partial_failure() {
    let tracker = BorrowTracker::with_spin_limit(0);

    // Pre-lock component 1 with a write so that the guard's read
    // acquisition for component 1 will fail.
    tracker.acquire_write(1).unwrap();

    // Try to acquire reads on [0, 1] and writes on [2].
    // Write on 2 should succeed, read on 0 should succeed,
    // read on 1 should fail — and then 0 and 2 should be released.
    let result = BorrowGuard::new(&tracker, &[0, 1], &[2]);
    assert!(result.is_err());

    // Component 0 and 2 should have been rolled back to unlocked.
    assert_eq!(tracker.states[0].load(Ordering::Relaxed), 0);
    assert_eq!(tracker.states[2].load(Ordering::Relaxed), 0);

    // Component 1 should still be write-locked (by our pre-lock).
    assert_eq!(tracker.states[1].load(Ordering::Relaxed), 1);

    // Clean up.
    tracker.release_write(1);
}

// Concurrent tests (require std::thread)

#[test]
fn concurrent_readers_do_not_block() {
    use std::sync::Arc;

    let tracker = Arc::new(BorrowTracker::new());
    let id: ComponentID = 42;
    let num_threads = 8;

    let barrier = Arc::new(std::sync::Barrier::new(num_threads));
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let t = Arc::clone(&tracker);
            let b = Arc::clone(&barrier);
            std::thread::spawn(move || {
                t.acquire_read(id).unwrap();
                b.wait();
                // All threads should be here simultaneously with read locks held.
                t.release_read(id);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tracker.states[id as usize].load(Ordering::Relaxed), 0);
}

#[test]
fn concurrent_release_read_never_produces_state_one() {
    // Stress test: many readers acquire and release concurrently.
    // At no point should the state ever be observed as 1 (write-locked),
    // which was possible with the old fetch_sub + store implementation.
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    let tracker = Arc::new(BorrowTracker::new());
    let id: ComponentID = 99;
    let stop = Arc::new(AtomicBool::new(false));

    // Observer thread: polls the state and asserts it's never 1
    // while only readers are active.
    let observer_tracker = Arc::clone(&tracker);
    let observer_stop = Arc::clone(&stop);
    let observer = std::thread::spawn(move || {
        while !observer_stop.load(Ordering::Relaxed) {
            let s = observer_tracker.states[id as usize].load(Ordering::Relaxed);
            assert_ne!(
                s, 1,
                "observed transient write-lock state (1) during read-only access"
            );
        }
    });

    // Worker threads: acquire and release reads in a tight loop.
    let num_workers = 4;
    let iterations = 10_000;
    let handles: Vec<_> = (0..num_workers)
        .map(|_| {
            let t = Arc::clone(&tracker);
            std::thread::spawn(move || {
                for _ in 0..iterations {
                    t.acquire_read(id).unwrap();
                    t.release_read(id);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    stop.store(true, Ordering::Relaxed);
    observer.join().unwrap();

    assert_eq!(tracker.states[id as usize].load(Ordering::Relaxed), 0);
}
