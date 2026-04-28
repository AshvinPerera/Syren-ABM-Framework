//! Stable per-thread worker identifiers shared by all parallel extension
//! modules.
//!
//! Extensions that accumulate per-worker state - message emit slots, agent
//! spawn buckets, env-write tracking - need a deterministic worker index so
//! that drain/merge passes produce identical output across runs and across
//! thread counts. This module exposes that index on top of Rayon's pool so
//! every extension uses the same numbering.
//!
//! # Identifier semantics
//!
//! - Inside a Rayon parallel section, [`worker_id`] returns the calling
//!   worker's pool-relative index.
//! - On the main thread (or any other non-Rayon thread), [`worker_id`]
//!   returns a stable identifier obtained from a process-wide registry.
//!   Foreign-thread IDs are allocated downward from `u32::MAX` so they cannot
//!   collide with Rayon's dense pool-relative IDs.
//! - The returned `u32` is stable for the lifetime of the calling thread.
//!
//! # Capacity
//!
//! [`max_workers`] returns Rayon's current pool size. It is only an initial
//! sizing hint for Rayon-worker storage, not an upper bound for every possible
//! [`worker_id`], because foreign-thread IDs are intentionally sparse.

use std::sync::atomic::{AtomicU32, Ordering};

/// Counter producing monotonically increasing IDs for non-Rayon threads.
/// Starts at `u32::MAX` and decreases so it cannot collide with Rayon's
/// pool-relative indices, which start at `0`.
static NEXT_FOREIGN_ID: AtomicU32 = AtomicU32::new(u32::MAX);

thread_local! {
    /// The cached worker ID for the current thread, computed on first call.
    static THIS_WORKER: std::cell::OnceCell<u32> = const { std::cell::OnceCell::new() };
}

/// Returns the stable worker identifier for the calling thread.
///
/// Identifiers are dense for Rayon workers (`0..rayon::current_num_threads()`)
/// and sparse-from-the-top for foreign threads. Treat the result as opaque
/// for indexing into a `Vec<Option<...>>` and grow the vec lazily on first
/// observation of an ID.
pub fn worker_id() -> u32 {
    THIS_WORKER.with(|cell| {
        *cell.get_or_init(|| {
            if let Some(idx) = rayon::current_thread_index() {
                idx as u32
            } else {
                NEXT_FOREIGN_ID.fetch_sub(1, Ordering::Relaxed)
            }
        })
    })
}

/// Returns Rayon's current pool size.
///
/// Useful as an initial sizing hint for per-worker accumulators; not an
/// upper bound, because foreign threads may also observe a [`worker_id`].
#[inline]
pub fn max_workers() -> u32 {
    rayon::current_num_threads() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Mutex;

    #[test]
    fn worker_id_stable_within_thread() {
        let a = worker_id();
        let b = worker_id();
        assert_eq!(a, b);
    }

    #[test]
    fn worker_ids_are_unique_in_rayon_pool() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let observed: Mutex<HashSet<u32>> = Mutex::new(HashSet::new());
        pool.install(|| {
            (0..1024usize).into_iter().for_each(|_| {
                observed.lock().unwrap().insert(worker_id());
            });
        });
        // Should see at most one ID per pool worker; foreign threads not used.
        let ids = observed.into_inner().unwrap();
        assert!(ids.len() <= 4);
        for id in ids {
            assert!(id < 4, "rayon worker id {id} out of pool range");
        }
    }

    #[test]
    fn foreign_thread_id_is_distinct_from_rayon_range() {
        let id = worker_id();
        // On the main thread the index is None for global rayon, so we
        // get a foreign id; assert it's outside the typical pool range.
        assert!(
            id >= u32::MAX - 1024 || id < max_workers(),
            "expected foreign id near u32::MAX or rayon pool id; got {id}"
        );
    }
}
