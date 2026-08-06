//! Per-worker staging buffers for parallel accumulation.
//!
//! Extension modules that collect items from inside parallel `for_each`
//! bodies (space reindexing, network mutation deltas) need a place to write
//! without locks. [`WorkerStage`] gives every Rayon worker its own slot,
//! indexed by [`worker_id`](crate::engine::workers::worker_id), with the same
//! phase discipline the messaging emit path uses: **Phase A** (systems
//! running inside a stage) writes only the calling worker's slot; **Phase B**
//! (boundary `finalise`, which holds `&mut self` through the boundary
//! resource write lock) drains all slots exclusively.
//!
//! Threads whose worker id falls outside the Rayon pool range (foreign
//! threads such as tests or `main`) fall back to a mutex-guarded overflow
//! vector - correct, just not lock-free.
//!
//! Determinism note: which worker stages which item depends on Rayon's
//! work-stealing, so the concatenated drain order is *not* stable across
//! runs. Consumers that need run-to-run stable output must impose an order
//! after draining (the space index sorts each cell's occupants by entity id,
//! the network store sorts its deltas).

use std::cell::UnsafeCell;
use std::sync::Mutex;

use crate::engine::workers::{max_workers, worker_id};

/// Lock-free per-worker staging for values produced inside parallel stages.
pub(crate) struct WorkerStage<T> {
    /// One slot per Rayon pool worker, indexed by `worker_id()`.
    slots: Vec<UnsafeCell<Vec<T>>>,
    /// Fallback for foreign-thread worker ids (sparse, near `u32::MAX`).
    overflow: Mutex<Vec<T>>,
}

// SAFETY: Phase discipline, identical to `WorkerEmitSlots` in
// `messaging::thread_local_emit`:
// - `push` writes only the slot belonging to the calling worker thread
//   (worker ids are stable per thread, and a thread runs one Rayon task at a
//   time), so no two threads ever write one slot concurrently;
// - `drain_into` / `clear` take `&mut self`, which boundary resources only
//   obtain under their per-resource write lock after all systems of the
//   stage have returned.
unsafe impl<T: Send> Send for WorkerStage<T> {}
unsafe impl<T: Send> Sync for WorkerStage<T> {}

impl<T: Send> WorkerStage<T> {
    /// Creates a stage sized for the current Rayon pool.
    pub(crate) fn new() -> Self {
        let workers = max_workers() as usize;
        Self {
            slots: (0..workers).map(|_| UnsafeCell::new(Vec::new())).collect(),
            overflow: Mutex::new(Vec::new()),
        }
    }

    /// Appends `value` to the calling worker's slot (Phase A).
    #[inline]
    pub(crate) fn push(&self, value: T) {
        let id = worker_id() as usize;
        if let Some(slot) = self.slots.get(id) {
            // SAFETY: this slot belongs exclusively to the calling worker
            // during Phase A; see the impl-level safety comment.
            let items = unsafe { &mut *slot.get() };
            items.push(value);
        } else if let Ok(mut overflow) = self.overflow.lock() {
            overflow.push(value);
        }
        // A poisoned overflow mutex would silently drop foreign-thread items;
        // poisoning requires a panic mid-push, which itself aborts the tick.
    }

    /// Moves every staged item into `out`, slot order then overflow (Phase B).
    pub(crate) fn drain_into(&mut self, out: &mut Vec<T>) {
        for slot in &mut self.slots {
            out.append(slot.get_mut());
        }
        if let Ok(mut overflow) = self.overflow.lock() {
            out.append(&mut overflow);
        }
    }

    /// Discards every staged item without deallocating slot capacity.
    pub(crate) fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.get_mut().clear();
        }
        if let Ok(mut overflow) = self.overflow.lock() {
            overflow.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_pushes_all_arrive() {
        let stage = WorkerStage::<usize>::new();
        rayon::scope(|s| {
            for task in 0..64usize {
                let stage = &stage;
                s.spawn(move |_| {
                    for i in 0..100 {
                        stage.push(task * 100 + i);
                    }
                });
            }
        });

        let mut stage = stage;
        let mut out = Vec::new();
        stage.drain_into(&mut out);
        assert_eq!(out.len(), 6_400);
        out.sort_unstable();
        assert_eq!(out, (0..6_400).collect::<Vec<_>>());

        // Drained slots are empty; clear() keeps them reusable.
        let mut again = Vec::new();
        stage.drain_into(&mut again);
        assert!(again.is_empty());
    }
}
