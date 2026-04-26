//! Zero-cost, lock-free message emission via thread-local storage.
//!
//! # Design
//!
//! Message **emission** happens on Rayon worker threads inside a parallel
//! `for_each` invocation (Phase A).  Message **draining** happens in the
//! boundary stage that immediately follows (Phase B), on the main thread.
//! These two phases are **mutually exclusive** by scheduler design: the
//! boundary stage cannot start until all workers from the previous parallel
//! phase have returned.
//!
//! This phase discipline allows us to use [`UnsafeCell`] without any locking
//! during the emit path: a worker writes its own slot and no other thread
//! reads it until the drain phase begins.
//!
//! # Worker registration
//!
//! When a Rayon worker first calls [`emit`] it registers itself in
//! `GLOBAL_EMIT_REGISTRY` (protected by a `Mutex`, but only on the very first
//! call per worker — effectively once-per-thread).  The drain path iterates
//! this registry to collect all per-worker buffers.
//!
//! # Memory layout
//!
//! Each worker has a [`WorkerEmitSlots`] containing one
//! `Option<AlignedBuffer>` slot per registered message type (indexed by
//! [`MessageTypeID`]).  Slots are populated lazily on first emit; the drain
//! path calls [`AlignedBuffer::extend_from`] to merge them into the central
//! per-type buffer owned by a [`MessageBufferSet`](crate::messaging::MessageBufferSet).

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use super::aligned_buffer::AlignedBuffer;
use super::error::MessagingError;
use super::message::{Message, MessageTypeID};
use crate::engine::error::{ECSError, ECSResult};

/// Private identifier for one [`MessageBufferSet`](crate::messaging::MessageBufferSet)
/// runtime. It prevents two models with matching `MessageTypeID`s from sharing
/// thread-local emit slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MessageRuntimeID(u64);

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn alloc_runtime_id() -> MessageRuntimeID {
    MessageRuntimeID(NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed))
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-worker slot container
// ─────────────────────────────────────────────────────────────────────────────

/// One set of emit buffers per Rayon worker thread.
///
/// `slots[i]` is the pending buffer for message type with index `i`.
pub(crate) struct WorkerEmitSlots {
    /// Indexed by `MessageTypeID::index()`.  `None` means the buffer has not
    /// been created yet for this worker.
    slots: UnsafeCell<Vec<Option<AlignedBuffer>>>,
}

// SAFETY: The emit path writes only to the calling thread's own slots (no
// aliasing).  The drain path accesses every slot exactly once, after all
// workers have returned from Phase A (exclusive access by phase discipline).
unsafe impl Send for WorkerEmitSlots {}
unsafe impl Sync for WorkerEmitSlots {}

impl WorkerEmitSlots {
    fn new(num_message_types: usize) -> Self {
        let slots: Vec<Option<AlignedBuffer>> = (0..num_message_types).map(|_| None).collect();
        WorkerEmitSlots {
            slots: UnsafeCell::new(slots),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global registry of all workers
// ─────────────────────────────────────────────────────────────────────────────

/// All workers that have registered themselves.  Guarded by a `Mutex` but
/// only written to once per worker (first emit call on that thread).
static GLOBAL_EMIT_REGISTRY: LazyLock<Mutex<HashMap<MessageRuntimeID, Vec<Arc<WorkerEmitSlots>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

thread_local! {
    /// The current thread's slot container.  Initialised lazily on first use.
    static THIS_WORKER: std::cell::RefCell<HashMap<MessageRuntimeID, Arc<WorkerEmitSlots>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Initialises the thread-local worker slots for the current thread.
///
/// Must be called once before the first `emit` on each worker thread.  In
/// practice this is handled automatically by [`emit`]'s lazy initialisation.
fn ensure_worker_registered_fallible(
    runtime_id: MessageRuntimeID,
    num_message_types: usize,
) -> ECSResult<Arc<WorkerEmitSlots>> {
    THIS_WORKER.with(|cell| {
        if let Some(existing) = cell.borrow().get(&runtime_id).cloned() {
            return Ok(existing);
        }

        let slots = Arc::new(WorkerEmitSlots::new(num_message_types));
        cell.borrow_mut().insert(runtime_id, Arc::clone(&slots));
        GLOBAL_EMIT_REGISTRY
            .lock()
            .map_err(|_| ECSError::from(MessagingError::LockPoisoned("global emit registry")))?
            .entry(runtime_id)
            .or_default()
            .push(Arc::clone(&slots));
        Ok(slots)
    })
}
// ─────────────────────────────────────────────────────────────────────────────
// Emit (Phase A — called from worker threads)
// ─────────────────────────────────────────────────────────────────────────────

/// Emits a message into the calling thread's local buffer.
///
/// This function acquires **no locks** after the first call per thread
/// (worker registration happens once).  It is intended to be called inside
/// Rayon parallel sections.
///
/// # Panics
///
/// Panics if `mtid` is out of range for the registry (i.e. the message type
/// was not registered before the registry was frozen).
pub(crate) fn emit<M: Message>(
    runtime_id: MessageRuntimeID,
    num_message_types: usize,
    mtid: MessageTypeID,
    item_size: usize,
    item_align: usize,
    capacity: usize,
    msg: M,
) -> ECSResult<()> {
    let worker = ensure_worker_registered_fallible(runtime_id, num_message_types)?;

    // SAFETY: We are the only thread writing to this slot during Phase A.
    // No drain path is running concurrently.
    let slots = unsafe { &mut *worker.slots.get() };

    if slots[mtid.index()].is_none() {
        slots[mtid.index()] = Some(AlignedBuffer::with_capacity(
            item_size, item_align, capacity,
        ));
    }

    let buf = slots[mtid.index()].as_mut().unwrap();
    // SAFETY: M is the type registered for mtid; item_size and align match.
    unsafe { buf.push(msg) };
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Drain (Phase B — called from boundary stage, main thread only)
// ─────────────────────────────────────────────────────────────────────────────

/// Drains all per-worker buffers for `mtid` into `out`.
///
/// Called once per message type per tick boundary, after all parallel emit
/// phases have completed.  Resets each worker's slot to `None` so workers
/// reallocate fresh buffers for the next tick (we keep capacity for reuse
/// if we do `clear` instead — see [`drain_reuse`] below for that variant).
///
/// # Safety contract
///
/// No worker is actively emitting when this is called (Phase A is finished).
pub(crate) fn drain_into(
    runtime_id: MessageRuntimeID,
    mtid: MessageTypeID,
    out: &mut AlignedBuffer,
) -> ECSResult<()> {
    let registry = GLOBAL_EMIT_REGISTRY
        .lock()
        .map_err(|_| ECSError::from(MessagingError::LockPoisoned("global emit registry")))?;

    let Some(workers) = registry.get(&runtime_id) else {
        return Ok(());
    };

    for worker in workers {
        // SAFETY: Phase B is exclusive; no worker writes to these slots now.
        let slots = unsafe { &mut *worker.slots.get() };
        if let Some(ref buf) = slots[mtid.index()] {
            out.extend_from(buf);
        }
    }
    Ok(())
}

/// Clears all per-worker emit buffers for `mtid` without deallocating.
///
/// Called at the start of each tick (before workers begin emitting) so that
/// stale messages from the previous tick are discarded.
///
/// # Safety contract
///
/// No worker is actively emitting when this is called.
pub(crate) fn clear_for_tick(runtime_id: MessageRuntimeID, mtid: MessageTypeID) -> ECSResult<()> {
    let registry = GLOBAL_EMIT_REGISTRY
        .lock()
        .map_err(|_| ECSError::from(MessagingError::LockPoisoned("global emit registry")))?;

    let Some(workers) = registry.get(&runtime_id) else {
        return Ok(());
    };

    for worker in workers {
        // SAFETY: No emit is in progress.
        let slots = unsafe { &mut *worker.slots.get() };
        if let Some(ref mut buf) = slots[mtid.index()] {
            buf.clear();
        }
    }
    Ok(())
}

/// Returns a snapshot of how many workers are currently registered.  Useful
/// for diagnostics and tests.
#[allow(dead_code)]
pub(crate) fn registered_worker_count() -> usize {
    GLOBAL_EMIT_REGISTRY
        .lock()
        .map(|g| g.values().map(Vec::len).sum())
        .unwrap_or(0)
}
