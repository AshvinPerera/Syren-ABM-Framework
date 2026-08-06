//! Zero-cost, lock-free message emission via thread-local storage.
//!
//! # Design
//!
//! Message **emission** happens on scheduler worker threads inside a system
//! stage (Phase A). Message **draining** happens in the boundary stage that
//! finalises the produced message channel (Phase B), after GPU sync and
//! deferred-command draining for that boundary. These two phases are
//! mutually exclusive by scheduler design: the boundary stage cannot start
//! until all systems from the previous stage have returned.
//!
//! This phase discipline allows us to use [`UnsafeCell`] without any locking
//! during the emit path: a worker writes its own slot and no other thread
//! reads it until the drain phase begins.
//!
//! # Worker registration
//!
//! When a thread first calls [`emit`] for a message runtime it registers a
//! slot container in `GLOBAL_EMIT_REGISTRY` (protected by a `Mutex`, but only
//! on first use for that runtime/thread pair). The drain path iterates this
//! registry to collect all registered buffers.
//!
//! # Memory layout
//!
//! Each registered thread has a [`WorkerEmitSlots`] containing one
//! `Option<AlignedBuffer>` slot per registered message type (indexed by
//! [`MessageTypeID`]). Slots are populated lazily on first emit; the drain
//! path calls [`AlignedBuffer::extend_from`] to merge them into the central
//! per-type buffer owned by a [`MessageBufferSet`](crate::messaging::MessageBufferSet).
//! The drain step does not clear worker slots. [`clear_for_tick`] runs from
//! `MessageBufferSet::begin_tick` before the next stage can emit, clearing
//! buffers in place so capacity can be reused safely.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Weak};

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

// -----------------------------------------------------------------------------
// Per-thread slot container
// -----------------------------------------------------------------------------

/// One set of emit buffers per registered emitting thread.
///
/// `slots[i]` is the pending buffer for message type with index `i`.
pub(crate) struct WorkerEmitSlots {
    /// Indexed by `MessageTypeID::index()`.  `None` means the buffer has not
    /// been created yet for this worker.
    slots: UnsafeCell<Vec<Option<AlignedBuffer>>>,
    /// Stable worker identifier, from [`crate::engine::workers::worker_id`].
    ///
    /// The drain path concatenates every worker's staged messages into one
    /// buffer, so the order workers are visited *is* the message order that
    /// systems observe. Registration happens on a thread's first emit, which
    /// is a race, so the registry is kept sorted by this id instead: it is
    /// dense and stable across runs for Rayon workers.
    worker_id: u32,
}

// SAFETY: The emit path writes only to the calling thread's own slots (no
// aliasing).  The drain path accesses every slot exactly once, after all
// workers have returned from Phase A (exclusive access by phase discipline).
unsafe impl Send for WorkerEmitSlots {}
unsafe impl Sync for WorkerEmitSlots {}

impl WorkerEmitSlots {
    fn new(num_message_types: usize, worker_id: u32) -> Self {
        let slots: Vec<Option<AlignedBuffer>> = (0..num_message_types).map(|_| None).collect();
        WorkerEmitSlots {
            slots: UnsafeCell::new(slots),
            worker_id,
        }
    }
}

// -----------------------------------------------------------------------------
// Global registry of all registered emitting threads
// -----------------------------------------------------------------------------

/// All threads that have registered emit slots. Guarded by a `Mutex` but only
/// written once per runtime/thread pair.
static GLOBAL_EMIT_REGISTRY: LazyLock<Mutex<HashMap<MessageRuntimeID, Vec<Arc<WorkerEmitSlots>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

thread_local! {
    /// The current thread's slot container.  Initialised lazily on first use.
    static THIS_WORKER: std::cell::RefCell<HashMap<MessageRuntimeID, Weak<WorkerEmitSlots>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Initialises the thread-local emit slots for the current thread.
///
/// Must be called once before the first `emit` on each thread/runtime pair. In
/// practice this is handled automatically by [`emit`]'s lazy initialisation;
/// [`MessageEmitter`] calls it once at construction and caches the result.
pub(crate) fn ensure_worker_registered_fallible(
    runtime_id: MessageRuntimeID,
    num_message_types: usize,
) -> ECSResult<Arc<WorkerEmitSlots>> {
    THIS_WORKER.with(|cell| {
        let existing = {
            let workers = cell.borrow();
            workers.get(&runtime_id).and_then(Weak::upgrade)
        };
        if let Some(existing) = existing {
            return Ok(existing);
        }

        let worker_id = crate::engine::workers::worker_id();
        let slots = Arc::new(WorkerEmitSlots::new(num_message_types, worker_id));
        cell.borrow_mut().insert(runtime_id, Arc::downgrade(&slots));
        let mut registry = GLOBAL_EMIT_REGISTRY
            .lock()
            .map_err(|_| ECSError::from(MessagingError::LockPoisoned("global emit registry")))?;
        let workers = registry.entry(runtime_id).or_default();
        // Insert in `worker_id` order so the drain concatenation is identical
        // on every run regardless of which thread emitted first. Registration
        // is once per thread per runtime, so the linear scan is negligible.
        let at = workers.partition_point(|w| w.worker_id < worker_id);
        workers.insert(at, Arc::clone(&slots));
        drop(registry);
        Ok(slots)
    })
}

/// Removes all globally registered worker slots for a message runtime.
///
/// Called when the owning [`MessageBufferSet`](crate::messaging::MessageBufferSet)
/// is dropped. The current thread's weak cache entry is also removed as a
/// best-effort cleanup; other worker threads hold only weak references, so they
/// do not keep the buffers alive after the global registry entry is removed.
pub(crate) fn deregister_runtime(runtime_id: MessageRuntimeID) {
    if let Ok(mut registry) = GLOBAL_EMIT_REGISTRY.lock() {
        registry.remove(&runtime_id);
    }

    THIS_WORKER.with(|cell| {
        cell.borrow_mut().remove(&runtime_id);
    });
}

#[cfg(test)]
pub(crate) fn registered_worker_count_for_test(runtime_id: MessageRuntimeID) -> usize {
    GLOBAL_EMIT_REGISTRY
        .lock()
        .ok()
        .and_then(|registry| registry.get(&runtime_id).map(Vec::len))
        .unwrap_or(0)
}

#[cfg(test)]
pub(crate) fn current_thread_has_worker_for_test(runtime_id: MessageRuntimeID) -> bool {
    THIS_WORKER.with(|cell| cell.borrow().contains_key(&runtime_id))
}
// -----------------------------------------------------------------------------
// Emit (Phase A - called from worker threads)
// -----------------------------------------------------------------------------

/// Emits a message into the calling thread's local buffer.
///
/// This function acquires **no locks** after the first call per runtime/thread
/// pair. It is intended to be called from systems scheduled by the engine.
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

// -----------------------------------------------------------------------------
// Cached emitter (Phase A fast path)
// -----------------------------------------------------------------------------

/// Marker bundling the emitter's auto-trait shape: `*mut ()` pins it to its
/// constructing thread (`!Send + !Sync`); `fn() -> M` binds the message type
/// without adding further constraints.
type EmitterMarker<M> = std::marker::PhantomData<(*mut (), fn() -> M)>;

/// Cached per-thread emitter for one message type.
///
/// The per-call emit path resolves the calling thread's slot container on
/// every message:
/// a thread-local access, a `RefCell` borrow, a `HashMap` lookup, and a
/// `Weak::upgrade`. A `MessageEmitter` performs that resolution **once** at
/// construction, so each [`MessageEmitter::emit`] is a bounds-checked slot
/// access plus a buffer push - the right shape for per-agent hot loops.
///
/// Obtain one via
/// [`MessageBufferSet::emitter`](crate::messaging::MessageBufferSet::emitter)
/// inside the system that emits, and drop it before the system returns.
///
/// # Thread affinity
///
/// The cached slot belongs to the constructing thread, so the emitter is
/// deliberately `!Send + !Sync`. Construct it *inside* the closure or system
/// body running on the emitting thread - one emitter per thread, not one
/// shared across a parallel iteration.
///
/// # Lifetime
///
/// The emitter borrows the
/// [`MessageBufferSet`](crate::messaging::MessageBufferSet) it came from, so
/// it cannot outlive the boundary handle - the same Phase A discipline that makes the
/// per-thread buffers sound applies unchanged.
pub struct MessageEmitter<'a, M: Message> {
    worker: Arc<WorkerEmitSlots>,
    slot_index: usize,
    item_size: usize,
    item_align: usize,
    initial_capacity: usize,
    /// Borrow of the owning buffer set (keeps Phase A discipline).
    _owner: std::marker::PhantomData<&'a ()>,
    _marker: EmitterMarker<M>,
}

impl<'a, M: Message> MessageEmitter<'a, M> {
    pub(crate) fn new(
        runtime_id: MessageRuntimeID,
        num_message_types: usize,
        mtid: MessageTypeID,
        item_size: usize,
        item_align: usize,
        initial_capacity: usize,
    ) -> ECSResult<Self> {
        let worker = ensure_worker_registered_fallible(runtime_id, num_message_types)?;
        Ok(Self {
            worker,
            slot_index: mtid.index(),
            item_size,
            item_align,
            initial_capacity,
            _owner: std::marker::PhantomData,
            _marker: std::marker::PhantomData,
        })
    }

    /// Emits one message into the calling thread's staging buffer.
    #[inline]
    pub fn emit(&self, msg: M) {
        // SAFETY: Phase A discipline - this emitter is `!Send`, so the slot
        // container belongs to the current thread and no drain can run
        // concurrently (drains happen in boundary stages after all systems
        // have returned, and the emitter cannot outlive its boundary handle).
        let slots = unsafe { &mut *self.worker.slots.get() };

        if slots[self.slot_index].is_none() {
            slots[self.slot_index] = Some(AlignedBuffer::with_capacity(
                self.item_size,
                self.item_align,
                self.initial_capacity,
            ));
        }

        let buffer = slots[self.slot_index]
            .as_mut()
            .expect("slot initialised above");
        // SAFETY: `M` is the type registered for this slot's message type id;
        // size and alignment were taken from the registry descriptor.
        unsafe { buffer.push(msg) };
    }
}

// -----------------------------------------------------------------------------
// Drain (Phase B - called from boundary stage, main thread only)
// -----------------------------------------------------------------------------

/// Drains all per-worker buffers for `mtid` into `out`.
///
/// Called once per message type per tick boundary, after all emit phases for
/// that boundary have completed. It copies each registered thread's buffer
/// into `out` and leaves the thread-local buffer intact; [`clear_for_tick`]
/// clears those buffers before the next tick so allocation capacity is reused.
///
/// # Safety contract
///
/// No thread is actively emitting when this is called (Phase A is finished).
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

/// Clears all per-thread emit buffers for `mtid` without deallocating.
///
/// Called at the start of each tick (before workers begin emitting) so that
/// stale messages from the previous tick are discarded.
///
/// # Safety contract
///
/// No thread is actively emitting when this is called.
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
