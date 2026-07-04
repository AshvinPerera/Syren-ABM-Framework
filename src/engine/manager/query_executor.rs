//! Internal query execution helpers for chunk-parallel ECS iteration.
//!
//! `ECSData` owns storage and structural mutation. This module owns the
//! mechanics of turning a built query into locked column guards, chunk pointer
//! views, Rayon work items, and deterministic fallible/reduction dispatch.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLockReadGuard, RwLockWriteGuard};

use smallvec::SmallVec;

use crate::engine::activation::{ActivationContext, ActivationOrder};
use crate::engine::archetype::Archetype;
use crate::engine::entity::Entity;
use crate::engine::error::{ECSError, ECSResult, ExecutionError};
use crate::engine::query::BuiltQuery;
use crate::engine::random::splitmix64;
use crate::engine::storage::TypeErasedAttribute;
use crate::engine::types::{ArchetypeID, ChunkID, ComponentID};

#[cfg(feature = "gpu")]
use crate::engine::dirty::{DirtyChunks, Entry};

use super::iteration::ChunkView;

type ReadGuard<'a> = (
    ComponentID,
    RwLockReadGuard<'a, Box<dyn TypeErasedAttribute>>,
);
type WriteGuard<'a> = (
    ComponentID,
    RwLockWriteGuard<'a, Box<dyn TypeErasedAttribute>>,
);

/// Minimum number of rows a work range may contain when a chunk is split
/// across tasks. Even trivial per-row work amortises a Rayon spawn well
/// before this many rows, and larger floors would re-create the coarse
/// granularity this planner exists to avoid.
const MIN_ROWS_PER_RANGE: usize = 2048;

/// A contiguous run of rows within one chunk, executed by exactly one task.
///
/// `ordinal` is the range's position in the deterministic flattened work
/// order; fallible runs use it to latch the lowest-ordinal error so the
/// surfaced error is independent of thread count.
#[derive(Clone, Copy, Debug)]
struct WorkRange {
    chunk: usize,
    row_lo: usize,
    row_hi: usize,
    ordinal: usize,
}

/// Splits an archetype's chunks into parallel tasks of contiguous row ranges.
///
/// Sizing policy:
/// - aim for `threads * 2` tasks (oversubscription smooths load imbalance),
/// - never below one whole chunk of work when chunks are plentiful,
/// - when there are fewer chunks than target tasks, split chunks into row
///   ranges no smaller than [`MIN_ROWS_PER_RANGE`], so small populations
///   still spread across the machine.
///
/// Chunk visit order honours `ShuffleChunks`; empty chunks are dropped.
/// Ranges are packed into tasks contiguously, preserving the deterministic
/// flattened order that `ordinal` records.
fn plan_work_tasks(
    chunk_lens: &[usize],
    threads: usize,
    activation: ActivationContext,
    archetype_id: ArchetypeID,
) -> Vec<Vec<WorkRange>> {
    let total_rows: usize = chunk_lens.iter().sum();
    if total_rows == 0 {
        return Vec::new();
    }

    let order = chunk_order(chunk_lens.len(), activation, archetype_id);
    let target_tasks = threads.saturating_mul(2).max(1);
    let rows_per_range = (total_rows / target_tasks).max(MIN_ROWS_PER_RANGE);

    let mut ranges: Vec<WorkRange> = Vec::new();
    let mut ordinal = 0usize;
    for &chunk in &order {
        let len = chunk_lens[chunk];
        if len == 0 {
            continue;
        }
        // Number of near-equal ranges for this chunk (floor division keeps
        // every range at or above `rows_per_range` except when len is small,
        // where the whole chunk becomes a single range).
        let splits = (len / rows_per_range).max(1);
        let base = len / splits;
        let remainder = len % splits;
        let mut row_lo = 0usize;
        for i in 0..splits {
            let size = base + usize::from(i < remainder);
            ranges.push(WorkRange {
                chunk,
                row_lo,
                row_hi: row_lo + size,
                ordinal,
            });
            ordinal += 1;
            row_lo += size;
        }
    }

    let task_count = ranges.len().min(target_tasks).max(1);
    let per_task = ranges.len().div_ceil(task_count);
    ranges
        .chunks(per_task)
        .map(|group| group.to_vec())
        .collect()
}

pub(super) fn for_each_unchecked(
    archetypes: &[Archetype],
    matches: &[ArchetypeID],
    query: BuiltQuery,
    #[cfg(feature = "gpu")] dirty_chunks: &DirtyChunks,
    activation: ActivationContext,
    f: impl Fn(&[&[u8]], &mut [&mut [u8]]) + Send + Sync,
) -> Result<(), ExecutionError> {
    let f = Arc::new(f);

    for &archetype_id in matches {
        let archetype = &archetypes[archetype_id as usize];
        let (read_guards, mut write_guards) = lock_columns(archetype, &query)?;
        let views = build_chunk_view(archetype, &query, &read_guards, &mut write_guards)?;
        #[cfg(feature = "gpu")]
        let dirty_entries =
            resolve_dirty_entries(archetype, &query, views.chunk_lens.len(), dirty_chunks);

        #[cfg(feature = "gpu")]
        run_chunks(
            &views,
            &query,
            archetype_id,
            activation,
            &*f,
            &dirty_entries,
        );

        #[cfg(not(feature = "gpu"))]
        run_chunks(&views, &query, archetype_id, activation, &*f);

        drop(read_guards);
        drop(write_guards);
    }

    Ok(())
}

pub(super) fn for_each_entity_unchecked(
    archetypes: &[Archetype],
    matches: &[ArchetypeID],
    query: BuiltQuery,
    #[cfg(feature = "gpu")] dirty_chunks: &DirtyChunks,
    activation: ActivationContext,
    f: impl Fn(&[Entity], &[&[u8]], &mut [&mut [u8]]) + Send + Sync,
) -> Result<(), ExecutionError> {
    let f = Arc::new(f);

    for &archetype_id in matches {
        let archetype = &archetypes[archetype_id as usize];
        let (read_guards, mut write_guards) = lock_columns(archetype, &query)?;
        let views = build_chunk_view(archetype, &query, &read_guards, &mut write_guards)?;
        let entity_chunks = archetype
            .entity_chunks(&views.chunk_lens)
            .map_err(|_| ExecutionError::InternalExecutionError)?;
        #[cfg(feature = "gpu")]
        let dirty_entries =
            resolve_dirty_entries(archetype, &query, views.chunk_lens.len(), dirty_chunks);

        #[cfg(feature = "gpu")]
        run_chunks_entity(
            &views,
            &entity_chunks,
            &query,
            archetype_id,
            activation,
            &*f,
            &dirty_entries,
        );

        #[cfg(not(feature = "gpu"))]
        run_chunks_entity(
            &views,
            &entity_chunks,
            &query,
            archetype_id,
            activation,
            &*f,
        );

        drop(read_guards);
        drop(write_guards);
    }

    Ok(())
}

pub(super) fn for_each_fallible_unchecked(
    archetypes: &[Archetype],
    matches: &[ArchetypeID],
    query: BuiltQuery,
    #[cfg(feature = "gpu")] dirty_chunks: &DirtyChunks,
    activation: ActivationContext,
    f: impl Fn(&[&[u8]], &mut [&mut [u8]]) -> ECSResult<()> + Send + Sync,
) -> ECSResult<()> {
    let f = Arc::new(f);
    let abort = Arc::new(AtomicBool::new(false));
    let err: Arc<Mutex<Option<(usize, ECSError)>>> = Arc::new(Mutex::new(None));

    for &archetype_id in matches {
        let archetype = &archetypes[archetype_id as usize];
        let (read_guards, mut write_guards) =
            lock_columns(archetype, &query).map_err(ECSError::from)?;
        let views = build_chunk_view(archetype, &query, &read_guards, &mut write_guards)
            .map_err(ECSError::from)?;
        #[cfg(feature = "gpu")]
        let dirty_entries =
            resolve_dirty_entries(archetype, &query, views.chunk_lens.len(), dirty_chunks);

        #[cfg(feature = "gpu")]
        run_chunks_fallible(
            &views,
            &query,
            archetype_id,
            activation,
            &*f,
            &abort,
            &err,
            &dirty_entries,
        );

        #[cfg(not(feature = "gpu"))]
        run_chunks_fallible(&views, &query, archetype_id, activation, &*f, &abort, &err);

        if abort.load(Ordering::Acquire) {
            let guard = err.lock().map_err(|_| {
                ECSError::from(ExecutionError::LockPoisoned {
                    what: "job error latch",
                })
            })?;
            return Err(guard
                .as_ref()
                .map(|(_, e)| e.clone())
                .unwrap_or_else(|| ECSError::from(ExecutionError::InternalExecutionError)));
        }

        drop(read_guards);
        drop(write_guards);
    }

    Ok(())
}

pub(super) fn for_each_entity_fallible_unchecked(
    archetypes: &[Archetype],
    matches: &[ArchetypeID],
    query: BuiltQuery,
    #[cfg(feature = "gpu")] dirty_chunks: &DirtyChunks,
    activation: ActivationContext,
    f: impl Fn(&[Entity], &[&[u8]], &mut [&mut [u8]]) -> ECSResult<()> + Send + Sync,
) -> ECSResult<()> {
    let f = Arc::new(f);
    let abort = Arc::new(AtomicBool::new(false));
    let err: Arc<Mutex<Option<(usize, ECSError)>>> = Arc::new(Mutex::new(None));

    for &archetype_id in matches {
        let archetype = &archetypes[archetype_id as usize];
        let (read_guards, mut write_guards) =
            lock_columns(archetype, &query).map_err(ECSError::from)?;
        let views = build_chunk_view(archetype, &query, &read_guards, &mut write_guards)
            .map_err(ECSError::from)?;
        let entity_chunks = archetype.entity_chunks(&views.chunk_lens)?;
        #[cfg(feature = "gpu")]
        let dirty_entries =
            resolve_dirty_entries(archetype, &query, views.chunk_lens.len(), dirty_chunks);

        #[cfg(feature = "gpu")]
        run_chunks_entity_fallible(
            &views,
            &entity_chunks,
            &query,
            archetype_id,
            activation,
            &*f,
            &abort,
            &err,
            &dirty_entries,
        );

        #[cfg(not(feature = "gpu"))]
        run_chunks_entity_fallible(
            &views,
            &entity_chunks,
            &query,
            archetype_id,
            activation,
            &*f,
            &abort,
            &err,
        );

        if abort.load(Ordering::Acquire) {
            let guard = err.lock().map_err(|_| {
                ECSError::from(ExecutionError::LockPoisoned {
                    what: "job error latch",
                })
            })?;
            return Err(guard
                .as_ref()
                .map(|(_, e)| e.clone())
                .unwrap_or_else(|| ECSError::from(ExecutionError::InternalExecutionError)));
        }

        drop(read_guards);
        drop(write_guards);
    }

    Ok(())
}

pub(super) fn reduce_unchecked<R>(
    archetypes: &[Archetype],
    matches: &[ArchetypeID],
    query: BuiltQuery,
    init: impl Fn() -> R + Send + Sync,
    fold_chunk: impl Fn(&mut R, &[&[u8]], usize) + Send + Sync,
    combine: impl Fn(&mut R, R) + Send + Sync,
) -> Result<R, ExecutionError>
where
    R: Send + 'static,
{
    let init = Arc::new(init);
    let fold_chunk = Arc::new(fold_chunk);
    let combine = Arc::new(combine);
    let partials: Arc<Mutex<Vec<(usize, usize, R)>>> = Arc::new(Mutex::new(Vec::new()));
    // Set when a worker cannot record its partial because the mutex was
    // poisoned by a panicking fold on another thread.
    let partials_poisoned = Arc::new(AtomicBool::new(false));

    for (archetype_order, &archetype_id) in matches.iter().enumerate() {
        let archetype = &archetypes[archetype_id as usize];

        let mut sorted_reads: Vec<ComponentID> = query.read_ids().to_vec();
        sorted_reads.sort_unstable();

        let mut read_guards: Vec<ReadGuard<'_>> = Vec::with_capacity(query.read_ids().len());

        for &cid in &sorted_reads {
            let locked = archetype
                .component_locked(cid)
                .ok_or(ExecutionError::MissingComponent { component_id: cid })?;
            let guard = locked.read().map_err(|_| ExecutionError::LockPoisoned {
                what: "component column (read)",
            })?;
            read_guards.push((cid, guard));
        }

        let chunk_count = archetype
            .chunk_count()
            .map_err(|_| ExecutionError::InternalExecutionError)?;
        if chunk_count == 0 {
            continue;
        }

        let chunk_lens = collect_chunk_lens(archetype, chunk_count)?;
        let n_reads = query.read_ids().len();
        let mut read_ptrs: Vec<(*const u8, usize)> = Vec::with_capacity(chunk_count * n_reads);

        for (chunk, len) in chunk_lens.iter().copied().enumerate() {
            if len == 0 {
                for _ in 0..n_reads {
                    read_ptrs.push((std::ptr::null(), 0));
                }
                continue;
            }
            collect_read_ptrs_by_id(
                &read_guards,
                query.read_ids(),
                chunk as ChunkID,
                len,
                &mut read_ptrs,
            )?;
        }

        let views = ChunkView {
            chunk_lens,
            n_reads,
            n_writes: 0,
            read_ptrs,
            write_ptrs: Vec::new(),
        };

        let threads = rayon::current_num_threads().max(1);
        // Reductions always fold in sequential chunk order; the default
        // activation context yields the identity chunk ordering.
        let tasks = plan_work_tasks(
            &views.chunk_lens,
            threads,
            ActivationContext::default(),
            archetype_id,
        );

        if tasks.len() <= 1 {
            if let Some(task) = tasks.first() {
                let mut local = init();
                let mut read_views: SmallVec<[&[u8]; 8]> = SmallVec::with_capacity(views.n_reads);
                fold_reduce_task(&views, &query, task, &mut local, &*fold_chunk, &mut read_views);
                partials
                    .lock()
                    .map_err(|_| ExecutionError::LockPoisoned {
                        what: "reduce partials",
                    })?
                    .push((archetype_order, task[0].ordinal, local));
            }
        } else {
            let views_ref = &views;
            let query_ref = &query;

            rayon::scope(|s| {
                for task in &tasks {
                    let init = init.clone();
                    let fold_chunk = fold_chunk.clone();
                    let partials = partials.clone();
                    let partials_poisoned = partials_poisoned.clone();
                    let views = views_ref;

                    s.spawn(move |_| {
                        let mut local = init();
                        let mut read_views: SmallVec<[&[u8]; 8]> =
                            SmallVec::with_capacity(views.n_reads);
                        fold_reduce_task(
                            views,
                            query_ref,
                            task,
                            &mut local,
                            &*fold_chunk,
                            &mut read_views,
                        );

                        match partials.lock() {
                            Ok(mut guard) => guard.push((archetype_order, task[0].ordinal, local)),
                            Err(_) => partials_poisoned.store(true, Ordering::Release),
                        }
                    });
                }
            });
        }

        drop(read_guards);
    }

    if partials_poisoned.load(Ordering::Acquire) {
        return Err(ExecutionError::LockPoisoned {
            what: "reduce partials",
        });
    }

    let mut parts = partials.lock().map_err(|_| ExecutionError::LockPoisoned {
        what: "reduce partials",
    })?;
    parts.sort_by_key(|(archetype_order, start, _)| (*archetype_order, *start));
    let mut out = init();
    for (_, _, p) in parts.drain(..) {
        combine(&mut out, p);
    }
    Ok(out)
}

/// Folds every range in `task` into `local`, passing the range's row count
/// as the fold's valid length.
fn fold_reduce_task<R>(
    views: &ChunkView,
    query: &BuiltQuery,
    task: &[WorkRange],
    local: &mut R,
    fold_chunk: &(impl Fn(&mut R, &[&[u8]], usize) + Send + Sync),
    read_views: &mut SmallVec<[&[u8]; 8]>,
) {
    for range in task {
        let rows = range.row_hi - range.row_lo;
        read_views.clear();
        let base = range.chunk * views.n_reads;
        for i in 0..views.n_reads {
            let (ptr, bytes) = views.read_ptrs[base + i];
            let size = query.reads()[i].size();
            debug_assert!(size > 0);
            debug_assert_eq!(bytes, views.chunk_lens[range.chunk] * size);
            unsafe {
                read_views.push(std::slice::from_raw_parts(
                    ptr.add(range.row_lo * size),
                    rows * size,
                ));
            }
        }
        fold_chunk(local, read_views, rows);
    }
}

fn lock_columns<'a>(
    archetype: &'a Archetype,
    query: &BuiltQuery,
) -> Result<(Vec<ReadGuard<'a>>, Vec<WriteGuard<'a>>), ExecutionError> {
    let mut lock_order: Vec<(ComponentID, bool)> =
        Vec::with_capacity(query.read_ids().len() + query.write_ids().len());
    for &cid in query.read_ids() {
        lock_order.push((cid, false));
    }
    for &cid in query.write_ids() {
        lock_order.push((cid, true));
    }
    lock_order.sort_unstable_by_key(|(cid, _)| *cid);
    lock_order.dedup_by_key(|(cid, _)| *cid);

    let mut read_guards: Vec<ReadGuard<'a>> = Vec::new();
    let mut write_guards: Vec<WriteGuard<'a>> = Vec::new();

    for (cid, is_write) in &lock_order {
        let locked = archetype
            .component_locked(*cid)
            .ok_or(ExecutionError::MissingComponent { component_id: *cid })?;
        if *is_write {
            let g = locked.write().map_err(|_| ExecutionError::LockPoisoned {
                what: "component column (write)",
            })?;
            write_guards.push((*cid, g));
        } else {
            let g = locked.read().map_err(|_| ExecutionError::LockPoisoned {
                what: "component column (read)",
            })?;
            read_guards.push((*cid, g));
        }
    }

    Ok((read_guards, write_guards))
}

fn build_chunk_view(
    archetype: &Archetype,
    query: &BuiltQuery,
    read_guards: &[ReadGuard<'_>],
    write_guards: &mut [WriteGuard<'_>],
) -> Result<ChunkView, ExecutionError> {
    let chunk_count = archetype
        .chunk_count()
        .map_err(|_| ExecutionError::InternalExecutionError)?;
    let chunk_lens = collect_chunk_lens(archetype, chunk_count)?;

    let n_reads = query.read_ids().len();
    let n_writes = query.write_ids().len();

    let mut read_ptrs: Vec<(*const u8, usize)> = Vec::with_capacity(chunk_count * n_reads);
    let mut write_ptrs: Vec<(*mut u8, usize)> = Vec::with_capacity(chunk_count * n_writes);

    for (chunk, len) in chunk_lens.iter().copied().enumerate() {
        let chunk_id = chunk as ChunkID;

        if len == 0 {
            for _ in 0..n_reads {
                read_ptrs.push((std::ptr::null(), 0));
            }
            for _ in 0..n_writes {
                write_ptrs.push((std::ptr::null_mut(), 0));
            }
            continue;
        }

        collect_read_ptrs_by_id(read_guards, query.read_ids(), chunk_id, len, &mut read_ptrs)?;

        for &cid in query.write_ids() {
            let (_, g) = write_guards
                .iter_mut()
                .find(|(id, _)| *id == cid)
                .ok_or(ExecutionError::InternalExecutionError)?;
            let (ptr, bytes) = g
                .chunk_bytes_mut(chunk_id, len)
                .ok_or(ExecutionError::InternalExecutionError)?;
            write_ptrs.push((ptr, bytes));
        }
    }

    Ok(ChunkView {
        chunk_lens,
        n_reads,
        n_writes,
        read_ptrs,
        write_ptrs,
    })
}

fn collect_chunk_lens(
    archetype: &Archetype,
    chunk_count: usize,
) -> Result<Vec<usize>, ExecutionError> {
    let mut chunk_lens = Vec::with_capacity(chunk_count);
    for c in 0..chunk_count {
        let len = archetype
            .chunk_valid_length(c)
            .map_err(|_| ExecutionError::InternalExecutionError)?;
        chunk_lens.push(len);
    }
    Ok(chunk_lens)
}

fn collect_read_ptrs_by_id(
    guards: &[ReadGuard<'_>],
    declaration_order: &[ComponentID],
    chunk_id: ChunkID,
    len: usize,
    out: &mut Vec<(*const u8, usize)>,
) -> Result<(), ExecutionError> {
    for &cid in declaration_order {
        let (_, g) = guards
            .iter()
            .find(|(id, _)| *id == cid)
            .ok_or(ExecutionError::InternalExecutionError)?;
        let (ptr, bytes) = g
            .chunk_bytes(chunk_id, len)
            .ok_or(ExecutionError::InternalExecutionError)?;
        out.push((ptr, bytes));
    }
    Ok(())
}

fn run_chunks(
    views: &ChunkView,
    query: &BuiltQuery,
    archetype_id: crate::engine::types::ArchetypeID,
    activation: ActivationContext,
    f: &(impl Fn(&[&[u8]], &mut [&mut [u8]]) + Send + Sync),
    #[cfg(feature = "gpu")] dirty_entries: &[Arc<Entry>],
) {
    let threads = rayon::current_num_threads().max(1);
    let tasks = plan_work_tasks(&views.chunk_lens, threads, activation, archetype_id);

    rayon::scope(|s| {
        for task in &tasks {
            s.spawn(move |_| {
                let mut read_views: SmallVec<[&[u8]; 8]> = SmallVec::new();
                let mut write_views: SmallVec<[&mut [u8]; 8]> = SmallVec::new();

                for range in task {
                    #[cfg(feature = "gpu")]
                    mark_dirty_entries(dirty_entries, range.chunk);

                    read_views.clear();
                    write_views.clear();
                    match activation.order {
                        ActivationOrder::ShuffleFull => {
                            // The shuffle covers the whole chunk so results
                            // are independent of how the chunk was ranged;
                            // this task visits its slice of that order.
                            let len = views.chunk_lens[range.chunk];
                            let rows = row_order(len, activation, archetype_id, range.chunk);
                            for &row in &rows[range.row_lo..range.row_hi] {
                                read_views.clear();
                                write_views.clear();
                                fill_row_slices(
                                    views,
                                    query,
                                    range.chunk,
                                    row,
                                    &mut read_views,
                                    &mut write_views,
                                );
                                f(&read_views, &mut write_views);
                            }
                        }
                        ActivationOrder::Sequential | ActivationOrder::ShuffleChunks => {
                            fill_range_slices(views, query, range, &mut read_views, &mut write_views);
                            f(&read_views, &mut write_views);
                        }
                    }
                }
            });
        }
    });
}

fn run_chunks_entity(
    views: &ChunkView,
    entity_chunks: &[Vec<Entity>],
    query: &BuiltQuery,
    archetype_id: crate::engine::types::ArchetypeID,
    activation: ActivationContext,
    f: &(impl Fn(&[Entity], &[&[u8]], &mut [&mut [u8]]) + Send + Sync),
    #[cfg(feature = "gpu")] dirty_entries: &[Arc<Entry>],
) {
    let threads = rayon::current_num_threads().max(1);
    let tasks = plan_work_tasks(&views.chunk_lens, threads, activation, archetype_id);

    rayon::scope(|s| {
        for task in &tasks {
            s.spawn(move |_| {
                let mut row_entities: SmallVec<[Entity; 1]> = SmallVec::new();
                let mut read_views: SmallVec<[&[u8]; 8]> = SmallVec::new();
                let mut write_views: SmallVec<[&mut [u8]; 8]> = SmallVec::new();

                for range in task {
                    #[cfg(feature = "gpu")]
                    mark_dirty_entries(dirty_entries, range.chunk);

                    read_views.clear();
                    write_views.clear();
                    match activation.order {
                        ActivationOrder::ShuffleFull => {
                            let len = views.chunk_lens[range.chunk];
                            let rows = row_order(len, activation, archetype_id, range.chunk);
                            for &row in &rows[range.row_lo..range.row_hi] {
                                row_entities.clear();
                                row_entities.push(entity_chunks[range.chunk][row]);
                                read_views.clear();
                                write_views.clear();
                                fill_row_slices(
                                    views,
                                    query,
                                    range.chunk,
                                    row,
                                    &mut read_views,
                                    &mut write_views,
                                );
                                f(&row_entities, &read_views, &mut write_views);
                            }
                        }
                        ActivationOrder::Sequential | ActivationOrder::ShuffleChunks => {
                            fill_range_slices(views, query, range, &mut read_views, &mut write_views);
                            f(
                                &entity_chunks[range.chunk][range.row_lo..range.row_hi],
                                &read_views,
                                &mut write_views,
                            );
                        }
                    }
                }
            });
        }
    });
}

// Keep the hot-path helper flat: grouping these parameters obscures the two
// closure shapes and the GPU-only dirty-tracking slice.
#[allow(clippy::too_many_arguments)]
fn run_chunks_fallible(
    views: &ChunkView,
    query: &BuiltQuery,
    archetype_id: crate::engine::types::ArchetypeID,
    activation: ActivationContext,
    f: &(impl Fn(&[&[u8]], &mut [&mut [u8]]) -> ECSResult<()> + Send + Sync),
    abort: &Arc<AtomicBool>,
    err: &Arc<Mutex<Option<(usize, ECSError)>>>,
    #[cfg(feature = "gpu")] dirty_entries: &[Arc<Entry>],
) {
    let threads = rayon::current_num_threads().max(1);
    let tasks = plan_work_tasks(&views.chunk_lens, threads, activation, archetype_id);

    rayon::scope(|s| {
        for task in &tasks {
            let abort = Arc::clone(abort);
            let err = Arc::clone(err);

            s.spawn(move |_| {
                let mut read_views: SmallVec<[&[u8]; 8]> = SmallVec::new();
                let mut write_views: SmallVec<[&mut [u8]; 8]> = SmallVec::new();

                for range in task {
                    let ordinal = range.ordinal;
                    if abort.load(Ordering::Acquire) {
                        let latched = err
                            .lock()
                            .map(|g| g.as_ref().map(|(c, _)| *c))
                            .unwrap_or(None);
                        if latched.is_some_and(|c| c <= ordinal) {
                            return;
                        }
                    }

                    #[cfg(feature = "gpu")]
                    mark_dirty_entries(dirty_entries, range.chunk);

                    read_views.clear();
                    write_views.clear();
                    match activation.order {
                        ActivationOrder::ShuffleFull => {
                            let len = views.chunk_lens[range.chunk];
                            let rows = row_order(len, activation, archetype_id, range.chunk);
                            for &row in &rows[range.row_lo..range.row_hi] {
                                read_views.clear();
                                write_views.clear();
                                fill_row_slices(
                                    views,
                                    query,
                                    range.chunk,
                                    row,
                                    &mut read_views,
                                    &mut write_views,
                                );
                                if let Err(e) = f(&read_views, &mut write_views) {
                                    latch_iteration_error(&err, ordinal, e);
                                    abort.store(true, Ordering::Release);
                                    return;
                                }
                            }
                        }
                        ActivationOrder::Sequential | ActivationOrder::ShuffleChunks => {
                            fill_range_slices(views, query, range, &mut read_views, &mut write_views);

                            if let Err(e) = f(&read_views, &mut write_views) {
                                latch_iteration_error(&err, ordinal, e);
                                abort.store(true, Ordering::Release);
                                return;
                            }
                        }
                    }
                }
            });
        }
    });
}

// Keep the hot-path helper flat: grouping these parameters obscures the two
// closure shapes and the GPU-only dirty-tracking slice.
#[allow(clippy::too_many_arguments)]
fn run_chunks_entity_fallible(
    views: &ChunkView,
    entity_chunks: &[Vec<Entity>],
    query: &BuiltQuery,
    archetype_id: crate::engine::types::ArchetypeID,
    activation: ActivationContext,
    f: &(impl Fn(&[Entity], &[&[u8]], &mut [&mut [u8]]) -> ECSResult<()> + Send + Sync),
    abort: &Arc<AtomicBool>,
    err: &Arc<Mutex<Option<(usize, ECSError)>>>,
    #[cfg(feature = "gpu")] dirty_entries: &[Arc<Entry>],
) {
    let threads = rayon::current_num_threads().max(1);
    let tasks = plan_work_tasks(&views.chunk_lens, threads, activation, archetype_id);

    rayon::scope(|s| {
        for task in &tasks {
            let abort = Arc::clone(abort);
            let err = Arc::clone(err);

            s.spawn(move |_| {
                let mut row_entities: SmallVec<[Entity; 1]> = SmallVec::new();
                let mut read_views: SmallVec<[&[u8]; 8]> = SmallVec::new();
                let mut write_views: SmallVec<[&mut [u8]; 8]> = SmallVec::new();

                for range in task {
                    let ordinal = range.ordinal;
                    if abort.load(Ordering::Acquire) {
                        let latched = err
                            .lock()
                            .map(|g| g.as_ref().map(|(c, _)| *c))
                            .unwrap_or(None);
                        if latched.is_some_and(|c| c <= ordinal) {
                            return;
                        }
                    }

                    #[cfg(feature = "gpu")]
                    mark_dirty_entries(dirty_entries, range.chunk);

                    read_views.clear();
                    write_views.clear();
                    match activation.order {
                        ActivationOrder::ShuffleFull => {
                            let len = views.chunk_lens[range.chunk];
                            let rows = row_order(len, activation, archetype_id, range.chunk);
                            for &row in &rows[range.row_lo..range.row_hi] {
                                row_entities.clear();
                                row_entities.push(entity_chunks[range.chunk][row]);
                                read_views.clear();
                                write_views.clear();
                                fill_row_slices(
                                    views,
                                    query,
                                    range.chunk,
                                    row,
                                    &mut read_views,
                                    &mut write_views,
                                );
                                if let Err(e) = f(&row_entities, &read_views, &mut write_views) {
                                    latch_iteration_error(&err, ordinal, e);
                                    abort.store(true, Ordering::Release);
                                    return;
                                }
                            }
                        }
                        ActivationOrder::Sequential | ActivationOrder::ShuffleChunks => {
                            fill_range_slices(views, query, range, &mut read_views, &mut write_views);

                            if let Err(e) = f(
                                &entity_chunks[range.chunk][range.row_lo..range.row_hi],
                                &read_views,
                                &mut write_views,
                            ) {
                                latch_iteration_error(&err, ordinal, e);
                                abort.store(true, Ordering::Release);
                                return;
                            }
                        }
                    }
                }
            });
        }
    });
}

#[cfg(feature = "gpu")]
fn resolve_dirty_entries(
    archetype: &Archetype,
    query: &BuiltQuery,
    chunk_count: usize,
    dirty_chunks: &DirtyChunks,
) -> Vec<Arc<Entry>> {
    query
        .write_ids()
        .iter()
        .map(|&component_id| {
            dirty_chunks.resolve_entry(archetype.archetype_id(), component_id, chunk_count)
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn mark_dirty_entries(entries: &[Arc<Entry>], chunk: usize) {
    for entry in entries {
        entry.mark_dirty(chunk);
    }
}

fn chunk_order(
    chunk_count: usize,
    activation: ActivationContext,
    archetype_id: crate::engine::types::ArchetypeID,
) -> Vec<usize> {
    let mut chunks: Vec<usize> = (0..chunk_count).collect();
    if activation.order == ActivationOrder::ShuffleChunks {
        let seed = activation_seed(activation, archetype_id, 0);
        shuffle_with_seed(&mut chunks, seed);
    }
    chunks
}

fn row_order(
    len: usize,
    activation: ActivationContext,
    archetype_id: crate::engine::types::ArchetypeID,
    chunk: usize,
) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..len).collect();
    let seed = activation_seed(activation, archetype_id, chunk as u64);
    shuffle_with_seed(&mut rows, seed);
    rows
}

fn activation_seed(
    activation: ActivationContext,
    archetype_id: crate::engine::types::ArchetypeID,
    salt: u64,
) -> u64 {
    splitmix64(
        activation.seed
            ^ ((activation.system_id as u64) << 32)
            ^ ((archetype_id as u64) << 16)
            ^ salt,
    )
}

fn shuffle_with_seed(values: &mut [usize], seed: u64) {
    if values.len() <= 1 {
        return;
    }
    for i in (1..values.len()).rev() {
        let r = splitmix64(seed ^ i as u64);
        let j = (r as usize) % (i + 1);
        values.swap(i, j);
    }
}

fn latch_iteration_error(
    err: &Arc<Mutex<Option<(usize, ECSError)>>>,
    ordinal: usize,
    error: ECSError,
) {
    if let Ok(mut guard) = err.lock() {
        let take = match guard.as_ref() {
            Some((existing_ordinal, _)) => ordinal < *existing_ordinal,
            None => true,
        };
        if take {
            *guard = Some((ordinal, error));
        }
    }
}

/// Pushes byte views covering `range`'s rows of each queried column.
///
/// A whole chunk is simply the range `0..chunk_len`. Sub-chunk ranges from
/// [`plan_work_tasks`] produce disjoint sub-slices of the same chunk, so two
/// tasks may safely hold mutable views into one chunk concurrently.
fn fill_range_slices<'a>(
    views: &ChunkView,
    query: &BuiltQuery,
    range: &WorkRange,
    read_views: &mut SmallVec<[&'a [u8]; 8]>,
    write_views: &mut SmallVec<[&'a mut [u8]; 8]>,
) {
    let rows = range.row_hi - range.row_lo;

    let rbase = range.chunk * views.n_reads;
    for i in 0..views.n_reads {
        let (ptr, bytes) = views.read_ptrs[rbase + i];
        let size = query.reads()[i].size();
        debug_assert!(size > 0);
        debug_assert_eq!(bytes, views.chunk_lens[range.chunk] * size);
        debug_assert!(range.row_hi * size <= bytes);
        unsafe {
            read_views.push(std::slice::from_raw_parts(
                ptr.add(range.row_lo * size),
                rows * size,
            ));
        }
    }

    let wbase = range.chunk * views.n_writes;
    for i in 0..views.n_writes {
        let (ptr, bytes) = views.write_ptrs[wbase + i];
        let size = query.writes()[i].size();
        debug_assert!(size > 0);
        debug_assert_eq!(bytes, views.chunk_lens[range.chunk] * size);
        debug_assert!(range.row_hi * size <= bytes);
        unsafe {
            write_views.push(std::slice::from_raw_parts_mut(
                ptr.add(range.row_lo * size),
                rows * size,
            ));
        }
    }
}

fn fill_row_slices<'a>(
    views: &ChunkView,
    query: &BuiltQuery,
    chunk: usize,
    row: usize,
    read_views: &mut SmallVec<[&'a [u8]; 8]>,
    write_views: &mut SmallVec<[&'a mut [u8]; 8]>,
) {
    let rbase = chunk * views.n_reads;
    for i in 0..views.n_reads {
        let (ptr, bytes) = views.read_ptrs[rbase + i];
        let size = query.reads()[i].size();
        debug_assert!(size > 0);
        debug_assert_eq!(bytes, views.chunk_lens[chunk] * size);
        unsafe {
            read_views.push(std::slice::from_raw_parts(ptr.add(row * size), size));
        }
    }

    let wbase = chunk * views.n_writes;
    for i in 0..views.n_writes {
        let (ptr, bytes) = views.write_ptrs[wbase + i];
        let size = query.writes()[i].size();
        debug_assert!(size > 0);
        debug_assert_eq!(bytes, views.chunk_lens[chunk] * size);
        unsafe {
            write_views.push(std::slice::from_raw_parts_mut(ptr.add(row * size), size));
        }
    }
}
