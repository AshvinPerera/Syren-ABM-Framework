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
use crate::engine::error::{ECSError, ECSResult, ExecutionError};
use crate::engine::query::BuiltQuery;
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

pub(super) fn for_each_unchecked(
    archetypes: &[Archetype],
    matches: Vec<ArchetypeID>,
    query: BuiltQuery,
    #[cfg(feature = "gpu")] dirty_chunks: &DirtyChunks,
    activation: ActivationContext,
    f: impl Fn(&[&[u8]], &mut [&mut [u8]]) + Send + Sync,
) -> Result<(), ExecutionError> {
    let f = Arc::new(f);

    for archetype_id in matches {
        let archetype = &archetypes[archetype_id as usize];
        let (read_guards, mut write_guards) = lock_columns(archetype, &query)?;
        let views = build_chunk_view(archetype, &query, &read_guards, &mut write_guards)?;
        #[cfg(feature = "gpu")]
        let dirty_entries =
            resolve_dirty_entries(archetype, &query, views.chunk_count, dirty_chunks);

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

pub(super) fn for_each_fallible_unchecked(
    archetypes: &[Archetype],
    matches: Vec<ArchetypeID>,
    query: BuiltQuery,
    #[cfg(feature = "gpu")] dirty_chunks: &DirtyChunks,
    activation: ActivationContext,
    f: impl Fn(&[&[u8]], &mut [&mut [u8]]) -> ECSResult<()> + Send + Sync,
) -> ECSResult<()> {
    let f = Arc::new(f);
    let abort = Arc::new(AtomicBool::new(false));
    let err: Arc<Mutex<Option<(usize, ECSError)>>> = Arc::new(Mutex::new(None));

    for archetype_id in matches {
        let archetype = &archetypes[archetype_id as usize];
        let (read_guards, mut write_guards) =
            lock_columns(archetype, &query).map_err(ECSError::from)?;
        let views = build_chunk_view(archetype, &query, &read_guards, &mut write_guards)
            .map_err(ECSError::from)?;
        #[cfg(feature = "gpu")]
        let dirty_entries =
            resolve_dirty_entries(archetype, &query, views.chunk_count, dirty_chunks);

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

pub(super) fn reduce_unchecked<R>(
    archetypes: &[Archetype],
    matches: Vec<ArchetypeID>,
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

    for (archetype_order, archetype_id) in matches.into_iter().enumerate() {
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
            chunk_count,
            chunk_lens,
            n_reads,
            n_writes: 0,
            read_ptrs,
            write_ptrs: Vec::new(),
        };

        let threads = rayon::current_num_threads().max(1);
        let grainsize = (views.chunk_count / threads).max(8);

        if views.chunk_count <= grainsize {
            let mut local = init();
            let mut read_views: SmallVec<[&[u8]; 8]> = SmallVec::with_capacity(views.n_reads);
            fold_reduce_range(
                &views,
                0,
                views.chunk_count,
                &mut local,
                &*fold_chunk,
                &mut read_views,
            );
            partials.lock().unwrap().push((archetype_order, 0, local));
        } else {
            let views_ref = &views;

            rayon::scope(|s| {
                let mut start = 0usize;
                while start < views_ref.chunk_count {
                    let end = (start + grainsize).min(views_ref.chunk_count);
                    let init = init.clone();
                    let fold_chunk = fold_chunk.clone();
                    let partials = partials.clone();
                    let views = views_ref;

                    s.spawn(move |_| {
                        let mut local = init();
                        let mut read_views: SmallVec<[&[u8]; 8]> =
                            SmallVec::with_capacity(views.n_reads);
                        fold_reduce_range(
                            views,
                            start,
                            end,
                            &mut local,
                            &*fold_chunk,
                            &mut read_views,
                        );

                        partials
                            .lock()
                            .unwrap()
                            .push((archetype_order, start, local));
                    });

                    start = end;
                }
            });
        }

        drop(read_guards);
    }

    let mut parts = partials.lock().unwrap();
    parts.sort_by_key(|(archetype_order, start, _)| (*archetype_order, *start));
    let mut out = init();
    for (_, _, p) in parts.drain(..) {
        combine(&mut out, p);
    }
    Ok(out)
}

fn fold_reduce_range<R>(
    views: &ChunkView,
    start: usize,
    end: usize,
    local: &mut R,
    fold_chunk: &(impl Fn(&mut R, &[&[u8]], usize) + Send + Sync),
    read_views: &mut SmallVec<[&[u8]; 8]>,
) {
    for chunk in start..end {
        let len = views.chunk_lens[chunk];
        if len == 0 {
            continue;
        }
        read_views.clear();
        let base = chunk * views.n_reads;
        for i in 0..views.n_reads {
            let (ptr, bytes) = views.read_ptrs[base + i];
            unsafe {
                read_views.push(std::slice::from_raw_parts(ptr, bytes));
            }
        }
        fold_chunk(local, read_views, len);
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
        chunk_count,
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
    let grainsize = (views.chunk_count / threads).max(8);
    let chunk_order = chunk_order(views.chunk_count, activation, archetype_id);

    rayon::scope(|s| {
        let mut start = 0usize;
        while start < chunk_order.len() {
            let end = (start + grainsize).min(chunk_order.len());
            let views = views;
            let query = query;
            let chunks = &chunk_order[start..end];

            s.spawn(move |_| {
                let mut read_views: SmallVec<[&[u8]; 8]> = SmallVec::new();
                let mut write_views: SmallVec<[&mut [u8]; 8]> = SmallVec::new();

                for &chunk in chunks {
                    let len = views.chunk_lens[chunk];
                    if len == 0 {
                        continue;
                    }

                    #[cfg(feature = "gpu")]
                    mark_dirty_entries(dirty_entries, chunk);

                    read_views.clear();
                    write_views.clear();
                    match activation.order {
                        ActivationOrder::ShuffleFull => {
                            let mut rows = row_order(len, activation, archetype_id, chunk);
                            for row in rows.drain(..) {
                                read_views.clear();
                                write_views.clear();
                                fill_row_slices(
                                    views,
                                    query,
                                    chunk,
                                    row,
                                    &mut read_views,
                                    &mut write_views,
                                );
                                f(&read_views, &mut write_views);
                            }
                        }
                        ActivationOrder::Sequential | ActivationOrder::ShuffleChunks => {
                            fill_chunk_slices(views, chunk, &mut read_views, &mut write_views);
                            f(&read_views, &mut write_views);
                        }
                    }
                }
            });

            start = end;
        }
    });
}

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
    let grainsize = (views.chunk_count / threads).max(8);
    let chunk_order = chunk_order(views.chunk_count, activation, archetype_id);

    rayon::scope(|s| {
        let mut start = 0usize;
        while start < chunk_order.len() {
            let end = (start + grainsize).min(chunk_order.len());
            let abort = Arc::clone(abort);
            let err = Arc::clone(err);
            let views = views;
            let query = query;
            let chunks = &chunk_order[start..end];

            s.spawn(move |_| {
                let mut read_views: SmallVec<[&[u8]; 8]> = SmallVec::new();
                let mut write_views: SmallVec<[&mut [u8]; 8]> = SmallVec::new();

                for (ordinal_offset, &chunk) in chunks.iter().enumerate() {
                    let ordinal = start + ordinal_offset;
                    if abort.load(Ordering::Acquire) {
                        let latched = err
                            .lock()
                            .map(|g| g.as_ref().map(|(c, _)| *c))
                            .unwrap_or(None);
                        if latched.map_or(false, |c| c <= ordinal) {
                            return;
                        }
                    }

                    let len = views.chunk_lens[chunk];
                    if len == 0 {
                        continue;
                    }

                    #[cfg(feature = "gpu")]
                    mark_dirty_entries(dirty_entries, chunk);

                    read_views.clear();
                    write_views.clear();
                    match activation.order {
                        ActivationOrder::ShuffleFull => {
                            let mut rows = row_order(len, activation, archetype_id, chunk);
                            for row in rows.drain(..) {
                                read_views.clear();
                                write_views.clear();
                                fill_row_slices(
                                    views,
                                    query,
                                    chunk,
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
                            fill_chunk_slices(views, chunk, &mut read_views, &mut write_views);

                            if let Err(e) = f(&read_views, &mut write_views) {
                                latch_iteration_error(&err, ordinal, e);
                                abort.store(true, Ordering::Release);
                                return;
                            }
                        }
                    }
                }
            });

            start = end;
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

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
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

fn fill_chunk_slices<'a>(
    views: &ChunkView,
    chunk: usize,
    read_views: &mut SmallVec<[&'a [u8]; 8]>,
    write_views: &mut SmallVec<[&'a mut [u8]; 8]>,
) {
    let rbase = chunk * views.n_reads;
    for i in 0..views.n_reads {
        let (ptr, bytes) = views.read_ptrs[rbase + i];
        unsafe {
            read_views.push(std::slice::from_raw_parts(ptr, bytes));
        }
    }

    let wbase = chunk * views.n_writes;
    for i in 0..views.n_writes {
        let (ptr, bytes) = views.write_ptrs[wbase + i];
        unsafe {
            write_views.push(std::slice::from_raw_parts_mut(ptr, bytes));
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
