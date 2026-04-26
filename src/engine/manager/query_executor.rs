//! Internal query execution helpers for chunk-parallel ECS iteration.
//!
//! `ECSData` owns storage and structural mutation. This module owns the
//! mechanics of turning a built query into locked column guards, chunk pointer
//! views, Rayon work items, and deterministic fallible/reduction dispatch.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLockReadGuard, RwLockWriteGuard};

use smallvec::SmallVec;

use crate::engine::archetype::{Archetype, ArchetypeMatch};
use crate::engine::error::{ECSError, ECSResult, ExecutionError};
use crate::engine::query::{BuiltQuery, QuerySignature};
use crate::engine::storage::TypeErasedAttribute;
use crate::engine::types::{ChunkID, ComponentID};

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
    query: BuiltQuery,
    f: impl Fn(&[&[u8]], &mut [&mut [u8]]) + Send + Sync,
) -> Result<(), ExecutionError> {
    let matches = matching_archetypes(archetypes, &query.signature)?;
    let f = Arc::new(f);
    let abort = Arc::new(AtomicBool::new(false));
    let err: Arc<Mutex<Option<ExecutionError>>> = Arc::new(Mutex::new(None));

    for matched_archetype in matches {
        let archetype = &archetypes[matched_archetype.archetype_id as usize];
        let (read_guards, mut write_guards) = lock_columns(archetype, &query)?;
        let views = build_chunk_view(archetype, &query, &read_guards, &mut write_guards)?;

        run_chunks(&views, &*f, &abort);

        if abort.load(Ordering::Acquire) {
            let guard = err.lock().map_err(|_| ExecutionError::LockPoisoned {
                what: "job error latch",
            })?;
            return Err(guard
                .clone()
                .unwrap_or(ExecutionError::InternalExecutionError));
        }

        drop(read_guards);
        drop(write_guards);
    }

    Ok(())
}

pub(super) fn for_each_fallible_unchecked(
    archetypes: &[Archetype],
    query: BuiltQuery,
    f: impl Fn(&[&[u8]], &mut [&mut [u8]]) -> ECSResult<()> + Send + Sync,
) -> ECSResult<()> {
    let matches = matching_archetypes(archetypes, &query.signature).map_err(ECSError::from)?;
    let f = Arc::new(f);
    let abort = Arc::new(AtomicBool::new(false));
    let err: Arc<Mutex<Option<(usize, ECSError)>>> = Arc::new(Mutex::new(None));

    for matched_archetype in matches {
        let archetype = &archetypes[matched_archetype.archetype_id as usize];
        let (read_guards, mut write_guards) =
            lock_columns(archetype, &query).map_err(ECSError::from)?;
        let views = build_chunk_view(archetype, &query, &read_guards, &mut write_guards)
            .map_err(ECSError::from)?;

        run_chunks_fallible(&views, &*f, &abort, &err);

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
    query: BuiltQuery,
    init: impl Fn() -> R + Send + Sync,
    fold_chunk: impl Fn(&mut R, &[&[u8]], usize) + Send + Sync,
    combine: impl Fn(&mut R, R) + Send + Sync,
) -> Result<R, ExecutionError>
where
    R: Send + 'static,
{
    let matches = matching_archetypes(archetypes, &query.signature)?;
    let init = Arc::new(init);
    let fold_chunk = Arc::new(fold_chunk);
    let combine = Arc::new(combine);
    let partials: Arc<Mutex<Vec<(usize, R)>>> = Arc::new(Mutex::new(Vec::new()));

    for matched in matches {
        let archetype = &archetypes[matched.archetype_id as usize];

        let mut sorted_reads: Vec<ComponentID> = query.reads.clone();
        sorted_reads.sort_unstable();

        let mut read_guards: Vec<ReadGuard<'_>> = Vec::with_capacity(query.reads.len());

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
        let n_reads = query.reads.len();
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
                &query.reads,
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
                        fold_chunk(&mut local, &read_views, len);
                    }

                    partials.lock().unwrap().push((start, local));
                });

                start = end;
            }
        });

        drop(read_guards);
    }

    let mut parts = partials.lock().unwrap();
    parts.sort_by_key(|(start, _)| *start);
    let mut out = init();
    for (_, p) in parts.drain(..) {
        combine(&mut out, p);
    }
    Ok(out)
}

fn lock_columns<'a>(
    archetype: &'a Archetype,
    query: &BuiltQuery,
) -> Result<(Vec<ReadGuard<'a>>, Vec<WriteGuard<'a>>), ExecutionError> {
    let mut lock_order: Vec<(ComponentID, bool)> =
        Vec::with_capacity(query.reads.len() + query.writes.len());
    for &cid in &query.reads {
        lock_order.push((cid, false));
    }
    for &cid in &query.writes {
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

    let n_reads = query.reads.len();
    let n_writes = query.writes.len();

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

        collect_read_ptrs_by_id(read_guards, &query.reads, chunk_id, len, &mut read_ptrs)?;

        for &cid in &query.writes {
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
    f: &(impl Fn(&[&[u8]], &mut [&mut [u8]]) + Send + Sync),
    abort: &Arc<AtomicBool>,
) {
    let threads = rayon::current_num_threads().max(1);
    let grainsize = (views.chunk_count / threads).max(8);

    rayon::scope(|s| {
        let mut start = 0usize;
        while start < views.chunk_count {
            let end = (start + grainsize).min(views.chunk_count);
            let abort = Arc::clone(abort);
            let views = views;

            s.spawn(move |_| {
                if abort.load(Ordering::Acquire) {
                    return;
                }

                let mut read_views: SmallVec<[&[u8]; 8]> = SmallVec::new();
                let mut write_views: SmallVec<[&mut [u8]; 8]> = SmallVec::new();

                for chunk in start..end {
                    let len = views.chunk_lens[chunk];
                    if len == 0 {
                        continue;
                    }

                    read_views.clear();
                    write_views.clear();
                    fill_chunk_slices(views, chunk, &mut read_views, &mut write_views);
                    f(&read_views, &mut write_views);
                }
            });

            start = end;
        }
    });
}

fn run_chunks_fallible(
    views: &ChunkView,
    f: &(impl Fn(&[&[u8]], &mut [&mut [u8]]) -> ECSResult<()> + Send + Sync),
    abort: &Arc<AtomicBool>,
    err: &Arc<Mutex<Option<(usize, ECSError)>>>,
) {
    let threads = rayon::current_num_threads().max(1);
    let grainsize = (views.chunk_count / threads).max(8);

    rayon::scope(|s| {
        let mut start = 0usize;
        while start < views.chunk_count {
            let end = (start + grainsize).min(views.chunk_count);
            let abort = Arc::clone(abort);
            let err = Arc::clone(err);
            let views = views;

            s.spawn(move |_| {
                let mut read_views: SmallVec<[&[u8]; 8]> = SmallVec::new();
                let mut write_views: SmallVec<[&mut [u8]; 8]> = SmallVec::new();

                for chunk in start..end {
                    if abort.load(Ordering::Acquire) {
                        let latched = err
                            .lock()
                            .map(|g| g.as_ref().map(|(c, _)| *c))
                            .unwrap_or(None);
                        if latched.map_or(false, |c| c <= chunk) {
                            return;
                        }
                    }

                    let len = views.chunk_lens[chunk];
                    if len == 0 {
                        continue;
                    }

                    read_views.clear();
                    write_views.clear();
                    fill_chunk_slices(views, chunk, &mut read_views, &mut write_views);

                    if let Err(e) = f(&read_views, &mut write_views) {
                        if let Ok(mut guard) = err.lock() {
                            let take = match guard.as_ref() {
                                Some((existing_chunk, _)) => chunk < *existing_chunk,
                                None => true,
                            };
                            if take {
                                *guard = Some((chunk, e));
                            }
                        }
                        abort.store(true, Ordering::Release);
                        return;
                    }
                }
            });

            start = end;
        }
    });
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

fn matching_archetypes(
    archetypes: &[Archetype],
    query: &QuerySignature,
) -> Result<Vec<ArchetypeMatch>, ExecutionError> {
    let mut out = Vec::new();
    for a in archetypes {
        if !query.requires_all(a.signature()) {
            continue;
        }
        let chunks = a
            .chunk_count()
            .map_err(|_| ExecutionError::InternalExecutionError)?;
        out.push(ArchetypeMatch {
            archetype_id: a.archetype_id(),
            chunks,
        });
    }
    Ok(out)
}
