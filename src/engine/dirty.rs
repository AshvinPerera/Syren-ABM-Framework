//! # Dirty Chunk Tracking
//!
//! This module provides chunk-granular dirty tracking for component columns.
//!
//! ## Purpose
//! Track which (archetype, component, chunk) ranges were written by CPU systems,
//! so the GPU backend can upload only the modified chunks.
//!
//! ## Concurrency
//! Writes occur inside parallel query execution (Rayon). This tracker is designed
//! to be thread-safe with minimal contention.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::engine::types::{ArchetypeID, ComponentID, COMPONENT_CAP};

// ---------------------------------------------------------------------------
// Small-vec bitset for Entry
// ---------------------------------------------------------------------------

/// Bitset storage for dirty chunk tracking.
///
/// For typical archetype sizes (<= 64 chunks), a single `AtomicU64` is
/// sufficient and avoids a heap allocation.  Larger archetypes fall back
/// to a heap-allocated `Vec<AtomicU64>`.
#[derive(Debug)]
enum Words {
    /// <= 64 chunks - single inline atomic word.
    Inline(AtomicU64),
    /// > 64 chunks - heap-allocated word array.
    Heap(Vec<AtomicU64>),
}

impl Words {
    fn new(chunk_count: usize) -> Self {
        if chunk_count <= 64 {
            Words::Inline(AtomicU64::new(0))
        } else {
            let word_count = chunk_count.div_ceil(64);
            let mut words = Vec::with_capacity(word_count);
            for _ in 0..word_count {
                words.push(AtomicU64::new(0));
            }
            Words::Heap(words)
        }
    }

    #[inline]
    fn get(&self, index: usize) -> Option<&AtomicU64> {
        match self {
            Words::Inline(w) => {
                if index == 0 {
                    Some(w)
                } else {
                    None
                }
            }
            Words::Heap(v) => v.get(index),
        }
    }

    /// Iterates over all words in the bitset.
    fn iter(&self) -> WordsIter<'_> {
        WordsIter {
            words: self,
            index: 0,
        }
    }
}

struct WordsIter<'a> {
    words: &'a Words,
    index: usize,
}

impl<'a> Iterator for WordsIter<'a> {
    type Item = &'a AtomicU64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let w = self.words.get(self.index)?;
        self.index += 1;
        Some(w)
    }
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

/// Per-(archetype, component) dirty bitset.
#[derive(Debug)]
pub struct Entry {
    /// Number of chunks the bitset was built for.
    chunk_count: usize,
    /// Bitset: 1 bit per chunk.
    words: Words,
}

impl Entry {
    fn new(chunk_count: usize) -> Self {
        Self {
            chunk_count,
            words: Words::new(chunk_count),
        }
    }

    /// Marks a single chunk as dirty (lock-free, atomic).
    #[inline]
    pub(crate) fn mark_dirty(&self, chunk: usize) {
        let word_idx = chunk / 64;
        let bit_offset = chunk % 64;
        if let Some(word) = self.words.get(word_idx) {
            word.fetch_or(1u64 << bit_offset, Ordering::Relaxed);
        }
    }

    /// Marks all chunks as dirty.
    #[inline]
    fn mark_all_dirty(&self) {
        for word in self.words.iter() {
            word.store(u64::MAX, Ordering::Relaxed);
        }
    }

    /// Atomically takes the set of dirty chunk indices and clears the bitset.
    fn take_dirty_chunks_and_clear(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for (word_index, word) in self.words.iter().enumerate() {
            let bits = word.swap(0, Ordering::AcqRel);
            if bits == 0 {
                continue;
            }

            let base = word_index * 64;
            let mut remaining = bits;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let index = base + bit;
                if index < self.chunk_count {
                    out.push(index);
                }
                remaining &= remaining - 1;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Flat, lock-free DirtyChunks
// ---------------------------------------------------------------------------

/// Chunk-granular dirty tracker.
///
/// ## Flat lock-free structure
///
/// The backing store is a flat `Vec<Option<Arc<Entry>>>` indexed by
/// `archetype_id * COMPONENT_CAP + component_id`.  This eliminates the
/// `HashMap` and makes the hot path (`mark_chunk_dirty` during parallel
/// iteration) fully lock-free - callers index directly into the vec and
/// operate on the `Arc<Entry>` without acquiring any lock.
///
/// The `RwLock` protects only structural mutations (entry creation,
/// archetype-scoped invalidation).  It is never acquired on the per-chunk
/// dirty-marking hot path when entries are pre-resolved.
#[derive(Debug)]
pub struct DirtyChunks {
    /// Flat storage indexed by `archetype_id * COMPONENT_CAP + component_id`.
    entries: RwLock<Vec<Option<Arc<Entry>>>>,
}

impl Default for DirtyChunks {
    fn default() -> Self {
        Self::new()
    }
}

impl DirtyChunks {
    /// Creates a new, empty dirty tracker.
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
        }
    }

    /// Computes the flat index for a (archetype, component) pair.
    #[inline]
    fn flat_index(archetype: ArchetypeID, component: ComponentID) -> usize {
        (archetype as usize) * COMPONENT_CAP + (component as usize)
    }

    /// Ensures the flat vec is large enough to hold `index`.
    fn ensure_vec_capacity(vec: &mut Vec<Option<Arc<Entry>>>, index: usize) {
        if index >= vec.len() {
            vec.resize_with(index + 1, || None);
        }
    }

    /// Called when structural changes occur for a specific archetype
    /// (spawn, despawn, add/remove component).
    pub fn notify_archetype_changed(&self, archetype_id: ArchetypeID) {
        if let Ok(entries) = self.entries.read() {
            let base = (archetype_id as usize) * COMPONENT_CAP;
            if base >= entries.len() {
                return; // No entries have been created for this archetype yet
            }
            let end = (base + COMPONENT_CAP).min(entries.len());
            for entry in entries[base..end].iter().flatten() {
                entry.mark_all_dirty();
            }
        }
    }

    /// Ensure an entry exists for (archetype, component) with the right
    /// chunk_count.
    fn ensure_entry(
        &self,
        archetype: ArchetypeID,
        component: ComponentID,
        chunk_count: usize,
    ) -> Arc<Entry> {
        let index = Self::flat_index(archetype, component);

        // Fast path: read lock - no allocation, no write lock.
        if let Ok(entries) = self.entries.read() {
            if let Some(Some(entry)) = entries.get(index) {
                if entry.chunk_count == chunk_count {
                    return entry.clone();
                }
            }
        }

        // Slow path: write lock - create or replace entry.
        let mut entries = self.entries.write().expect("DirtyChunks lock poisoned");
        Self::ensure_vec_capacity(&mut entries, index);

        // Double-check after acquiring write lock.
        if let Some(entry) = &entries[index] {
            if entry.chunk_count == chunk_count {
                return entry.clone();
            }
        }

        let entry = Arc::new(Entry::new(chunk_count));
        entry.mark_all_dirty();
        entries[index] = Some(entry.clone());
        entry
    }

    /// Resolves and returns an `Arc<Entry>` for the given
    /// (archetype, component) pair.
    #[inline]
    pub fn resolve_entry(
        &self,
        archetype: ArchetypeID,
        component: ComponentID,
        chunk_count: usize,
    ) -> Arc<Entry> {
        self.ensure_entry(archetype, component, chunk_count)
    }

    /// Mark a specific chunk dirty (thread-safe).
    ///
    /// This still works as before for callers that have not pre-resolved
    /// entries, but the hot path in `for_each_abstraction_unchecked` now
    /// uses `resolve_entry` + direct `entry.mark_dirty()` instead.
    #[inline]
    pub fn mark_chunk_dirty(
        &self,
        archetype: ArchetypeID,
        component: ComponentID,
        chunk: usize,
        chunk_count: usize,
    ) {
        let entry = self.ensure_entry(archetype, component, chunk_count);
        entry.mark_dirty(chunk);
    }

    /// Mark all chunks dirty for a component in an archetype.
    pub fn mark_all_dirty(
        &self,
        archetype: ArchetypeID,
        component: ComponentID,
        chunk_count: usize,
    ) {
        let entry = self.ensure_entry(archetype, component, chunk_count);
        entry.mark_all_dirty();
    }

    /// Take and clear the dirty chunk list for this (archetype, component).
    pub fn take_dirty_chunks(
        &self,
        archetype: ArchetypeID,
        component: ComponentID,
        chunk_count: usize,
    ) -> Vec<usize> {
        let entry = self.ensure_entry(archetype, component, chunk_count);
        entry.take_dirty_chunks_and_clear()
    }
}
