//! Internal iteration primitives for chunk-parallel ECS queries.
//!
//! Contains the precomputed chunk pointer view (`ChunkView`) and the RAII
//! iteration scope guard (`IterationScope`).  Both are `pub(super)` - they
//! are implementation details shared between `data` and `ecs_reference`.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Per-archetype precomputed view into chunk memory.
///
/// Layout:
/// - read_ptrs[(chunk * n_reads) + i]  => (ptr, bytes) for read component i in that chunk
/// - write_ptrs[(chunk * n_writes) + i] => (ptr, bytes) for write component i in that chunk
pub(super) struct ChunkView {
    pub chunk_count: usize,
    pub chunk_lens: Vec<usize>,
    pub n_reads: usize,
    pub n_writes: usize,
    pub read_ptrs: Vec<(*const u8, usize)>,
    pub write_ptrs: Vec<(*mut u8, usize)>,
}

// ChunkView contains raw pointers into archetype-owned component storage.
// These pointers are:
// - valid for the duration of for_each_abstraction_unchecked
// - protected by phase discipline (no structural mutation)
// - protected by BorrowGuard (no aliasing violations)
// - chunk-disjoint across parallel tasks

unsafe impl Send for ChunkView {}
unsafe impl Sync for ChunkView {}

pub(super) struct IterationScope<'a>(pub &'a AtomicUsize);

impl<'a> IterationScope<'a> {
    #[inline]
    pub fn new(counter: &'a AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self(counter)
    }
}

impl Drop for IterationScope<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
