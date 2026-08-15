#[cfg(test)]
mod tests {
    use crate::engine::storage::Attribute;
    use crate::engine::types::CHUNK_CAP;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// A value that increments a shared counter when dropped, allowing tests to
    /// verify that every destructor is called exactly once.
    #[derive(Clone)]
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    // -----------------------------------------------------------------------
    // Empty attribute
    // -----------------------------------------------------------------------

    #[test]
    fn empty_attribute_invariants() {
        let attr: Attribute<i32> = Attribute::default();
        assert_eq!(attr.length, 0);
        assert_eq!(attr.chunk_count(), 0);
        assert_eq!(attr.last_chunk_length, 0);
        assert_eq!(attr.iter().count(), 0);
    }

    #[test]
    fn clear_on_empty_is_noop() {
        let mut attr: Attribute<i32> = Attribute::default();
        attr.clear();
        assert_eq!(attr.length, 0);
        assert_eq!(attr.chunk_count(), 0);
    }

    #[test]
    fn get_on_empty_returns_none() {
        let attr: Attribute<i32> = Attribute::default();
        assert!(attr.get(0, 0).is_none());
    }

    // -----------------------------------------------------------------------
    // push / get
    // -----------------------------------------------------------------------

    #[test]
    fn push_and_get_single_element() {
        let mut attr: Attribute<i32> = Attribute::default();
        let (chunk, row) = attr.push(42).unwrap();
        assert_eq!(chunk, 0);
        assert_eq!(row, 0);
        assert_eq!(attr.length, 1);
        assert_eq!(*attr.get(chunk, row).unwrap(), 42);
    }

    #[test]
    fn push_fills_first_chunk_then_spills_to_second() {
        let mut attr: Attribute<i32> = Attribute::default();

        // Fill exactly one chunk.
        for i in 0..CHUNK_CAP {
            let (chunk, row) = attr.push(i as i32).unwrap();
            assert_eq!(chunk as usize, 0);
            assert_eq!(row as usize, i);
        }
        assert_eq!(attr.chunk_count(), 1);
        assert_eq!(attr.length, CHUNK_CAP);
        assert_eq!(attr.last_chunk_length, CHUNK_CAP);

        // One more push must spill into a new chunk.
        let (chunk, row) = attr.push(999).unwrap();
        assert_eq!(chunk as usize, 1);
        assert_eq!(row as usize, 0);
        assert_eq!(attr.chunk_count(), 2);
        assert_eq!(attr.length, CHUNK_CAP + 1);
        assert_eq!(attr.last_chunk_length, 1);
        assert_eq!(*attr.get(chunk, row).unwrap(), 999);
    }

    // -----------------------------------------------------------------------
    // swap_remove
    // -----------------------------------------------------------------------

    #[test]
    fn swap_remove_only_element() {
        let mut attr: Attribute<i32> = Attribute::default();
        attr.push(7).unwrap();
        let moved = attr.swap_remove(0, 0).unwrap();
        assert!(
            moved.is_none(),
            "no element should be moved when removing the only one"
        );
        assert_eq!(attr.length, 0);
        assert_eq!(attr.chunk_count(), 0);
    }

    #[test]
    fn swap_remove_last_element() {
        let mut attr: Attribute<i32> = Attribute::default();
        attr.push(1).unwrap();
        attr.push(2).unwrap();
        attr.push(3).unwrap();
        // Remove the last element (chunk=0, row=2).
        let moved = attr.swap_remove(0, 2).unwrap();
        assert!(moved.is_none());
        assert_eq!(attr.length, 2);
        assert_eq!(*attr.get(0, 0).unwrap(), 1);
        assert_eq!(*attr.get(0, 1).unwrap(), 2);
    }

    #[test]
    fn swap_remove_first_element() {
        let mut attr: Attribute<i32> = Attribute::default();
        attr.push(10).unwrap();
        attr.push(20).unwrap();
        attr.push(30).unwrap();
        // Remove element at (chunk=0, row=0); last element (30) should fill the gap.
        let moved = attr.swap_remove(0, 0).unwrap();
        assert_eq!(moved, Some((0, 2)));
        assert_eq!(attr.length, 2);
        assert_eq!(*attr.get(0, 0).unwrap(), 30);
        assert_eq!(*attr.get(0, 1).unwrap(), 20);
    }

    #[test]
    fn swap_remove_middle_element() {
        let mut attr: Attribute<i32> = Attribute::default();
        for v in [1, 2, 3, 4, 5] {
            attr.push(v).unwrap();
        }
        // Remove element at row 2 (value=3); last element (5 at row 4) fills the gap.
        let moved = attr.swap_remove(0, 2).unwrap();
        assert_eq!(moved, Some((0, 4)));
        assert_eq!(attr.length, 4);
        assert_eq!(*attr.get(0, 2).unwrap(), 5);
    }

    #[test]
    fn swap_remove_across_chunk_boundary() {
        // Fill more than one chunk so we can test cross-boundary swap-remove.
        let mut attr: Attribute<i32> = Attribute::default();
        for i in 0..(CHUNK_CAP + 1) {
            attr.push(i as i32).unwrap();
        }
        // The last element lives in chunk 1, row 0.
        // Remove the very first element (chunk 0, row 0).
        let moved = attr.swap_remove(0, 0).unwrap();
        assert_eq!(
            moved,
            Some((1, 0)),
            "last element from chunk 1 should fill the hole"
        );
        assert_eq!(attr.length, CHUNK_CAP);
        // The value that was at chunk 1, row 0 (= CHUNK_CAP as i32) is now at (0, 0).
        assert_eq!(*attr.get(0, 0).unwrap(), CHUNK_CAP as i32);
        // The second chunk should have been dropped since it is now empty.
        assert_eq!(attr.chunk_count(), 1);
    }

    #[test]
    fn swap_remove_out_of_bounds_returns_error() {
        let mut attr: Attribute<i32> = Attribute::default();
        attr.push(1).unwrap();
        // Row 1 does not exist.
        assert!(attr.swap_remove(0, 1).is_err());
        // Chunk 1 does not exist.
        assert!(attr.swap_remove(1, 0).is_err());
    }

    // -----------------------------------------------------------------------
    // push_from - success path
    // -----------------------------------------------------------------------

    #[test]
    fn push_from_success_moves_element() {
        let mut src: Attribute<i32> = Attribute::default();
        let mut dst: Attribute<i32> = Attribute::default();

        src.push(100).unwrap();
        src.push(200).unwrap();

        // Move src[0, 0] (= 100) into dst.
        let ((dst_chunk, dst_row), moved_src) = dst.push_from(&mut src, 0, 0).unwrap();

        // Destination received the value.
        assert_eq!(*dst.get(dst_chunk, dst_row).unwrap(), 100);
        assert_eq!(dst.length, 1);

        // Source had element 200 swap-filled into the hole.
        assert_eq!(src.length, 1);
        assert_eq!(moved_src, Some((0, 1)));
        assert_eq!(*src.get(0, 0).unwrap(), 200);
    }

    #[test]
    fn push_from_moving_last_element_leaves_no_moved_indicator() {
        let mut src: Attribute<i32> = Attribute::default();
        let mut dst: Attribute<i32> = Attribute::default();

        src.push(42).unwrap();

        let (_, moved_src) = dst.push_from(&mut src, 0, 0).unwrap();
        assert!(moved_src.is_none());
        assert_eq!(src.length, 0);
        assert_eq!(dst.length, 1);
        assert_eq!(*dst.get(0, 0).unwrap(), 42);
    }

    // -----------------------------------------------------------------------
    // push_from - source unchanged after invalid position
    // -----------------------------------------------------------------------

    #[test]
    fn push_from_invalid_source_position_leaves_source_unchanged() {
        let mut src: Attribute<i32> = Attribute::default();
        let mut dst: Attribute<i32> = Attribute::default();

        src.push(1).unwrap();
        src.push(2).unwrap();

        // Request an out-of-bounds position in the source.
        let result = dst.push_from(&mut src, 0, 5);
        assert!(result.is_err());

        // Source must be completely unchanged.
        assert_eq!(src.length, 2);
        assert_eq!(*src.get(0, 0).unwrap(), 1);
        assert_eq!(*src.get(0, 1).unwrap(), 2);
        // Destination must also be unchanged.
        assert_eq!(dst.length, 0);
    }

    // -----------------------------------------------------------------------
    // clear and Drop - destructor accounting
    // -----------------------------------------------------------------------

    #[test]
    fn clear_drops_all_elements() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let mut attr: Attribute<DropCounter> = Attribute::default();

        const N: usize = CHUNK_CAP + 3; // spans two chunks
        for _ in 0..N {
            attr.push(DropCounter(Arc::clone(&drop_count))).unwrap();
        }
        assert_eq!(drop_count.load(Ordering::Relaxed), 0);

        attr.clear();
        assert_eq!(
            drop_count.load(Ordering::Relaxed),
            N,
            "all {N} elements must be dropped by clear"
        );
        assert_eq!(attr.length, 0);
        assert!(attr.chunks.is_empty());
    }

    #[test]
    fn drop_calls_all_destructors() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        {
            let mut attr: Attribute<DropCounter> = Attribute::default();
            const N: usize = CHUNK_CAP * 2 + 1;
            for _ in 0..N {
                attr.push(DropCounter(Arc::clone(&drop_count))).unwrap();
            }
            // `attr` drops here.
        }
        // Every element's destructor must have fired.
        let expected = CHUNK_CAP * 2 + 1;
        assert_eq!(drop_count.load(Ordering::Relaxed), expected);
    }

    #[test]
    fn swap_remove_drops_removed_element() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let mut attr: Attribute<DropCounter> = Attribute::default();

        attr.push(DropCounter(Arc::clone(&drop_count))).unwrap();
        attr.push(DropCounter(Arc::clone(&drop_count))).unwrap();

        // swap_remove should drop the element at (0, 0).
        attr.swap_remove(0, 0).unwrap();
        assert_eq!(drop_count.load(Ordering::Relaxed), 1);

        // Dropping the attribute drops the remaining element.
        drop(attr);
        assert_eq!(drop_count.load(Ordering::Relaxed), 2);
    }

    // -----------------------------------------------------------------------
    // Capacity / chunk growth
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_count_grows_correctly() {
        let mut attr: Attribute<u32> = Attribute::default();

        for i in 0..(CHUNK_CAP * 3) {
            attr.push(i as u32).unwrap();
            let expected_chunks = i / CHUNK_CAP + 1;
            assert_eq!(
                attr.chunk_count(),
                expected_chunks,
                "after pushing {} elements, chunk_count should be {}",
                i + 1,
                expected_chunks
            );
        }
    }

    #[test]
    fn iter_visits_all_elements_in_order() {
        let mut attr: Attribute<i32> = Attribute::default();
        let n = CHUNK_CAP + 5;
        for i in 0..n {
            attr.push(i as i32).unwrap();
        }
        let collected: Vec<i32> = attr.iter().copied().collect();
        let expected: Vec<i32> = (0..n as i32).collect();
        assert_eq!(collected, expected);
    }

    // -----------------------------------------------------------------------
    // Position validation (regression: row >= CHUNK_CAP must error, not panic)
    // -----------------------------------------------------------------------

    #[test]
    fn swap_remove_rejects_row_beyond_chunk_capacity() {
        let mut attr: Attribute<u64> = Attribute::default();
        // Two full chunks, so `chunk*CHUNK_CAP + row < length` holds for the
        // out-of-range row below and the old index-only check would pass.
        for i in 0..(2 * CHUNK_CAP) {
            attr.push(i as u64).unwrap();
        }

        let result = attr.swap_remove(0, (CHUNK_CAP + 5) as u32);
        assert!(
            matches!(
                result,
                Err(crate::engine::error::AttributeError::Position(_))
            ),
            "expected Position error, got {result:?}"
        );
        assert_eq!(attr.length, 2 * CHUNK_CAP, "attribute must be unchanged");
    }

    #[test]
    fn take_swap_remove_rejects_row_beyond_chunk_capacity() {
        let mut attr: Attribute<u64> = Attribute::default();
        for i in 0..(2 * CHUNK_CAP) {
            attr.push(i as u64).unwrap();
        }

        let result = attr.take_swap_remove(0, (CHUNK_CAP + 5) as u32);
        assert!(
            matches!(
                result,
                Err(crate::engine::error::AttributeError::Position(_))
            ),
            "expected Position error, got Ok/other"
        );
        assert_eq!(attr.length, 2 * CHUNK_CAP, "attribute must be unchanged");
    }

    // -----------------------------------------------------------------------
    // Chunk hysteresis
    // -----------------------------------------------------------------------

    #[test]
    fn boundary_oscillation_reuses_the_spare_chunk() {
        let mut attr: Attribute<u64> = Attribute::default();
        for i in 0..CHUNK_CAP {
            attr.push(i as u64).unwrap();
        }

        // Cross the boundary once to allocate chunk 1.
        attr.push(0).unwrap();
        assert_eq!(attr.chunk_count(), 2);

        // Retiring the trailing chunk parks it in the spare slot...
        attr.swap_remove(1, 0).unwrap();
        assert_eq!(attr.chunk_count(), 1);
        assert!(attr.spare_chunk.is_some(), "popped chunk must be retained");

        // ...and the next boundary crossing takes it back without allocating.
        for _ in 0..100 {
            attr.push(0).unwrap();
            assert!(attr.spare_chunk.is_none(), "spare must be reused");
            attr.swap_remove(1, 0).unwrap();
            assert!(attr.spare_chunk.is_some(), "spare must be recaptured");
        }
        assert_eq!(attr.length, CHUNK_CAP);
    }

    #[test]
    fn truncate_to_drops_tail_and_keeps_prefix() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut attr: Attribute<DropCounter> = Attribute::default();
        for _ in 0..(CHUNK_CAP + 10) {
            attr.push(DropCounter(Arc::clone(&counter))).unwrap();
        }

        attr.truncate_to(5).unwrap();
        assert_eq!(attr.length, 5);
        assert_eq!(counter.load(Ordering::Relaxed), CHUNK_CAP + 5);
        assert_eq!(attr.chunk_count(), 1);

        // Truncating beyond the current length is rejected.
        assert!(attr.truncate_to(6).is_err());

        attr.truncate_to(0).unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), CHUNK_CAP + 10);
        assert_eq!(attr.length, 0);
    }

    // -----------------------------------------------------------------------
    // Batched-migration primitives: exactly-one-drop accounting
    // -----------------------------------------------------------------------

    #[test]
    fn batch_row_move_primitives_drop_each_value_exactly_once() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut source: Attribute<DropCounter> = Attribute::default();
        for _ in 0..100 {
            source.push(DropCounter(Arc::clone(&counter))).unwrap();
        }
        let mut destination: Attribute<DropCounter> = Attribute::default();

        // Copy phase: bitwise copies, no ownership transfer yet.
        let rows = [(0u16, 99u32), (0, 50), (0, 5)]; // descending
        let (start, count) = destination.extend_from_rows(&source, &rows).unwrap();
        assert_eq!((start, count), (0, 3));
        assert_eq!(
            counter.load(Ordering::Relaxed),
            0,
            "copy phase must not drop"
        );

        // Commit phase: forgotten removal transfers ownership to destination.
        for &(chunk, row) in &rows {
            source.swap_remove_forgotten(chunk, row).unwrap();
        }
        assert_eq!(counter.load(Ordering::Relaxed), 0, "commit must not drop");
        assert_eq!(source.length, 97);
        assert_eq!(destination.length, 3);

        drop(source);
        assert_eq!(counter.load(Ordering::Relaxed), 97);
        drop(destination);
        assert_eq!(
            counter.load(Ordering::Relaxed),
            100,
            "every value dropped exactly once"
        );
    }

    #[test]
    fn truncate_forgotten_discards_copies_without_dropping() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut source: Attribute<DropCounter> = Attribute::default();
        for _ in 0..10 {
            source.push(DropCounter(Arc::clone(&counter))).unwrap();
        }
        let mut destination: Attribute<DropCounter> = Attribute::default();
        let rows: Vec<(u16, u32)> = (0..10).map(|row| (0u16, row as u32)).collect();
        destination.extend_from_rows(&source, &rows).unwrap();

        // Rollback: the copies are discarded, source still owns everything.
        destination.truncate_forgotten(0).unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        assert_eq!(destination.length, 0);

        drop(destination);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        drop(source);
        assert_eq!(counter.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn extend_permuted_from_vec_applies_the_order() {
        let mut attr: Attribute<u32> = Attribute::default();
        attr.extend_permuted_from_vec(vec![10, 11, 12, 13], &[3, 1, 2, 0])
            .unwrap();
        let observed: Vec<u32> = attr.iter().copied().collect();
        assert_eq!(observed, vec![13, 11, 12, 10]);

        // Length mismatch and out-of-range indices are rejected untouched.
        assert!(attr.extend_permuted_from_vec(vec![1, 2], &[0]).is_err());
        assert!(attr.extend_permuted_from_vec(vec![1, 2], &[0, 5]).is_err());
        assert_eq!(attr.length, 4);
    }

    // -----------------------------------------------------------------------
    // Debug impl
    // -----------------------------------------------------------------------

    #[test]
    fn debug_impl_does_not_panic() {
        let mut attr: Attribute<i32> = Attribute::default();
        attr.push(1).unwrap();
        let s = format!("{:?}", attr);
        assert!(s.contains("Attribute"));
        assert!(s.contains("length"));
        assert!(s.contains("chunk_count"));
        assert!(s.contains("last_chunk_length"));
    }
}
