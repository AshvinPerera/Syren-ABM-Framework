
#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    use crate::engine::storage::Attribute;
    use crate::engine::types::CHUNK_CAP;

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
        assert!(moved.is_none(), "no element should be moved when removing the only one");
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
        assert_eq!(moved, Some((1, 0)), "last element from chunk 1 should fill the hole");
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
    // push_from – success path
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
    // push_from – source unchanged after invalid position
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
    // clear and Drop – destructor accounting
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
        assert_eq!(drop_count.load(Ordering::Relaxed), N, "all {N} elements must be dropped by clear");
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
