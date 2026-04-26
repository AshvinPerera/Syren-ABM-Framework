//! A heap-allocated, type-erased byte buffer that guarantees the alignment
//! required by any stored type.
//!
//! # Why not `Vec<u8>`?
//!
//! `Vec<u8>` is allocated with alignment 1.  Reinterpreting its contents as a
//! slice of a type `M` with `align_of::<M>() > 1` is **undefined behaviour**.
//! [`AlignedBuffer`] solves this by allocating with the correct alignment via
//! [`std::alloc::alloc`] and [`Layout::from_size_align`].
//!
//! # Safety contract for callers
//!
//! - `push<T>` and `as_slice<T>` must be called with the same type `T` that
//!   was used to construct the buffer (i.e. `item_size == size_of::<T>()` and
//!   `align == align_of::<T>()`).  The buffer does **not** track `TypeId`;
//!   that responsibility belongs to the call sites in the registry and
//!   buffer-set modules.

use std::alloc::{alloc, dealloc, realloc, Layout};
use std::ptr::NonNull;

/// A type-erased, alignment-aware byte buffer for message storage.
pub(crate) struct AlignedBuffer {
    /// Pointer to the first byte of the allocation.
    /// When `cap == 0` this is a dangling pointer; no reads/writes allowed.
    ptr: NonNull<u8>,
    /// Number of items currently stored.
    len: usize,
    /// Number of items the current allocation can hold.
    cap: usize,
    /// Size of one item in bytes (`size_of::<M>()`).
    item_size: usize,
    /// Required alignment of one item (`align_of::<M>()`).
    align: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Safety: AlignedBuffer owns its allocation.  No aliased access is created
// internally; callers in the messaging subsystem ensure read/write phases are
// exclusive (see `thread_local_emit` phase-discipline notes).
// ─────────────────────────────────────────────────────────────────────────────
unsafe impl Send for AlignedBuffer {}
// SAFETY: The buffer is only read via shared references during the consumption
// phase, when no writers are active (scheduler enforces phase ordering).
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Creates an empty buffer with no initial allocation.
    ///
    /// The buffer will allocate on the first [`push`](AlignedBuffer::push).
    #[inline]
    pub(crate) fn new(item_size: usize, align: usize) -> Self {
        debug_assert!(align.is_power_of_two(), "align must be a power of two");
        debug_assert!(item_size > 0, "item_size must be > 0");
        AlignedBuffer {
            // SAFETY: align is a power of two and > 0, so this is a valid
            // dangling pointer for the given alignment.
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            item_size,
            align,
        }
    }

    /// Creates a buffer pre-allocated for `capacity` items.
    pub(crate) fn with_capacity(item_size: usize, align: usize, capacity: usize) -> Self {
        let mut buf = Self::new(item_size, align);
        if capacity > 0 {
            buf.grow_to(capacity);
        }
        buf
    }

    /// Returns the number of items currently stored.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if no items are stored.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clears all items without deallocating.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    /// Appends a copy of `value` to the end of the buffer.
    ///
    /// # Safety
    ///
    /// `T` must be the exact type this buffer was created for:
    /// `size_of::<T>() == self.item_size` and `align_of::<T>() == self.align`.
    #[inline]
    pub(crate) unsafe fn push<T: Copy>(&mut self, value: T) {
        debug_assert_eq!(std::mem::size_of::<T>(), self.item_size);
        debug_assert_eq!(std::mem::align_of::<T>(), self.align);
        if self.len == self.cap {
            self.grow();
        }
        // SAFETY: we just ensured len < cap, so this slot is in-bounds and
        // properly aligned.
        unsafe {
            let dst = self.ptr.as_ptr().add(self.len * self.item_size) as *mut T;
            dst.write(value);
        }
        self.len += 1;
    }

    /// Sets the logical length of the buffer.
    ///
    /// # Safety
    ///
    /// `new_len` must not exceed `self.cap`.  Bytes `[len*item_size,
    /// new_len*item_size)` must have been initialised by the caller before
    /// any subsequent read (e.g. after a `copy_nonoverlapping` scatter).
    #[inline]
    pub(crate) unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.cap);
        self.len = new_len;
    }

    /// Returns a shared slice over the stored items.
    ///
    /// # Safety
    ///
    /// `T` must be the exact type this buffer was created for.
    #[inline]
    pub(crate) unsafe fn as_slice<T>(&self) -> &[T] {
        debug_assert_eq!(std::mem::size_of::<T>(), self.item_size);
        debug_assert_eq!(std::mem::align_of::<T>(), self.align);
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ptr is valid for self.len items of type T, properly aligned,
        // and no mutable reference to this data exists while &self is live.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const T, self.len) }
    }

    /// Returns a raw pointer to the item at position `index` (in bytes).
    ///
    /// # Safety
    ///
    /// `index` must be less than `self.len`.
    #[inline]
    pub(crate) unsafe fn as_ptr_at(&self, index: usize) -> *const u8 {
        debug_assert!(index < self.len);
        // SAFETY: index < len ≤ cap, so this offset is within the allocation.
        unsafe { self.ptr.as_ptr().add(index * self.item_size) }
    }

    /// Returns a mutable raw pointer to the item at position `index`.
    ///
    /// # Safety
    ///
    /// `index` must be less than `self.cap` and the caller must ensure
    /// exclusive access (no concurrent reads or writes).
    #[inline]
    pub(crate) unsafe fn as_mut_ptr_at(&mut self, index: usize) -> *mut u8 {
        debug_assert!(index < self.cap);
        // SAFETY: index < cap, so this offset is within the allocation.
        unsafe { self.ptr.as_ptr().add(index * self.item_size) }
    }

    /// Ensures the buffer can hold at least `new_cap` items, reallocating if
    /// necessary without changing `len`.
    pub(crate) fn reserve(&mut self, new_cap: usize) {
        if new_cap > self.cap {
            self.grow_to(new_cap);
        }
    }

    /// Appends all bytes from `other` into `self`, item by item.
    ///
    /// Both buffers must have been created with the same `item_size` and
    /// `align`.
    pub(crate) fn extend_from(&mut self, other: &AlignedBuffer) {
        debug_assert_eq!(self.item_size, other.item_size);
        debug_assert_eq!(self.align, other.align);
        if other.len == 0 {
            return;
        }
        let needed = self.len + other.len;
        if needed > self.cap {
            self.grow_to(needed);
        }
        // SAFETY: both pointers are valid; destination range does not overlap
        // source (different allocations or disjoint regions within the same one).
        unsafe {
            let dst = self.ptr.as_ptr().add(self.len * self.item_size);
            let src = other.ptr.as_ptr();
            std::ptr::copy_nonoverlapping(src, dst, other.len * other.item_size);
        }
        self.len += other.len;
    }

    /// Returns the initialized bytes backing the stored items.
    #[cfg_attr(not(feature = "messaging_gpu"), allow(dead_code))]
    #[inline]
    pub(crate) fn as_bytes(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: the first `len * item_size` bytes are initialized items.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len * self.item_size) }
    }

    /// Appends initialized item bytes.
    ///
    /// `bytes.len()` must be a multiple of `item_size`.
    #[cfg_attr(not(feature = "messaging_gpu"), allow(dead_code))]
    pub(crate) fn extend_from_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(bytes.len() % self.item_size, 0);
        if bytes.is_empty() {
            return;
        }
        let items = bytes.len() / self.item_size;
        let needed = self.len + items;
        if needed > self.cap {
            self.grow_to(needed);
        }
        // SAFETY: destination is valid for `bytes.len()` bytes after reserve.
        unsafe {
            let dst = self.ptr.as_ptr().add(self.len * self.item_size);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
        self.len += items;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Allocation helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Doubles capacity (or sets to 4 if currently 0).
    fn grow(&mut self) {
        let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
        self.grow_to(new_cap);
    }

    fn grow_to(&mut self, new_cap: usize) {
        debug_assert!(new_cap > self.cap);
        let new_layout = Layout::from_size_align(new_cap * self.item_size, self.align)
            .expect("AlignedBuffer: layout overflow");

        let new_ptr = if self.cap == 0 {
            // SAFETY: new_layout has non-zero size (new_cap > 0, item_size > 0).
            unsafe { alloc(new_layout) }
        } else {
            let old_layout = Layout::from_size_align(self.cap * self.item_size, self.align)
                .expect("AlignedBuffer: old layout overflow");
            // SAFETY: ptr was allocated with old_layout; new_layout has a
            // larger size.
            unsafe { realloc(self.ptr.as_ptr(), old_layout, new_layout.size()) }
        };

        self.ptr = NonNull::new(new_ptr).expect("AlignedBuffer: allocation failed");
        self.cap = new_cap;
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.cap > 0 {
            let layout = Layout::from_size_align(self.cap * self.item_size, self.align)
                .expect("AlignedBuffer: drop layout overflow");
            // SAFETY: ptr was allocated with this layout and cap > 0.
            unsafe { dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_read_u32() {
        let mut buf = AlignedBuffer::new(std::mem::size_of::<u32>(), std::mem::align_of::<u32>());
        for i in 0u32..8 {
            unsafe { buf.push(i) };
        }
        assert_eq!(buf.len(), 8);
        let slice: &[u32] = unsafe { buf.as_slice() };
        assert_eq!(slice, &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn push_and_read_aligned_struct() {
        #[repr(C, align(16))]
        #[derive(Copy, Clone, Debug, PartialEq)]
        struct Wide {
            x: f64,
            y: f64,
        }

        let mut buf = AlignedBuffer::new(std::mem::size_of::<Wide>(), std::mem::align_of::<Wide>());
        unsafe { buf.push(Wide { x: 1.0, y: 2.0 }) };
        unsafe { buf.push(Wide { x: 3.0, y: 4.0 }) };
        let slice: &[Wide] = unsafe { buf.as_slice() };
        assert_eq!(slice[0], Wide { x: 1.0, y: 2.0 });
        assert_eq!(slice[1], Wide { x: 3.0, y: 4.0 });
    }

    #[test]
    fn extend_from() {
        let mut a = AlignedBuffer::new(4, 4);
        let mut b = AlignedBuffer::new(4, 4);
        for i in 0u32..4 {
            unsafe { a.push(i) }
        }
        for i in 4u32..8 {
            unsafe { b.push(i) }
        }
        a.extend_from(&b);
        assert_eq!(a.len(), 8);
        let slice: &[u32] = unsafe { a.as_slice() };
        assert_eq!(slice, &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn clear_and_reuse() {
        let mut buf = AlignedBuffer::new(4, 4);
        for i in 0u32..4 {
            unsafe { buf.push(i) }
        }
        buf.clear();
        assert_eq!(buf.len(), 0);
        unsafe { buf.push(99u32) };
        let slice: &[u32] = unsafe { buf.as_slice() };
        assert_eq!(slice, &[99]);
    }
}
