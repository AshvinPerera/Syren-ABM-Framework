/// A chunked, contiguous, column-oriented storage container for elements of type `T`.
///
/// `Attribute<T>` stores elements in fixed-size chunks of capacity `CHUNK_CAP`,
/// each chunk represented as an array of `MaybeUninit<T>`. All elements are stored
/// densely, without gaps, and indexing is performed using `(ChunkID, RowID)`
/// coordinates.
///
/// # Storage Layout
///
/// Elements are stored in a vector of boxed arrays:
///
/// ```text
/// chunks: Vec<Box<[MaybeUninit<T>; CHUNK_CAP]>>
/// ```
///
/// A chunk is filled from row `0` upward. Chunks are filled in order, and only the
/// **last chunk** may be partially filled. All earlier chunks **must be completely
/// full**.
///
/// A visual example (`CHUNK_CAP = 4`):
///
/// ```text
/// Chunk 0: [ T, T, T, T ]   (full)
/// Chunk 1: [ T, T, T, T ]   (full)
/// Chunk 2: [ T, T, -, - ]   (partially full, last_chunk_length = 2)
/// ```
///
/// # Invariants
///
/// These invariants must hold at all times:
///
/// - **Full chunks rule:**
///   All chunks except possibly the last contain exactly `CHUNK_CAP` initialized
///   elements.
///
/// - **Last chunk rule:**
///   Only the last chunk may be partially initialized, with its initialized prefix
///   length stored in `last_chunk_length`, where:
///
///   ```text
///   0 <= last_chunk_length <= CHUNK_CAP
///   ```
///
/// - **Length rule:**
///   `length` is the **total number of initialized elements** across all chunks, and:
///
///   ```text
///   length = (chunks.len() - 1) * CHUNK_CAP + last_chunk_length
///   ```
///
/// - **Initialization rule:**
///   Only the first `last_chunk_length` elements of the last chunk may be
///   `assume_init_*()`-safe. All other elements must remain uninitialized.
///
/// These invariants allow the implementation to use `assume_init_ref`,
/// `assume_init_mut`, and `assume_init_drop` safely.
///
/// # Safety Notes
///
/// - This structure internally uses `MaybeUninit<T>` to avoid initializing unused
///   memory, and relies on strict invariant preservation to maintain safety.
/// - Operations such as `push`, `swap_remove`, and `push_from` use `unsafe` blocks
///   but remain sound because they preserve the invariants above.
/// - Iteration and indexed access (`get`, `get_mut`) rely on `valid_position()` to
///   determine whether a slot is initialized.
///
/// # Use Cases
///
/// - ECS component storage
/// - High-performance columnar databases
/// - Memory-dense simulation data
/// - Chunked array storage for incremental growth
///
/// # Performance Characteristics
///
/// - Appending (`push`) is amortized O(1).
/// - Removing with `swap_remove` is O(1).
/// - Indexed read/write is O(1).
/// - Iteration is cache-friendly and chunk-aligned.
///
/// # Type Parameters
/// - `T`: the stored element type.
///
/// # Fields
///
/// - `chunks` - The chunked backing storage.
/// - `last_chunk_length` - The number of initialized elements in the final chunk.
/// - `length` - Total number of initialized elements across all chunks.
use std::{array, convert::TryInto, fmt, mem::MaybeUninit, ptr};

use crate::engine::error::{AttributeError, AttributeInvariantViolation, PositionOutOfBoundsError};
use crate::engine::types::{ChunkID, RowID, CHUNK_CAP};

/// Typed, chunked storage for a single component column in an archetype.
///
/// Stores values in fixed-size chunks of [`CHUNK_CAP`] to allow cache-friendly
/// iteration and incremental allocation without full reallocation.
pub struct Attribute<T> {
    pub(crate) chunks: Vec<Box<[MaybeUninit<T>; CHUNK_CAP]>>,
    pub(crate) last_chunk_length: usize, // number of initialized elements in the last chunk
    pub(crate) length: usize,
}

impl<T: 'static + Send + Sync> fmt::Debug for Attribute<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Attribute")
            .field("length", &self.length)
            .field("chunk_count", &self.chunks.len())
            .field("last_chunk_length", &self.last_chunk_length)
            .finish()
    }
}

impl<T> Attribute<T> {
    /// Returns the number of allocated chunks in this attribute.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Ensures that there is a writable chunk at the end of the attribute.
    ///
    /// If:
    /// - no chunks exist, or
    /// - the last chunk is full (`last_chunk_length == CHUNK_CAP`),
    ///
    /// a new chunk is allocated and `last_chunk_length` is reset to zero.
    ///
    /// This function does **not** modify `length`; it only guarantees that
    /// writes to `(last_chunk, last_chunk_length)` are valid.

    #[inline]
    fn ensure_last_chunk(&mut self) {
        if self.chunks.is_empty() || self.last_chunk_length == CHUNK_CAP {
            self.chunks
                .push(Box::new(array::from_fn(|_| MaybeUninit::<T>::uninit())));
            self.last_chunk_length = 0;
        }
    }

    /// Returns `true` if `(chunk,row)` refers to an initialized element.
    ///
    /// For all chunks **except the last**, all rows `< CHUNK_CAP` are valid.
    /// For the last chunk, only rows `< last_chunk_length` are initialized.

    #[inline]
    pub(crate) fn valid_position(&self, chunk: ChunkID, row: RowID) -> bool {
        let chunk = chunk as usize;
        let row = row as usize;
        if chunk >= self.chunk_count() {
            return false;
        }
        if chunk + 1 == self.chunk_count() {
            row < self.last_chunk_length
        } else {
            row < CHUNK_CAP
        }
    }

    fn position_error(&self, chunk: ChunkID, row: RowID) -> AttributeError {
        AttributeError::Position(PositionOutOfBoundsError {
            chunk,
            row,
            chunks: self.chunks.len(),
            capacity: CHUNK_CAP,
            last_chunk_length: self.last_chunk_length,
        })
    }

    fn next_push_position(&self) -> Result<(ChunkID, RowID), AttributeError> {
        let chunk: ChunkID = (self.length / CHUNK_CAP)
            .try_into()
            .map_err(|_| AttributeError::IndexOverflow("ChunkID"))?;
        let row: RowID = (self.length % CHUNK_CAP)
            .try_into()
            .map_err(|_| AttributeError::IndexOverflow("RowID"))?;
        Ok((chunk, row))
    }

    /// Returns a mutable reference to the `MaybeUninit<T>` slot at `(chunk,row)`
    /// without performing any bounds checks.
    ///
    /// # Safety
    /// - `chunk < self.chunk_count()` must hold.
    /// - `row < CHUNK_CAP` must hold.
    /// - The caller must ensure that the usage of the returned slot obeys
    ///   initialization and aliasing rules.
    ///
    /// Debug asserts fire in debug mode, but no runtime checks exist in release.

    #[inline]
    pub(crate) unsafe fn get_slot_unchecked(
        &mut self,
        chunk: usize,
        row: usize,
    ) -> &mut MaybeUninit<T> {
        debug_assert!(chunk < self.chunk_count());
        debug_assert!(row < CHUNK_CAP);
        &mut self.chunks[chunk][row]
    }

    /// Reserves space for additional chunks in the underlying vector.
    ///
    /// This does **not** allocate or initialize new chunks; it simply increases
    /// the allocation capacity of the internal `Vec<Box<[MaybeUninit<T>; CHUNK_CAP]>>`.
    ///
    /// Useful for amortizing allocation cost before large inserts or bulk loads.

    pub fn reserve_chunks(&mut self, additional: usize) {
        self.chunks.reserve(additional);
    }

    /// Adjusts `chunks` and `last_chunk_length` after `length` has already been
    /// decremented. Pops any chunks that are now entirely past the new last element,
    /// and recomputes `last_chunk_length`.
    ///
    /// # Precondition
    /// `self.length` must already reflect the post-removal count.

    fn fixup_after_length_decrement(&mut self) {
        if self.length == 0 {
            self.chunks.clear();
            self.last_chunk_length = 0;
        } else {
            let new_last_chunk = (self.length - 1) / CHUNK_CAP;
            let new_last_row = (self.length - 1) % CHUNK_CAP;

            while self.chunks.len() - 1 > new_last_chunk {
                self.chunks.pop();
            }

            self.last_chunk_length = new_last_row + 1;
        }
    }

    /// Returns a shared reference to an initialized element at `(chunk, row)`,
    /// or `None` if the position is invalid.

    pub fn get(&self, chunk: ChunkID, row: RowID) -> Option<&T> {
        if !self.valid_position(chunk, row) {
            return None;
        }
        // SAFETY: `valid_position` guarantees that `(chunk, row)` refers to an
        // initialized slot within bounds. The element was written by a prior `push`
        // or `push_from`, so `assume_init_ref` is sound.
        Some(unsafe { self.chunks[chunk as usize][row as usize].assume_init_ref() })
    }

    /// Returns a mutable reference to an initialized element at `(chunk, row)`,
    /// or `None` if the position is invalid.

    pub fn get_mut(&mut self, chunk: ChunkID, row: RowID) -> Option<&mut T> {
        if !self.valid_position(chunk, row) {
            return None;
        }
        // SAFETY: `valid_position` guarantees that `(chunk, row)` refers to an
        // initialized slot within bounds. No other mutable reference can exist because
        // we hold `&mut self`.
        Some(unsafe { self.chunks[chunk as usize][row as usize].assume_init_mut() })
    }

    /// Returns an iterator over all initialized elements in the attribute.
    ///
    /// The iterator visits all chunks in order and yields references to elements
    /// in the order they were inserted. Only initialized elements are visited;
    /// the uninitialized tail of the final chunk is skipped.
    ///
    /// Returns an empty iterator if the attribute contains no elements.

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let last_chunk_length = self.last_chunk_length;
        let chunk_count = self.chunks.len();

        self.chunks.iter().enumerate().flat_map(move |(i, chunk)| {
            let initialized = if chunk_count == 0 {
                0
            } else if i == chunk_count - 1 {
                last_chunk_length
            } else {
                CHUNK_CAP
            };
            chunk[..initialized]
                .iter()
                // SAFETY: All chunks before the last are fully initialized (CHUNK_CAP
                // elements). The last chunk has exactly `last_chunk_length` initialized
                // elements. We only iterate over the initialized prefix.
                .map(|mu| unsafe { mu.assume_init_ref() })
        })
    }

    /// Appends a new element to the end of the attribute.
    ///
    /// If the final chunk is full, a new chunk is allocated before insertion.
    ///
    /// # Returns
    /// Returns the `(ChunkID, RowID)` location where the value was inserted.
    ///
    /// # Errors
    /// Returns [`AttributeError::IndexOverflow`] if the computed chunk or row
    /// index cannot be represented in their respective ID types.

    pub fn push(&mut self, value: T) -> Result<(ChunkID, RowID), AttributeError> {
        self.ensure_last_chunk();
        let chunk_index = self.chunks.len() - 1;
        let row_index = self.last_chunk_length;

        let chunk_id: ChunkID = chunk_index
            .try_into()
            .map_err(|_| AttributeError::IndexOverflow("ChunkID"))?;
        let row_id: RowID = row_index
            .try_into()
            .map_err(|_| AttributeError::IndexOverflow("RowID"))?;

        // SAFETY: `ensure_last_chunk` guarantees that `chunks[chunk_index]` exists
        // and `row_index < CHUNK_CAP` (because we reset `last_chunk_length` to 0
        // when a new chunk is created, and only reach here if it was < CHUNK_CAP).
        // The slot at `[chunk_index][row_index]` is uninitialized, so `ptr::write`
        // does not drop any existing value.
        unsafe {
            self.get_slot_unchecked(chunk_index, row_index)
                .as_mut_ptr()
                .write(value);
        }

        self.last_chunk_length += 1;
        self.length += 1;

        Ok((chunk_id, row_id))
    }

    /// Removes an element at the given `(chunk, row)` using a constant-time
    /// swap-remove strategy.
    ///
    /// This operation removes the element at the specified position. If the removed
    /// element is not the final element in the attribute, the last element is moved
    /// into the removed slot. The last element is then logically removed by
    /// decrementing `length`, adjusting `last_chunk_length`, and possibly popping an
    /// empty chunk.
    ///
    /// # Parameters
    /// - `chunk`: The chunk index of the element to remove.
    /// - `row`:   The row within the chunk.
    ///
    /// # Returns
    /// Returns `Some((ChunkID, RowID))` if another element was moved to fill
    /// the removed slot, or `None` if the removed element was already last.
    ///
    /// # Errors
    /// Returns [`AttributeError::Position`] if `(chunk, row)` does not identify
    /// a valid, initialized element.
    ///
    /// # Complexity
    /// Constant time: `O(1)`.

    pub fn swap_remove(
        &mut self,
        chunk: ChunkID,
        row: RowID,
    ) -> Result<Option<(ChunkID, RowID)>, AttributeError> {
        let chunk_count = self.chunks.len();

        if chunk as usize >= self.chunks.len() {
            return Err(AttributeError::Position(PositionOutOfBoundsError {
                chunk,
                row,
                chunks: chunk_count,
                capacity: CHUNK_CAP,
                last_chunk_length: self.last_chunk_length,
            }));
        }

        let index = chunk as usize * CHUNK_CAP + row as usize;
        if index >= self.length {
            return Err(AttributeError::Position(PositionOutOfBoundsError {
                chunk,
                row,
                chunks: chunk_count,
                capacity: CHUNK_CAP,
                last_chunk_length: self.last_chunk_length,
            }));
        }

        let last_index = self.length - 1;
        let last_chunk = last_index / CHUNK_CAP;
        let last_row = last_index % CHUNK_CAP;

        let is_last = (chunk as usize == last_chunk) && (row as usize == last_row);

        // SAFETY: The position has been validated against `self.length`, so the slot
        // at `(chunk, row)` is initialized. `assume_init_drop` runs the destructor
        // for the value in-place without creating a temporary owned `T`, which is the
        // correct approach when we intend to discard the removed value.
        if is_last {
            unsafe {
                self.get_slot_unchecked(chunk as usize, row as usize)
                    .assume_init_drop();
            }
        } else {
            // SAFETY: `last_index < self.length` and `index < last_index`, so both
            // `(chunk, row)` and `(last_chunk, last_row)` are initialized slots.
            // `ptr::read` moves ownership of the last element out of the slot.
            // We then drop the removed element in-place and write the last element
            // into the vacated slot, transferring ownership there. The last slot
            // is then overwritten with uninit to reflect that it no longer holds a
            // valid value.
            unsafe {
                let last_value = ptr::read(self.get_slot_unchecked(last_chunk, last_row).as_ptr());

                self.get_slot_unchecked(chunk as usize, row as usize)
                    .assume_init_drop();

                ptr::write(
                    self.get_slot_unchecked(chunk as usize, row as usize)
                        .as_mut_ptr(),
                    last_value,
                );

                *self.get_slot_unchecked(last_chunk, last_row) = MaybeUninit::uninit();
            }
        }

        let mut moved_from: Option<(ChunkID, RowID)> = None;
        if !is_last {
            moved_from = Some((
                last_chunk
                    .try_into()
                    .map_err(|_| AttributeError::IndexOverflow("chunk"))?,
                last_row
                    .try_into()
                    .map_err(|_| AttributeError::IndexOverflow("row"))?,
            ));
        }

        self.length -= 1;
        self.fixup_after_length_decrement();

        Ok(moved_from)
    }

    /// Removes a value with swap-remove and returns ownership of the removed value.
    pub(crate) fn take_swap_remove(
        &mut self,
        chunk: ChunkID,
        row: RowID,
    ) -> Result<(T, Option<(ChunkID, RowID)>), AttributeError> {
        let chunk_count = self.chunks.len();

        if chunk as usize >= self.chunks.len() {
            return Err(AttributeError::Position(PositionOutOfBoundsError {
                chunk,
                row,
                chunks: chunk_count,
                capacity: CHUNK_CAP,
                last_chunk_length: self.last_chunk_length,
            }));
        }

        let index = chunk as usize * CHUNK_CAP + row as usize;
        if index >= self.length {
            return Err(AttributeError::Position(PositionOutOfBoundsError {
                chunk,
                row,
                chunks: chunk_count,
                capacity: CHUNK_CAP,
                last_chunk_length: self.last_chunk_length,
            }));
        }

        let last_index = self.length - 1;
        let last_chunk = last_index / CHUNK_CAP;
        let last_row = last_index % CHUNK_CAP;
        let is_last = (chunk as usize == last_chunk) && (row as usize == last_row);

        let removed = unsafe {
            ptr::read(
                self.get_slot_unchecked(chunk as usize, row as usize)
                    .as_ptr(),
            )
        };

        let moved_from = if is_last {
            unsafe {
                *self.get_slot_unchecked(chunk as usize, row as usize) = MaybeUninit::uninit();
            }
            None
        } else {
            let last_value =
                unsafe { ptr::read(self.get_slot_unchecked(last_chunk, last_row).as_ptr()) };
            unsafe {
                *self.get_slot_unchecked(last_chunk, last_row) = MaybeUninit::uninit();
                ptr::write(
                    self.get_slot_unchecked(chunk as usize, row as usize)
                        .as_mut_ptr(),
                    last_value,
                );
            }
            Some((
                last_chunk
                    .try_into()
                    .map_err(|_| AttributeError::IndexOverflow("chunk"))?,
                last_row
                    .try_into()
                    .map_err(|_| AttributeError::IndexOverflow("row"))?,
            ))
        };

        self.length -= 1;
        self.fixup_after_length_decrement();

        Ok((removed, moved_from))
    }

    /// Restores a value removed by [`take_swap_remove`](Self::take_swap_remove).
    pub(crate) fn restore_swap_removed(
        &mut self,
        chunk: ChunkID,
        row: RowID,
        value: T,
        moved_from: Option<(ChunkID, RowID)>,
    ) -> Result<(), AttributeError> {
        match moved_from {
            Some(expected_append) => {
                if self.next_push_position()? != expected_append {
                    return Err(AttributeError::InternalInvariant(
                        AttributeInvariantViolation::LengthMismatch,
                    ));
                }
                if !self.valid_position(chunk, row) {
                    return Err(self.position_error(chunk, row));
                }

                let displaced = unsafe {
                    ptr::read(
                        self.get_slot_unchecked(chunk as usize, row as usize)
                            .as_ptr(),
                    )
                };
                unsafe {
                    ptr::write(
                        self.get_slot_unchecked(chunk as usize, row as usize)
                            .as_mut_ptr(),
                        value,
                    );
                }
                let pos = self.push(displaced)?;
                debug_assert_eq!(pos, expected_append);
                Ok(())
            }
            None => {
                if self.next_push_position()? != (chunk, row) {
                    return Err(AttributeError::InternalInvariant(
                        AttributeInvariantViolation::LengthMismatch,
                    ));
                }
                let pos = self.push(value)?;
                debug_assert_eq!(pos, (chunk, row));
                Ok(())
            }
        }
    }

    /// Pops and returns the last value, verifying its position first.
    pub(crate) fn pop_last_at(&mut self, expected: (ChunkID, RowID)) -> Result<T, AttributeError> {
        if self.length == 0 {
            return Err(AttributeError::InternalInvariant(
                AttributeInvariantViolation::SwapRemoveOnEmpty,
            ));
        }

        let last_index = self.length - 1;
        let last_chunk = last_index / CHUNK_CAP;
        let last_row = last_index % CHUNK_CAP;
        let actual = (
            last_chunk
                .try_into()
                .map_err(|_| AttributeError::IndexOverflow("chunk"))?,
            last_row
                .try_into()
                .map_err(|_| AttributeError::IndexOverflow("row"))?,
        );

        if actual != expected {
            return Err(AttributeError::InternalInvariant(
                AttributeInvariantViolation::LengthMismatch,
            ));
        }

        let value = unsafe { ptr::read(self.get_slot_unchecked(last_chunk, last_row).as_ptr()) };
        unsafe {
            *self.get_slot_unchecked(last_chunk, last_row) = MaybeUninit::uninit();
        }

        self.length -= 1;
        self.fixup_after_length_decrement();

        Ok(value)
    }

    /// Moves an element from a source attribute into this attribute.
    ///
    /// The value at `(source_chunk, source_row)` is removed from `source` and
    /// appended to `self`. If the removed source element is not the last one,
    /// a swap-remove is performed in the source attribute.
    ///
    /// # Parameters
    /// - `source`: The attribute to move the value from.
    /// - `source_chunk`: Chunk index of the source element.
    /// - `source_row`: Row index of the source element.
    ///
    /// # Returns
    /// Returns:
    /// - the destination `(ChunkID, RowID)` where the value was inserted, and
    /// - an optional `(ChunkID, RowID)` indicating which source element was
    ///   moved during swap-remove, if any.
    ///
    /// # Errors
    /// - [`AttributeError::Position`] if the source position is invalid.
    /// - Any error returned by [`Attribute::push`] on the destination.
    ///
    /// # Failure Semantics
    ///
    /// If the push into the destination fails after the source value has been
    /// consumed, the source attribute performs a compensating swap-remove to
    /// fill the resulting hole. This preserves the source's dense-packing
    /// invariant. The consumed value itself is lost (it was dropped by `push`'s
    /// error path), but no uninitialized memory is left in the source's
    /// initialized range.
    ///
    /// # Complexity
    /// `O(1)` for both the transfer and the source swap-remove.

    pub fn push_from(
        &mut self,
        source: &mut Attribute<T>,
        source_chunk: ChunkID,
        source_row: RowID,
    ) -> Result<((ChunkID, RowID), Option<(ChunkID, RowID)>), AttributeError> {
        let source_chunk_count = source.chunks.len();
        if !source.valid_position(source_chunk, source_row) {
            return Err(AttributeError::Position(PositionOutOfBoundsError {
                chunk: source_chunk,
                row: source_row,
                chunks: source_chunk_count,
                capacity: CHUNK_CAP,
                last_chunk_length: source.last_chunk_length,
            }));
        }

        // SAFETY: `valid_position` confirmed the source slot is initialized.
        // `ptr::read` moves ownership out of the slot. We immediately overwrite the
        // slot with `MaybeUninit::uninit()` to mark it as logically empty, preventing
        // any double-drop if we return early.
        let moved_value = unsafe {
            let value = ptr::read(
                source
                    .get_slot_unchecked(source_chunk as usize, source_row as usize)
                    .as_ptr(),
            );
            *source.get_slot_unchecked(source_chunk as usize, source_row as usize) =
                MaybeUninit::uninit();
            value
        };

        let (destination_chunk, destination_row) = match self.push(moved_value) {
            Ok(pos) => pos,
            Err(e) => {
                // Push failed. The value was consumed by `push` (either written
                // and cleaned up, or dropped when the `value` parameter went out
                // of scope on the IndexOverflow path). The source slot at
                // (source_chunk, source_row) is now uninit, creating a hole.
                //
                // To maintain the source's dense-packing invariant, we perform a
                // compensating swap-remove: move the source's last element into
                // the hole and decrement the source's length. If the hole *is*
                // the last element, we just decrement.

                let hole_index = source_chunk as usize * CHUNK_CAP + source_row as usize;
                let last_index = source.length - 1;

                if hole_index != last_index {
                    let last_chunk = last_index / CHUNK_CAP;
                    let last_row = last_index % CHUNK_CAP;

                    // SAFETY: `last_index` points to an initialized slot that is
                    // different from the hole. `ptr::read` moves ownership out;
                    // `ptr::write` places it into the hole, filling it.
                    unsafe {
                        let last_value =
                            ptr::read(source.get_slot_unchecked(last_chunk, last_row).as_ptr());
                        *source.get_slot_unchecked(last_chunk, last_row) = MaybeUninit::uninit();
                        ptr::write(
                            source
                                .get_slot_unchecked(source_chunk as usize, source_row as usize)
                                .as_mut_ptr(),
                            last_value,
                        );
                    }
                }

                // Decrement length and fix up bookkeeping.
                source.length -= 1;
                source.fixup_after_length_decrement();

                return Err(e);
            }
        };

        let last_index = source.length - 1;
        let last_chunk = last_index / CHUNK_CAP;
        let last_row = last_index % CHUNK_CAP;

        let mut moved_from_source: Option<(ChunkID, RowID)> = None;

        let source_index = source_chunk as usize * CHUNK_CAP + source_row as usize;

        if source_index != last_index {
            // SAFETY: `last_index` is the index of the last initialized element in
            // the source (`source.length - 1`), and `source_index != last_index`, so
            // the last slot is a different, initialized slot. `ptr::read` moves
            // ownership out. We then overwrite the source slot (which was already
            // marked uninit above) with the last value via `ptr::write`, completing
            // the swap-remove.
            let last_value = unsafe {
                let value = ptr::read(source.get_slot_unchecked(last_chunk, last_row).as_ptr());
                *source.get_slot_unchecked(last_chunk, last_row) = MaybeUninit::uninit();
                value
            };

            moved_from_source = Some((
                last_chunk
                    .try_into()
                    .map_err(|_| AttributeError::IndexOverflow("chunk"))?,
                last_row
                    .try_into()
                    .map_err(|_| AttributeError::IndexOverflow("row"))?,
            ));

            // SAFETY: The source slot at `(source_chunk, source_row)` was marked
            // uninit above, so writing into it does not cause a double-drop. The
            // `last_value` was moved out of the last slot, so ownership transfers
            // cleanly into the source slot.
            unsafe {
                ptr::write(
                    source
                        .get_slot_unchecked(source_chunk as usize, source_row as usize)
                        .as_mut_ptr(),
                    last_value,
                );
            }
        }

        source.length -= 1;
        source.fixup_after_length_decrement();

        Ok(((destination_chunk, destination_row), moved_from_source))
    }

    /// Extends the attribute by pushing all elements from the iterator.
    ///
    /// Bulk extension is simply repeated calls to `push`.
    /// If any insert fails, the function returns the error immediately.

    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iterator: I) -> Result<(), AttributeError> {
        for v in iterator {
            self.push(v)?;
        }
        Ok(())
    }

    /// Drops all initialized elements in all chunks without modifying the chunk
    /// structure.
    ///
    /// This is used internally by [`clear`] and during destruction.

    fn drop_all_initialized_elements(&mut self) {
        if self.length == 0 {
            return;
        }

        let mut remaining = self.length;
        let chunk_count = self.chunks.len();
        let last_chunk_len = self.last_chunk_length;

        for (chunk_idx, chunk) in self.chunks.iter_mut().enumerate() {
            let init_in_chunk = if chunk_idx + 1 == chunk_count {
                last_chunk_len
            } else {
                CHUNK_CAP
            };

            let to_drop = init_in_chunk.min(remaining);
            for i in 0..to_drop {
                // SAFETY: We only iterate over the initialized prefix of each chunk.
                // Full chunks have `CHUNK_CAP` initialized elements; the last chunk
                // has `last_chunk_len` initialized elements. `remaining` tracks how
                // many elements are left to drop across all chunks.
                unsafe {
                    chunk[i].assume_init_drop();
                }
            }

            if remaining <= init_in_chunk {
                break;
            }
            remaining -= init_in_chunk;
        }
    }

    /// Clears the attribute by dropping all initialized elements and freeing all
    /// allocated chunks.
    ///
    /// After calling this method:
    /// - `length == 0`,
    /// - `chunks.is_empty()`,
    /// - `last_chunk_length == 0`.
    ///
    /// Equivalent to resetting the attribute to its initial state.

    pub fn clear(&mut self) {
        if self.length == 0 {
            return;
        }

        self.drop_all_initialized_elements();
        self.chunks.clear();
        self.length = 0;
        self.last_chunk_length = 0;
    }
}

impl<T> Default for Attribute<T> {
    fn default() -> Self {
        Self {
            chunks: Vec::new(),
            last_chunk_length: 0,
            length: 0,
        }
    }
}

impl<T> Drop for Attribute<T> {
    fn drop(&mut self) {
        self.drop_all_initialized_elements();
    }
}
