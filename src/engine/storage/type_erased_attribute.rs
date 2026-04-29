//! Type-erased interface for chunked attribute storage.
//!
//! This module defines the [`TypeErasedAttribute`] trait and its blanket
//! implementation for [`Attribute<T>`], allowing attribute containers to be
//! managed in heterogeneous collections without knowing `T` at compile time.

/// A type-erased interface for chunked attribute storage.
///
/// `TypeErasedAttribute` provides a dynamically-typed, reflection-based API for
/// interacting with attribute containers without knowing their underlying element
/// type `T`.
///
/// Attributes implementing this trait must internally store elements in fixed-size
/// chunks (`CHUNK_CAP` rows per chunk), and must maintain the following invariants:
///
/// - `length()` returns the total number of initialised elements.
/// - `chunk_count()` returns the number of allocated chunks.
/// - `last_chunk_length()` returns the number of initialised elements in the final
///   chunk and must satisfy:
///   `0 < last_chunk_length <= CHUNK_CAP`, unless `length() == 0`.
/// - All indices below `length()` correspond to initialised, valid elements.
/// - Types exposed through `chunk_slice_ref`, `chunk_slice_mut`, and `push_dyn` must
///   correspond to the attribute's concrete element type.
///
/// This interface allows:
/// - querying storage structure,
/// - retrieving typed or untyped chunk slices,
/// - downcasting back to concrete attribute types,
/// - inserting, removing, and transferring elements using dynamically-typed APIs.
///
/// # Safety
/// Although the trait uses no explicit `unsafe` methods, implementing it requires
/// ensuring memory correctness around uninitialized regions and type alignment.
/// Callers must only request typed slices (`chunk_slice_ref`, `chunk_slice_mut`) that
/// match the implementing attribute's element type. Mismatches must return `None`.
///
/// # Downcasting
/// Implementers must return `self` cast to `&dyn Any` / `&mut dyn Any` so callers can
/// attempt a `downcast::<Attribute<T>>()` when necessary.
///
/// ## Example
/// ```text
/// if let Some(attr) = erased.as_any().downcast_ref::<Attribute<f32>>() {
///     println!("attribute stores {} f32 values", attr.length());
/// }
/// ```
///
/// # Element transfer and mutation
///
/// - `swap_remove_dyn` removes an element using swap-remove.
/// - `push_dyn` inserts a dynamically-typed element.
/// - `push_from_dyn` moves an element between attributes.
///
/// # Intended usage
/// This trait is suitable when:
/// - components or attributes must be managed in a type-erased container,
/// - serializers/deserializers need read/write access to raw storage,
/// - low-level systems operate on contiguous chunks of memory for performance.
use std::any::{type_name, Any, TypeId};
use std::mem::size_of;

use crate::engine::error::{AttributeError, TypeMismatchError};
use crate::engine::storage::attribute::Attribute;
use crate::engine::storage::{PushFromOutcome, TakeSwapRemoveOutcome};
use crate::engine::types::{ChunkID, RowID, CHUNK_CAP};

/// Type-erased interface for a component column, enabling archetype storage
/// to hold heterogeneous component types without monomorphisation at the
/// archetype level.
pub trait TypeErasedAttribute: Any + Send + Sync {
    /// Returns the number of allocated chunks in this attribute.
    fn chunk_count(&self) -> usize;

    /// Reserves additional chunk slots in the backing storage vector.
    fn reserve_chunks(&mut self, additional: usize);

    /// Returns the total number of initialised elements stored.
    fn length(&self) -> usize;

    /// Returns the number of initialised elements in the final chunk.
    fn last_chunk_length(&self) -> usize;

    /// Returns an immutable type-erased reference for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns a mutable type-erased reference for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Returns the `TypeId` of the element type stored by this attribute.
    fn element_type_id(&self) -> TypeId;

    /// Returns the human-readable name of the element type stored.
    fn element_type_name(&self) -> &'static str;

    /// Returns a raw byte pointer and size for a chunk slice.
    fn chunk_bytes(&self, chunk_id: ChunkID, length: usize) -> Option<(*const u8, usize)>;

    /// Returns a mutable raw byte pointer and size for a chunk slice.
    fn chunk_bytes_mut(&mut self, chunk_id: ChunkID, length: usize) -> Option<(*mut u8, usize)>;

    /// Returns a typed immutable slice into a chunk if the requested type matches.
    ///
    /// Note: These methods have `where Self: Sized` bounds, making them
    /// uncallable through trait objects. For trait-object usage, use the
    /// `_dyn` variants (`chunk_bytes`, `chunk_bytes_mut`).
    fn chunk_slice<T: 'static>(&self, chunk_id: ChunkID, length: usize) -> Option<&[T]>
    where
        Self: Sized;

    /// Returns a typed mutable slice into a chunk if the requested type matches.
    ///
    /// Note: These methods have `where Self: Sized` bounds, making them
    /// uncallable through trait objects. For trait-object usage, use the
    /// `_dyn` variants (`chunk_bytes`, `chunk_bytes_mut`).
    fn chunk_slice_mut<T: 'static>(&mut self, chunk_id: ChunkID, length: usize) -> Option<&mut [T]>
    where
        Self: Sized;

    /// Removes an element using swap-remove through a type-erased interface.
    fn swap_remove_dyn(
        &mut self,
        chunk: ChunkID,
        row: RowID,
    ) -> Result<Option<(ChunkID, RowID)>, AttributeError>;

    /// Inserts a dynamically-typed value into the attribute.
    fn push_dyn(&mut self, value: Box<dyn Any>) -> Result<(ChunkID, RowID), AttributeError>;

    /// Transfers an element from another attribute into this one.
    fn push_from_dyn(
        &mut self,
        source: &mut dyn TypeErasedAttribute,
        source_chunk: ChunkID,
        source_row: RowID,
    ) -> Result<PushFromOutcome, AttributeError>;

    /// Removes a value with swap-remove and returns the removed value.
    ///
    /// The returned position is the source position of the row that was moved
    /// into the removed slot, if the removed row was not already last.
    fn take_swap_remove_dyn(
        &mut self,
        chunk: ChunkID,
        row: RowID,
    ) -> Result<TakeSwapRemoveOutcome, AttributeError>;

    /// Reverses a previous [`take_swap_remove_dyn`](Self::take_swap_remove_dyn).
    ///
    /// The `moved_from` argument must be the value returned by that prior take.
    fn restore_swap_removed_dyn(
        &mut self,
        chunk: ChunkID,
        row: RowID,
        value: Box<dyn Any>,
        moved_from: Option<(ChunkID, RowID)>,
    ) -> Result<(), AttributeError>;

    /// Pops the current last value, asserting that it is at `expected`.
    fn pop_last_dyn(&mut self, expected: (ChunkID, RowID)) -> Result<Box<dyn Any>, AttributeError>;

    /// Replaces the value at `(chunk, row)` **in place**, dropping the old
    /// value correctly and consuming the provided boxed replacement.
    ///
    /// Semantics:
    /// - No archetype transition, no growth, no swap-remove.
    /// - The caller must guarantee `(chunk, row)` refers to an initialised
    ///   slot (i.e., a live entity row).
    /// - The dynamic type of `value` must match the column's stored type;
    ///   mismatches return [`AttributeError::TypeMismatch`] and leave the
    ///   slot untouched.
    ///
    /// Used by [`Command::Set`](crate::engine::commands::Command::Set) to
    /// overwrite a single component without archetype migration. Correctly
    /// drops the old value, so components do not need to be `Copy`.
    ///
    /// # Errors
    /// - [`AttributeError::TypeMismatch`] - `value`'s type does not match `T`.
    /// - [`AttributeError::Position`] - `(chunk, row)` is not an initialised
    ///   slot in this attribute.
    fn replace_slot_dyn(
        &mut self,
        chunk: ChunkID,
        row: RowID,
        value: Box<dyn Any>,
    ) -> Result<(), AttributeError>;
}

impl<T: 'static + Send + Sync> TypeErasedAttribute for Attribute<T> {
    fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
    fn reserve_chunks(&mut self, additional: usize) {
        Attribute::reserve_chunks(self, additional);
    }
    fn length(&self) -> usize {
        self.length
    }
    fn last_chunk_length(&self) -> usize {
        self.last_chunk_length
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn element_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }
    fn element_type_name(&self) -> &'static str {
        type_name::<T>()
    }

    fn chunk_bytes(&self, chunk_id: ChunkID, valid_length: usize) -> Option<(*const u8, usize)> {
        let chunk = self.chunks.get(chunk_id as usize)?;
        let len = if chunk_id as usize + 1 == self.chunks.len() {
            self.last_chunk_length
        } else {
            CHUNK_CAP
        }
        .min(valid_length);

        let ptr = chunk.as_ptr() as *const u8;
        let bytes = len * size_of::<T>();
        Some((ptr, bytes))
    }

    fn chunk_bytes_mut(
        &mut self,
        chunk_id: ChunkID,
        valid_length: usize,
    ) -> Option<(*mut u8, usize)> {
        let chunk_index = chunk_id as usize;
        let chunk_count = self.chunks.len();
        let len = if chunk_index + 1 == chunk_count {
            self.last_chunk_length
        } else {
            CHUNK_CAP
        }
        .min(valid_length);

        let chunk = self.chunks.get_mut(chunk_index)?;

        let ptr = chunk.as_mut_ptr() as *mut u8;
        let bytes = len * size_of::<T>();
        Some((ptr, bytes))
    }

    /// Returns a typed immutable slice of the elements stored in the specified
    /// chunk, if and only if the requested type `U` matches the attribute's actual
    /// element type `T`.
    ///
    /// # Parameters
    /// - `chunk_id`: The chunk index from which to retrieve elements.
    /// - `valid_length`: The maximum number of elements the caller intends to
    ///   operate on. The returned slice length is clamped to the chunk's actual
    ///   initialized length.
    ///
    /// # Returns
    /// - `Some(&[U])` if `U == T` and `chunk_id` is a valid chunk index.
    /// - `None` if the type does not match or the chunk index is out of bounds.
    fn chunk_slice<U: 'static>(&self, chunk_id: ChunkID, valid_length: usize) -> Option<&[U]> {
        if TypeId::of::<U>() != TypeId::of::<T>() {
            return None;
        }

        let chunk = self.chunks.get(chunk_id as usize)?;
        let len = if chunk_id as usize + 1 == self.chunks.len() {
            self.last_chunk_length
        } else {
            CHUNK_CAP
        }
        .min(valid_length);

        // SAFETY: The type check above guarantees `U == T`, so the pointer cast is
        // layout-compatible. `len` is bounded by the chunk's initialised prefix.
        // The slice borrows from `self`, so it cannot outlive the attribute.
        Some(unsafe { std::slice::from_raw_parts(chunk.as_ptr() as *const U, len) })
    }

    /// Returns a typed mutable slice of the elements stored in the specified
    /// chunk, if and only if the requested type `U` matches the attribute's actual
    /// element type `T`.
    ///
    /// # Parameters
    /// - `chunk_id`: The chunk index from which to retrieve elements.
    /// - `valid_length`: The maximum number of elements the caller intends to
    ///   operate on. The returned slice length is clamped to the chunk's actual
    ///   initialized length.
    ///
    /// # Returns
    /// - `Some(&mut [U])` if `U == T` and `chunk_id` is a valid chunk index.
    /// - `None` if the type does not match or the chunk index is out of bounds.
    fn chunk_slice_mut<U: 'static>(
        &mut self,
        chunk_id: ChunkID,
        valid_length: usize,
    ) -> Option<&mut [U]> {
        if TypeId::of::<U>() != TypeId::of::<T>() {
            return None;
        }

        let is_last = chunk_id as usize + 1 == self.chunks.len();
        let len = if is_last {
            self.last_chunk_length
        } else {
            CHUNK_CAP
        }
        .min(valid_length);

        let chunk = self.chunks.get_mut(chunk_id as usize)?;
        // SAFETY: The type check above guarantees `U == T`, so the pointer cast is
        // layout-compatible. `len` is bounded by the chunk's initialised prefix.
        // We hold `&mut self`, so no aliasing mutable references can exist.
        Some(unsafe { std::slice::from_raw_parts_mut(chunk.as_mut_ptr() as *mut U, len) })
    }

    /// Removes an element at the given position using swap-remove through a
    /// type-erased interface.
    ///
    /// # Returns
    /// Returns the moved-from position (if any).
    ///
    /// # Errors
    /// - [`AttributeError::Position`] if the `(chunk, row)` position is invalid.
    fn swap_remove_dyn(
        &mut self,
        chunk: ChunkID,
        row: RowID,
    ) -> Result<Option<(ChunkID, RowID)>, AttributeError> {
        self.swap_remove(chunk, row)
    }

    /// Attempts to push a dynamically-typed value into the attribute.
    ///
    /// If the provided `Box<dyn Any>` contains a value of type `T`, it is extracted
    /// and forwarded to [`Attribute::push`].
    ///
    /// # Returns
    /// Returns the insertion location `(ChunkID, RowID)`.
    ///
    /// # Errors
    /// - [`AttributeError::TypeMismatch`] if the dynamic value has the wrong type.
    fn push_dyn(&mut self, value: Box<dyn Any>) -> Result<(ChunkID, RowID), AttributeError> {
        let actual = value.as_ref().type_id();
        if let Ok(v) = value.downcast::<T>() {
            let (chunk, row) = self.push(*v)?;
            return Ok((chunk, row));
        }

        Err(AttributeError::TypeMismatch(TypeMismatchError {
            expected: TypeId::of::<T>(),
            actual,
            expected_name: type_name::<T>(),
            actual_name: "",
        }))
    }

    /// Moves an element from a source attribute into this attribute through a
    /// type-erased interface.
    ///
    /// # Returns
    /// Returns the destination insertion position and information about any
    /// swap-remove that occurred in the source.
    ///
    /// # Errors
    /// - [`AttributeError::TypeMismatch`] if the source attribute doesn't store
    ///   the same element type as this attribute.
    /// - [`AttributeError::Position`] if the `(source_chunk, source_row)` position
    ///   is invalid in the source attribute.
    fn push_from_dyn(
        &mut self,
        source: &mut dyn TypeErasedAttribute,
        source_chunk: ChunkID,
        source_row: RowID,
    ) -> Result<PushFromOutcome, AttributeError> {
        // Capture immutable info FIRST before taking the mutable borrow
        let actual_type = source.element_type_id();
        let actual_name = source.element_type_name();

        let source_attribute = match source.as_any_mut().downcast_mut::<Attribute<T>>() {
            Some(attr) => attr,
            None => {
                return Err(AttributeError::TypeMismatch(TypeMismatchError {
                    expected: TypeId::of::<T>(),
                    actual: actual_type,
                    expected_name: type_name::<T>(),
                    actual_name,
                }));
            }
        };

        self.push_from(source_attribute, source_chunk, source_row)
    }

    fn take_swap_remove_dyn(
        &mut self,
        chunk: ChunkID,
        row: RowID,
    ) -> Result<TakeSwapRemoveOutcome, AttributeError> {
        let (value, moved_from) = self.take_swap_remove(chunk, row)?;
        Ok((Box::new(value), moved_from))
    }

    fn restore_swap_removed_dyn(
        &mut self,
        chunk: ChunkID,
        row: RowID,
        value: Box<dyn Any>,
        moved_from: Option<(ChunkID, RowID)>,
    ) -> Result<(), AttributeError> {
        let actual_type = value.as_ref().type_id();
        let value = value.downcast::<T>().map_err(|_| {
            AttributeError::TypeMismatch(TypeMismatchError {
                expected: TypeId::of::<T>(),
                actual: actual_type,
                expected_name: type_name::<T>(),
                actual_name: "",
            })
        })?;
        self.restore_swap_removed(chunk, row, *value, moved_from)
    }

    fn pop_last_dyn(&mut self, expected: (ChunkID, RowID)) -> Result<Box<dyn Any>, AttributeError> {
        self.pop_last_at(expected)
            .map(|value| Box::new(value) as Box<dyn Any>)
    }

    /// Replaces the value at `(chunk, row)` in place.
    ///
    /// Drops the old value using the concrete `T`'s drop glue before writing
    /// the new one, so this is correct for any `T: 'static + Send + Sync`
    /// regardless of whether `T: Copy`.
    ///
    /// Type mismatch is reported without mutating the slot.
    fn replace_slot_dyn(
        &mut self,
        chunk: ChunkID,
        row: RowID,
        value: Box<dyn Any>,
    ) -> Result<(), AttributeError> {
        // Type-check first; on mismatch the slot is untouched.
        let actual_type = value.as_ref().type_id();
        if actual_type != TypeId::of::<T>() {
            return Err(AttributeError::TypeMismatch(TypeMismatchError {
                expected: TypeId::of::<T>(),
                actual: actual_type,
                expected_name: type_name::<T>(),
                actual_name: "",
            }));
        }

        // Bounds check - the caller promises the slot is live, but verify.
        if !self.valid_position(chunk, row) {
            return Err(AttributeError::Position(
                crate::engine::error::PositionOutOfBoundsError {
                    chunk,
                    row,
                    chunks: self.chunks.len(),
                    capacity: CHUNK_CAP,
                    last_chunk_length: self.last_chunk_length,
                },
            ));
        }

        // Downcast the box so we own a `Box<T>` whose deallocation runs
        // through the correct allocator without invoking the `Any` vtable.
        let typed_box = match value.downcast::<T>() {
            Ok(b) => b,
            Err(_) => {
                // Unreachable given the TypeId check above; defensive path.
                return Err(AttributeError::TypeMismatch(TypeMismatchError {
                    expected: TypeId::of::<T>(),
                    actual: actual_type,
                    expected_name: type_name::<T>(),
                    actual_name: "",
                }));
            }
        };
        let new_value: T = *typed_box;

        // SAFETY:
        // - `valid_position` confirmed `(chunk, row)` refers to an
        //   initialised slot, so `assume_init_drop` will correctly drop
        //   the previously-stored `T` using its real drop glue.
        // - After dropping, we write the new `T` into the now-uninit slot
        //   via `MaybeUninit::write`, which does not drop the destination
        //   a second time (the slot is uninit at that point).
        // - No aliasing: we hold `&mut self`, so nothing else can see
        //   the slot during this swap.
        unsafe {
            let slot = self.get_slot_unchecked(chunk as usize, row as usize);
            slot.assume_init_drop();
            slot.write(new_value);
        }

        Ok(())
    }
}
