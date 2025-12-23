//! Chunked attribute storage and type-erased access for ECS-style column data.
//!
//! This module implements a high-performance, column-oriented container,
//! [`Attribute<T>`], which stores values densely in fixed-capacity chunks
//! (`CHUNK_CAP` rows per chunk). The design targets ECS/component storage and
//! similar workloads where predictable layout, cache-friendly iteration, and
//! constant-time insert/remove are more important than stable ordering.
//!
//! # What this module provides
//!
//! - **`Attribute<T>`**: A chunked, contiguous storage container for a single
//!   element type `T`.
//! - **`TypeErasedAttribute`**: A dynamically-typed interface for interacting
//!   with attributes without knowing `T` at compile time (for heterogeneous
//!   containers, reflection-like tooling, serialization, etc.).
//! - **Raw chunk access** (`chunk_bytes`, `chunk_bytes_mut`) and helper casting
//!   utilities (`cast_slice`, `cast_slice_mut`) for low-level, zero-copy
//!   operations.
//! - **Optional rollback support** behind the `rollback` feature flag for
//!   fully undoing mutating operations.
//!
//! # Storage model
//!
//! Internally, an attribute stores its values as:
//!
//! ```text
//! Vec<Box<[MaybeUninit<T>; CHUNK_CAP]>>
//! ```
//!
//! Values are written densely from the beginning of chunk 0 upward, with no gaps.
//! All chunks except the final chunk are fully initialized. Only the last chunk
//! may be partially filled, tracked by `last_chunk_length`.
//!
//! Positions are addressed using `(ChunkID, RowID)` coordinates rather than a
//! single linear index.
//!
//! # Core operations
//!
//! - **Append**: `push` writes into the last chunk, allocating a new chunk if the
//!   previous one is full.
//! - **Remove**: `swap_remove` deletes an element in `O(1)` by moving the last
//!   element into the removed slot (unless the removed slot is already last).
//! - **Transfer**: `push_from` moves a value from one attribute into another,
//!   performing swap-remove in the source when necessary.
//!
//! These operations are constant-time and preserve dense packing, but they do
//! **not** preserve element order.
//!
//! # Type erasure
//!
//! The [`TypeErasedAttribute`] trait allows working with attributes stored behind
//! trait objects (`Box<dyn TypeErasedAttribute>`). It provides:
//!
//! - the element [`TypeId`] and human-readable element type name,
//! - downcasting hooks via `as_any` / `as_any_mut`,
//! - typed chunk views (`chunk_slice` / `chunk_slice_mut`) guarded by type checks,
//! - mutation APIs that mirror the typed operations (`push_dyn`, `swap_remove_dyn`,
//!   `push_from_dyn`).
//!
//! Typed chunk slice access succeeds only when the requested type matches the
//! attribute’s real element type; otherwise it returns `None`.
//!
//! # Rollback feature
//!
//! When compiled with the `rollback` feature, mutating operations produce
//! rollback actions that describe how to restore the prior state.
//!
//! - `push` / `push_dyn` produce actions that remove the inserted value and restore
//!   metadata/chunk allocation.
//! - `swap_remove` / `swap_remove_dyn` produce actions that restore the removed value
//!   and any displaced last element.
//! - `push_from` / `push_from_dyn` produce actions that undo both the destination
//!   insertion and the source removal/displacement.
//!
//! Rollback actions are **one-shot** state deltas and must only be applied to the
//! attribute(s) that produced them, exactly once.
//!
//! # Safety and invariants
//!
//! This module uses `MaybeUninit<T>` and raw pointer reads/writes internally to avoid
//! unnecessary initialization. Soundness relies on maintaining these invariants:
//!
//! - `length` equals the total number of initialized elements stored.
//! - All chunks except the last are fully initialized (`CHUNK_CAP` elements).
//! - Only `0..last_chunk_length` in the last chunk are initialized.
//! - No method exposes references to uninitialized memory.
//!
//! The `cast_slice` / `cast_slice_mut` helpers are `unsafe` because they interpret raw
//! bytes as typed slices; callers must ensure alignment, length, initialization, and
//! aliasing requirements are satisfied.
//!
//! # Intended usage
//!
//! Use this module when you need:
//! - fast, dense, chunked storage for a single component/column type,
//! - type-erased management of heterogeneous component stores,
//! - chunk-level access for serialization or bulk processing,
//! - optional rollback/undo semantics for structural mutations.

use std::{
    ptr,
    array,
    slice,
    any::{Any, TypeId, type_name},
    mem::MaybeUninit,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
    convert::TryInto
};

use crate::engine::types::{
    ChunkID, 
    RowID, 
    CHUNK_CAP
};

use crate::engine::error::{
    PositionOutOfBoundsError, 
    TypeMismatchError, 
    AttributeError
};


/// Describes how to fully undo a mutating operation performed on an
/// `Attribute<T>` container.
///
/// Each mutating method (`push`, `swap_remove`, and `push_from`)
/// produces a corresponding `RollbackAction<T>` that captures the data
/// required to restore the attribute to its prior state.
///
/// A rollback action stores:
/// - All element values that were moved or removed,
/// - All positions involved in the mutation,
/// - Metadata needed to re-expand chunks if necessary,
/// - Complete information to restore source and destination attributes during
///   cross-attribute operations.
///
/// Rollback actions are *one-shot* state deltas: applying them exactly restores
/// the data structure to its previous state, but applying them more than once is
/// invalid.
///
/// All variants assume that the attribute’s underlying invariants were maintained
/// at the time the action was created.

#[cfg(feature = "rollback")]
enum RollbackAction<T> {

    /// Reverts a `swap_remove` operation.
    ///
    /// A swap-remove behaves as follows:
    /// - The element at `removed_position` is extracted,
    /// - If it was not the final element, the last element in the attribute is
    ///   moved into its position,
    /// - The attribute length is reduced, and a chunk may be popped.
    ///
    /// This rollback action contains all data needed to reverse that:
    ///
    /// - `removed_position` — The position of the element originally removed.
    /// - `removed_value` — The value that was removed from that position.
    /// - `moved_from` — If `Some((c, r))`, the value originally at `(c, r)` was
    ///   moved into `removed_position`. Otherwise, the removed element was already the
    ///   last element and no move occurred.
    /// - `moved_value` — The value that was moved from `(c, r)` into
    ///   `removed_position`. This is `None` if no move occurred.
    ///
    /// Applying this rollback:
    /// 1. Restores `removed_value` to `removed_position`,
    /// 2. Restores `moved_value` back to its original position,
    /// 3. Re-increases length and reconstructs chunks if needed.
    
    SwapRemove {
        previous_length: usize,
        previous_last_chunk_length: usize,
        previous_chunk_count: usize,        
        removed_position: (ChunkID, RowID),
        removed_value: T,
        moved_from: Option<(ChunkID, RowID)>,
        moved_value: Option<T>,
    },

    /// Reverts a `push` or `push_dyn` operation.
    ///
    /// A push always appends a new element at the end of the attribute,
    /// potentially growing a chunk or adding a new chunk.
    ///
    /// The rollback action simply stores:
    /// - `inserted_position` — Where the pushed value was inserted.
    ///
    /// Applying this rollback removes the inserted value (using swap-remove
    /// internally), restores `length` and `last_chunk_length`, and pops any chunk
    /// that was newly created during the push.

    Push {
        previous_length: usize,
        previous_last_chunk_length: usize,
        previous_chunk_count: usize,    
        inserted_position: (ChunkID, RowID),
    },

    /// Reverts a `push_from` operation, which transfers a value from one
    /// attribute to another.
    ///
    /// A `push_from` operation:
    /// - Removes a value from the source attribute,
    /// - Inserts it into the destination attribute,
    /// - Performs swap-remove in the source if the removed position was not the
    ///   last element.
    ///
    /// This rollback action contains everything needed to undo *both* sides of
    /// the transfer:
    ///
    /// - `destination_position` — Where the transferred value was inserted in the
    ///   destination.
    /// - `source_position` — The original location of the value in the source.
    /// - `value_moved` — The value that was transferred from source to destination.
    /// - `last_displaced_source_information` — If the source performed a
    ///   swap-remove, this is:
    ///     `Some(((c, r), v))` where `(c, r)` is the position of the source’s
    ///     last element before the operation, and `v` is its value.
    ///     Otherwise `None` if no displacement occurred.
    ///
    /// Applying this rollback:
    /// 1. Removes the inserted value from the destination,
    /// 2. Restores `value_moved` to its original `source_position`,
    /// 3. Restores the displaced last element in the source (if any),
    /// 4. Repairs metadata in both attributes.

    PushFrom {
        destination_previous_length: usize,
        destination_previous_last_chunk_length: usize,
        destination_previous_chunk_count: usize,
        source_previous_length: usize,
        source_previous_last_chunk_length: usize,
        source_previous_chunk_count: usize,    
        destination_position: (ChunkID, RowID),
        source_position: (ChunkID, RowID),
        value_moved: T,
        last_displaced_source_information: Option<( (ChunkID, RowID), T )>,
    },
}

/// # Rollback Support
///
/// Rollback functionality is conditionally compiled:
/// - **With `rollback` feature**: Methods return `RollbackAction` for undo operations
/// - **Without `rollback` feature**: Methods return simplified results, no undo capability
///
/// When the `rollback` feature is disabled, the following simplifications occur:
/// - `push()` returns only the insertion position
/// - `swap_remove()` returns only the moved-from position (if any)
/// - `push_from()` returns only destination and moved-from positions
/// - `rollback()` method is not available

#[cfg(feature = "rollback")]
#[inline]
fn new_uninit_chunk<T>() -> Box<[MaybeUninit<T>; CHUNK_CAP]> {
    Box::new(array::from_fn(|_| MaybeUninit::uninit()))
}

/// A type-erased interface for chunked attribute storage.
///
/// `TypeErasedAttribute` provides a dynamically-typed, reflection-based API for
/// interacting with attribute containers without knowing their underlying element
/// type `T`.
///
/// Attributes implementing this trait must internally store elements in fixed-size
/// chunks (`CHUNK_CAP` rows per chunk), and must maintain the following invariants:
///
/// - `length()` returns the total number of initialized elements.
/// - `chunk_count()` returns the number of allocated chunks.
/// - `last_chunk_length()` returns the number of initialized elements in the final
///   chunk and must satisfy:  
///   `0 < last_chunk_length <= CHUNK_CAP`, unless `length() == 0`.
/// - All indices below `length()` correspond to initialized, valid elements.
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
/// ```ignore
/// if let Some(attr) = erased.as_any().downcast_ref::<Attribute<f32>>() {
///     println!("attribute stores {} f32 values", attr.length());
/// }
/// ```
///
/// # Element transfer and mutation
///
/// The `swap_remove`, `push_dyn`, and `push_from` methods provide a type-erased
/// mutation interface compatible with the typed equivalents on concrete attribute
/// implementations.
///
/// #[cfg(feature = "rollback")]
/// **With rollback feature enabled:**
/// - `swap_remove_dyn` removes an element and returns a rollback action.
/// - `push_dyn` inserts a dynamically-typed element and returns a rollback action.
/// - `push_from_dyn` moves an element between attributes and returns a rollback action.
/// - `rollback_dyn` applies a previously generated rollback action.
///
/// #[cfg(not(feature = "rollback"))]
/// **With rollback feature disabled:**
/// - `swap_remove_dyn` removes an element without rollback capability.
/// - `push_dyn` inserts a dynamically-typed element without rollback capability.
/// - `push_from_dyn` moves an element between attributes without rollback capability.
/// - `rollback_dyn` is a no-op that always succeeds.
///
/// # Intended usage
/// This trait is suitable when:
/// - components or attributes must be managed in a type-erased container,
/// - serializers/deserializers need read/write access to raw storage,
/// - low-level systems operate on contiguous chunks of memory for performance.
///
/// Implementing this trait correctly enables efficient, safe, and flexible
/// manipulation of heterogeneous attribute storage.

pub trait TypeErasedAttribute: Any + Send + Sync {
    /// Returns the number of allocated chunks in this attribute.
    fn chunk_count(&self) -> usize;
    
    /// Returns the total number of initialized elements stored.
    fn length(&self) -> usize;

    /// Returns the number of initialized elements in the final chunk.
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
    fn chunk_bytes(
        &self,
        chunk_id: ChunkID,
        length: usize,
    ) -> Option<(*const u8, usize)>;

    /// Returns a mutable raw byte pointer and size for a chunk slice.
    fn chunk_bytes_mut(
        &mut self,
        chunk_id: ChunkID,
        length: usize,
    ) -> Option<(*mut u8, usize)>;

    /// Returns a typed immutable slice into a chunk if the requested type matches.
    fn chunk_slice<T: 'static>(
        &self,
        chunk_id: ChunkID,
        length: usize,
    ) -> Option<&[T]>
    where
        Self: Sized;

    /// Returns a typed mutable slice into a chunk if the requested type matches.
    fn chunk_slice_mut<T: 'static>(
        &mut self,
        chunk_id: ChunkID,
        length: usize,
    ) -> Option<&mut [T]>
    where
        Self: Sized;

    /// Removes an element using swap-remove through a type-erased interface.    
    #[cfg(feature = "rollback")]
    fn swap_remove_dyn(
        &mut self, 
        chunk: ChunkID, 
        row: RowID
    ) -> Result<(Option<(ChunkID, RowID)>, Box<dyn Any>), AttributeError>;
    
    /// Removes an element using swap-remove through a type-erased interface.
    #[cfg(not(feature = "rollback"))]
    fn swap_remove_dyn(
        &mut self, 
        chunk: ChunkID, 
        row: RowID
    ) -> Result<Option<(ChunkID, RowID)>, AttributeError>;
    
    /// Inserts a dynamically-typed value into the attribute.
    #[cfg(feature = "rollback")]
    fn push_dyn(
        &mut self, 
        value: Box<dyn Any>
    ) -> Result<((ChunkID, RowID), Box<dyn Any>), AttributeError>;
    
    /// Inserts a dynamically-typed value into the attribute.
    #[cfg(not(feature = "rollback"))]
    fn push_dyn(
        &mut self, 
        value: Box<dyn Any>
    ) -> Result<(ChunkID, RowID), AttributeError>;
    
    #[cfg(feature = "rollback")]
    fn push_from_dyn(
        &mut self, 
        source: &mut dyn TypeErasedAttribute, 
        source_chunk: ChunkID, 
        source_row: RowID
    ) -> Result<((ChunkID, RowID), Option<(ChunkID, RowID)>, Box<dyn Any>), AttributeError>;

    /// Transfers an element from another attribute into this one.
    #[cfg(not(feature = "rollback"))]
    fn push_from_dyn(
        &mut self, 
        source: &mut dyn TypeErasedAttribute, 
        source_chunk: ChunkID, 
        source_row: RowID
    ) -> Result<((ChunkID, RowID), Option<(ChunkID, RowID)>), AttributeError>;

    /// Transfers an element from another attribute into this one.
    #[cfg(feature = "rollback")]
    fn rollback_dyn(
        &mut self, 
        action: Box<dyn Any>, 
        source: Option<&mut dyn TypeErasedAttribute>
    ) -> Result<(), AttributeError>;
}

/// A chunked, contiguous, column-oriented storage container for elements of type `T`.
///
/// `Attribute<T>` stores elements in fixed-size chunks of capacity `CHUNK_CAP`,
/// each chunk represented as an array of `MaybeUninit<T>`. All elements are stored
/// densely, without gaps, and indexing is performed using `(ChunkID, RowID)`
/// coordinates.
///
/// This structure is designed for high-performance ECS or columnar data storage,
/// where fast iteration, predictable layout, and constant-time append/delete
/// operations are required.
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
/// Chunk 2: [ T, T, —, — ]   (partially full, last_chunk_length = 2)
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
///   0 ≤ last_chunk_length ≤ CHUNK_CAP
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
/// - `chunks` — The chunked backing storage.  
/// - `last_chunk_length` — The number of initialized elements in the final chunk.  
/// - `length` — Total number of initialized elements across all chunks.
///

pub struct Attribute<T> {
    chunks: Vec<Box<[MaybeUninit<T>; CHUNK_CAP]>>,
    last_chunk_length: usize, // number of initialized elements in the last chunk
    length: usize
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
            self.chunks.push(Box::new(array::from_fn(|_| MaybeUninit::<T>::uninit())));
            self.last_chunk_length = 0;
        }
    }

    /// Converts a linear index into `(chunk, row)` coordinates.
    ///
    /// The mapping is:
    ///
    /// ```text
    /// chunk = index / CHUNK_CAP
    /// row   = index % CHUNK_CAP
    /// ```
    ///
    /// This is a pure arithmetic helper and performs no bounds checking.

    #[cfg(feature = "rollback")]
    #[inline]
    fn get_chunk_position(&self, index: usize) -> (ChunkID, RowID) {
        let chunk = (index / CHUNK_CAP) as ChunkID;
        let row = (index % CHUNK_CAP) as RowID;
        (chunk, row)
    }

    /// Returns a mutable reference to the `MaybeUninit<T>` slot at `(chunk,row)`
    /// without performing any bounds checks.
    ///
    /// # Safety
    /// Equivalent to indexing raw memory:
    /// - `chunk < self.chunk_count()` must hold,
    /// - `row < CHUNK_CAP` must hold,
    /// - the caller must ensure that the usage of the returned slot obeys
    ///   initialization and aliasing rules.
    ///
    /// Debug asserts fire in debug mode, but no runtime checks exist in release.

    #[inline]
    fn get_slot_unchecked(&mut self, chunk: usize, row: usize) -> &mut MaybeUninit<T> {
        debug_assert!(chunk < self.chunk_count());
        debug_assert!(row < CHUNK_CAP);
        &mut self.chunks[chunk][row]
    }

    /// Converts `(usize, usize)` indices into `(ChunkID, RowID)` and checks for
    /// integer overflow.
    ///
    /// Returns an error if either index cannot fit into the narrower integer type.

    #[cfg(feature = "rollback")]
    #[inline]
    fn to_ids(chunk: usize, row: usize) -> Result<(ChunkID, RowID), AttributeError> {
        let chunk: ChunkID = chunk.try_into().map_err(|_| AttributeError::IndexOverflow("ChunkID"))?;
        let row: RowID     = row.try_into().map_err(|_| AttributeError::IndexOverflow("RowID"))?;
        Ok((chunk, row))
    }  

    /// Returns `true` if `(chunk,row)` refers to an initialized element.
    ///
    /// For all chunks **except the last**, all rows `< CHUNK_CAP` are valid.
    /// For the last chunk, only rows `< last_chunk_length` are initialized.

    #[inline]
    fn valid_position(&self, chunk: ChunkID, row: RowID) -> bool {
        let chunk = chunk as usize;
        let row = row as usize;
        if chunk >= self.chunk_count() { return false; }
        if chunk + 1 == self.chunk_count() {
            row < self.last_chunk_length
        } else {
            row < CHUNK_CAP
        }
    }

    /// Returns the `(chunk,row)` of the last initialized element, or `None`
    /// if the attribute is empty.
    ///
    /// Equivalent to decomposing `length - 1` into chunk and row.

    #[cfg(feature = "rollback")]    
    #[inline]
    fn last_filled_position(&self) -> Option<(usize, usize)> {
        if self.length == 0 { return None; }
        let index = self.length - 1;
        Some((index / CHUNK_CAP, index % CHUNK_CAP))
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

    /// Returns a shared reference to an initialized element at `(chunk, row)`,
    /// or `None` if the position is invalid.
    ///
    /// # Safety
    /// Uses `assume_init_ref()`, relying on `valid_position` to ensure the slot
    /// has been written to previously.

    pub fn get(&self, chunk: ChunkID, row: RowID) -> Option<&T> {
        if !self.valid_position(chunk, row) { return None; }
        Some(unsafe { self.chunks[chunk as usize][row as usize].assume_init_ref() })
    }

    /// Returns a mutable reference to an initialized element at `(chunk, row)`,
    /// or `None` if the position is invalid.
    ///
    /// # Safety
    /// Uses `assume_init_mut()`; caller must ensure they do not create aliasing
    /// references elsewhere.

    pub fn get_mut(&mut self, chunk: ChunkID, row: RowID) -> Option<&mut T> {
        if !self.valid_position(chunk, row) { return None; }
        Some(unsafe { self.chunks[chunk as usize][row as usize].assume_init_mut() })
    }

    /// Returns an iterator over all initialized elements in the attribute.
    ///
    /// The iterator visits all chunks in order and yields references to elements
    /// in the order they were inserted. Only initialized elements are visited;
    /// the uninitialized tail of the final chunk is skipped.
    ///
    /// # Safety
    /// Each element is accessed using `assume_init_ref()`, guarded by known
    /// initialization boundaries (`last_chunk_length`).

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.chunks.iter().enumerate().flat_map(move |(i, chunk)| {
            let initialized = if i == self.chunks.len() - 1 {
                self.last_chunk_length
            } else {
                CHUNK_CAP
            };
            chunk[..initialized]
                .iter()
                .map(|mu| unsafe { mu.assume_init_ref() })
        })
    }

    /// Appends a new element to the attribute.
    ///
    /// Automatically creates a new chunk if the last chunk is full.
    ///
    /// # Returns
    /// #[cfg(feature = "rollback")]
    /// Returns both the assigned `(ChunkID, RowID)` location and a
    /// [`RollbackAction::Push`] describing how to undo the insertion.
    ///
    /// #[cfg(not(feature = "rollback"))]
    /// Returns only the assigned `(ChunkID, RowID)` location.
    ///
    /// # Errors
    /// Returns [`AttributeError::IndexOverflow`] if either the chunk index or row
    /// index cannot be represented in their respective integer types.
    ///
    /// # Safety
    /// Writes into a `MaybeUninit<T>` slot using raw pointer write. The slot is
    /// guaranteed valid due to `ensure_last_chunk()`.

    #[cfg(feature = "rollback")]
    pub fn push(
        &mut self, 
        value: T
    ) -> Result<((ChunkID, RowID), RollbackAction<T>), AttributeError> {
        let previous_length = self.length;
        let previous_last_chunk_length = self.last_chunk_length;
        let previous_chunk_count = self.chunks.len();

        self.ensure_last_chunk();
        let chunk_index = self.chunks.len() - 1;
        let row_index = self.last_chunk_length;

        let chunk_id: ChunkID = chunk_index.try_into()
            .map_err(|_| AttributeError::IndexOverflow("ChunkID"))?;
        let row_id: RowID = row_index.try_into()
            .map_err(|_| AttributeError::IndexOverflow("RowID"))?;

        unsafe {
            self
            .get_slot_unchecked(chunk_index, row_index)
            .as_mut_ptr()
            .write(value);
        }
        self.last_chunk_length += 1;
        self.length += 1;

        Ok(
            (
                (chunk_id, row_id),
                RollbackAction::Push {
                    previous_length,
                    previous_last_chunk_length,
                    previous_chunk_count,                
                    inserted_position: (chunk_id, row_id)
                }
            )
        )
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
    ///
    /// # Safety
    /// Internally writes into a `MaybeUninit<T>` slot. Correctness relies on
    /// `ensure_last_chunk` guaranteeing that the target slot is valid.

    #[cfg(not(feature = "rollback"))]
    pub fn push(
        &mut self, 
        value: T
    ) -> Result<(ChunkID, RowID), AttributeError> {
        self.ensure_last_chunk();
        let chunk_index = self.chunks.len() - 1;
        let row_index = self.last_chunk_length;

        let chunk_id: ChunkID = chunk_index.try_into()
            .map_err(|_| AttributeError::IndexOverflow("ChunkID"))?;
        let row_id: RowID = row_index.try_into()
            .map_err(|_| AttributeError::IndexOverflow("RowID"))?;

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
    /// #[cfg(feature = "rollback")]
    /// Returns both the structural effects of the removal and a rollback action
    /// capable of fully restoring the previous state.
    ///
    /// #[cfg(not(feature = "rollback"))]
    /// Returns only information about the structural effects of the removal.
    ///
    /// # Errors
    /// Returns a [`PositionOutOfBounds`] error if `(chunk, row)` does not identify
    /// a valid, initialized element.
    ///
    /// # Safety
    /// Internally uses `unsafe` memory operations (`ptr::read`, `ptr::write`).
    /// The attribute guarantees that:
    /// - all slots up to `length` are initialized,  
    /// - moved-from positions are valid,  
    /// - chunk boundaries match `last_chunk_length`.
    ///
    /// # Complexity
    /// Constant time: `O(1)`.

    #[cfg(feature = "rollback")]
    pub fn swap_remove(
        &mut self, 
        chunk: ChunkID, 
        row: RowID
    ) -> Result<(Option<(ChunkID, RowID)>, RollbackAction<T>), AttributeError> {
        let chunk_count = self.chunks.len();
        
        if chunk as usize >= self.chunks.len() {
            return Err(
                AttributeError::Position(
                        PositionOutOfBoundsError {
                        chunk, 
                        row, 
                        chunks: chunk_count,
                        capacity: CHUNK_CAP,
                        last_chunk_length: self.last_chunk_length,
                    }
                )
            );
        }

        let index = chunk as usize * CHUNK_CAP + row as usize;    
        if index >= self.length {
            return Err(
                AttributeError::Position(
                    PositionOutOfBoundsError {
                        chunk,
                        row,
                        chunks: chunk_count,
                        capacity: CHUNK_CAP,
                        last_chunk_length: self.last_chunk_length,
                    }
                )
            );
        }

        let previous_length = self.length;
        let previous_last_chunk_length = self.last_chunk_length;
        let previous_chunk_count = self.chunks.len();

        let last_index = self.length - 1;
        let last_chunk = last_index / CHUNK_CAP;
        let last_row = last_index % CHUNK_CAP;
        
        let is_last = (chunk as usize == last_chunk) && (row as usize == last_row);

        let removed_value = unsafe {
            ptr::read(
                self
                .get_slot_unchecked(chunk as usize, row as usize)
                .as_ptr() as *const T
            )
        };

        let mut moved: Option<((ChunkID, RowID), T)> = None;

        if !is_last {
            let last_value = unsafe {
                ptr::read(
                    self
                    .get_slot_unchecked(last_chunk, last_row)
                    .as_ptr() as *const T
                )
            };

            unsafe {
                ptr::write(
                    self.get_slot_unchecked(chunk as usize, row as usize)
                        .as_mut_ptr(),
                    last_value
                );
            }

            moved = Some((
                (last_chunk as ChunkID, last_row as RowID),
                last_value
            ));
        
            unsafe {
                self.get_slot_unchecked(last_chunk, last_row)
                    .as_mut_ptr()
                    .write(MaybeUninit::uninit());
            }
        } else {
            unsafe {
                self.get_slot_unchecked(chunk as usize, row as usize)
                    .as_mut_ptr()
                    .write(MaybeUninit::uninit());
            }
        }

        self.length -= 1;

        if self.length == 0 {
            self.last_chunk_length = 0;
            self.chunks.clear();
        } else {
            let new_last_chunk = (self.length - 1) / CHUNK_CAP;
            let new_last_row = (self.length - 1) % CHUNK_CAP;

            while self.chunks.len() - 1 > new_last_chunk {
                self.chunks.pop();
            }

            self.last_chunk_length = new_last_row + 1;
        }

        let (moved_from, moved_value) = match moved {
            Some(((c, r), v)) => (Some((c, r)), Some(v)),
            None => (None, None),
        };

        Ok(
            (
                moved_from,
                RollbackAction::SwapRemove {
                    previous_length,
                    previous_last_chunk_length,
                    previous_chunk_count,                  
                    removed_position: (chunk, row),
                    removed_value,
                    moved_from,
                    moved_value,
                }
            )
        )
    }   

    /// Removes an element using a constant-time swap-remove strategy.
    ///
    /// If the removed element is not the last initialized element, the final
    /// element in the attribute is moved into the removed position.
    ///
    /// # Parameters
    /// - `chunk`: Chunk index of the element to remove.
    /// - `row`: Row index within the chunk.
    ///
    /// # Returns
    /// Returns `Some((ChunkID, RowID))` if another element was moved to fill
    /// the removed slot, or `None` if the removed element was already last.
    ///
    /// # Errors
    /// Returns [`AttributeError::Position`] if `(chunk, row)` does not refer
    /// to a valid, initialized element.
    ///
    /// # Complexity
    /// Runs in constant time: `O(1)`.
    ///
    /// # Safety
    /// Uses raw memory reads and writes. Soundness relies on the attribute’s
    /// invariants regarding initialization and chunk boundaries.

    #[cfg(not(feature = "rollback"))]
    pub fn swap_remove(
        &mut self, 
        chunk: ChunkID, 
        row: RowID
    ) -> Result<Option<(ChunkID, RowID)>, AttributeError> {
        let chunk_count = self.chunks.len();
        
        if chunk as usize >= self.chunks.len() {
            return Err(
                AttributeError::Position(
                        PositionOutOfBoundsError {
                        chunk, 
                        row, 
                        chunks: chunk_count,
                        capacity: CHUNK_CAP,
                        last_chunk_length: self.last_chunk_length,
                    }
                )
            );
        }

        let index = chunk as usize * CHUNK_CAP + row as usize;    
        if index >= self.length {
            return Err(
                AttributeError::Position(
                    PositionOutOfBoundsError {
                        chunk,
                        row,
                        chunks: chunk_count,
                        capacity: CHUNK_CAP,
                        last_chunk_length: self.last_chunk_length,
                    }
                )
            );
        }

        let last_index = self.length - 1;
        let last_chunk = last_index / CHUNK_CAP;
        let last_row = last_index % CHUNK_CAP;
        
        let is_last = (chunk as usize == last_chunk) && (row as usize == last_row);
        
        unsafe {
            ptr::read(
                self
                .get_slot_unchecked(chunk as usize, row as usize)
                .as_ptr() as *const T
            );
        }

        let mut moved_from: Option<(ChunkID, RowID)> = None;

        if !is_last {
            let last_value = unsafe {
                ptr::read(
                    self
                    .get_slot_unchecked(last_chunk, last_row)
                    .as_ptr() as *const T
                )
            };

            unsafe {
                ptr::write(
                    self.get_slot_unchecked(chunk as usize, row as usize)
                        .as_mut_ptr(),
                    last_value
                );
            }

            moved_from = Some((
                last_chunk as ChunkID,
                last_row as RowID
            ));
        
            *self.get_slot_unchecked(last_chunk as usize, last_row as usize) = MaybeUninit::uninit();
        } else {
            *self.get_slot_unchecked(last_chunk as usize, last_row as usize) = MaybeUninit::uninit();
        }

        self.length -= 1;

        if self.length == 0 {
            self.last_chunk_length = 0;
            self.chunks.clear();
        } else {
            let new_last_chunk = (self.length - 1) / CHUNK_CAP;
            let new_last_row = (self.length - 1) % CHUNK_CAP;

            while self.chunks.len() - 1 > new_last_chunk {
                self.chunks.pop();
            }

            self.last_chunk_length = new_last_row + 1;
        }

        Ok(moved_from)
    }

    /// Moves a value out of a source attribute and inserts it into `self`.
    ///
    /// This operation:
    ///
    /// 1. Validates that `source` is an attribute of the same concrete type `T`.
    /// 2. Reads (moves) the element from `(source_chunk, source_row)`.
    /// 3. Pushes it into the destination (`self`), producing an insertion position.
    /// 4. Performs a swap-remove in the source attribute, which may move the source's
    ///    last element to fill the removed slot.
    ///
    /// # Parameters
    /// - `source`: A mutable reference to the source attribute.
    /// - `source_chunk`: Chunk index of the element to transfer.
    /// - `source_row`: Row index of the element to transfer.
    ///
    /// # Returns
    /// #[cfg(feature = "rollback")]
    /// Returns the destination insertion position, information about any swap-remove
    /// that occurred in the source, and a rollback action capable of undoing the
    /// entire cross-attribute move.
    ///
    /// #[cfg(not(feature = "rollback"))]
    /// Returns only the destination insertion position and information about any
    /// swap-remove that occurred in the source.
    ///
    /// # Errors
    /// - [`PositionOutOfBounds`] if `(source_chunk, source_row)` is invalid.
    /// - Any error that may arise from [`Attribute::push`] on the destination.
    ///
    /// If inserting into the destination fails, the value is safely written back
    /// into the source before the error is returned.
    ///
    /// # Safety
    /// Uses unsafe memory access for moving values across chunked, uninitialized
    /// storage regions, but the attribute invariant ensures memory validity.
    ///
    /// # Complexity
    /// `O(1)` for both the transfer and the source swap-remove.

    #[cfg(feature = "rollback")]
    pub fn push_from(
        &mut self, 
        source: &mut Attribute<T>, 
        source_chunk: ChunkID, 
        source_row: RowID
    ) -> Result<((ChunkID, RowID), Option<(ChunkID, RowID)>, RollbackAction<T>), AttributeError> {

        let source_chunk_count = source.chunks.len();
        if !source.valid_position(source_chunk, source_row) {
            return Err(
                AttributeError::Position(
                    PositionOutOfBoundsError {
                        chunk: source_chunk,
                        row: source_row,
                        chunks: source_chunk_count,
                        capacity: CHUNK_CAP,
                        last_chunk_length: source.last_chunk_length,
                    }
                )
            );
        }

        let destination_previous_length = self.length;
        let destination_previous_last_chunk_length = self.last_chunk_length;
        let dest_previous_chunk_count = self.chunks.len();

        let source_previous_length = source.length;
        let source_previous_last_chunk_length = source.last_chunk_length;
        let source_previous_chunk_count = source.chunks.len();

        let moved_value = unsafe {
            let value = ptr::read(
                source
                    .get_slot_unchecked(source_chunk as usize, source_row as usize)
                    .as_ptr() as *const T
            );
            source
                .get_slot_unchecked(source_chunk as usize, source_row as usize)
                .as_mut_ptr()
                .write(MaybeUninit::uninit());
            value
        };
        
        let (destination_chunk, destination_row) = match self.push(moved_value) {
            Ok((pos, _)) => pos,
            Err(e) => {
                unsafe {
                    source
                        .get_slot_unchecked(source_chunk as usize, source_row as usize)
                        .as_mut_ptr()
                        .write(moved_value);
                }
                return Err(e);
            }
        };

        let last_index = source.length - 1;
        let last_chunk = last_index / CHUNK_CAP;
        let last_row = last_index % CHUNK_CAP;

        let mut moved_from_source: Option<(ChunkID, RowID)> = None;
        let mut displaced_information: Option<((ChunkID, RowID), T)> = None;

        let source_index = source_chunk as usize * CHUNK_CAP + source_row as usize;
        
        if source_index != last_index {       
            let last_value = unsafe {
                let value = ptr::read(
                    source
                        .get_slot_unchecked(last_chunk, last_row)
                        .as_ptr() as *const T
                );
                source
                    .get_slot_unchecked(last_chunk, last_row)
                    .as_mut_ptr()
                    .write(MaybeUninit::uninit());
                value
            };
            
            moved_from_source = Some((
                last_chunk.try_into().unwrap(),
                last_row.try_into().unwrap(),
            ));

            displaced_information = Some((
                (
                    last_chunk.try_into().unwrap(),
                    last_row.try_into().unwrap(),
                ),
                last_value,
            ));

            if let Some(((_, _), last_value)) = displaced_information {
                unsafe {
                    ptr::write(
                        source
                        .get_slot_unchecked(source_chunk as usize, source_row as usize)
                        .as_mut_ptr(),
                        last_value
                    );
                }
            }

        }

        source.length -= 1;
        
        if source.length == 0 {
            source.chunks.clear();
            source.last_chunk_length = 0;
        } else {
            let new_last_chunk = (source.length - 1) / CHUNK_CAP;
            let new_last_row = (source.length - 1) % CHUNK_CAP;

            while source.chunks.len() - 1 > new_last_chunk {
                source.chunks.pop();
            }

            source.last_chunk_length = new_last_row + 1;
        }

        Ok(
            (
                (destination_chunk, destination_row),
                moved_from_source,
                RollbackAction::PushFrom {
                    destination_previous_length: destination_previous_length,
                    destination_previous_last_chunk_length: destination_previous_last_chunk_length,
                    destination_previous_chunk_count: destination_previous_chunk_count,
                    source_previous_length: source_previous_length,
                    source_previous_last_chunk_length: source_previous_last_chunk_length,
                    source_previous_chunk_count: source_previous_chunk_count,
                    destination_position: (destination_chunk, destination_row),
                    source_position: (source_chunk, source_row),
                    value_moved: moved_value,
                    last_displaced_source_information: displaced_information, 
                }
            )
        )  
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
    /// If insertion into the destination fails, the value is written back
    /// into the source before returning the error.
    ///
    /// # Safety
    /// Uses raw memory operations to move values between attributes. Correctness
    /// depends on both attributes maintaining their internal invariants.

    #[cfg(not(feature = "rollback"))]
    pub fn push_from(
        &mut self, 
        source: &mut Attribute<T>, 
        source_chunk: ChunkID, 
        source_row: RowID
    ) -> Result<((ChunkID, RowID), Option<(ChunkID, RowID)>), AttributeError> {
        let source_chunk_count = source.chunks.len();
        if !source.valid_position(source_chunk, source_row) {
            return Err(
                AttributeError::Position(
                    PositionOutOfBoundsError {
                        chunk: source_chunk,
                        row: source_row,
                        chunks: source_chunk_count,
                        capacity: CHUNK_CAP,
                        last_chunk_length: source.last_chunk_length,
                    }
                )
            );
        }

        let mut moved_value = Some(
            unsafe {
                let value = ptr::read(
                    source
                        .get_slot_unchecked(source_chunk as usize, source_row as usize)
                        .as_ptr() as *const T
                );
                *source.get_slot_unchecked(source_chunk as usize, source_row as usize) = MaybeUninit::uninit();
                value
            }
        );
        
        let value = moved_value.take().unwrap();

        let (destination_chunk, destination_row) = match self.push(value) {
            Ok(pos) => pos,
            Err(e) => {
                let value = moved_value.take().unwrap();
                unsafe {
                    source
                        .get_slot_unchecked(source_chunk as usize, source_row as usize)
                        .as_mut_ptr()
                        .write(value);
                }
                return Err(e);
            }
        };

        let last_index = source.length - 1;
        let last_chunk = last_index / CHUNK_CAP;
        let last_row = last_index % CHUNK_CAP;

        let mut moved_from_source: Option<(ChunkID, RowID)> = None;

        let source_index = source_chunk as usize * CHUNK_CAP + source_row as usize;
        
        if source_index != last_index {       
            let last_value = unsafe {
                let value = ptr::read(
                    source
                        .get_slot_unchecked(last_chunk, last_row)
                        .as_ptr() as *const T
                );
                *source.get_slot_unchecked(source_chunk as usize, source_row as usize) = MaybeUninit::uninit();
                value
            };
            
            moved_from_source = Some((
                last_chunk.try_into().map_err(|_| AttributeError::IndexOverflow("chunk"))?,
                last_row.try_into().map_err(|_| AttributeError::IndexOverflow("row"))?
            ));

            unsafe {
                ptr::write(
                    source
                    .get_slot_unchecked(source_chunk as usize, source_row as usize)
                    .as_mut_ptr(),
                    last_value
                );
            }
        }

        source.length -= 1;
        
        if source.length == 0 {
            source.chunks.clear();
            source.last_chunk_length = 0;
        } else {
            let new_last_chunk = (source.length - 1) / CHUNK_CAP;
            let new_last_row = (source.length - 1) % CHUNK_CAP;

            while source.chunks.len() - 1 > new_last_chunk {
                source.chunks.pop();
            }

            source.last_chunk_length = new_last_row + 1;
        }

        Ok(((destination_chunk, destination_row), moved_from_source))
    }

    #[cfg(feature = "rollback")]
    /// Reverts a previously executed mutation represented by a [`RollbackAction`].  
    ///
    /// This function is the inverse of all mutating operations that produce
    /// rollback actions (`push`, `push_dyn`, `swap_remove`, and `push_from`).  
    ///
    /// Each rollback action fully restores:
    /// - element values,
    /// - chunk and row metadata,
    /// - `length` and `last_chunk_length`,
    /// - chunk allocation and deallocation,
    /// - cross-attribute transfers.
    ///
    /// # Parameters
    /// - `action`: The rollback action describing how to restore state.
    /// - `source`: Required *only* for [`RollbackAction::PushFrom`], where the
    ///   original value came from a different attribute. For all other operations,
    ///   this should be `None`.
    ///
    /// # Behavior
    /// Depending on the variant of `RollbackAction<T>`, rollback performs:
    ///
    /// ## `SwapRemove`
    /// Restores:
    /// - the removed value to its original `(chunk, row)`
    /// - the moved last value (if any)
    /// - metadata (`length`, chunk boundaries)
    ///
    /// ## `Push`
    /// Simply removes the inserted element using a bounded swap-remove.
    ///
    /// ## `PushFrom`
    /// Restores *both* the destination and the original source:
    /// - removes the inserted destination element,
    /// - rebuilds source metadata,
    /// - restores the moved value to its original location,
    /// - restores the displaced final source value if swap-remove occurred.
    ///
    /// # Safety
    /// Uses unsafe writes to restore values into uninitialized/storage slots.  
    /// Rollback is correct only if the provided `RollbackAction` originated from
    /// this exact attribute and reflects its *current* state.
    ///
    /// # Notes
    /// Rollback is idempotent **only if never applied twice**.  
    /// Rollback actions represent *state deltas*, not reversible transactions.

    #[cfg(feature = "rollback")]
    pub fn rollback(&mut self, action: RollbackAction<T>, source: Option<&mut Attribute<T>>) {
        match action {
            RollbackAction::SwapRemove {
                previous_length,
                previous_last_chunk_length,
                previous_chunk_count,
                removed_position,
                removed_value,
                moved_from,
                moved_value,
            } => {
                self.length = previous_length;
                self.last_chunk_length = previous_last_chunk_length;
                self.chunks.resize(previous_chunk_count, new_uninit_chunk::<T>());

                unsafe {
                    let (c, r) = removed_position;
                    self.get_slot_unchecked(c as usize, r as usize)
                        .as_mut_ptr()
                        .write(removed_value);
                }

                if let (Some((mc, mr)), Some(value)) = (moved_from, moved_value) {
                    unsafe {
                        self.get_slot_unchecked(mc as usize, mr as usize)
                            .as_mut_ptr()
                            .write(value);
                    }
                }
            }

            RollbackAction::Push {
                previous_length,
                previous_last_chunk_length,
                previous_chunk_count,
                inserted_position: _,
            } => {
                self.length = previous_length;
                self.last_chunk_length = previous_last_chunk_length;
                self.chunks.resize(previous_chunk_count, new_uninit_chunk::<T>());
            }

            RollbackAction::PushFrom {
                destination_previous_length,
                destination_previous_last_chunk_length,
                destination_previous_chunk_count,
                source_previous_length,
                source_previous_last_chunk_length,
                source_previous_chunk_count,                
                destination_position,
                source_position,
                value_moved,
                last_displaced_source_information,
            } => {
                self.length = destination_previous_length;
                self.last_chunk_length = destination_previous_last_chunk_length;
                self.chunks.resize(destination_previous_chunk_count, new_uninit_chunk::<T>());

                let source = source.expect("source missing in rollback");

                source.length = source_previous_length;
                source.last_chunk_length = source_previous_last_chunk_length;
                source.chunks.resize(source_previous_chunk_count, new_uninit_chunk::<T>());

                unsafe {
                    let (chunk, row) = source_position;
                    source
                    .get_slot_unchecked(chunk as usize, row as usize)
                    .as_mut_ptr()
                    .write(value_moved);
                }

                if let Some(((last_chunk, last_row), last_value)) = last_displaced_source_information {
                    unsafe {
                        source
                        .get_slot_unchecked(last_chunk as usize, last_row as usize)
                        .as_mut_ptr()
                        .write(last_value);
                    }
                }
            }
        }
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
    ///
    /// # Safety
    /// Elements are dropped with `assume_init_drop()`, which requires that only
    /// initialized slots are visited.

    fn drop_all_initialized_elements(&mut self) {
        if self.length == 0 { return; }

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
                unsafe { chunk[i].assume_init_drop(); }
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
        if self.length == 0 { return; }

        self.drop_all_initialized_elements();
        self.chunks.clear();
        self.length = 0;
        self.last_chunk_length = 0;
    }    

}

impl<T: 'static + Send + Sync> TypeErasedAttribute for Attribute<T> {
    fn chunk_count(&self) -> usize { self.chunks.len() } // Returns the number of allocated chunks in this attribute.
    fn length(&self) -> usize { self.length } // Returns the total number of initialized elements stored across all chunks.
    fn last_chunk_length(&self) -> usize { self.last_chunk_length } // Returns the number of initialized elements in the last chunk.

    fn as_any(&self) -> &dyn Any { self } // Returns an immutable `&dyn Any` reference to this attribute.
    fn as_any_mut(&mut self) -> &mut dyn Any { self } // Returns a mutable `&mut dyn Any` reference to this attribute.

    fn element_type_id(&self) -> TypeId {TypeId::of::<T>()} // Returns the `TypeId` of the element type stored by this attribute.
    fn element_type_name(&self) -> &'static str {type_name::<T>()} // Returns the human-readable name of the element type stored in this attribute.

    fn chunk_bytes(
        &self,
        chunk_id: ChunkID,
        valid_length: usize,
    ) -> Option<(*const u8, usize)> {
        let chunk = self.chunks.get(chunk_id as usize)?;
        let len = if chunk_id as usize + 1 == self.chunks.len() {
            self.last_chunk_length
        } else {
            CHUNK_CAP
        }.min(valid_length);

        let ptr = chunk.as_ptr() as *const u8;
        let bytes = len * std::mem::size_of::<T>();
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
        let bytes = len * std::mem::size_of::<T>();
        Some((ptr, bytes))
    }

    /// Returns a typed immutable slice of the elements stored in the specified
    /// chunk, if and only if the requested type `U` matches the attribute's actual
    /// element type `T`.
    ///
    /// This function provides efficient, zero-copy access to a portion of a chunk's
    /// memory. It is intended for bulk iteration, serialization, and other
    /// performance-critical operations.
    ///
    /// # Parameters
    /// - `chunk_id`: The chunk index from which to retrieve elements.
    /// - `valid_length`: The maximum number of elements the caller intends to
    ///   operate on. The returned slice will have length:
    ///
    ///   ```ignore
    ///   min(valid_length, actual_initialized_length_of_chunk)
    ///   ```
    ///
    ///   This allows callers to request a subset of a chunk without needing to know
    ///   its exact populated length.
    ///
    /// # Returns
    /// - `Some(&[U])` if:
    ///   - `U` is the concrete element type stored in this attribute (`U == T`), and  
    ///   - `chunk_id` is a valid chunk index.
    /// - `None` if the type does not match or the chunk index is out of bounds.
    ///
    /// # Type Matching
    /// Since this method is exposed through a type-erased interface, it can only
    /// return a slice if the caller requests exactly the type stored internally.
    /// Type mismatches return `None` instead of attempting any conversion.
    ///
    /// # Safety
    /// Internally, this function performs bounds-checked chunk lookup and creates
    /// a slice from initialized memory. These operations are safe because:
    /// - The attribute guarantees that the first `len` elements of the chunk are
    ///   fully initialized,
    /// - The type check ensures layout compatibility,
    /// - The slice never outlives the attribute.
    ///
    /// # Usage Example
    /// ```ignore
    /// if let Some(values) = attr.chunk_slice::<f32>(chunk_id, 128) {
    ///     for v in values {
    ///         println!("value = {}", v);
    ///     }
    /// }
    /// ```

    fn chunk_slice<U: 'static>(
        &self,
        chunk_id: ChunkID,
        valid_length: usize,
    ) -> Option<&[U]> {
        if TypeId::of::<U>() != TypeId::of::<T>() {
            return None;
        }

        let chunk = self.chunks.get(chunk_id as usize)?;
        let len = if chunk_id as usize + 1 == self.chunks.len() {
            self.last_chunk_length
        } else {
            CHUNK_CAP
        }.min(valid_length);

        Some(unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const U,
                len,
            )
        })
    }

    /// Returns a typed mutable slice of the elements stored in the specified
    /// chunk, if and only if the requested type `U` matches the attribute's actual
    /// element type `T`.
    ///
    /// This function provides efficient, zero-copy access to a portion of a chunk's
    /// memory. It is intended for bulk iteration, serialization, and other
    /// performance-critical operations.
    ///
    /// # Parameters
    /// - `chunk_id`: The chunk index from which to retrieve elements.
    /// - `valid_length`: The maximum number of elements the caller intends to
    ///   operate on. The returned slice will have length:
    ///
    ///   ```ignore
    ///   min(valid_length, actual_initialized_length_of_chunk)
    ///   ```
    ///
    ///   This allows callers to request a subset of a chunk without needing to know
    ///   its exact populated length.
    ///
    /// # Returns
    /// - `Some(&[U])` if:
    ///   - `U` is the concrete element type stored in this attribute (`U == T`), and  
    ///   - `chunk_id` is a valid chunk index.
    /// - `None` if the type does not match or the chunk index is out of bounds.
    ///
    /// # Type Matching
    /// Since this method is exposed through a type-erased interface, it can only
    /// return a slice if the caller requests exactly the type stored internally.
    /// Type mismatches return `None` instead of attempting any conversion.
    ///
    /// # Safety
    /// Internally, this function performs bounds-checked chunk lookup and creates
    /// a slice from initialized memory. These operations are safe because:
    /// - The attribute guarantees that the first `len` elements of the chunk are
    ///   fully initialized,
    /// - The type check ensures layout compatibility,
    /// - The slice never outlives the attribute.
    ///
    /// # Usage Example
    /// ```ignore
    /// if let Some(values) = attr.chunk_slice_mut::<f32>(chunk_id, 128) {
    ///     for v in values {
    ///         println!("value = {}", v);
    ///     }
    /// }
    /// ```

    fn chunk_slice_mut<U: 'static>(
        &mut self,
        chunk_id: ChunkID,
        valid_length: usize,
    ) -> Option<&mut [U]> {
        if TypeId::of::<U>() != TypeId::of::<T>() {
            return None;
        }

        let is_last = chunk_id as usize + 1 == self.chunks.len();
        let len = if is_last { self.last_chunk_length } else { CHUNK_CAP }
            .min(valid_length);

        let chunk = self.chunks.get_mut(chunk_id as usize)?;
        Some(unsafe {
            std::slice::from_raw_parts_mut(
                chunk.as_mut_ptr() as *mut U,
                len,
            )
        })
    }

    /// Attempts to push a dynamically-typed value into the attribute.
    ///
    /// If the provided `Box<dyn Any>` contains a value of type `T`, it is extracted
    /// and forwarded to [`Attribute::push`].
    ///
    /// # Returns
    /// #[cfg(feature = "rollback")]
    /// Returns both the insertion location and a rollback action.
    ///
    /// #[cfg(not(feature = "rollback"))]
    /// Returns only the insertion location.
    ///
    /// # Errors
    /// - [`AttributeError::TypeMismatch`] if the dynamic value has the wrong type.
    ///
    /// # Notes
    /// This method is used when type-erasure is required (e.g., in ECS systems or
    /// heterogeneous attribute stores).
    
    #[cfg(feature = "rollback")]
    fn push_dyn(
        &mut self, 
        value: Box<dyn Any>
    ) -> Result<((ChunkID, RowID), Box<dyn Any>), AttributeError> {
        if let Ok(v) = value.downcast::<T>() {
            let (position, rollback) = self.push(*v)?;
            return Ok((position, Box::new(rollback)));
        }
        let expected = TypeId::of::<T>();
        let actual = value.as_ref().type_id();
        Err(AttributeError::TypeMismatch(TypeMismatchError { expected, actual }))
    }

    #[cfg(not(feature = "rollback"))]
    fn push_dyn(
        &mut self, 
        value: Box<dyn Any>
    ) -> Result<(ChunkID, RowID), AttributeError> {
        let actual = value.as_ref().type_id();
        if let Ok(v) = value.downcast::<T>() {
            let (chunk, row) = self.push(*v)?;
            return Ok((chunk, row));
        }

        Err(AttributeError::TypeMismatch(TypeMismatchError {
            expected: TypeId::of::<T>(),
            actual,
        }))
    }

    /// Attempts to remove an element at the given position using swap-remove.
    ///
    /// If the provided `(chunk, row)` is valid, this method removes the element
    /// and returns information about the structural changes.
    ///
    /// # Returns
    /// #[cfg(feature = "rollback")]
    /// Returns both the moved-from position (if any) and a rollback action that
    /// can restore the removed element and any structural changes.
    ///
    /// #[cfg(not(feature = "rollback"))]
    /// Returns only the moved-from position (if any) without rollback capability.
    ///
    /// # Errors
    /// - [`AttributeError::Position`] if the `(chunk, row)` position is invalid.
    /// - Any error that may arise from the underlying `swap_remove` operation.
    ///
    /// # Notes
    /// This method provides type-erased access to the swap-remove operation,
    /// allowing dynamic attribute manipulation without knowing the concrete type.    
    
    #[cfg(feature = "rollback")]
    fn swap_remove_dyn(
        &mut self,
        chunk: ChunkID,
        row: RowID,
    ) -> Result<(Option<(ChunkID, RowID)>, Box<dyn Any>), AttributeError> {
        let (moved, rollback) = self.swap_remove(chunk, row)?;
        Ok((moved, Box::new(rollback)))
    }

    #[cfg(not(feature = "rollback"))]
    fn swap_remove_dyn(
        &mut self,
        chunk: ChunkID,
        row: RowID,
    ) -> Result<Option<(ChunkID, RowID)>, AttributeError> {
        self.swap_remove(chunk, row)
    }    

    /// Moves an element from a source attribute into this attribute.
    ///
    /// This method transfers an element from the specified position in the source
    /// attribute to this attribute, performing necessary swap-remove operations
    /// in the source if the removed element wasn't the last.
    ///
    /// # Returns
    /// #[cfg(feature = "rollback")]
    /// Returns the destination insertion position, information about any
    /// swap-remove that occurred in the source, and a rollback action that can
    /// undo the entire cross-attribute transfer.
    ///
    /// #[cfg(not(feature = "rollback"))]
    /// Returns only the destination insertion position and information about any
    /// swap-remove that occurred in the source, without rollback capability.
    ///
    /// # Errors
    /// - [`AttributeError::TypeMismatch`] if the source attribute doesn't store
    ///   the same element type as this attribute.
    /// - [`AttributeError::Position`] if the `(source_chunk, source_row)` position
    ///   is invalid in the source attribute.
    /// - Any error that may arise from the underlying `push_from` operation.
    ///
    /// # Notes
    /// This method enables type-erased movement of elements between attributes,
    /// which is useful for entity component migration or bulk transfers in ECS.

    #[cfg(feature = "rollback")]
    fn push_from_dyn(
        &mut self,
        source: &mut dyn TypeErasedAttribute,
        source_chunk: ChunkID,
        source_row: RowID,
    ) -> Result<((ChunkID, RowID), Option<(ChunkID, RowID)>, Box<dyn Any>), AttributeError> {
        let source_attribute = source
            .as_any_mut()
            .downcast_mut::<Attribute<T>>()
            .ok_or_else(|| {
                AttributeError::TypeMismatch(
                    TypeMismatchError {
                        expected: TypeId::of::<T>(),
                        actual: source.element_type_id()
                    }
                )
            })?;

        let (position, moved, rollback) = self.push_from(source_attribute, source_chunk, source_row)?;
        Ok((position, moved, Box::new(rollback)))
    }    

    #[cfg(not(feature = "rollback"))]
    fn push_from_dyn(
        &mut self,
        source: &mut dyn TypeErasedAttribute,
        source_chunk: ChunkID,
        source_row: RowID,
    ) -> Result<((ChunkID, RowID), Option<(ChunkID, RowID)>), AttributeError> {

        // Capture immutable info FIRST
        let actual_type = source.element_type_id();

        // Then take the mutable borrow
        let source_attribute = match source.as_any_mut().downcast_mut::<Attribute<T>>() {
            Some(attr) => attr,
            None => {
                return Err(
                    AttributeError::TypeMismatch(
                        TypeMismatchError {
                            expected: TypeId::of::<T>(),
                            actual: actual_type,
                        }
                    )
                );
            }
        };

        self.push_from(source_attribute, source_chunk, source_row)
    }

    #[cfg(feature = "rollback")]
    /// Applies a previously generated rollback action to restore attribute state.
    ///
    /// This method is the type-erased counterpart to [`Attribute::rollback`],
    /// allowing rollback actions to be applied through the dynamic trait interface.
    ///
    /// # Parameters
    /// - `action`: A boxed rollback action that was previously generated by
    ///   `push_dyn`, `swap_remove_dyn`, or `push_from_dyn`.
    /// - `source`: Required only for `PushFrom` rollback actions, where the
    ///   original value came from a different attribute. For other actions,
    ///   this should be `None`.
    ///
    /// # Returns
    /// Returns `Ok(())` if the rollback was successfully applied, or an error
    /// if the action type doesn't match this attribute's element type.
    ///
    /// # Errors
    /// - [`AttributeError::TypeMismatch`] if the provided action doesn't contain
    ///   a `RollbackAction<T>` matching this attribute's element type `T`.
    ///
    /// # Notes
    /// Rollback actions are one-shot and should only be applied once to the
    /// exact attribute that generated them. Applying a rollback multiple times
    /// or to a different attribute will lead to incorrect state.

    #[cfg(feature = "rollback")]    
    fn rollback_dyn(
        &mut self, 
        action: Box<dyn Any>, 
        source: Option<&mut dyn TypeErasedAttribute>
    ) -> Result<(), AttributeError> {
        if let Ok(concrete_action) = action.downcast::<RollbackAction<T>>() {
            let concrete_source = source.and_then(|s| s.as_any_mut().downcast_mut::<Attribute<T>>());
            self.rollback(*concrete_action, concrete_source);
            Ok(())
        } else {
            Err(
                AttributeError::TypeMismatch(
                    TypeMismatchError {
                        expected: TypeId::of::<T>(),
                        actual: action.as_ref().type_id()
                    }
                )
            )
        }
    }   
}

impl<T> Default for Attribute<T> {
    fn default() -> Self {
        Self { chunks: Vec::new(), last_chunk_length: 0, length: 0 }
    }
}

impl<T> Drop for Attribute<T> {
    fn drop(&mut self) {
        self.drop_all_initialized_elements();
    }
}

/// A thread-safe wrapper around type-erased attribute.
#[derive(Clone)]
pub struct LockedAttribute {
    inner: Arc<RwLock<Box<dyn TypeErasedAttribute>>>,
}

impl LockedAttribute {
    /// Creates a new `LockedAttribute` wrapping the given type-erased attribute.
    pub fn new(attribute: Box<dyn TypeErasedAttribute>) -> Self {
        Self { inner: Arc::new(RwLock::new(attribute)) }
    }

    /// Returns a read guard to the inner attribute.
    #[inline]
    pub fn read(
        &self,
    ) -> Result<RwLockReadGuard<'_, Box<dyn TypeErasedAttribute>>, AttributeError> {
        self.inner
            .read()
            .map_err(|_| AttributeError::InternalInvariant("LockedAttribute read lock poisoned"))
    }

    /// Returns a write guard to the inner attribute.
    #[inline]
    pub fn write(
        &self,
    ) -> Result<RwLockWriteGuard<'_, Box<dyn TypeErasedAttribute>>, AttributeError> {
        self.inner
            .write()
            .map_err(|_| AttributeError::InternalInvariant("LockedAttribute write lock poisoned"))
    }

    /// Returns a clone of the internal `Arc<RwLock<Box<dyn TypeErasedAttribute>>>`.
    #[inline]
    pub fn arc(&self) -> Arc<RwLock<Box<dyn TypeErasedAttribute>>> {
        self.inner.clone()
    }

    /// Consumes the `LockedAttribute`, returning the inner attribute.
    pub fn into_inner(self) -> Result<Box<dyn TypeErasedAttribute>, AttributeError> {
        match Arc::try_unwrap(self.inner) {
            Ok(lock) => lock
                .into_inner()
                .map_err(|_| AttributeError::InternalInvariant("LockedAttribute poisoned during unwrap")),
            Err(_) => Err(AttributeError::InternalInvariant(
                "LockedAttribute still shared",
            )),
        }
    }   
}

/// Interprets a raw byte slice as a typed slice.
///
/// # Safety
/// - `ptr` must be properly aligned for `T`
/// - `bytes` must be a multiple of `size_of::<T>()`
/// - the memory region must contain fully initialized `T` values
/// - the returned slice must not outlive the backing storage

#[inline]
pub unsafe fn cast_slice<'a, T>(ptr: *const u8, bytes: usize) -> &'a [T] {
    let len = bytes / std::mem::size_of::<T>();
    unsafe{slice::from_raw_parts(ptr as *const T, len)}
}

/// Interprets a mutable raw byte slice as a mutable typed slice.
///
/// # Safety
/// - `ptr` must be properly aligned for `T`
/// - `bytes` must be a multiple of `size_of::<T>()`
/// - the memory region must contain fully initialized `T` values
/// - no aliasing mutable references may exist

#[inline]
pub unsafe fn cast_slice_mut<'a, T>(ptr: *mut u8, bytes: usize) -> &'a mut [T] {
    let len = bytes / std::mem::size_of::<T>();
    unsafe{slice::from_raw_parts_mut(ptr as *mut T, len)}
}
