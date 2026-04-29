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
//! attribute's real element type; otherwise it returns `None`.
//!
//! # Safety and invariants
//!
//! This module uses `MaybeUninit<T>` and raw pointer reads/writes internally to avoid
//! unnecessary initialization. Soundness relies on maintaining these invariants:
//!
//! - `length` equals the total number of initialized elements stored.
//! - All chunks except the last are fully initialized (`CHUNK_CAP` elements).
//! - Only `0 to last_chunk_length` in the last chunk are initialized.
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
//! - chunk-level access for serialization or bulk processing.

mod attribute;
mod locked_attribute;
mod slice;
mod tests;
mod type_erased_attribute;

use crate::engine::types::{ChunkID, RowID};

pub(crate) type StoragePosition = (ChunkID, RowID);
pub(crate) type MovedStoragePosition = Option<StoragePosition>;
pub(crate) type PushFromOutcome = (StoragePosition, MovedStoragePosition);
pub(crate) type TakeSwapRemoveOutcome = (Box<dyn std::any::Any>, MovedStoragePosition);

pub use attribute::Attribute;
pub use locked_attribute::LockedAttribute;
pub use slice::{cast_slice, cast_slice_mut};
pub use type_erased_attribute::TypeErasedAttribute;
