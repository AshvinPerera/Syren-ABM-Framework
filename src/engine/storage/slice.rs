//! Raw byte slice reinterpretation utilities.
//!
//! This module provides two `unsafe` helper functions — [`cast_slice`] and
//! [`cast_slice_mut`] — for reinterpreting a raw byte pointer and length as a
//! typed slice without copying any data.
//!
//! # Purpose
//!
//! [`Attribute<T>`] exposes chunk data as raw byte pointers via
//! [`TypeErasedAttribute::chunk_bytes`] and [`TypeErasedAttribute::chunk_bytes_mut`].
//! These helpers allow callers such as serializers, GPU upload paths, or
//! bulk-processing systems to reinterpret that raw memory as a concrete `&[T]`
//! or `&mut [T]` in a single, auditable place rather than scattering
//! `slice::from_raw_parts` calls across the codebase.
//!
//! # Safety contract
//!
//! Both functions require the caller to uphold the following:
//!
//! - The pointer must be correctly aligned for `T`.
//! - `bytes` must be an exact multiple of `size_of::<T>()`.
//! - The entire memory region must contain fully initialized, valid `T` values.
//! - The returned slice must not outlive the allocation the pointer was derived from.
//! - For [`cast_slice_mut`]: no other live references — mutable or shared — may
//!   alias the same memory region.
//!
//! Alignment and byte-length preconditions are verified with `assert_eq!` at
//! runtime (in both debug and release builds), turning contract violations into
//! a panic rather than undefined behaviour. Lifetime and initialization
//! correctness remain entirely the caller's responsibility.
//!
//! # Zero-sized types
//!
//! Both functions return an empty slice immediately when `size_of::<T>() == 0`,
//! avoiding a division by zero and matching the semantics of
//! `slice::from_raw_parts` for ZSTs.

use std::slice;
use std::mem::{size_of, align_of};


/// Interprets a raw byte slice as a typed slice.
///
/// # Safety
/// - `pointer` must be properly aligned for `T`.
/// - `bytes` must be a multiple of `size_of::<T>()`.
/// - The memory region must contain fully initialized `T` values.
/// - The returned slice must not outlive the backing storage.

#[inline]
pub unsafe fn cast_slice<'a, T>(pointer: *const u8, bytes: usize) -> &'a [T] {
    let size = size_of::<T>();
    if size == 0 { return &[]; }
    assert_eq!(bytes % size, 0, "bytes not multiple of element size");
    assert_eq!(
        pointer as usize % align_of::<T>(), 0,
        "pointer is not properly aligned for type"
    );
    let len = bytes / size;
    // SAFETY: Caller guarantees that `pointer` points to `len` fully initialized
    // values of type `T`, that the pointer is aligned (verified above), and that
    // the returned slice will not outlive the backing storage.
    unsafe { slice::from_raw_parts(pointer as *const T, len) }
}

/// Interprets a mutable raw byte slice as a mutable typed slice.
///
/// # Safety
/// - `pointer` must be properly aligned for `T`.
/// - `bytes` must be a multiple of `size_of::<T>()`.
/// - The memory region must contain fully initialized `T` values.
/// - No aliasing mutable references may exist.

#[inline]
pub unsafe fn cast_slice_mut<'a, T>(pointer: *mut u8, bytes: usize) -> &'a mut [T] {
    let size = size_of::<T>();
    if size == 0 { return &mut []; }
    assert_eq!(bytes % size, 0, "bytes not multiple of element size");
    assert_eq!(
        pointer as usize % align_of::<T>(), 0,
        "pointer is not properly aligned for type"
    );
    let len = bytes / size;
    // SAFETY: Caller guarantees that `pointer` points to `len` fully initialized
    // values of type `T`, that the pointer is aligned (verified above), that no
    // aliasing references exist, and that the returned slice will not outlive the
    // backing storage.
    unsafe { slice::from_raw_parts_mut(pointer as *mut T, len) }
}
