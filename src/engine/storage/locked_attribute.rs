//! Thread-safe shared ownership of a type-erased attribute column.
//!
//! This module provides [`LockedAttribute`], a thin wrapper that combines
//! `Arc` and `RwLock` to allow multiple owners to safely share a single
//! [`TypeErasedAttribute`] across threads - typically the component columns
//! stored inside an [`Archetype`].
//!
//! # Design
//!
//! [`LockedAttribute`] wraps a `Box<dyn TypeErasedAttribute>` in an
//! `Arc<RwLock<...>>`. Cloning a [`LockedAttribute`] is cheap: it increments
//! the reference count rather than copying the underlying storage. Multiple
//! clones all refer to the same data, guarded by the same lock.
//!
//! Read access is obtained via [`LockedAttribute::read`] and write access via
//! [`LockedAttribute::write`]. Both return the standard library's
//! `RwLockReadGuard` / `RwLockWriteGuard` wrapped in a `Result`, converting
//! a poisoned lock into an [`AttributeError::InternalInvariant`] rather than
//! propagating a panic.
//!
//! # Ownership and unwrapping
//!
//! When a [`LockedAttribute`] is the sole remaining owner of its data (i.e.
//! the internal `Arc` reference count is 1), [`LockedAttribute::into_inner`]
//! can recover the underlying `Box<dyn TypeErasedAttribute>`. If other clones
//! still exist the call fails with [`AttributeError::InternalInvariant`].
//!
//! [`LockedAttribute::arc`] exposes the raw `Arc` for callers that need to
//! store or pass the lock handle directly.
//!
//! # Errors
//!
//! Lock-poisoning caused by a thread panicking while holding the write lock is
//! treated as an unrecoverable internal invariant violation and surfaces as
//! [`AttributeError::InternalInvariant`].

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::engine::error::AttributeError;
use crate::engine::error::AttributeInvariantViolation;
use crate::engine::storage::type_erased_attribute::TypeErasedAttribute;

type AttributeReadGuard<'a> = RwLockReadGuard<'a, Box<dyn TypeErasedAttribute>>;
type AttributeWriteGuard<'a> = RwLockWriteGuard<'a, Box<dyn TypeErasedAttribute>>;
type TryAttributeReadError<'a> = std::sync::TryLockError<AttributeReadGuard<'a>>;

/// A thread-safe wrapper around a type-erased attribute.
#[derive(Clone)]
pub struct LockedAttribute {
    inner: Arc<RwLock<Box<dyn TypeErasedAttribute>>>,
}

impl LockedAttribute {
    /// Creates a new `LockedAttribute` wrapping the given type-erased attribute.
    pub fn new(attribute: Box<dyn TypeErasedAttribute>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(attribute)),
        }
    }

    /// Returns a read guard to the inner attribute.
    #[inline]
    pub fn read(&self) -> Result<AttributeReadGuard<'_>, AttributeError> {
        self.inner.read().map_err(|_| {
            AttributeError::InternalInvariant(AttributeInvariantViolation::LockPoisoned)
        })
    }

    /// Returns a write guard to the inner attribute.
    #[inline]
    pub fn write(&self) -> Result<AttributeWriteGuard<'_>, AttributeError> {
        self.inner.write().map_err(|_| {
            AttributeError::InternalInvariant(AttributeInvariantViolation::LockPoisoned)
        })
    }

    /// Returns a clone of the internal `Arc<RwLock<Box<dyn TypeErasedAttribute>>>`.
    #[inline]
    pub fn arc(&self) -> Arc<RwLock<Box<dyn TypeErasedAttribute>>> {
        self.inner.clone()
    }

    /// Consumes the `LockedAttribute`, returning the inner attribute.
    pub fn into_inner(self) -> Result<Box<dyn TypeErasedAttribute>, AttributeError> {
        match Arc::try_unwrap(self.inner) {
            Ok(lock) => lock.into_inner().map_err(|_| {
                AttributeError::InternalInvariant(AttributeInvariantViolation::LockPoisoned)
            }),
            Err(_) => Err(AttributeError::InternalInvariant(
                AttributeInvariantViolation::StillShared,
            )),
        }
    }

    /// Returns a read guard if the lock is not currently write-held.
    ///
    /// Unlike [`read`](Self::read), this method does not block. It returns
    /// `Err(TryLockError::WouldBlock)` immediately if a writer holds the
    /// lock, preventing same-thread deadlocks when called from inside a
    /// `for_each` callback.
    #[inline]
    pub fn try_read(&self) -> Result<AttributeReadGuard<'_>, TryAttributeReadError<'_>> {
        self.inner.try_read()
    }
}
