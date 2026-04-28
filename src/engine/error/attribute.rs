//! Aggregate error type for component column (attribute) storage operations.
//!
//! This module defines [`AttributeError`], a unified error enum that surfaces
//! precise, low-level failures arising from pushing or writing component data
//! into ECS attribute storage. It consolidates three categories of failure:
//!
//! | Variant | Source |
//! |---|---|
//! | [`AttributeError::Position`] | `(ChunkID, RowID)` address out of bounds |
//! | [`AttributeError::TypeMismatch`] | Runtime type mismatch against column's declared type |
//! | [`AttributeError::IndexOverflow`] | Arithmetic overflow building a storage index |
//! | [`AttributeError::InternalInvariant`] | Violated internal storage invariant |
//!
//! ## Error propagation
//!
//! `From` conversions are provided for [`PositionOutOfBoundsError`] and
//! [`TypeMismatchError`], so call sites can propagate with `?` without manual
//! wrapping.
//!
//! ## Display vs Debug
//!
//! `Display` produces concise, log-friendly messages. `Debug` exposes full
//! variant structure and is preferred for diagnostics and test output.

use std::fmt;

use super::primitives::{PositionOutOfBoundsError, TypeMismatchError};

/// Identifies which internal storage invariant was violated.
///
/// Each variant corresponds to a specific consistency guarantee that
/// `Attribute<T>` and `LockedAttribute` rely on for safe operation. These
/// violations indicate a bug in the storage implementation rather than
/// recoverable user error.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeInvariantViolation {
    /// A chunk index exceeded the number of allocated chunks.
    ChunkIndexOutOfRange,
    /// The cached last-chunk length does not match the actual chunk contents.
    LastChunkLengthInconsistent,
    /// The total stored length disagrees with the sum of chunk lengths.
    LengthMismatch,
    /// An operation that requires at least one chunk found the chunk vector empty.
    EmptyChunkVec,
    /// A swap-remove was attempted on an empty attribute.
    SwapRemoveOnEmpty,
    /// A `push_from` operation encountered a type mismatch between attributes.
    PushFromTypeMismatch,
    /// The `RwLock` protecting this attribute was poisoned by a panicking writer.
    LockPoisoned,
    /// The `Arc` wrapping this attribute still has multiple owners and cannot be unwrapped.
    StillShared,
}

impl fmt::Display for AttributeInvariantViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttributeInvariantViolation::ChunkIndexOutOfRange => {
                write!(f, "chunk index out of range")
            }
            AttributeInvariantViolation::LastChunkLengthInconsistent => {
                write!(f, "last chunk length inconsistent with stored data")
            }
            AttributeInvariantViolation::LengthMismatch => {
                write!(f, "total length does not match sum of chunk lengths")
            }
            AttributeInvariantViolation::EmptyChunkVec => {
                write!(
                    f,
                    "chunk vector is empty when at least one chunk is required"
                )
            }
            AttributeInvariantViolation::SwapRemoveOnEmpty => {
                write!(f, "swap-remove attempted on empty attribute")
            }
            AttributeInvariantViolation::PushFromTypeMismatch => {
                write!(f, "push_from source attribute has a different element type")
            }
            AttributeInvariantViolation::LockPoisoned => {
                write!(f, "RwLock poisoned by a panicking writer")
            }
            AttributeInvariantViolation::StillShared => {
                write!(f, "Arc still has multiple owners and cannot be unwrapped")
            }
        }
    }
}

/// Aggregate error for attribute (component column) operations.
///
/// This wraps precise, low-level failures that can occur when pushing or
/// writing component data into storage. Typical sources include:
///
/// * index math issues while constructing column positions (e.g. index overflow),
/// * position bounds problems when addressing `(ChunkID, RowID)` pairs,
/// * type mismatches between a column's declared element type and a provided value.
///
/// Conversions (`From<T>`) are implemented for common low-level errors so callers
/// can write `?` and still return a single, expressive type.
///
/// ### Display
/// `Display` messages are concise and suitable for logs. For deep inspection,
/// prefer `Debug` which includes full structure.
///
/// ### Example
/// ```text
/// fn push_component<T: 'static>(col: &mut Column, value: T) -> Result<(), AttributeError> {
///     // may become AttributeError::TypeMismatch via From<TypeMismatchError>
///     check_type::<T>(col)?;
///
///     // may become AttributeError::IndexOverflow("row") etc.
///     let pos = compute_position(col.len())?;
///
///     // may become AttributeError via From<PositionOutOfBoundsError>
///     col.write(pos, value)?;
///     Ok(())
/// }
/// ```

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeError {
    /// A `(ChunkID, RowID)` addressed storage outside valid bounds.
    Position(PositionOutOfBoundsError),

    /// The dynamic type of value did not match the component storage type.
    TypeMismatch(TypeMismatchError),

    /// Index arithmetic overflow occurred while constructing a storage index.
    ///
    /// The string identifies which index overflowed (e.g. `"row"` or `"chunk"`).
    IndexOverflow(&'static str),

    /// An internal storage invariant was violated.
    InternalInvariant(AttributeInvariantViolation),
}

impl fmt::Display for AttributeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttributeError::Position(e) => write!(f, "{e}"),
            AttributeError::TypeMismatch(e) => write!(f, "{e}"),
            AttributeError::IndexOverflow(which) => {
                write!(f, "index overflow constructing {}", which)
            }
            AttributeError::InternalInvariant(violation) => {
                write!(f, "internal storage invariant violated: {}", violation)
            }
        }
    }
}

impl std::error::Error for AttributeError {}

impl From<PositionOutOfBoundsError> for AttributeError {
    fn from(e: PositionOutOfBoundsError) -> Self {
        AttributeError::Position(e)
    }
}

impl From<TypeMismatchError> for AttributeError {
    fn from(e: TypeMismatchError) -> Self {
        AttributeError::TypeMismatch(e)
    }
}
