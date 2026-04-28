//! Tuple-based query parameter trait for generic ECS iteration.
//!
//! Defines the `Read<T>` / `Write<T>` marker types and the `QueryParam` trait
//! that allows a single generic `for_each` entry point.
//!
//! # Overview
//!
//! `QueryParam` is implemented for the marker types [`Read<T>`] and
//! [`Write<T>`], as well as for tuples of those markers. Each implementation
//! knows how to:
//!
//! 1. **Validate** - confirm that a [`BuiltQuery`] has the correct number of
//!    read and write columns for this parameter shape.
//! 2. **Iterate** - reinterpret raw byte slices as typed component slices and
//!    invoke a caller-provided closure once per entity.
//!
//! Together these enable [`ECSReference::for_each`] to accept a single type
//! parameter that encodes the full read/write signature of the iteration:
//!
//! ```text
//! ecs.for_each::<(Read<Position>, Read<Velocity>, Write<Acceleration>)>(
//!     query,
//!     |(pos, vel, acc)| { /* ... */ },
//! )?;
//! ```
//!
//! # Closure signature
//!
//! Each `QueryParam` implementation defines the concrete reference types
//! the user closure receives via the `Closure` bound on `for_each_chunk`.
//! For a single `Read<A>` the closure takes `&A`; for `(Read<A>, Write<B>)`
//! it takes `(&A, &mut B)`, and so on.
//!
//! # Safety
//!
//! `QueryParam` is an `unsafe` trait.  Implementors must:
//!
//! - correctly report the expected read/write counts in `validate`,
//! - only reinterpret byte slices as the declared component types in
//!   `for_each_chunk`,
//! - uphold the aliasing guarantees required by `cast_slice` /
//!   `cast_slice_mut`.

use crate::engine::error::{ECSError, ECSResult, InternalViolation};
use crate::engine::query::BuiltQuery;
use crate::engine::storage::{cast_slice, cast_slice_mut};

/// Marker for a read-only component parameter in a tuple-based query.
pub struct Read<T>(std::marker::PhantomData<T>);

/// Marker for a mutable component parameter in a tuple-based query.
pub struct Write<T>(std::marker::PhantomData<T>);

/// Trait implemented by query parameter markers (`Read<T>`, `Write<T>`) and
/// tuples thereof to enable a single generic `for_each` entry point.
///
/// # Safety
///
/// Implementors must correctly report whether they are read or write, must
/// only interpret raw byte slices as the declared type `T`, and must uphold
/// the aliasing guarantees required by [`cast_slice`] / [`cast_slice_mut`].
pub unsafe trait QueryParam: Send + Sync {
    /// The closure type that `for_each_chunk` accepts.
    ///
    /// Each implementation defines this as a concrete `Fn` signature matching
    /// the typed references for the parameter tuple. For example:
    /// - `Read<A>` -> `Fn(&A)`
    /// - `(Read<A>, Write<B>)` -> `Fn((&A, &mut B))`
    type Closure: ?Sized;

    /// Validates that the query shape (read/write counts) matches this param tuple.
    fn validate(query: &BuiltQuery) -> ECSResult<()>;

    /// Iterates over a single chunk, reinterpreting the raw byte slices as
    /// typed component slices and invoking `f` once per entity.
    ///
    /// # Safety
    ///
    /// The caller guarantees that:
    /// - each byte slice in `reads` / `writes` is correctly typed and aligned
    ///   for the corresponding component,
    /// - all byte slices represent the same number of entities,
    /// - read/write aliasing rules are upheld by the borrow tracker.
    unsafe fn for_each_chunk(reads: &[&[u8]], writes: &mut [&mut [u8]], f: &Self::Closure);
}

// --- Implementations for single Read/Write ---

unsafe impl<A: 'static + Send + Sync> QueryParam for Read<A> {
    type Closure = dyn Fn(&A) + Send + Sync;

    fn validate(query: &BuiltQuery) -> ECSResult<()> {
        if query.reads().len() != 1 || !query.writes().is_empty() {
            return Err(ECSError::from(InternalViolation::QueryShapeMismatch {
                method: "for_each<Read<A>>",
                expected_reads: 1,
                expected_writes: 0,
            }));
        }
        query.validate_read_type::<A>(0, "for_each<Read<A>>")?;
        Ok(())
    }

    unsafe fn for_each_chunk(reads: &[&[u8]], _writes: &mut [&mut [u8]], f: &Self::Closure) {
        // SAFETY: Caller guarantees the byte slice is correctly typed for A.
        let a = unsafe { cast_slice::<A>(reads[0].as_ptr(), reads[0].len()) };
        for v in a {
            f(v);
        }
    }
}

unsafe impl<A: 'static + Send + Sync> QueryParam for Write<A> {
    type Closure = dyn Fn(&mut A) + Send + Sync;

    fn validate(query: &BuiltQuery) -> ECSResult<()> {
        if !query.reads().is_empty() || query.writes().len() != 1 {
            return Err(ECSError::from(InternalViolation::QueryShapeMismatch {
                method: "for_each<Write<A>>",
                expected_reads: 0,
                expected_writes: 1,
            }));
        }
        query.validate_write_type::<A>(0, "for_each<Write<A>>")?;
        Ok(())
    }

    unsafe fn for_each_chunk(_reads: &[&[u8]], writes: &mut [&mut [u8]], f: &Self::Closure) {
        // SAFETY: Caller guarantees the byte slice is correctly typed for A.
        let a = unsafe { cast_slice_mut::<A>(writes[0].as_mut_ptr(), writes[0].len()) };
        for v in a {
            f(v);
        }
    }
}

/// Helper macro to generate `QueryParam` implementations for tuples of
/// reads and writes.  Each expansion produces a `validate` that checks
/// column counts and a `for_each_chunk` that casts the raw byte slices to
/// typed component slices and invokes the caller's closure once per entity.
macro_rules! impl_query_param_tuple {
    (reads=[$($R:ident : $ri:tt),*], writes=[$($W:ident : $wi:tt),*], method=$method:expr) => {
        unsafe impl<$($R: 'static + Send + Sync,)* $($W: 'static + Send + Sync,)*>
            QueryParam for ($(Read<$R>,)* $(Write<$W>,)*)
        {
            type Closure = dyn Fn(($(&$R,)* $(&mut $W,)*)) + Send + Sync;

            fn validate(query: &BuiltQuery) -> ECSResult<()> {
                let expected_reads = impl_query_param_tuple!(@count $($R)*);
                let expected_writes = impl_query_param_tuple!(@count $($W)*);
                if query.reads().len() != expected_reads || query.writes().len() != expected_writes {
                    return Err(ECSError::from(InternalViolation::QueryShapeMismatch {
                        method: $method,
                        expected_reads,
                        expected_writes,
                    }));
                }
                $(
                    query.validate_read_type::<$R>($ri, $method)?;
                )*
                $(
                    query.validate_write_type::<$W>($wi, $method)?;
                )*
                Ok(())
            }

            #[allow(unused_variables, unused_assignments, non_snake_case)]
            unsafe fn for_each_chunk(
                reads: &[&[u8]],
                writes: &mut [&mut [u8]],
                f: &Self::Closure,
            ) {
                // SAFETY: Caller guarantees byte slices are correctly typed
                // and aligned for the corresponding component types.

                // Cast each read column to its typed slice.
                $(
                    let $R = unsafe {
                        cast_slice::<$R>(reads[$ri].as_ptr(), reads[$ri].len())
                    };
                )*

                // Cast each write column to its typed mutable slice.
                $(
                    let $W = unsafe {
                        cast_slice_mut::<$W>(writes[$wi].as_mut_ptr(), writes[$wi].len())
                    };
                )*

                // Determine entity count from the first available column.
                let _len: usize;
                impl_query_param_tuple!(@first_len _len, [$($R),*], [$($W),*]);

                // Debug-assert all columns have the same entity count.
                $( debug_assert_eq!($R.len(), _len); )*
                $( debug_assert_eq!($W.len(), _len); )*

                for _i in 0.._len {
                    f(($(&$R[_i],)* $(&mut $W[_i],)*));
                }
            }
        }
    };

    // Counting helper.
    (@count) => { 0usize };
    (@count $head:ident $($tail:ident)*) => { 1usize + impl_query_param_tuple!(@count $($tail)*) };

    // Pick the first available column to determine entity count.
    (@first_len $len:ident, [$first:ident $(, $rest:ident)*], [$($W:ident),*]) => {
        $len = $first.len();
    };
    (@first_len $len:ident, [], [$first:ident $(, $rest:ident)*]) => {
        $len = $first.len();
    };
    (@first_len $len:ident, [], []) => {
        $len = 0;
    };
}

// Generate tuple impls matching the supported iteration signatures:
impl_query_param_tuple!(reads=[A:0], writes=[], method="for_each<(Read<A>,)>");
impl_query_param_tuple!(reads=[A:0, B:1], writes=[], method="for_each<(Read<A>, Read<B>)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2], writes=[], method="for_each<(Read<A>, Read<B>, Read<C>)>");
impl_query_param_tuple!(reads=[A:0], writes=[B:0], method="for_each<(Read<A>, Write<B>)>");
impl_query_param_tuple!(reads=[A:0, B:1], writes=[C:0], method="for_each<(Read<A>, Read<B>, Write<C>)>");
impl_query_param_tuple!(reads=[A:0, B:1], writes=[C:0, D:1], method="for_each<(Read<A>, Read<B>, Write<C>, Write<D>)>");
impl_query_param_tuple!(reads=[], writes=[A:0], method="for_each<(Write<A>,)>");
