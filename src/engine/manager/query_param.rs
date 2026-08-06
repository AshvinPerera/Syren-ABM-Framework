//! Tuple-based query parameter trait for generic ECS iteration.
//!
//! Defines the `Read<T>` / `Write<T>` marker types and the `QueryParam` trait
//! that allows a single generic `for_each` entry point.
//!
//! # Overview
//!
//! `QueryParam` is implemented for the marker types [`Read<T>`] and
//! [`Write<T>`], as well as for tuples of those markers (up to four reads and
//! four writes). Each implementation knows how to:
//!
//! 1. **Validate** - confirm that a [`BuiltQuery`] has the correct number of
//!    read and write columns for this parameter shape.
//! 2. **Iterate** - reinterpret raw byte slices as typed component slices and
//!    invoke a caller-provided closure once per entity.
//!
//! Together these enable [`ECSReference::for_each`] to accept a type parameter
//! that encodes the full read/write signature of the iteration:
//!
//! ```text
//! ecs.for_each::<(Read<Position>, Read<Velocity>, Write<Acceleration>), _>(
//!     query,
//!     |(pos, vel, acc)| { /* ... */ },
//! )?;
//! ```
//!
//! # Closure items and static dispatch
//!
//! Each implementation defines [`QueryParam::Item`], the typed value the user
//! closure receives per row: `&A` for a bare `Read<A>`, `(&A, &mut B)` for
//! `(Read<A>, Write<B>)`, and so on. Closures are **statically dispatched** -
//! the generic `F: Fn(P::Item<'_>)` bound lets the compiler inline the closure
//! body into the per-row loop and vectorise it, unlike the former
//! `&dyn Fn`-based design which paid an indirect call per row.
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

use crate::engine::entity::Entity;
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
pub unsafe trait QueryParam: 'static {
    /// The typed value handed to the closure for one row.
    ///
    /// Examples: `&'a A` for `Read<A>`, `(&'a A, &'a mut B)` for
    /// `(Read<A>, Write<B>)`.
    type Item<'a>;

    /// Validates that the query shape (read/write counts) matches this param tuple.
    fn validate(query: &BuiltQuery) -> ECSResult<()>;

    /// Iterates over a single chunk range, reinterpreting the raw byte slices
    /// as typed component slices and invoking `f` once per entity.
    ///
    /// # Safety
    ///
    /// The caller guarantees that:
    /// - each byte slice in `reads` / `writes` is correctly typed and aligned
    ///   for the corresponding component,
    /// - all byte slices represent the same number of entities,
    /// - read/write aliasing rules are upheld by the borrow tracker.
    unsafe fn for_each_chunk<F>(reads: &[&[u8]], writes: &mut [&mut [u8]], f: &F)
    where
        F: for<'a> Fn(Self::Item<'a>);
}

/// Entity-aware variant of [`QueryParam`].
///
/// Implementations receive a row-aligned entity slice in addition to the
/// component byte slices. The slice contains exactly one [`Entity`] for each
/// component row in the range, allowing generated systems to map a component
/// reference back to its owning entity without changing component storage.
///
/// # Safety
///
/// Implementors inherit the [`QueryParam`] safety requirements and must also
/// keep the entity slice indexed in lockstep with the component slices.
pub unsafe trait EntityQueryParam: QueryParam {
    /// The typed value handed to entity-aware closures for one row:
    /// the owning [`Entity`] followed by the component references.
    ///
    /// Examples: `(Entity, &'a A)` for `Read<A>`,
    /// `(Entity, &'a A, &'a mut B)` for `(Read<A>, Write<B>)`.
    type EntityItem<'a>;

    /// Iterates one chunk range and invokes `f` once per entity row.
    ///
    /// # Safety
    ///
    /// The caller guarantees that `entities`, `reads`, and `writes` describe
    /// the same chunk range and row count.
    unsafe fn for_each_entity_chunk<F>(
        entities: &[Entity],
        reads: &[&[u8]],
        writes: &mut [&mut [u8]],
        f: &F,
    ) where
        F: for<'a> Fn(Self::EntityItem<'a>);

    /// Iterates one chunk range with a fallible closure.
    ///
    /// # Safety
    ///
    /// The caller guarantees that `entities`, `reads`, and `writes` describe
    /// the same chunk range and row count.
    unsafe fn for_each_entity_chunk_fallible<F>(
        entities: &[Entity],
        reads: &[&[u8]],
        writes: &mut [&mut [u8]],
        f: &F,
    ) -> ECSResult<()>
    where
        F: for<'a> Fn(Self::EntityItem<'a>) -> ECSResult<()>;
}

// --- Implementations for bare Read / Write ---

unsafe impl<A: 'static + Send + Sync> QueryParam for Read<A> {
    type Item<'a> = &'a A;

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

    unsafe fn for_each_chunk<F>(reads: &[&[u8]], _writes: &mut [&mut [u8]], f: &F)
    where
        F: for<'a> Fn(Self::Item<'a>),
    {
        // SAFETY: Caller guarantees the byte slice is correctly typed for A.
        let a = unsafe { cast_slice::<A>(reads[0].as_ptr(), reads[0].len()) };
        for v in a {
            f(v);
        }
    }
}

unsafe impl<A: 'static + Send + Sync> EntityQueryParam for Read<A> {
    type EntityItem<'a> = (Entity, &'a A);

    unsafe fn for_each_entity_chunk<F>(
        entities: &[Entity],
        reads: &[&[u8]],
        _writes: &mut [&mut [u8]],
        f: &F,
    ) where
        F: for<'a> Fn(Self::EntityItem<'a>),
    {
        // SAFETY: Caller guarantees the byte slice is correctly typed for A.
        let a = unsafe { cast_slice::<A>(reads[0].as_ptr(), reads[0].len()) };
        debug_assert_eq!(entities.len(), a.len());
        for i in 0..a.len() {
            f((entities[i], &a[i]));
        }
    }

    unsafe fn for_each_entity_chunk_fallible<F>(
        entities: &[Entity],
        reads: &[&[u8]],
        _writes: &mut [&mut [u8]],
        f: &F,
    ) -> ECSResult<()>
    where
        F: for<'a> Fn(Self::EntityItem<'a>) -> ECSResult<()>,
    {
        // SAFETY: Caller guarantees the byte slice is correctly typed for A.
        let a = unsafe { cast_slice::<A>(reads[0].as_ptr(), reads[0].len()) };
        debug_assert_eq!(entities.len(), a.len());
        for i in 0..a.len() {
            f((entities[i], &a[i]))?;
        }
        Ok(())
    }
}

unsafe impl<A: 'static + Send + Sync> QueryParam for Write<A> {
    type Item<'a> = &'a mut A;

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

    unsafe fn for_each_chunk<F>(_reads: &[&[u8]], writes: &mut [&mut [u8]], f: &F)
    where
        F: for<'a> Fn(Self::Item<'a>),
    {
        // SAFETY: Caller guarantees the byte slice is correctly typed for A.
        let a = unsafe { cast_slice_mut::<A>(writes[0].as_mut_ptr(), writes[0].len()) };
        for v in a {
            f(v);
        }
    }
}

unsafe impl<A: 'static + Send + Sync> EntityQueryParam for Write<A> {
    type EntityItem<'a> = (Entity, &'a mut A);

    unsafe fn for_each_entity_chunk<F>(
        entities: &[Entity],
        _reads: &[&[u8]],
        writes: &mut [&mut [u8]],
        f: &F,
    ) where
        F: for<'a> Fn(Self::EntityItem<'a>),
    {
        // SAFETY: Caller guarantees the byte slice is correctly typed for A.
        let a = unsafe { cast_slice_mut::<A>(writes[0].as_mut_ptr(), writes[0].len()) };
        debug_assert_eq!(entities.len(), a.len());
        for i in 0..a.len() {
            f((entities[i], &mut a[i]));
        }
    }

    unsafe fn for_each_entity_chunk_fallible<F>(
        entities: &[Entity],
        _reads: &[&[u8]],
        writes: &mut [&mut [u8]],
        f: &F,
    ) -> ECSResult<()>
    where
        F: for<'a> Fn(Self::EntityItem<'a>) -> ECSResult<()>,
    {
        // SAFETY: Caller guarantees the byte slice is correctly typed for A.
        let a = unsafe { cast_slice_mut::<A>(writes[0].as_mut_ptr(), writes[0].len()) };
        debug_assert_eq!(entities.len(), a.len());
        for i in 0..a.len() {
            f((entities[i], &mut a[i]))?;
        }
        Ok(())
    }
}

/// Generates `QueryParam` / `EntityQueryParam` implementations for tuples of
/// reads and writes.  Each expansion produces a `validate` that checks column
/// counts and iteration bodies that cast the raw byte slices to typed
/// component slices and invoke the caller's closure once per entity, fully
/// monomorphised.
macro_rules! impl_query_param_tuple {
    (reads=[$($R:ident : $ri:tt),*], writes=[$($W:ident : $wi:tt),*], method=$method:expr) => {
        unsafe impl<$($R: 'static + Send + Sync,)* $($W: 'static + Send + Sync,)*>
            QueryParam for ($(Read<$R>,)* $(Write<$W>,)*)
        {
            type Item<'a> = ($(&'a $R,)* $(&'a mut $W,)*);

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
            unsafe fn for_each_chunk<Func>(
                reads: &[&[u8]],
                writes: &mut [&mut [u8]],
                f: &Func,
            )
            where
                Func: for<'a> Fn(Self::Item<'a>),
            {
                // SAFETY: Caller guarantees byte slices are correctly typed
                // and aligned for the corresponding component types.

                $(
                    let $R = unsafe {
                        cast_slice::<$R>(reads[$ri].as_ptr(), reads[$ri].len())
                    };
                )*

                $(
                    let $W = unsafe {
                        cast_slice_mut::<$W>(writes[$wi].as_mut_ptr(), writes[$wi].len())
                    };
                )*

                let _len: usize;
                impl_query_param_tuple!(@first_len _len, [$($R),*], [$($W),*]);

                $( debug_assert_eq!($R.len(), _len); )*
                $( debug_assert_eq!($W.len(), _len); )*

                for _i in 0.._len {
                    f(($(&$R[_i],)* $(&mut $W[_i],)*));
                }
            }
        }

        unsafe impl<$($R: 'static + Send + Sync,)* $($W: 'static + Send + Sync,)*>
            EntityQueryParam for ($(Read<$R>,)* $(Write<$W>,)*)
        {
            type EntityItem<'a> = (Entity, $(&'a $R,)* $(&'a mut $W,)*);

            #[allow(unused_variables, unused_assignments, non_snake_case)]
            unsafe fn for_each_entity_chunk<Func>(
                entities: &[Entity],
                reads: &[&[u8]],
                writes: &mut [&mut [u8]],
                f: &Func,
            )
            where
                Func: for<'a> Fn(Self::EntityItem<'a>),
            {
                // SAFETY: Caller guarantees byte slices are correctly typed
                // and aligned for the corresponding component types.

                $(
                    let $R = unsafe {
                        cast_slice::<$R>(reads[$ri].as_ptr(), reads[$ri].len())
                    };
                )*

                $(
                    let $W = unsafe {
                        cast_slice_mut::<$W>(writes[$wi].as_mut_ptr(), writes[$wi].len())
                    };
                )*

                let _len: usize;
                impl_query_param_tuple!(@first_len _len, [$($R),*], [$($W),*]);

                debug_assert_eq!(entities.len(), _len);
                $( debug_assert_eq!($R.len(), _len); )*
                $( debug_assert_eq!($W.len(), _len); )*

                for _i in 0.._len {
                    f((entities[_i], $(&$R[_i],)* $(&mut $W[_i],)*));
                }
            }

            #[allow(unused_variables, unused_assignments, non_snake_case)]
            unsafe fn for_each_entity_chunk_fallible<Func>(
                entities: &[Entity],
                reads: &[&[u8]],
                writes: &mut [&mut [u8]],
                f: &Func,
            ) -> ECSResult<()>
            where
                Func: for<'a> Fn(Self::EntityItem<'a>) -> ECSResult<()>,
            {
                // SAFETY: Caller guarantees byte slices are correctly typed
                // and aligned for the corresponding component types.

                $(
                    let $R = unsafe {
                        cast_slice::<$R>(reads[$ri].as_ptr(), reads[$ri].len())
                    };
                )*

                $(
                    let $W = unsafe {
                        cast_slice_mut::<$W>(writes[$wi].as_mut_ptr(), writes[$wi].len())
                    };
                )*

                let _len: usize;
                impl_query_param_tuple!(@first_len _len, [$($R),*], [$($W),*]);

                debug_assert_eq!(entities.len(), _len);
                $( debug_assert_eq!($R.len(), _len); )*
                $( debug_assert_eq!($W.len(), _len); )*

                for _i in 0.._len {
                    f((entities[_i], $(&$R[_i],)* $(&mut $W[_i],)*))?;
                }
                Ok(())
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

// Generate tuple impls for every read/write arity combination up to four of
// each (the empty combination is not a meaningful query shape).
impl_query_param_tuple!(reads=[A:0], writes=[], method="for_each<(Read,)>");
impl_query_param_tuple!(reads=[A:0, B:1], writes=[], method="for_each<(Read, Read)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2], writes=[], method="for_each<(Read x3)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3], writes=[], method="for_each<(Read x4)>");

impl_query_param_tuple!(reads=[], writes=[A:0], method="for_each<(Write,)>");
impl_query_param_tuple!(reads=[], writes=[A:0, B:1], method="for_each<(Write x2)>");
impl_query_param_tuple!(reads=[], writes=[A:0, B:1, C:2], method="for_each<(Write x3)>");
impl_query_param_tuple!(reads=[], writes=[A:0, B:1, C:2, D:3], method="for_each<(Write x4)>");

impl_query_param_tuple!(reads=[A:0], writes=[B:0], method="for_each<(Read, Write)>");
impl_query_param_tuple!(reads=[A:0], writes=[B:0, C:1], method="for_each<(Read, Write x2)>");
impl_query_param_tuple!(reads=[A:0], writes=[B:0, C:1, D:2], method="for_each<(Read, Write x3)>");
impl_query_param_tuple!(reads=[A:0], writes=[B:0, C:1, D:2, E:3], method="for_each<(Read, Write x4)>");

impl_query_param_tuple!(reads=[A:0, B:1], writes=[C:0], method="for_each<(Read x2, Write)>");
impl_query_param_tuple!(reads=[A:0, B:1], writes=[C:0, D:1], method="for_each<(Read x2, Write x2)>");
impl_query_param_tuple!(reads=[A:0, B:1], writes=[C:0, D:1, E:2], method="for_each<(Read x2, Write x3)>");
impl_query_param_tuple!(reads=[A:0, B:1], writes=[C:0, D:1, E:2, F:3], method="for_each<(Read x2, Write x4)>");

impl_query_param_tuple!(reads=[A:0, B:1, C:2], writes=[D:0], method="for_each<(Read x3, Write)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2], writes=[D:0, E:1], method="for_each<(Read x3, Write x2)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2], writes=[D:0, E:1, F:2], method="for_each<(Read x3, Write x3)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2], writes=[D:0, E:1, F:2, G:3], method="for_each<(Read x3, Write x4)>");

impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3], writes=[E:0], method="for_each<(Read x4, Write)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3], writes=[E:0, F:1], method="for_each<(Read x4, Write x2)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3], writes=[E:0, F:1, G:2], method="for_each<(Read x4, Write x3)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3], writes=[E:0, F:1, G:2, H:3], method="for_each<(Read x4, Write x4)>");

// Extended arities, total reads + writes up to 9.
//
// The 4x4 grid above covers most systems; these cover the rest. Unused
// combinations cost only a trait impl -- monomorphisation happens per concrete
// type combination actually instantiated, so the binary does not grow for
// shapes nobody calls.
impl_query_param_tuple!(reads=[], writes=[A:0, B:1, C:2, D:3, E:4], method="for_each<(Read x0, Write x5)>");
impl_query_param_tuple!(reads=[], writes=[A:0, B:1, C:2, D:3, E:4, F:5], method="for_each<(Read x0, Write x6)>");
impl_query_param_tuple!(reads=[], writes=[A:0, B:1, C:2, D:3, E:4, F:5, G:6], method="for_each<(Read x0, Write x7)>");
impl_query_param_tuple!(reads=[], writes=[A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7], method="for_each<(Read x0, Write x8)>");
impl_query_param_tuple!(reads=[A:0], writes=[B:0, C:1, D:2, E:3, F:4], method="for_each<(Read x1, Write x5)>");
impl_query_param_tuple!(reads=[A:0], writes=[B:0, C:1, D:2, E:3, F:4, G:5], method="for_each<(Read x1, Write x6)>");
impl_query_param_tuple!(reads=[A:0], writes=[B:0, C:1, D:2, E:3, F:4, G:5, H:6], method="for_each<(Read x1, Write x7)>");
impl_query_param_tuple!(reads=[A:0], writes=[B:0, C:1, D:2, E:3, F:4, G:5, H:6, I:7], method="for_each<(Read x1, Write x8)>");
impl_query_param_tuple!(reads=[A:0, B:1], writes=[C:0, D:1, E:2, F:3, G:4], method="for_each<(Read x2, Write x5)>");
impl_query_param_tuple!(reads=[A:0, B:1], writes=[C:0, D:1, E:2, F:3, G:4, H:5], method="for_each<(Read x2, Write x6)>");
impl_query_param_tuple!(reads=[A:0, B:1], writes=[C:0, D:1, E:2, F:3, G:4, H:5, I:6], method="for_each<(Read x2, Write x7)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2], writes=[D:0, E:1, F:2, G:3, H:4], method="for_each<(Read x3, Write x5)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2], writes=[D:0, E:1, F:2, G:3, H:4, I:5], method="for_each<(Read x3, Write x6)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3], writes=[E:0, F:1, G:2, H:3, I:4], method="for_each<(Read x4, Write x5)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4], writes=[], method="for_each<(Read x5, Write x0)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4], writes=[F:0], method="for_each<(Read x5, Write x1)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4], writes=[F:0, G:1], method="for_each<(Read x5, Write x2)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4], writes=[F:0, G:1, H:2], method="for_each<(Read x5, Write x3)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4], writes=[F:0, G:1, H:2, I:3], method="for_each<(Read x5, Write x4)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4, F:5], writes=[], method="for_each<(Read x6, Write x0)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4, F:5], writes=[G:0], method="for_each<(Read x6, Write x1)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4, F:5], writes=[G:0, H:1], method="for_each<(Read x6, Write x2)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4, F:5], writes=[G:0, H:1, I:2], method="for_each<(Read x6, Write x3)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4, F:5, G:6], writes=[], method="for_each<(Read x7, Write x0)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4, F:5, G:6], writes=[H:0], method="for_each<(Read x7, Write x1)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4, F:5, G:6], writes=[H:0, I:1], method="for_each<(Read x7, Write x2)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7], writes=[], method="for_each<(Read x8, Write x0)>");
impl_query_param_tuple!(reads=[A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7], writes=[I:0], method="for_each<(Read x8, Write x1)>");
