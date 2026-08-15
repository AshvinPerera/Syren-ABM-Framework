# Safety invariants

Syren uses `unsafe` in a small number of places to get columnar storage and
lock-free per-worker staging. Each use rests on an invariant that the surrounding
safe API upholds. This chapter records those invariants so that changes near them
are made deliberately.

## Type-erased columns

Component columns are stored type-erased and cast back to their concrete type
during iteration (`cast_slice` / `cast_slice_mut`). The invariant: a column is
only ever cast to the type it was registered with. Registration records the type
for each `ComponentID`, queries validate the requested type against the column,
and the registry is frozen before the world runs, so the mapping cannot change
underneath a cast.

## Runtime borrow checking

Because the compiler cannot see which components a query touches, aliasing is
enforced at runtime by the borrow tracker. The invariant: no two live borrows
alias a component incompatibly (two writes, or a read overlapping a write). The
scheduler avoids conflicts through declared access; the borrow tracker is the
backstop, turning a conflicting borrow into an error instead of a data race.

## Per-worker staging

Values produced inside a parallel stage are written to per-worker slots
(`WorkerStage`) without locking. The invariant: during the parallel phase, each
slot is written only by the worker it belongs to, and the slots are drained only
under an exclusive borrow taken after the stage completes. Worker ids are stable
per thread and a thread runs one task at a time, so no two threads write one slot
concurrently.

## Batch atomicity

Batch spawn and despawn apply columnar changes and roll back on error by
truncating to the pre-batch length. The invariant: a batch either applies in full
or leaves storage exactly as it was, so a failure never exposes a partially
constructed cohort.

## GPU byte copies

GPU components are copied to and from device buffers as raw bytes. The invariant:
only `GPUPod` types — fixed-layout, plain-old-data — are mirrored, so a byte copy
reconstructs a valid value. `register_gpu_component` is the gate that enforces the
bound.

## When editing near unsafe

Each `unsafe` block in the source carries a comment stating the invariant it
relies on. A change near one must re-establish that the invariant still holds; the
surrounding safe API maintains it.
