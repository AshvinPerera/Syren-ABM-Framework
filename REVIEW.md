# Rust Code Review: abm_framework (Syren)

## Summary

Syren is a substantial (~22 kLOC) archetype-based ECS aimed at large-scale agent-based modelling, with a feature-gated GPU backend, an extension-style messaging/environment/agents/model layer, and a custom borrow tracker for parallel system execution. The architecture is coherent and the safety story is on the whole well-thought-out: phase discipline, an iteration-scope counter, per-component runtime borrow tracking, sorted column locking, sharded entities, and an explicit deferred-command boundary. Documentation is unusually thorough, and the unsafe blocks are nearly all paired with reasoning that is correct.

That said, there are real findings. The most important is that the `ActivationOrder` shuffle variants and the scheduler RNG seed are documented public features but **never actually applied** anywhere in the iteration path — users opting into `ShuffleFull` get sequential iteration with no warning. There are also a handful of soundness footguns in the lower-level storage primitives (panic-during-drop, `usize` multiplication wrap in `AlignedBuffer`), one deliberate-but-acknowledged consistency hole in cross-archetype migration, and several smaller correctness/diagnostic regressions in `Bundle`, the type-erased registry, and a couple of error paths. Nothing here will silently corrupt data on the typical happy path; the issues cluster around edge cases (panic-on-drop, overflow, internal-violation rollback) and dead/half-wired features.

## Critical findings

No findings in this category.

## Major findings

### `ActivationOrder` shuffle variants and `Scheduler::seed` are dead code
- **Location:** `src/engine/scheduler.rs:183` (`activation_orders`), `src/engine/scheduler.rs:285–287` (`seed`), and the entire iteration path in `src/engine/manager/query_executor.rs`.
- **Issue:** `Scheduler::set_activation_order` writes into a `Vec<(SystemID, ActivationOrder)>`, and `Scheduler::seed` writes into `self.seed`, but a project-wide grep (`activation_orders`, `self.seed`, `Shuffle`, `tl_rand_u64`) shows that **no iteration site ever reads either field**. `for_each_unchecked`, `for_each_fallible_unchecked`, and `reduce_unchecked` walk archetypes and chunks in storage order unconditionally. The `tl_rand_u64` function in `src/engine/random.rs:86` is never called either, despite being documented as the engine of the shuffle variants.
- **Impact:** A user who follows the published documentation and does `scheduler.set_activation_order(id, ActivationOrder::ShuffleFull)` to break ordering bias in their ABM will silently get sequential iteration instead. For agent-based models this is exactly the kind of bug that produces subtly wrong scientific output without ever erroring. It also makes the `seed` knob non-functional, defeating the documented determinism story for shuffled runs.
- **Suggested fix:** Either implement the shuffle in the chunk-dispatch path inside `query_executor` (consult `self.systems[idx]` for the per-system order, fan out chunks accordingly, and seed `tl_rand_u64` per worker from `Scheduler::seed`), or remove the public surface (`ActivationOrder::Shuffle*`, `set_activation_order`, `seed`) until it's wired up. Leaving the API documented but inert is the worst of the three options.

### Cross-archetype migration leaves both archetypes inconsistent on partial failure
- **Location:** `src/engine/archetype/migration.rs:295–340` (the four-phase body of `move_row_to_archetype`).
- **Issue:** After Phase 1 (`move_row_across_shared_components`) succeeds but Phase 2 (`add_row_in_components_at_destination`) or Phase 3 (`remove_row_in_components_at_source`) fails, the source archetype's shared columns have already been swap-removed while its source-only columns still hold the entity's data. Every column has a different row count, the entity's location metadata is stale, and the destination archetype has a partial row whose missing columns will propagate divergence into every later operation. The doc-comment on `move_row_to_archetype` openly acknowledges this ("both archetypes will be in an inconsistent state. A rollback mechanism is planned but not yet implemented").
- **Impact:** Any failure path inside `push_dyn` for a destination-only component (factory misregistration, allocation failure deep inside `AlignedBuffer`, type-mismatch from a malformed bundle, etc.) renders the world unusable: subsequent queries iterate columns of differing length, and the entity that was being moved is lost. Because the surfacing error is a normal `MoveError`, callers cannot distinguish "transient failure" from "world is now corrupt" — recovery is impossible without tearing the world down.
- **Suggested fix:** Either (a) implement the rollback the comment describes — the move is naturally reversible because all per-column operations are swap-removes/pushes whose inverses are pushes/swap-removes, and you have the `source_swap_position`/`destination_position` info to undo Phase 1 and Phase 2; or (b) until then, escalate any error past Phase 1 to a dedicated `ECSError::Internal(InternalViolation::ArchetypeStateCorrupt)` so callers get a hard-fail signal instead of a recoverable-looking `MoveError`. Burying a known corruption hole behind a routine error variant invites silent data loss.

## Minor findings

### `AlignedBuffer::grow_to` can wrap on capacity*item_size in release builds
- **Location:** `src/messaging/aligned_buffer.rs:200` (and the matching `Layout::from_size_align` call below).
- **Issue:** `new_cap * self.item_size` is unchecked; in release builds it wraps. If a caller pushes enough items that `cap * item_size` exceeds `usize::MAX` (very large `item_size`, e.g. a 4 KB message struct, plus exponential doubling), the multiplication wraps to a small or zero value before `Layout::from_size_align` ever sees it, and `alloc(new_layout)` allocates far less memory than the buffer's `cap` claims. Subsequent `push`/`as_mut_ptr_at` writes go past the allocation.
- **Impact:** Heap corruption under extreme but legal capacity growth. Hard to trigger today (requires both very large `item_size` and very high message volume per tick), but no compile-time bound prevents either. Determinism makes this reproducible if it does fire.
- **Suggested fix:** Use `new_cap.checked_mul(self.item_size)` and convert the `None` case into an `expect("AlignedBuffer: capacity overflow")` (or surface as a `MessagingError` if you want recoverability). Same fix applies to the matching `cap * item_size` in `Drop` and the `old_layout` line.

### `Bundle::take` does not clear the signature bit
- **Location:** `src/engine/component/bundle.rs:103–113`.
- **Issue:** `swap_remove`s the value out of `values` but leaves the corresponding bit set in `signature`. After taking, `signature.has(cid)` lies; `is_complete_for(...)` reports the bundle still satisfies its requirements; a subsequent `take(cid)` returns `None` despite the signature suggesting otherwise.
- **Impact:** In current usage (`apply_deferred_commands` consumes a Bundle and discards it), this is invisible. But `Bundle` is part of the public API and `DynamicBundle::take` is a public trait method; any caller that reuses a bundle, or composes bundles, sees broken signature semantics. It's also a latent foot-gun for future code in the same module that adds new consumers.
- **Suggested fix:** `self.signature.clear(component_id)` after the `swap_remove`. Two-line change with no observable downside.

### Unsafe storage paths leak a row when migration's misalignment rollback fires
- **Location:** `src/engine/archetype/lifecycle.rs:117–125` (the inner `if let Some(rp) = reference_position` rollback in `spawn_on`).
- **Issue:** When column N detects misalignment (`position != rp`), the rollback loop iterates `written_indices` (which contains `0..N-1`, *not* N itself) and calls `swap_remove_dyn(rp.0, rp.1)` on each — but column N has already pushed its value at `position`, not at `rp`, and that row is never removed. The current column ends up with one more row than the others; the next operation that touches it observes a permanent length divergence between columns of the same archetype.
- **Impact:** The leak is only reachable on `MisalignedStorage`, which itself only fires on an internal-violation precondition (the archetype's columns disagreed before this spawn). So in practice this is a downstream-of-bug-already condition. Still: the engine claims to "clean up all partial writes" in the doc-comment, and it doesn't. If anyone ever depends on that claim during recovery from a corrupted snapshot, they'll lose more state than expected.
- **Suggested fix:** Push `idx` into `written_indices` *before* the misalignment check, then in the rollback iterate over `written_indices[..len-1]` removing at `rp`, and remove the last column's row at `position` separately. Or simpler: rollback every written column individually using its own returned position, captured into a parallel vec.

### `Attribute` unsafe paths assume drop glue cannot panic
- **Location:** `src/engine/storage/attribute.rs:200–220` (`swap_remove`'s non-last branch), `src/engine/storage/type_erased_attribute.rs:340–360` (`replace_slot_dyn`).
- **Issue:** Both paths perform `assume_init_drop()` on a slot, then write a new value into the now-uninit slot, with no panic guard between the drop and the write. If `T::drop` panics, unwinding leaves the slot uninitialised but `length` unchanged; `Attribute::Drop::drop_all_initialized_elements` then runs `assume_init_drop` on the same slot during world teardown, reading freed memory and re-running the destructor on bytes whose ownership was already released.
- **Impact:** Use-after-free / double-drop. Only triggerable when `T`'s destructor panics, which is rare in well-behaved component types but is not actually forbidden by the `Send + Sync + 'static` bound. A user who registers a component with a fallible drop (e.g. a guard-style RAII type that aborts on a logic error) gets UB on first failure rather than a clean crash.
- **Suggested fix:** Either document a hard "components must not panic on drop" contract (and ideally enforce it with a `catch_unwind` in the unsafe block, aborting on detection), or restructure the swap to read the new value into a local first, then `assume_init_drop` and write — so if the drop panics, the local is dropped on the way up the stack and the slot ends up holding the correct (newly-written) value. The `replace_slot_dyn` path can also use `std::ptr::replace`/`mem::replace` semantics to avoid the bare-window.

### Bucket prefix-sum can overflow `u32`
- **Location:** `src/messaging/specialisations/bucket.rs:78` (and the matching `cell_starts[c+1] = cell_starts[c] + counts[c]` in `spatial.rs:80`).
- **Issue:** `bucket_starts: Vec<u32>` and the prefix-sum is plain `u32` arithmetic. `n` is a `usize`. If a tick emits more than `u32::MAX ≈ 4.29 B` bucket-classified messages (or messages in the last cells of a spatial grid), the prefix sum wraps in release. In debug it panics, which is the better outcome.
- **Impact:** With wrap, the scatter cursor walks off the end of `data` and `copy_nonoverlapping` writes past the buffer's allocation. Heap corruption. Realistically out of reach for most simulations, but the `Capacity::unbounded` API explicitly invites large emit counts.
- **Suggested fix:** Either widen `bucket_starts`/`cell_starts` to `u64` (cheap; arrays are bounded by `max_buckets`/`total_cells`, not by item count), or check `raw.len() <= u32::MAX as usize` at the top of `finalise` and surface `MessagingError::EmitCapacityExceeded`. The latter is consistent with the existing per-tick-cap mechanism.

### `mark_gpu_safe` returns the wrong `TypeId` in its error
- **Location:** `src/engine/component/registry.rs:240–248`.
- **Issue:** When `mark_gpu_safe(component_id)` is called for an unregistered ID, it returns `RegistryError::NotRegistered { type_id: TypeId::of::<()>() }`. The function takes a `ComponentID`, not a Rust type, so it has no real `TypeId` to report — but reporting the unit-type ID is actively misleading in logs ("type `()` not registered").
- **Impact:** Diagnostics regression; an operator chasing a registration bug reads a meaningless type name.
- **Suggested fix:** Add a dedicated variant — e.g. `RegistryError::NotRegisteredByID { component_id: ComponentID }` — or reshape `NotRegistered` to carry a `TypeId` *or* `ComponentID` discriminant. Either is a small change; the current placeholder is worse than no info.

### Instance `ComponentRegistry::register` accepts ZSTs; the global wrapper rejects them
- **Location:** `src/engine/component/registry.rs:178–198` vs. `src/engine/component/global.rs:88–100`.
- **Issue:** `register_component<T>()` in `global.rs` checks `size_of::<T>() == 0` and rejects with `ZeroSizedComponent`. The instance method `ComponentRegistry::register::<T>()` does no such check. Multi-world callers (which the docs explicitly steer toward instance registries) can register a ZST, get a `ComponentID`, and then run into trouble inside `Attribute<()>` (chunks become ZST allocations, the `length`/`last_chunk_length` accounting still increments, but `chunk.as_ptr()` casts produce dangling pointers when used through `chunk_bytes`).
- **Impact:** Inconsistency between the two supposedly-equivalent registration paths; a multi-world user gets a footgun the single-world user is shielded from.
- **Suggested fix:** Move the ZST check into `ComponentRegistry::register` and have the global function call through. The global path then becomes a thin wrapper, which is what its doc comment already claims.

### `EntityShards::despawn` can leave the archetype meta inconsistent if `set_location` fails mid-fixup
- **Location:** `src/engine/archetype/lifecycle.rs:255–280` (the `Some((moved_chunk, moved_row))` branch of `despawn_on`).
- **Issue:** Inside the meta write lock, after writing `meta.entity_positions[entity_chunk][entity_row] = Some(moved_entity)`, `shards.set_location(...)` is called. If `set_location` returns `Err` (only happens on shard-mutex poisoning), the function returns without clearing `meta.entity_positions[moved_chunk][moved_row]`. That entry now points to `moved_entity`, *and* so does `entity_positions[entity_chunk][entity_row]` — the same entity is in two slots. `meta.length` is still decremented after the function returns? Actually, it isn't: the `?` returns before the `meta.length` update, so length is *not* decremented either, but the column data has already been swap-removed. Length and column row count diverge.
- **Impact:** Reachable only on shard-mutex poisoning (which itself implies a panicking thread elsewhere), so the world is already in a degraded state. Still, this is the recovery path, and it deepens the corruption rather than minimising it.
- **Suggested fix:** Move the `set_location` call so it runs *after* both `entity_positions` slots are in their final state, or wrap the meta updates in a small helper that clears the moved-from slot first and only then calls `set_location`. Either way, the goal is that returning early from `set_location` leaves a recoverable archetype.

### `chunk_valid_length` and `chunk_count` race-prone under concurrent meta mutation
- **Location:** `src/engine/archetype/core.rs:285–305`.
- **Issue:** `chunk_valid_length` calls `chunk_count()` and `length()` separately, each acquiring a fresh read lock on `meta`. Phase discipline guarantees these two reads are not interleaved with a write under correct usage — but the function itself doesn't enforce that, and a future caller that uses it from a path with different invariants will silently get inconsistent values (e.g. `chunk_count` observed before a structural mutation, `length` observed after).
- **Impact:** No concrete bug today; potential foot-gun for the next refactor. Worth flagging because it's the only non-obvious place where the meta lock is acquired twice in one logical operation.
- **Suggested fix:** Take the meta read lock once and compute both values inside a single guard. Net effect: same lock cost, atomic snapshot.

### `LockedAttribute::write` is invoked with `Self::lock_write_spawn` even outside spawn paths
- **Location:** `src/engine/archetype/lifecycle.rs:89` (the despawn loop calls `Self::lock_write_spawn`).
- **Issue:** `lock_write_spawn` returns `SpawnError::StoragePushFailedWith(AttributeError::InternalInvariant(LockPoisoned))` on a poisoned lock. Despawn calls it, so a poisoned column lock during despawn surfaces as `SpawnError::StoragePushFailedWith` — wrong category, wrong helper name.
- **Impact:** Diagnostics: the user reads "storage push failed" while the actual operation was a despawn. Minor but confusing.
- **Suggested fix:** Add a `lock_write_despawn` variant returning `SpawnError::StorageSwapRemoveFailed(AttributeError::InternalInvariant(LockPoisoned))`, or generalise both to a single `lock_write_archetype(attr, op_kind)`.

## Nits & style

### Empty placeholder modules clutter the source tree
- **Location:** `src/batch/`, `src/logging/`, `src/network/`, `src/scripting/`.
- **Issue:** Each is an empty directory under `src/`. They are not declared in `lib.rs` and contain no code. They'll show up in any `find`/IDE tree as if they were real modules.
- **Suggested fix:** Delete the directories until they're real. Adding empty folders with intent communicates better through an `ARCHITECTURE.md` roadmap, not through source-tree squatters.

### `thiserror` is used in two files; the rest of the error types are hand-written
- **Location:** `src/messaging/error.rs`, `src/model/error.rs` use `thiserror`; everything in `src/engine/error/` uses manual `Display`/`From` impls.
- **Issue:** Inconsistent. `thiserror = "2"` is already a dependency, and the engine error types are repetitive enough that they would benefit from the macro most.
- **Suggested fix:** Migrate `engine/error/*.rs` to `#[derive(thiserror::Error)]`. Removes ~150 lines of mechanical Display/From implementations.

### `Entity::components` deprecation aliases live forever
- **Location:** `src/engine/entity/entity.rs:113–122`.
- **Issue:** `#[deprecated(since = "0.2.0")]` — the crate is at 0.3.0 per `Cargo.toml`. Pre-1.0, it's reasonable to remove deprecated APIs at minor bumps.
- **Suggested fix:** Drop `Entity::components` now; it doesn't carry a 1.0 stability promise.

### Unused `#[must_use]` on `Entity` doesn't extend to `from_raw`/`to_raw`
- **Location:** `src/engine/entity/entity.rs:80–95`.
- **Issue:** `Entity` is `#[must_use]`, but `from_raw` returns it without explicit `#[must_use]` on the function, and `to_raw` also returns a value the user is unlikely to want to discard. Minor, but the surrounding code style is otherwise fastidious about lints.
- **Suggested fix:** Add `#[must_use]` on `from_raw` and `to_raw` for symmetry, or rely on the type-level `must_use` for the constructor only and explicitly mark the projection.

### `register` rejects duplicate templates with `TemplateNotFound`
- **Location:** `src/agents/registry.rs:84–95`.
- **Issue:** Returns `AgentError::TemplateNotFound("template '{}' is already registered")` for a duplicate registration. Variant name says "not found", message says "already registered". The code comment acknowledges the awkwardness.
- **Suggested fix:** Add a `DuplicateTemplate(String)` variant to `AgentError`. The change is small and worth the variant cost.

### `Bundle::insert` and `Bundle::insert_boxed` linear-scan to update existing entries
- **Location:** `src/engine/component/bundle.rs:78–98`.
- **Issue:** `iter_mut().find(|(cid, _)| *cid == component_id)` is O(n) per insert. A bundle with K components built one-at-a-time is O(K²). K is typically small (≤16 in most ABMs), but the file documents the bundle as part of "scripting layers, serialization, and editor tooling" — none of which can rely on K staying small.
- **Suggested fix:** Either keep `values` sorted by `ComponentID` and `binary_search` (consistent with `Archetype::components`), or skip the scan and rely on `signature.has(cid)` to short-circuit duplicate inserts at registration time.

### `Scheduler::pack_graph` cycle detection is `O(systems³)` worst-case
- **Location:** `src/engine/scheduler.rs:435–447`.
- **Issue:** For each component-conflict pair `(a, b)` it runs two `has_path` BFS passes, each `O(n + edges)`. Net `O(systems² × (n + edges))`. With many systems and dense conflicts that's `O(n³)` to `O(n⁴)`. For typical n ≤ 50 this is invisible; for n in the hundreds it becomes noticeable at every `dirty` rebuild.
- **Suggested fix:** Compute reachability once via Floyd–Warshall or by running `has_path` lazily and caching. Probably not worth it until system counts grow; flag and move on.

## Architectural observations

The architecture is opinionated and largely consistent: archetype storage in chunked `MaybeUninit<T>` columns, per-column `RwLock`s acquired in sorted order, sharded entities, an explicit phase lock plus an iteration counter, deferred commands as the only structural mutation route, and a per-component `BorrowTracker` enforcing aliasing rules at runtime. The four-pass migration in `archetype/migration.rs` is the cleanest exposition I've seen of how to factor add/remove/move into independently testable phases — modulo the partial-failure issue noted as Major above.

Two design tensions worth surfacing:

* **Engine module visibility is inverted.** `pub(crate) mod engine` plus a long block of selective re-exports in `lib.rs` is the opposite of what most ECS crates do (`pub mod engine` with private internals). The current shape forces every internal change to also touch `lib.rs`, and re-exports flatten what is genuinely a deep module tree. If your goal is to insulate users from internal restructuring you've largely achieved it, but the cost is that `lib.rs` is now a 130-line manifest. Consider a `prelude` plus a thinner `pub use engine::{public_api}` line per submodule and let the module tree document itself.

* **Boundary resources, channels, and the GPU dispatch all share the same exclusive-phase rope.** `apply_deferred_commands`, `sync_pending_to_cpu`, and `finalise_boundaries` all serialise on `phase_write`. With many GPU systems and many boundary resources, this becomes the central contention point of every tick. No fix to suggest — this is the cost of doing structural mutation safely without per-archetype RW guards — but it's worth documenting as the headline scaling limit of the current design so users don't try to spread structural changes across many small boundaries.

* **The `messaging` GPU support is gated behind `messaging_gpu` but I see no GPU-specific path in `messaging/specialisations/*.rs`.** The flag exists in `Cargo.toml` and is wired through `lib.rs`, but the actual specialisations don't appear to differ. Either the GPU path is genuinely missing (in which case the flag should fail loudly when used) or it's elsewhere and I missed it; worth a docstring even in the latter case.

## What's done well

* **Borrow tracker design is tight.** The state encoding (`0` unlocked, `1` write, `>=2` reader count + 1) avoids the transient-`1`-during-reader-arrival race that catches a lot of hand-rolled equivalents. Spin-limit-with-yield + `BorrowConflict` surfacing means scheduling bugs become diagnostic errors instead of hangs. The dirty-bitset clear (`src/engine/borrow/tracker.rs:130–145`) is a smart amortisation — only touch components that were borrowed.
* **Lock-ordering contract is documented and enforced by data layout.** `Archetype::components` is sorted by `ComponentID`, so iterating it acquires column locks in ascending order naturally. The mod-level docs spell out the contract (column locks before meta lock, ascending IDs), and `query_executor::lock_columns` follows it.
* **Error types are factored along genuine semantic axes.** `SpawnError`, `MoveError`, `AttributeError`, `RegistryError`, `ExecutionError`, `InternalViolation` each carry distinct, structured context (e.g. `BorrowConflict { component_id, held, requested }`). `InternalViolation` is a clean way to separate framework-bug errors from user-recoverable ones.
* **Unsafe documentation is consistently good.** Every unsafe block in the storage layer carries a SAFETY comment that lists the invariants relied on. The `swap_remove` and `push_from` reasoning chains are unusually thorough; review-by-eye for soundness is genuinely possible because of this.
* **Phase discipline is enforced at types.** `PhaseRead`/`PhaseWrite` zero-cost guard tokens making `data_mut_unchecked` un-callable without the right token is exactly the right level of abstraction — the unsafe boundary is precisely where the philosophy says it should be.
