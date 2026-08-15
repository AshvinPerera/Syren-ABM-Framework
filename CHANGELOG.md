# Changelog

## 0.6.0 — 2026-08-15

First release under the crate name `syren`. It renames the crate, adds
a model-wide seed API, migrates the macroeconomy example onto it, introduces an
introductory example and a full documentation system, and establishes CI,
governance, and packaging. See the migration notes at the end of this section.

### Added

- **Introductory example `first_model`** (`cargo run --example first_model
  --features model`): a population of random walkers that demonstrates the
  smallest end-to-end path — component registration, `ModelBuilder`, a system
  whose access is derived from its query, per-entity deterministic RNG from the
  run context, and a `Welford` reduction for count, mean, and variance. Its
  source is marked with `ANCHOR` comments so the guide includes compiled code.
- **Model-wide seed API.** `ModelBuilder::with_seed(u64)` sets a global RNG
  seed and `Model::seed()` returns it. The seed is applied to the root
  scheduler and every shared sub-scheduler at build time, reaching systems as
  `RunContext::simulation_seed`; combined with `DetRng::from_context`, model
  draws are reproducible for a given seed independently of the thread count.
  Nested models are isolated worlds and keep the seed configured by their own
  builders. The default seed is `0`.

### Breaking changes and migration

- **Crate renamed `abm_framework` → `syren`.** Update dependency declarations
  and imports: `use abm_framework::…` becomes `use syren::…`. The library name,
  the public module paths, and the documentation URL all change accordingly.
  The crate was not previously published to crates.io, so no released version
  is affected.

### Changed

- Renamed the Sugarscape example source `sugarscape_v2.rs` to `sugarscape.rs`
  and corrected the target name in its usage text and README. The Cargo example
  target remains `sugarscape` (`cargo run --example sugarscape`).
- Migrated the macroeconomy example onto `ModelBuilder::with_seed(config.seed)`,
  removing its manual `MacroRng` seed fold. The model seed now reaches the draw
  sites through `RunContext::simulation_seed`. **This changes the trajectory
  produced for a given numeric seed** versus 0.5.0; determinism is unchanged
  (a fixed seed reproduces exactly at any thread count, and distinct seeds
  diverge). The measured 40-quarter fixture figures in the example docs were
  regenerated.

### Fixed

- A model with no GPU systems no longer initialises a GPU adapter during
  `tick`. The per-stage GPU download step returned early only after opening the
  device; it now returns before touching the device when there is no GPU work,
  so CPU-only models run under the `gpu` feature on machines without an adapter.
- Updated `crossbeam-epoch` to 0.9.20 to resolve RUSTSEC-2026-0204.
- The macroeconomy stdout CSV header no longer contains an embedded newline; it
  prints as a single physical line. The headline, aggregate-trace, and
  firm-trace column names are each defined once and shared with their row
  builders (`examples/macroeconomy/output.rs`), and the example tests assert
  every header is one line, has unique names, and matches its row's field count.

### Documentation

- Rewrote the README as a project landing page (purpose, status, capabilities,
  installation, a short example, and links) and moved the architecture,
  performance, testing, benchmarking, profiling, and layout material into the
  guide.
- Added an mdBook user and contributor guide under `docs/`: getting started,
  core concepts, how-to recipes, the science of reproducibility and provenance,
  a reference section, and contributor documentation. Getting-started code is
  included from the compiled `first_model` example.
- Rewrote the crate-level rustdoc page around installation, features, the
  first-model path, determinism, and stability.

### Internal

- Added community and governance files: `CONTRIBUTING.md` (setup, checks,
  documentation, pull-request process, and the API-stability and release
  policies), `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`
  (private vulnerability reporting), `CITATION.cff`, `CODEOWNERS`, issue forms,
  and a pull-request template.
- Added GitHub Actions CI (third-party actions pinned to commit SHA):
  formatting, clippy, the explicit feature matrix, MSRV checks on Rust 1.87
  (library, no-features and all-features), benchmark compilation, strict
  rustdoc, `cargo package` with content assertions, and an external-consumer
  smoke build against the packaged artifact. A manual workflow runs the GPU
  tests on a self-hosted runner.
- Added a Pages workflow that builds the mdBook guide, checks internal links,
  and deploys it to GitHub Pages on the default branch; a weekly RustSec audit
  workflow; and Dependabot updates for Cargo and GitHub Actions.
- Pinned the development toolchain to Rust 1.91.1 (`rust-toolchain.toml`) with
  `rustfmt` and `clippy`; the library MSRV remains 1.87.
- Resolved strict `clippy` (`--all-targets --all-features -D warnings`) and
  strict rustdoc findings without relying on APIs newer than the MSRV.
- Replaced the packaging `exclude` list with an `include` allow-list and added
  crates.io `keywords`/`categories` and docs.rs metadata.

## 0.5.0 — 2026-07-05

Performance-focused release following a full code review. Steady-state
iteration, parallel scaling for mid-sized populations, bulk spawning,
despawn batching, message emission, and metadata memory were all reworked.
Several APIs changed shape; migration notes below.

Reference numbers from a 24-thread machine (release builds):

| Path | 0.4.0 | 0.5.0 |
| --- | --- | --- |
| 100k-agent compute-heavy system pass | 18.0 ms (single-threaded) | 2.4 ms (7.7× vs single thread) |
| 1M-agent trivial pass, generic `for_each` | 0.85 ms | 0.26 ms |
| 1M-agent trivial pass, `for_each_r1w1` | 0.14 ms | 0.10 ms |
| Spawn 1M agents × 3 components | 795 ms | 34.7 ms |
| Despawn 4096 agents (batched command) | 818 µs | 358 µs |
| Emit 1M brute-force messages | 28.5 ms | 2.4 ms (via `MessageEmitter`) |
| Entity back-map metadata per chunk | 256 KiB | 128 KiB |

### Performance

- **Row-range work planner.** Parallel iteration no longer floors task
  granularity at 8 chunks (131,072 rows). Chunks are split into row ranges
  targeting `2 × threads` tasks with a 2,048-row minimum, so populations from
  ~10k agents up scale across all cores. Applies to all `for_each` variants
  and reductions; activation-order semantics and fallible error determinism
  are unchanged.
- **Columnar bulk spawn.** `SpawnBatch` columns are single type-erased
  `Vec<T>`s copied into chunk storage in runs; entities are allocated in bulk
  (one lock per shard) and metadata committed in one pass. Batch application
  is atomic: any failure truncates appended columns and releases allocated
  handles.
- **Batched despawn.** `Command::DespawnBatchTagged` resolves and validates
  every handle up front, then removes rows per archetype in descending row
  order under once-per-batch column locks, with shard operations grouped.
- **`MessageEmitter`.** `MessageBufferSet::emitter(handle)` resolves the
  calling thread's staging slot once; `emit()` on the returned (`!Send`)
  emitter is a direct buffer push.
- **Static-dispatch `for_each`.** The generic tuple API now takes the closure
  by value with a `Fn(P::Item<'_>)` bound (GATs) instead of `&dyn Fn`,
  letting the compiler inline and vectorise per-row bodies.
- **Entity back-map halved.** Archetype row→entity metadata stores `Entity`
  with a placeholder sentinel instead of `Option<Entity>` (8 bytes/slot
  instead of 16). The sentinel bit pattern is made unreachable by skipping
  `VersionID::MAX` during handle recycling.
- **Chunk hysteresis.** Component columns keep one retired chunk as a spare,
  eliminating alloc/free churn when populations oscillate across a chunk
  boundary.
- **GPU dirty precision.** Structural commands invalidate only the archetypes
  they touched instead of every archetype.
- Query-match cache hands out `Arc<[ArchetypeID]>`; boundary handles cache
  their typed pointer at construction.

### Fixed

- `Attribute::swap_remove` / `take_swap_remove` return
  `AttributeError::Position` instead of panicking when `row >= CHUNK_CAP`
  aliases past the bounds check.
- Spatial queries whose circle lies entirely outside the grid return no
  messages instead of aliasing into unrelated rows through the flat cell
  index.
- `QueryBuilder::read_id` / `write_id` reject descriptors with no assigned
  `ComponentID` instead of silently querying component `0`.
- Borrow-conflict diagnostics report the actual holder under CAS contention.
- Reduction partials no longer `unwrap()` a possibly poisoned mutex.
- The `tick` benchmark no longer times world teardown; a steady-state
  variant was added for regression tracking.

### Added

- `DetRng`: deterministic, seed-keyed RNG for model systems
  (`DetRng::from_context(run_context, salt)`), replacing the orphaned and
  never-compiled thread-local RNG the docs used to describe. Thread-local
  RNGs are not reproducible under Rayon work stealing; `DetRng` is.
- Tuple iteration now covers every arity up to four reads and four writes
  (previously capped at two of each, with gaps such as 3-read + 1-write).
- `SpawnBatch` / `BatchColumn` re-exported at the crate root.
- Benchmarks: `parallel_scaling`, `spawn_batch_*`, `despawn_batch_4096`,
  `emit_per_call_1M` / `emit_cached_emitter_1M`.

### Breaking changes and migration

- **Generic `for_each` family** (`for_each`, `for_each_entity`,
  `for_each_entity_fallible`): closures are passed by value and the closure
  type is a second inferred parameter.

  ```rust
  // 0.4.0
  world.for_each::<(Read<Prod>, Write<Wealth>)>(q, &|(p, w)| { w.value += p.rate; })?;
  // 0.5.0
  world.for_each::<(Read<Prod>, Write<Wealth>), _>(q, |(p, w)| { w.value += p.rate; })?;
  ```

  Item shapes are unchanged (bare `Read<A>` yields `&A`; tuples yield tuples,
  including 1-tuples for `(Read<A>,)`).
- **`QueryBuilder::read_id` / `write_id`** return `ECSResult<Self>`; add `?`.
  `QueryComponent::from_desc` likewise returns a `Result`.
- **`AgentTemplateBuilder::with_component_factory`** is typed:
  `with_component_factory<T>(id, || T { .. })` instead of passing a boxed
  `DefaultFactory`. This lets templates derive columnar defaults for batch
  spawns.
- **`BatchColumn`** now carries `values: Box<dyn Any + Send>` (one `Vec<T>`
  per column) plus `len`, instead of `Vec<Box<dyn Any + Send>>`.
  `AgentBatch::set_column::<T>(id, Vec<T>)` is unchanged at the call site and
  is the recommended way to build batches.
- **`Command::DespawnBatchTagged`** is atomic: a stale or duplicate handle
  fails the whole batch before any despawn (previously it failed midway,
  leaving earlier despawns applied).
- `Entity` version space excludes `VersionID::MAX` (handles recycle through
  it transparently; only code that fabricated raw all-ones handles could
  notice).

## 0.4.0

Prior release. See git history.
