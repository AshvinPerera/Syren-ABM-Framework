# Repository layout

## Top level

| Path | Contents |
| --- | --- |
| `src/` | The library. |
| `examples/` | `first_model`, Sugarscape, and the macroeconomy model. |
| `tests/` | Integration tests. |
| `benches/` | Criterion benchmarks. |
| `docs/` | This mdBook guide (`book.toml`, `src/`). |
| `ci/` | The external-consumer smoke crate used by CI. |
| `.github/` | CI workflows, issue forms, and the pull-request template. |

Root files include `Cargo.toml`, `CHANGELOG.md`, `CONTRIBUTING.md`,
`CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, `CODEOWNERS`, the licence,
and `rust-toolchain.toml`.

## Inside `src/`

- **`engine/`** — the core ECS. Notable modules: `component` (registry and
  descriptors), `storage` and `archetype` (columnar storage and migration),
  `entity` (handles and shards), `query` (query building and resolution),
  `systems` (systems and access sets), `scheduler` (stage packing and execution),
  `activation` (the run context), `random` (`DetRng`), `reduce` (the
  accumulators), `boundary`, `commands` (deferred structural mutation), and
  `manager` (the world and its reference).
- **`model/`** — `ModelBuilder`, `Model`, sub-schedulers, and nested models.
- **`agents/`** — agent templates and lifecycle hooks.
- **`environment/`** — typed environment values and keys.
- **`messaging/`** — the message registry and the four specialisations.
- **`space/`** — the discrete grid and continuous space.
- **`gpu/`** — GPU context, mirroring, and dispatch.
- **`profiling/`** — tracing spans and Chrome Trace output.

The public surface is re-exported from `lib.rs`, with lower-level building blocks
under the `advanced` module.

## Examples are self-contained

Each example lives under its own directory (or file, for `first_model`) with its
sources, tests, and documentation together. The macroeconomy example, in
particular, keeps its model sources, equation mapping, parameters, deviations,
and limitations under `examples/macroeconomy/`.
