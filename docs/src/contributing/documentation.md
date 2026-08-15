# Documentation

Syren has three documentation surfaces, each with a distinct job:

- **rustdoc** is the authority for the public Rust API. Build it with all
  features and warnings denied.
- **This mdBook guide** (under `docs/`) is the user and contributor manual.
- The **examples** carry their own documentation next to their sources.

## Building

```bash
# API reference
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --open

# The guide
mdbook build docs
mdbook serve docs   # live preview at http://localhost:3000
```

## Rules

The single most important rule: **documentation describes the code as it is.** Do
not describe past behaviour, migration history, or implementation phases anywhere
in doc comments, guide prose, or code comments — the changelog is the place for
change history.

Beyond that:

- Map every statement to current source types, module docs, or tests before
  writing it.
- State the required Cargo features beside examples, and state limits beside
  guarantees.
- Quote a performance number only with its hardware, compiler, features, build
  profile, population, and command.
- Prefer present tense and factual wording. Avoid "fast", "easy", or
  "production-ready" unless a measurement defines the term.
- In the guide, link type and method names to their rustdoc rather than
  duplicating signatures, which drift.

## Code in the guide

Every code block in the guide is one of:

- **included from a compiled example** with `{{#include}}` and `ANCHOR` markers,
  so it always matches working code (the getting-started chapter does this from
  `first_model`),
- **a doctest** compiled by rustdoc, or
- **explicitly marked pseudocode** (a `rust,ignore` block) when it illustrates a
  pattern rather than a runnable program.

Prefer the first two. A snippet that compiles cannot go stale silently.

## Output schemas

Give each output schema a single source of truth — the column names defined once,
next to the row builder — and test that the header matches the row. See
`examples/macroeconomy/output.rs`.
