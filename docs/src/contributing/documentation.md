# Documentation

Syren has three documentation surfaces:

- **rustdoc** is the reference for the public Rust API.
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

## Code in the guide

Code blocks in the guide come from one of three places: a compiled example
included with `{{#include}}` and `ANCHOR` markers (the getting-started chapter
does this from `first_model`), a doctest compiled by rustdoc, or a `rust,ignore`
block used for an illustrative snippet.
