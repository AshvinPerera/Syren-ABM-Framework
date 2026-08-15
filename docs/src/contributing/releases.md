# Releases

Releases follow the process in
[`CONTRIBUTING.md`](https://github.com/AshvinPerera/Syren-ABM-Framework/blob/master/CONTRIBUTING.md).
This chapter summarises it and records the reasoning.

## Steps

1. **Version.** Update the version in `Cargo.toml` and keep `CITATION.cff` in sync
   with it. The citation must name the exact code a result came from.
2. **Changelog.** Finalise the entry for the version, including any breaking
   changes with migration notes and any trajectory changes for a fixed seed.
3. **Package inspection.** Run `cargo package --locked` and review
   `cargo package --list`. The archive must exclude local tooling, build output,
   profiling captures, and IDE settings; CI asserts this.
4. **Documentation.** Build rustdoc (all features, warnings denied) and the mdBook
   guide from the release commit.
5. **Full CI.** Run the whole pipeline from a clean checkout.
6. **Tag and release.** Tag the version and create the GitHub release.
7. **Publish.** `cargo publish`.
8. **DOI.** For a public, non-candidate release, mint a DOI and record it in
   `CITATION.cff`.

## Release candidates

A release candidate (`-rc.N`) carries out everything through building the package
and documentation and creating a **draft** release, but stops before tagging,
publishing to crates.io, and minting a DOI. These irreversible steps require a
separate approval.

## Versioning

Syren is pre-1.0. Patch releases do not break the public API; minor releases may,
with migration notes. See the [compatibility policy](../reference/compatibility.md)
and [API status](../reference/api-status.md).

## Documentation deployment

The guide is published to GitHub Pages from the default branch; docs.rs builds
the versioned API reference from the published crate. The two are independent: the
Pages site tracks the latest default-branch guide, and docs.rs tracks published
versions.
