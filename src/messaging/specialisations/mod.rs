//! Concrete message-buffer specialisations.
//!
//! Each sub-module implements one storage and indexing strategy:
//!
//! | Module | Buffer type | Query | Best for |
//! |---|---|---|---|
//! | [`brute_force`] | flat [`AlignedBuffer`] | iterate all | broadcast / global events |
//! | [`bucket`] | counting-sorted by key | one bucket | integer-keyed categories |
//! | [`spatial`] | counting-sorted by cell | radius circle | positional queries |
//! | [`targeted`] | sorted by full [`Entity`] | per-entity inbox | point-to-point messages |

pub(crate) mod brute_force;
pub(crate) mod bucket;
pub(crate) mod spatial;
pub(crate) mod targeted;

pub use brute_force::BruteForceIter;
pub use bucket::BucketIter;
pub use spatial::SpatialQueryIter;
pub use targeted::InboxIter;
