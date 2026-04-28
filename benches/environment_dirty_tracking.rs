use criterion::{criterion_group, criterion_main};

mod review_common;

criterion_group!(benches, review_common::environment_dirty_tracking_benchmark);
criterion_main!(benches);
