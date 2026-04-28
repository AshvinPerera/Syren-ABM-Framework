use criterion::{criterion_group, criterion_main};

mod review_common;

criterion_group!(benches, review_common::scheduler_packing_benchmark);
criterion_main!(benches);
