use criterion::{criterion_group, criterion_main};

mod review_common;

criterion_group!(benches, review_common::gpu_readback_benchmark);
criterion_main!(benches);
