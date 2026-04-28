use criterion::{criterion_group, criterion_main};

mod review_common;

criterion_group!(benches, review_common::gpu_dispatch_poll_benchmark);
criterion_main!(benches);
