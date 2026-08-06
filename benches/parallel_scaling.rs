//! Parallel scaling of a compute-heavy system across population sizes.
//!
//! Guards the work-planner's granularity: a ~186 ns/agent kernel must scale
//! with core count even when the population spans only a handful of chunks.
//! The original grain floor (8 chunks = 131,072 rows per task) ran 100k-agent
//! systems on a single thread; this bench would have caught that.

use criterion::*;

mod common;
use common::*;

/// Dependent floating-point kernel, ~64 fp ops per agent (~186 ns single
/// threaded on the reference machine).
#[inline]
fn heavy(x: f32) -> f32 {
    let mut acc = x;
    for _ in 0..64 {
        acc = acc.mul_add(1.000_1, 0.000_3).sqrt().max(0.01);
    }
    acc
}

fn scaling_benchmark(c: &mut Criterion) {
    let (registry, pos_id, wealth_id, prod_id) = make_registry();

    let mut group = c.benchmark_group("parallel_scaling");
    group.sample_size(20);

    for &agents in &[100_000usize, 500_000, 1_000_000] {
        group.throughput(Throughput::Elements(agents as u64));
        group.bench_function(BenchmarkId::new("heavy_kernel", agents), |b| {
            let ecs = make_world(4, registry.clone());
            populate(&ecs, agents, pos_id, wealth_id, prod_id).unwrap();

            let query = query_builder(&registry)
                .read::<Productivity>()
                .unwrap()
                .write::<Wealth>()
                .unwrap()
                .build()
                .unwrap();
            let world = ecs.world_ref();

            b.iter(|| {
                world
                    .for_each_r1w1::<Productivity, Wealth>(query.clone(), |p, w| {
                        w.value = heavy(w.value + p.rate);
                    })
                    .unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, scaling_benchmark);
criterion_main!(benches);
