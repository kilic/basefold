use basefold_study::mle::eq;
use criterion::*;
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Duration;

criterion_group!(benches, build_eq,);
criterion_main!(benches);

const NUM_SAMPLES: usize = 10;
const NUM_VARS: std::ops::Range<i32> = 20..24;

fn build_eq(c: &mut Criterion) {
    type F = BinomialExtensionField<Goldilocks, 2>;

    for nv in NUM_VARS {
        let mut group = c.benchmark_group(format!("{nv}"));
        group.sample_size(NUM_SAMPLES);

        let rng = SmallRng::seed_from_u64(1);
        let r = rng.random_iter().take(nv as usize).collect::<Vec<F>>();

        group.bench_function(BenchmarkId::new("eq", format!("num_var {nv}")), |b| {
            b.iter_custom(|iters| {
                let mut time = Duration::new(0, 0);
                for _ in 0..iters {
                    let instant = std::time::Instant::now();
                    let _ = eq(&r);
                    let elapsed = instant.elapsed();
                    time += elapsed;
                }
                time
            });
        });
        group.finish();
    }
}
