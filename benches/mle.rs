use basefold_study::{mle, utils::n_rand};
use criterion::*;
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;
use rand::{rngs::SmallRng, SeedableRng};

criterion_group!(benches, bench_eq, bench_eval_poly);
criterion_main!(benches);

type F = Goldilocks;
type Ext = BinomialExtensionField<F, 2>;

fn bench_eq(c: &mut Criterion) {
    const NUM_VARS: std::ops::Range<usize> = 23..24;
    let mut rng = SmallRng::seed_from_u64(1);

    let mut group = c.benchmark_group("eq");
    group.sample_size(10);
    for k in NUM_VARS {
        let zs: Vec<Ext> = n_rand(&mut rng, k);
        group.bench_function(BenchmarkId::new("eq", format!("k={k}")), |b| {
            b.iter(|| mle::eq(black_box(&zs)))
        });
    }
    group.finish();
}

fn bench_eval_poly(c: &mut Criterion) {
    const NUM_VARS: std::ops::Range<usize> = 23..24;
    let mut rng = SmallRng::seed_from_u64(1);

    let poly: Vec<Ext> = n_rand(&mut rng, 1 << NUM_VARS.last().unwrap());

    let mut group = c.benchmark_group("eval_poly");
    group.sample_size(10);

    for k in NUM_VARS {
        let zs: Vec<Ext> = n_rand(&mut rng, k);
        let poly = &poly[..(1 << k)];

        group.bench_function(BenchmarkId::new("naive par", format!("k={k}")), |b| {
            b.iter(|| mle::eval_poly(black_box(&zs), poly))
        });

        for split in 0..=k {
            group.bench_function(
                BenchmarkId::new("split", format!("k={k}, split={split}")),
                |b| {
                    b.iter(|| {
                        let split_eq = mle::SplitEq::new(black_box(&zs), split);
                        let _ = split_eq.eval_poly(&poly);
                    })
                },
            );
        }
    }
    group.finish();
}
