use crate::{
    data::MatrixOwn,
    utils::{unsafe_allocate_zero_vec, TwoAdicSlice, VecOps},
};
use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

pub fn eq<F: Field>(zs: &[F]) -> Vec<F> {
    eq_scaled(zs, F::ONE)
}

pub fn eq_scaled<F: Field>(zs: &[F], scale: F) -> Vec<F> {
    let k = zs.len();
    let mut eq = unsafe_allocate_zero_vec(1 << k);
    eq[0] = scale;
    for (i, &zi) in zs.iter().enumerate() {
        let (lo, hi) = eq.split_at_mut(1 << i);
        lo.par_iter_mut()
            .zip(hi.par_iter_mut())
            .for_each(|(lo, hi)| {
                *hi = *lo * zi;
                *lo -= *hi;
            });
    }
    eq
}

pub fn eq_tree<F: Field>(zs: &[F], scale: F) -> Vec<Vec<F>> {
    let k = zs.len();
    let mut eq = unsafe_allocate_zero_vec(1 << k);
    eq[0] = scale;
    let mut ladder = vec![vec![scale]];
    for (i, &zi) in zs.iter().enumerate() {
        let (lo, hi) = eq.split_at_mut(1 << i);
        lo.par_iter_mut()
            .zip(hi.par_iter_mut())
            .for_each(|(lo, hi)| {
                *hi = *lo * zi;
                *lo -= *hi;
            });
        ladder.push(eq[0..1 << (i + 1)].to_vec());
    }
    ladder
}

// pub fn eq_ladder<F: Field>(zs: &[F], scale: F) -> LadderVec<F> {
//     let k = zs.len();
//     let mut eq = vec![F::ZERO; 1 << k];
//     eq[0] = scale;
//     let mut ladder = LadderVec::new(k);
//     ladder.update_row(0, &eq[0..1]);
//     for (i, &zi) in zs.iter().enumerate() {
//         let (lo, hi) = eq.split_at_mut(1 << i);
//         lo.par_iter_mut()
//             .zip(hi.par_iter_mut())
//             .for_each(|(lo, hi)| {
//                 *hi = *lo * zi;
//                 *lo -= *hi;
//             });
//         // layout[off..off + lo.len()].copy_from_slice(&lo);
//         ladder.update_row(i + 1, &eq[0..1 << (i + 1)]);
//         // off += lo.len();
//     }
//     ladder
// }

pub fn fix_var<F: Field>(poly: &mut Vec<F>, zi: F) {
    let mid = poly.len() / 2;
    let (p0, p1) = poly.split_at_mut(mid);
    p0.par_iter_mut()
        .zip(p1.par_iter())
        .for_each(|(a0, a1)| *a0 += zi * (*a1 - *a0));
    poly.truncate(mid);
}

pub fn eval_poly<F: Field>(zs: &[F], poly: &[F]) -> F {
    assert_eq!(poly.k(), zs.len());
    *eval_poly_partial(zs, poly).first().unwrap()
}

pub fn eval_poly_partial<F: Field>(zs: &[F], poly: &[F]) -> Vec<F> {
    let k = poly.k();
    let d = zs.len();
    assert!(k >= d);
    let mut ml = poly.to_vec();

    for (i, &zi) in zs.iter().rev().enumerate() {
        let mid = 1 << (k - i - 1);
        let (lo, hi) = ml.split_at_mut(mid);
        lo.par_iter_mut()
            .zip(hi.par_iter())
            .for_each(|(a0, a1)| *a0 += zi * (*a1 - *a0));
        ml.truncate(mid);
    }
    assert_eq!(ml.k(), k - d);
    ml
}

pub fn eval_poly_sweet<F: Field>(zs: &[F], poly: &[F], sweetness: usize) -> F {
    assert_eq!(poly.k(), zs.len());

    let (z0, z1) = zs.split_at(poly.k().saturating_sub(sweetness));
    let eq0 = crate::mle::eq(z0);
    let eq1 = crate::mle::eq(z1);

    poly.chunks(eq0.len())
        .zip_eq(eq1.iter())
        .map(|(part, &c)| part.par_dot(&eq0) * c)
        .sum()
}

pub fn eval_mat<F: Field, E: ExtensionField<F>>(
    zs: &[E],
    mat: &MatrixOwn<F>,
    sweetness: usize,
) -> Vec<E> {
    let (z0, z1) = zs.split_at(mat.k().saturating_sub(sweetness));
    let eq0 = crate::mle::eq(z0);
    let eq1 = crate::mle::eq(z1);

    (0..mat.width())
        .map(|col| {
            mat.chunks(eq0.len())
                .zip_eq(eq1.iter())
                .map(|(part, &c)| {
                    part.par_iter()
                        .zip(eq0.par_iter())
                        .map(|(a, &b)| b * a[col])
                        .sum::<E>()
                        * c
                })
                .sum::<E>()
        })
        .collect()
}

pub fn eval_eq_xy<F: Field>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len());
    x.par_iter()
        .zip(y.par_iter())
        .map(|(&xi, &yi)| (xi * yi).double() - xi - yi + F::ONE)
        .product()
}

#[cfg(test)]
mod test {
    use p3_goldilocks::Goldilocks;
    use rand::Rng;

    use crate::{
        data::MatrixOwn,
        mle::{eval_mat, eval_poly_sweet},
        utils::{n_rand, VecOps},
    };

    #[test]
    fn test_eval_mat() {
        type F = Goldilocks;
        let k = 11;
        let width = 5;
        let mut rng = crate::test::seed_rng();
        // crate::test::init_tracing();

        let zs: Vec<F> = n_rand(&mut rng, k);
        let mat: Vec<F> = n_rand(&mut rng, (1 << k) * width);
        let mat = MatrixOwn::new(width, mat);

        let cols = mat.columns();
        let e0 = cols
            .iter()
            .map(|col| eval_poly_sweet(&zs, col, 3))
            .collect::<Vec<_>>();
        let e1 = eval_mat(&zs, &mat, 3);
        assert_eq!(e0, e1);
    }

    #[test]
    fn test_eval() {
        type F = Goldilocks;
        let k = 4;
        let mut rng = crate::test::seed_rng();
        // crate::test::init_tracing();

        let zs: Vec<F> = n_rand(&mut rng, k);
        let poly: Vec<F> = n_rand(&mut rng, 1 << k);

        let e0 = super::eval_poly(&zs, &poly);
        for sweetness in 0..k {
            let e1 = super::eval_poly_sweet(&zs, &poly, sweetness);
            assert_eq!(e0, e1);
        }

        let eq = super::eq(&zs);
        let e1 = poly.dot(&eq);
        assert_eq!(e0, e1);

        let beta = rng.random();
        let eq = super::eq_scaled(&zs, beta);
        let e1 = poly.dot(&eq);
        assert_eq!(e0 * beta, e1);
    }
}
