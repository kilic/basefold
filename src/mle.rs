use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    data::MatrixOwn,
    field::{ExtField, Field},
    utils::TwoAdicSlice,
};

pub fn eq<F: Field>(points: &[F]) -> Vec<F> {
    let k = points.len();
    let mut ml = vec![F::ZERO; 1 << k];
    ml[0] = F::ONE;
    for (i, &point) in points.iter().enumerate() {
        let (lo, hi) = ml.split_at_mut(1 << i);
        lo.par_iter_mut()
            .zip(hi.par_iter_mut())
            .for_each(|(lo, hi)| {
                *hi = *lo * point;
                *lo -= *hi;
            });
    }
    ml
}

pub fn eval_mat<F: Field, E: ExtField<F>>(mat: &MatrixOwn<F>, point: &[E]) -> Vec<E> {
    assert_eq!(point.len(), mat.k());
    let eq = eq(point);

    eq.par_iter()
        .zip(mat.par_iter())
        .map(|(&coeff, row)| row.iter().map(|&e| coeff * e).collect::<Vec<_>>())
        .reduce(
            || vec![E::ZERO; mat.k()],
            |a, b| a.iter().zip(b.iter()).map(|(&a, &b)| a + b).collect(),
        )
}

pub fn eval_eq_xy<F: Field>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len());
    // TODO: make it single mul
    x.par_iter()
        .zip(y.par_iter())
        .map(|(&xi, &yi)| xi * yi + (F::ONE - xi) * (F::ONE - yi))
        .product()
}

pub fn eval<F: Field>(poly: &[F], point: &[F]) -> F {
    assert_eq!(point.len(), poly.k());
    let eq = eq(point);
    eq.par_iter()
        .zip(poly.par_iter())
        .map(|(&a, &b)| a * b)
        .sum()
}

pub fn eval_partial<F: Field>(poly: &[F], point: &[F]) -> Vec<F> {
    assert!(point.len() <= poly.k());
    unimplemented!()
}
