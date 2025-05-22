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

pub fn eval_poly<F: Field>(zs: &[F], poly: &[F]) -> F {
    assert_eq!(poly.k(), zs.len());
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
    ml[0]
}

pub fn fix_var<F: Field>(poly: &mut Vec<F>, zi: F) {
    let mid = poly.len() / 2;
    let (p0, p1) = poly.split_at_mut(mid);
    p0.par_iter_mut()
        .zip(p1.par_iter())
        .for_each(|(a0, a1)| *a0 += zi * (*a1 - *a0));
    poly.truncate(mid);
}

pub struct SplitEq<F: Field, Ext: ExtensionField<F>> {
    pub(crate) left: Vec<Ext>,
    pub(crate) right: Vec<Ext>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> SplitEq<F, Ext> {
    pub fn new(zs: &[Ext], split: usize) -> Self {
        let (z0, z1) = zs.split_at(zs.len() - split);
        let left = crate::mle::eq(z0);
        let right = crate::mle::eq(z1);
        Self {
            left,
            right,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn eval_poly(&self, poly: &[F]) -> Ext {
        assert_eq!(poly.k(), self.left.k() + self.right.k());
        poly.chunks(self.left.len())
            .zip_eq(self.right.iter())
            .map(|(part, &c)| part.par_dot(&self.left) * c)
            .sum()
    }

    pub fn eval_mat(&self, mat: &MatrixOwn<F>) -> Vec<Ext> {
        (0..mat.width())
            .map(|col| {
                mat.chunks(self.left.len())
                    .zip_eq(self.right.iter())
                    .map(|(part, &c)| {
                        part.par_iter()
                            .zip(self.left.par_iter())
                            .map(|(a, &b)| b * a[col])
                            .sum::<Ext>()
                            * c
                    })
                    .sum::<Ext>()
            })
            .collect()
    }
}

pub(crate) fn eq_tree<F: Field>(zs: &[F], scale: F) -> Vec<Vec<F>> {
    let k = zs.len();
    let mut eq = unsafe_allocate_zero_vec(1 << k);
    eq[0] = scale;
    let mut tree = vec![vec![scale]];
    for (i, &zi) in zs.iter().enumerate() {
        let (lo, hi) = eq.split_at_mut(1 << i);
        lo.par_iter_mut()
            .zip(hi.par_iter_mut())
            .for_each(|(lo, hi)| {
                *hi = *lo * zi;
                *lo -= *hi;
            });
        tree.push(eq[0..1 << (i + 1)].to_vec());
    }
    tree
}

pub struct SplitEqTree<F: Field, Ext: ExtensionField<F>> {
    pub(crate) left: Vec<Vec<Ext>>,
    pub(crate) right: Vec<Vec<Ext>>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> SplitEqTree<F, Ext> {
    pub fn new(zs: &[Ext], split: usize) -> Self {
        let (z0, z1) = zs.split_at(zs.len() - split);
        let left = crate::mle::eq_tree(z0, Ext::ONE);
        let right = crate::mle::eq_tree(z1, Ext::ONE);
        Self {
            left,
            right,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn eval_mat(&self, mat: &MatrixOwn<F>) -> Vec<Ext> {
        let left = self.left.last().unwrap();
        let right = self.right.last().unwrap();
        (0..mat.width())
            .map(|col| {
                mat.chunks(left.len())
                    .zip_eq(right.iter())
                    .map(|(part, &c)| {
                        part.par_iter()
                            .zip(left.par_iter())
                            .map(|(a, &b)| b * a[col])
                            .sum::<Ext>()
                            * c
                    })
                    .sum::<Ext>()
            })
            .collect()
    }

    pub fn pop_left(&mut self) -> Vec<Ext> {
        self.left.pop().unwrap()
    }

    pub fn pop_right(&mut self) -> Vec<Ext> {
        self.right.pop().unwrap()
    }

    pub fn dot(&mut self, poly: &[Ext]) -> Ext {
        if !self.right.is_empty() {
            let eq1 = &self.pop_right();

            // if eq1 exhausted, drain eq0
            let eq0 = if self.right.is_empty() {
                &self.pop_left()
            } else {
                self.left.last().unwrap()
            };

            poly.chunks(eq0.len())
                .zip_eq(eq1.iter())
                .map(|(part, &c)| part.par_dot(eq0) * c)
                .sum()
        } else {
            poly.par_dot(&self.pop_left())
        }
    }
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
    use crate::{
        data::MatrixOwn,
        utils::{n_rand, VecOps},
    };
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;
    use rand::Rng;

    #[test]
    fn test_eq() {
        type F = Goldilocks;
        let k = 4;
        let mut rng = crate::test::seed_rng();

        let zs: Vec<F> = n_rand(&mut rng, k);
        let poly: Vec<F> = n_rand(&mut rng, 1 << k);

        let e0 = super::eq(&zs).dot(&poly);

        let beta = rng.random();
        let e1 = super::eq_scaled(&zs, beta).dot(&poly);
        assert_eq!(e0 * beta, e1);
    }

    #[test]
    fn test_eval() {
        type F = BinomialExtensionField<Goldilocks, 2>;

        let k = 25;
        let mut rng = crate::test::seed_rng();
        // crate::test::init_tracing();
        let zs: Vec<F> = n_rand(&mut rng, k);
        let poly = n_rand(&mut rng, 1 << k);

        let e0 = tracing::info_span!("naive eval").in_scope(|| super::eval_poly(&zs, &poly));
        let e1 = tracing::info_span!("eq mul").in_scope(|| super::eq(&zs).dot(&poly));
        assert_eq!(e0, e1);
        for split in 0..k / 2 {
            let e1 = tracing::info_span!("split", split).in_scope(|| {
                let split_eq = super::SplitEq::new(&zs, split);
                split_eq.eval_poly(&poly)
            });
            assert_eq!(e0, e1);
        }

        let width = 3;
        let mat: Vec<F> = n_rand(&mut rng, (1 << k) * width);
        let mat = MatrixOwn::new(width, mat);

        let cols = mat.columns();
        let e0 = cols
            .iter()
            .map(|col| super::eval_poly(&zs, col))
            .collect::<Vec<_>>();

        for split in 0..k / 2 {
            let e1 = tracing::info_span!("mat", split).in_scope(|| {
                let split_eq = super::SplitEq::new(&zs, split);
                split_eq.eval_mat(&mat)
            });
            assert_eq!(e0, e1);
        }
    }
}

// pub fn eq_tree2<F: Field>(zs: &[F], scale: F) -> LadderVec<F> {
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
