use itertools::Itertools;
use p3_field::ExtensionField;
use p3_field::Field;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub trait BatchInverse<F> {
    fn inverse(self) -> F;
}

impl<'a, F, I> BatchInverse<F> for I
where
    F: Field,
    I: IntoIterator<Item = &'a mut F>,
{
    fn inverse(self) -> F {
        let mut acc = F::ONE;
        let inter = self
            .into_iter()
            .map(|p| {
                let prev = acc;
                if !p.is_zero() {
                    acc *= *p
                }
                (prev, p)
            })
            .collect_vec();
        acc = acc.inverse();
        let prod = acc;
        for (mut tmp, p) in inter.into_iter().rev() {
            tmp *= acc;
            if !p.is_zero() {
                acc *= *p;
                *p = tmp;
            }
        }
        prod
    }
}

pub(crate) fn interpolate<F: Field, EF: ExtensionField<F>>(points: &[F], evals: &[EF]) -> Vec<EF> {
    assert_eq!(points.len(), evals.len());
    if points.len() == 1 {
        vec![evals[0]]
    } else {
        let mut denoms = Vec::with_capacity(points.len());
        points.iter().enumerate().for_each(|(j, x_j)| {
            let mut denom = Vec::with_capacity(points.len() - 1);
            points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .for_each(|x_k| denom.push(*x_j - *x_k));
            denoms.push(denom);
        });
        // Compute (x_j - x_k)^(-1) for each j != i
        denoms.iter_mut().flat_map(|v| v.iter_mut()).inverse();

        let mut final_poly = vec![EF::ZERO; points.len()];
        for (j, (denoms, eval)) in denoms.into_iter().zip(evals.iter()).enumerate() {
            let mut tmp: Vec<F> = Vec::with_capacity(points.len());
            let mut product = Vec::with_capacity(points.len() - 1);
            tmp.push(F::ONE);
            points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms.into_iter())
                .for_each(|(x_k, denom)| {
                    product.resize(tmp.len() + 1, F::ZERO);
                    tmp.iter()
                        .chain(core::iter::once(&F::ZERO))
                        .zip(core::iter::once(&F::ZERO).chain(tmp.iter()))
                        .zip(product.iter_mut())
                        .for_each(|((a, b), product)| *product = *a * (-denom * *x_k) + *b * denom);
                    core::mem::swap(&mut tmp, &mut product);
                });
            assert_eq!(tmp.len(), points.len());
            assert_eq!(product.len(), points.len() - 1);
            final_poly.iter_mut().zip(tmp.into_iter()).for_each(
                |(final_coeff, interpolation_coeff)| *final_coeff += *eval * interpolation_coeff,
            );
        }
        final_poly
    }
}

impl<V: Field> VecOps<V> for Vec<V> {}
impl<V: Field> VecOps<V> for &[V] {}
impl<V: Field> VecOps<V> for &mut [V] {}

pub trait VecOps<F: Field>: core::ops::Deref<Target = [F]> {
    fn hadamard<E: ExtensionField<F>>(&self, other: &[E]) -> Vec<E> {
        assert_eq!(self.len(), other.len());
        self.iter()
            .zip_eq(other.iter())
            .map(|(&a, &b)| b * a)
            .collect()
    }

    fn dot<E: ExtensionField<F>>(&self, other: &[E]) -> E {
        assert_eq!(self.len(), other.len());
        self.iter().zip_eq(other.iter()).map(|(&a, &b)| b * a).sum()
    }

    fn par_hadamard<E: ExtensionField<F>>(&self, other: &[E]) -> Vec<E> {
        assert_eq!(self.len(), other.len());
        self.par_iter()
            .zip_eq(other.par_iter())
            .map(|(&a, &b)| b * a)
            .collect()
    }

    fn par_dot<E: ExtensionField<F>>(&self, other: &[E]) -> E {
        assert_eq!(self.len(), other.len());
        self.par_iter()
            .zip_eq(other.par_iter())
            .map(|(&a, &b)| b * a)
            .sum()
    }

    fn horner<E: ExtensionField<F>>(&self, x: E) -> E {
        self.iter().rfold(E::ZERO, |acc, &coeff| acc * x + coeff)
    }
}
