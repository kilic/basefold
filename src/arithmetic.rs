use crate::field::{ExtField, Field};
use itertools::Itertools;

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
        acc = acc.inverse().unwrap();
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

impl<V: Field> VecOps<V> for Vec<V> {}
impl<V: Field> VecOps<V> for &[V] {}

pub trait VecOps<F: Field>: core::ops::Deref<Target = [F]> {
    fn hadamard<E: ExtField<F>>(&self, other: &[E]) -> Vec<E> {
        assert_eq!(self.len(), other.len());
        self.iter()
            .zip(other.iter())
            .map(|(&a, &b)| b * a)
            .collect()
    }

    fn dot<E: ExtField<F>>(&self, other: &[E]) -> E {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter()).map(|(&a, &b)| b * a).sum()
    }

    fn horner<E: ExtField<F>>(&self, x: E) -> E {
        self.iter().rfold(E::ZERO, |acc, &coeff| acc * x + coeff)
    }
}
