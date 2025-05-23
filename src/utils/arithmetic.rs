use itertools::Itertools;
use p3_field::ExtensionField;
use p3_field::Field;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::data::MatrixMut;
use crate::data::MatrixOwn;

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

pub fn ifft<F: Field>(mat: &mut MatrixOwn<F>, omega_inv: F) {
    let divisor = F::from_u64(1u64 << mat.k()).inverse();
    fft(mat, omega_inv);
    mat.par_iter_mut()
        .for_each(|a| a.iter_mut().for_each(|b| *b *= divisor));
}

pub fn fft<F: Field>(mat: &mut MatrixOwn<F>, omega: F) {
    let k = mat.k();
    let threads = rayon::current_num_threads();
    let log_threads = threads.ilog2() as usize;
    let n = mat.height();
    assert_eq!(n, 1 << k);
    mat.reverse_bits();

    let twiddles: Vec<_> = std::iter::successors(Some(F::ONE), |&x| Some(x * omega))
        .take(n >> 1)
        .collect();

    if k <= log_threads {
        let mut _k = 1;
        let mut twiddle_chunk = n >> 1;
        for _ in 0..k {
            mat.chunks_mut(1 << _k).for_each(|mut coeffs| {
                let (mut left, mut right) = coeffs.split_mut(_k - 1);
                left.iter_mut()
                    .zip(right.iter_mut())
                    .enumerate()
                    .for_each(|(i, (a, b))| {
                        a.iter_mut().zip(b.iter_mut()).for_each(|(a, b)| {
                            let t = *b * twiddles[i * twiddle_chunk];
                            *b = *a;
                            *a += t;
                            *b -= t;
                        });
                    });
            });
            _k += 1;
            twiddle_chunk /= 2;
        }
    } else {
        recursive_butterfly_arithmetic(&mut mat.as_mut(), k, 1, &twiddles)
    }
}

pub fn recursive_butterfly_arithmetic<F: Field>(
    mat: &mut MatrixMut<F>,
    k: usize,
    twiddle_chunk: usize,
    twiddles: &[F],
) {
    if k == 1 {
        let (mut left, mut right) = mat.split_mut(k - 1);
        let a = left.row_mut(0);
        let b = right.row_mut(0);

        a.iter_mut().zip(b.iter_mut()).for_each(|(a, b)| {
            let t = *b;
            *b = *a;
            *a += t;
            *b -= t;
        });
    } else {
        let (mut left, mut right) = mat.split_mut(k - 1);
        rayon::join(
            || recursive_butterfly_arithmetic(&mut left, k - 1, twiddle_chunk * 2, twiddles),
            || recursive_butterfly_arithmetic(&mut right, k - 1, twiddle_chunk * 2, twiddles),
        );
        left.iter_mut()
            .zip(right.iter_mut())
            .enumerate()
            .for_each(|(i, (a, b))| {
                a.iter_mut().zip(b.iter_mut()).for_each(|(a, b)| {
                    let t = *b * twiddles[i * twiddle_chunk];
                    *b = *a;
                    *a += t;
                    *b -= t;
                });
            });
    }
}

#[cfg(test)]
mod test {
    impl<S: Storage> Matrix<S> {
        pub fn eval<V, U>(&self, point: U) -> Vec<U>
        where
            S: AsRef<[V]>,
            V: Clone + Copy,
            U: core::ops::MulAssign<U> + core::ops::AddAssign<V>,
            U: Default + Clone + Copy,
        {
            let mut acc = vec![U::default(); self.width()];
            self.iter().rev().for_each(|row| {
                acc.iter_mut().for_each(|acc| *acc *= point);
                acc.iter_mut()
                    .zip(row.iter())
                    .for_each(|(acc, &row)| *acc += row);
            });
            acc
        }
    }

    use crate::data::{Matrix, MatrixOwn, Storage};
    use p3_field::Field;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::TwoAdicField;

    #[test]
    fn test_fft_roundtrip() {
        type F = p3_goldilocks::Goldilocks;
        let width = 4;
        let k = 5;

        let omega = F::two_adic_generator(k);
        let omega_inv = omega.inverse();

        let rng = &mut crate::test::seed_rng();
        let coeffs = MatrixOwn::<F>::rand(rng, width, 1 << k);

        let mut evals = coeffs.clone();
        super::fft(&mut evals, omega);

        let domain = omega.powers().take(1 << k).collect::<Vec<_>>();

        domain
            .iter()
            .zip(evals.iter())
            .for_each(|(&w, e0)| assert_eq!(e0.to_vec(), coeffs.eval(w)));

        let mut coeffs1 = evals.clone();
        super::ifft(&mut coeffs1, omega_inv);
        assert_eq!(coeffs, coeffs1);
    }
}
