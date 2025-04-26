pub mod goldilocks;

pub trait FieldOps<Rhs = Self, Output = Self>:
    Copy
    + Default
    + Send
    + Sync
    + core::ops::Add<Rhs, Output = Output>
    + core::ops::Sub<Rhs, Output = Output>
    + core::ops::Mul<Rhs, Output = Output>
{
}

pub trait FieldOpsAssigned<Rhs = Self>:
    Copy
    + Default
    + Send
    + Sync
    + Sized
    + core::ops::AddAssign<Rhs>
    + core::ops::SubAssign<Rhs>
    + core::ops::MulAssign<Rhs>
{
}

pub trait Field:
    Sized
    + core::ops::Neg<Output = Self>
    + FieldOps
    + FieldOpsAssigned
    + Eq
    + Copy
    + Clone
    + Default
    + Send
    + Sync
    + std::fmt::Debug
    + 'static
    + From<bool>
    + From<u64>
    + From<u32>
    + From<u8>
    + core::iter::Sum
    + core::iter::Product
    + for<'a> core::iter::Sum<&'a Self>
    + for<'a> core::iter::Product<&'a Self>
{
    const ZERO: Self;
    const ONE: Self;
    const NEG_ONE: Self;
    const TWO: Self;
    const TWO_INV: Self;
    const NUM_BITS: usize;
    const NUM_BYTES: usize = Self::NUM_BITS.div_ceil(8);
    const GENERATOR: Self;

    fn rand(rng: impl rand::RngCore) -> Self;

    #[inline(always)]
    fn double(&self) -> Self {
        *self + *self
    }

    #[inline(always)]
    fn double_assign(&mut self) {
        *self += *self;
    }

    #[inline(always)]
    fn square(&self) -> Self {
        *self * *self
    }

    #[inline(always)]
    fn square_assign(&mut self) {
        *self *= *self;
    }

    #[inline(always)]
    fn cube(&self) -> Self {
        *self * *self * *self
    }

    #[inline(always)]
    fn cube_assign(&mut self) {
        *self = *self * *self * *self;
    }

    fn inverse(&self) -> Option<Self>;

    fn is_zero(&self) -> bool;

    fn is_one(&self) -> bool;

    fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::ONE;
        for e in exp.as_ref().iter().rev() {
            for i in (0..64).rev() {
                res = res.square();
                (((*e >> i) & 1) == 1).then(|| res.mul_assign(*self));
            }
        }
        res
    }

    #[inline]
    fn vanishing(&self, k: usize) -> Self {
        self.pow2(k) - Self::ONE
    }

    #[inline]
    fn shifted_vanishing(&self, k: usize, shift: Self) -> Self {
        self.pow2(k) - shift.pow2(k)
    }

    #[inline]
    fn pow2(self, n: usize) -> Self {
        (0..n).fold(self, |acc, _| acc.square())
    }

    fn powers(&self) -> impl Iterator<Item = Self> {
        std::iter::successors(Some(Field::ONE), |&acc| Some(acc * *self))
    }

    fn div_two(&self) -> Self {
        *self * Self::TWO_INV
    }

    fn horner<'b, EF: Field, I>(&self, coeffs: I) -> Self
    where
        I: IntoIterator<Item = &'b EF>,
        I::IntoIter: DoubleEndedIterator,
        Self: std::ops::Add<EF, Output = Self>,
    {
        coeffs
            .into_iter()
            .rfold(Self::ZERO, |acc, &coeff| acc * *self + coeff)
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self>;
    fn from_uniform_bytes(bytes: &[u8]) -> Self;
    fn to_bytes(&self) -> Vec<u8>;
}

pub trait TwoAdicField: Field {
    const ROOT_OF_UNITY: Self;
    const TWO_ADICITY: usize;

    #[inline]
    fn omega(k: usize) -> Self {
        let t = Self::TWO_ADICITY.checked_sub(k).unwrap();
        Self::ROOT_OF_UNITY.pow2(t)
    }

    fn mul_subgroup(k: usize, shift: Self) -> Vec<Self> {
        Self::omega(k)
            .powers()
            .take(1 << k)
            .map(|v| v * shift)
            .collect()
    }
}

pub trait ExtField<F: Field>: Field + FieldOps<F, Self> + FieldOpsAssigned<F> + From<F> {
    const E: usize;
    fn as_slice(&self) -> &[F];
    fn from_base_slice(e: &[F]) -> Self;
    fn from_base_slice_parts(e: Vec<F>, new_size: usize) -> Vec<Self>;
}

impl<F: Field> ExtField<F> for F {
    const E: usize = 1;
    fn as_slice(&self) -> &[F] {
        std::slice::from_ref(self)
    }

    fn from_base_slice(e: &[F]) -> Self {
        assert_eq!(e.len(), 1);
        e[0]
    }

    fn from_base_slice_parts(e: Vec<F>, new_len: usize) -> Vec<Self> {
        let mut e: Vec<Self> = e;
        e.resize(new_len, Self::default());
        e
    }
}
