use rand::distributions::Standard;
use rand::prelude::Distribution;

use super::branch_hint;
use super::Goldilocks;
use crate::field::ExtField;
use crate::field::Field;
use crate::field::FieldOps;
use crate::field::FieldOpsAssigned;

#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord, Default)]
pub struct Goldilocks2(pub(crate) [Goldilocks; 2]);

impl FieldOps for Goldilocks2 {}
impl FieldOpsAssigned<Goldilocks2> for Goldilocks2 {}
impl FieldOps<Goldilocks, Goldilocks2> for Goldilocks2 {}
impl FieldOpsAssigned<Goldilocks> for Goldilocks2 {}

impl ExtField<Goldilocks> for Goldilocks2 {
    const E: usize = 2;
    fn as_slice(&self) -> &[Goldilocks] {
        &self.0
    }

    fn from_base_slice(e: &[Goldilocks]) -> Self {
        assert_eq!(e.len(), 2);
        Self([e[0], e[1]])
    }

    fn from_base_slice_parts(e: Vec<Goldilocks>, new_len: usize) -> Vec<Self> {
        let mut e = {
            let ptr = e.as_ptr() as *mut Goldilocks2;
            let new_len = e.len() / 2;
            std::mem::forget(e);
            unsafe { Vec::from_raw_parts(ptr, new_len, new_len) }
        };

        e.resize(new_len, Self::default());

        e
    }
}

impl Goldilocks2 {
    pub(crate) const fn from_base(e: Goldilocks) -> Self {
        let mut inner = [Goldilocks::ZERO; 2];
        inner[0] = e;
        Self(inner)
    }
}

impl From<bool> for Goldilocks2 {
    fn from(val: bool) -> Self {
        let val: Goldilocks = val.into();
        val.into()
    }
}

impl From<u64> for Goldilocks2 {
    fn from(val: u64) -> Self {
        let val: Goldilocks = val.into();
        val.into()
    }
}

impl From<u32> for Goldilocks2 {
    fn from(val: u32) -> Self {
        let val: Goldilocks = val.into();
        val.into()
    }
}

impl From<u8> for Goldilocks2 {
    fn from(val: u8) -> Self {
        let val: Goldilocks = val.into();
        val.into()
    }
}

impl From<Goldilocks> for Goldilocks2 {
    #[inline(always)]
    fn from(e: Goldilocks) -> Self {
        Self([e, Goldilocks::ZERO])
    }
}

impl From<[Goldilocks; 2]> for Goldilocks2 {
    #[inline(always)]
    fn from(inner: [Goldilocks; 2]) -> Self {
        Self(inner)
    }
}

impl From<Vec<Goldilocks>> for Goldilocks2 {
    #[inline(always)]
    fn from(inner: Vec<Goldilocks>) -> Self {
        let inner: [Goldilocks; 2] = inner.try_into().unwrap();
        inner.into()
    }
}

impl core::ops::Deref for Goldilocks2 {
    type Target = [Goldilocks];
    #[inline(always)]
    fn deref(&self) -> &[Goldilocks] {
        &self.0
    }
}

impl core::ops::DerefMut for Goldilocks2 {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [Goldilocks] {
        &mut self.0[..]
    }
}

impl Distribution<Goldilocks2> for Standard
where
    Standard: Distribution<Goldilocks>,
{
    #[inline(always)]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Goldilocks2 {
        Goldilocks2(std::array::from_fn(|_| rng.gen()))
    }
}

impl core::ops::Neg for Goldilocks2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self(std::array::from_fn(|i| -self.0[i]))
    }
}

impl core::ops::Add<Goldilocks2> for Goldilocks2 {
    type Output = Goldilocks2;
    #[inline(always)]
    fn add(self, rhs: Goldilocks2) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl core::ops::AddAssign for Goldilocks2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
    }
}

impl core::ops::Sub for Goldilocks2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl core::ops::SubAssign for Goldilocks2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
    }
}

impl core::ops::Mul for Goldilocks2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        // let v0 = self.0[0] * rhs.0[0];
        // let v1 = self.0[1] * rhs.0[1];
        // let u1 = (self.0[0] + self.0[1]) * (rhs.0[0] + rhs.0[1]) - (v0 + v1);
        // let u0 = v0 + v1.mul_by_nonresidue();
        // Goldilocks2([u0, u1])
        let c0 = ext2_add_prods0(&[self.0[0].0, self.0[1].0], &[rhs.0[0].0, rhs.0[1].0]);
        let c1 = ext2_add_prods1(&[self.0[0].0, self.0[1].0], &[rhs.0[0].0, rhs.0[1].0]);
        Goldilocks2([c0, c1])
    }
}

impl core::ops::Add<Goldilocks> for Goldilocks2 {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, rhs: Goldilocks) -> Self::Output {
        self.0[0] += rhs;
        self
    }
}

impl core::ops::AddAssign<Goldilocks> for Goldilocks2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Goldilocks) {
        self.0[0] += rhs;
    }
}

impl core::ops::Sub<Goldilocks> for Goldilocks2 {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, rhs: Goldilocks) -> Self::Output {
        self.0[0] -= rhs;
        self
    }
}

impl core::ops::SubAssign<Goldilocks> for Goldilocks2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Goldilocks) {
        self[0] -= rhs;
    }
}

impl core::ops::MulAssign for Goldilocks2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::Mul<Goldilocks> for Goldilocks2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Goldilocks) -> Self::Output {
        Goldilocks2([self.0[0] * rhs, self.0[1] * rhs])
    }
}

impl core::ops::MulAssign<Goldilocks> for Goldilocks2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Goldilocks) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
    }
}

impl core::iter::Sum for Goldilocks2 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<'a> core::iter::Sum<&'a Goldilocks2> for Goldilocks2 {
    fn sum<I: Iterator<Item = &'a Goldilocks2>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, &x| acc + x)
    }
}

impl core::iter::Product for Goldilocks2 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<'a> core::iter::Product<&'a Goldilocks2> for Goldilocks2 {
    fn product<I: Iterator<Item = &'a Goldilocks2>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, &x| acc * x)
    }
}

impl Field for Goldilocks2 {
    const ZERO: Self = Self::from_base(Goldilocks::ZERO);
    const ONE: Self = Self::from_base(Goldilocks::ONE);
    const NEG_ONE: Self = Self::from_base(Goldilocks::NEG_ONE);
    const TWO: Self = Self::from_base(Goldilocks::TWO);
    const TWO_INV: Self = Self::from_base(Goldilocks::TWO_INV);
    const NUM_BITS: usize = Goldilocks::NUM_BITS * 2;
    const GENERATOR: Self = Self([
        Goldilocks(0xfaeea08613fdecab),
        Goldilocks(0xdfbafc6bdb2869ae),
    ]);

    fn rand(mut rng: impl rand::RngCore) -> Self {
        std::array::from_fn(|_| Goldilocks::rand(&mut rng)).into()
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|e| e.is_zero())
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        (*self - Goldilocks::ONE).is_zero()
    }

    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let mut ret = Self::default();
        let v0 = self[0].square() - self[1].square() * Goldilocks(7);
        let t = v0.inverse().unwrap();
        ret[0] = self[0] * t;
        ret[1] = self[1] * -t;
        Some(ret)
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut ret = Self::ZERO;
        let mut off = 0;
        for e in ret.iter_mut() {
            *e = Goldilocks::from_bytes(&bytes[off..off + Goldilocks::NUM_BYTES])?;
            off += Goldilocks::NUM_BYTES;
        }
        Some(ret)
    }

    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        let els = bytes
            .chunks(Goldilocks::NUM_BYTES * 2)
            .map(Goldilocks::from_uniform_bytes)
            .chain(std::iter::repeat(Goldilocks::ZERO))
            .take(2)
            .collect::<Vec<_>>();
        els.into()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.iter().flat_map(|e| e.to_bytes()).collect()
    }
}

#[inline(always)]
unsafe fn add_no_canonicalize_trashing_input(x: u64, y: u64) -> u64 {
    let (res_wrapped, carry) = x.overflowing_add(y);
    // Below cannot overflow unless the assumption if x + y < 2**64 + ORDER is incorrect.
    res_wrapped + Goldilocks::NEGP * (carry as u64)
}

#[inline(always)]
pub(crate) unsafe fn reduce160(x_lo: u128, x_hi: u32) -> Goldilocks {
    let x_hi = (x_lo >> 96) as u64 + ((x_hi as u64) << 32); // shld to form x_hi
    let x_mid = (x_lo >> 64) as u32; // shr to form x_mid
    let x_lo = x_lo as u64;

    // sub + jc (should fuse)
    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi);
    if borrow {
        // The maximum possible value of x is (2^64 - 1)^2 * 4 * 7 < 2^133,
        // so x_hi < 2^37. A borrow will happen roughly one in 134 million
        // times, so it's best to branch.
        branch_hint();
        // NB: this assumes that x < 2^160 - 2^128 + 2^96.
        t0 -= Goldilocks::NEGP; // Cannot underflow if x_hi is canonical.
    }
    // imul
    let t1 = (x_mid as u64) * Goldilocks::NEGP;
    // add, sbb, add
    let t2 = add_no_canonicalize_trashing_input(t0, t1);
    t2.into()
}

#[inline(always)]
const fn u160_times_7(x: u128, y: u32) -> (u128, u32) {
    let (d, br) = (x << 3).overflowing_sub(x);
    (d, 7 * y + (x >> (128 - 3)) as u32 - br as u32)
}

#[inline(always)]
fn ext2_add_prods0(a: &[u64; 2], b: &[u64; 2]) -> Goldilocks {
    // Computes a0 * b0 + W * a1 * b1;
    let [a0, a1] = *a;
    let [b0, b1] = *b;

    let cy;

    // W * a1 * b1
    let (mut cumul_lo, mut cumul_hi) = u160_times_7((a1 as u128) * (b1 as u128), 0u32);

    // a0 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext2_add_prods1(a: &[u64; 2], b: &[u64; 2]) -> Goldilocks {
    // Computes a0 * b1 + a1 * b0;
    let [a0, a1] = *a;
    let [b0, b1] = *b;

    let cy;

    // a0 * b1
    let mut cumul_lo = (a0 as u128) * (b1 as u128);

    // a1 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b0 as u128));
    let cumul_hi = cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

impl Goldilocks2 {
    pub fn from_base_slice_parts(e: Vec<Goldilocks>) -> Vec<Goldilocks2> {
        let ptr = e.as_ptr() as *mut Goldilocks2;
        let new_len = e.len() / 2;
        std::mem::forget(e);
        unsafe { Vec::from_raw_parts(ptr, new_len, new_len) }
    }
}
