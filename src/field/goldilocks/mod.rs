pub mod ext;
pub use ext::*;

use super::{FieldOps, FieldOpsAssigned, TwoAdicField};
use crate::{field::Field, split128};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use std::{arch::asm, fmt::Debug};

#[derive(Copy, Clone, Default)]
pub struct Goldilocks(pub(crate) u64);

impl Debug for Goldilocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "{:016x?}", self.0)
        write!(f, "{}", self.0)
    }
}

// impl Extended<2> for Goldilocks {
//     const GENERATOR: [Self; 2] = [Self(0xfaeea08613fdecab), Self(0xdfbafc6bdb2869ae)];
//     const NON_RESIDUE: Self = Self(7);
// }

impl FieldOpsAssigned for Goldilocks {}
impl FieldOps for Goldilocks {}

#[inline(always)]
pub fn branch_hint() {
    unsafe {
        asm!("", options(nomem, nostack, preserves_flags));
    }
}

#[inline(always)]
pub(crate) fn reduce_wide(x: u128) -> Goldilocks {
    // x_lo + 2^64 * x_hi
    // x_lo + (2^32 - 1) * x_hi_lo - x_hi_hi

    let (x_lo, x_hi) = split128!(x);
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & Goldilocks::NEGP;

    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        branch_hint(); // A borrow is exceedingly rare. It is faster to branch.
        t0 -= Goldilocks::NEGP;
    }
    let t1 = x_hi_lo * Goldilocks::NEGP;

    let (res_wrapped, carry) = t0.overflowing_add(t1);
    let t2 = res_wrapped + Goldilocks::NEGP * u64::from(carry);

    Goldilocks(t2)
}

impl Goldilocks {
    pub const P: u64 = 0u64.wrapping_sub(1 << 32) + 1;
    pub const NEGP: u64 = (1 << 32) - 1;

    #[inline(always)]
    pub const fn value(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    pub const fn u64(&self) -> u64 {
        if self.0 >= Self::P {
            self.0 - Self::P
        } else {
            self.0
        }
    }

    #[inline(always)]
    pub const fn from_u64(e: u64) -> Self {
        // assert!(e < Self::P);
        Self(e)
    }

    #[inline(always)]
    pub const fn is_in_field(&self) -> bool {
        self.value() < Self::P
    }
}

impl Eq for Goldilocks {}

impl PartialEq for Goldilocks {
    fn eq(&self, other: &Self) -> bool {
        self.u64() == other.u64()
    }
}

impl Ord for Goldilocks {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.u64().cmp(&other.u64())
    }
}

impl PartialOrd for Goldilocks {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl From<u64> for Goldilocks {
    fn from(val: u64) -> Self {
        Self(val)
    }
}

impl From<u32> for Goldilocks {
    fn from(val: u32) -> Self {
        Self(val as u64)
    }
}

impl From<u8> for Goldilocks {
    fn from(val: u8) -> Self {
        Self(val as u64)
    }
}

impl From<bool> for Goldilocks {
    fn from(val: bool) -> Self {
        Self(val as u64)
    }
}

impl<T> From<&T> for Goldilocks
where
    T: Copy + Into<u64>,
{
    fn from(val: &T) -> Self {
        Self((*val).into())
    }
}

impl Distribution<Goldilocks> for Standard {
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Goldilocks {
        Goldilocks::rand(rng)
    }
}

impl TwoAdicField for Goldilocks {
    const ROOT_OF_UNITY: Self = Self(0x185629dcda58878c);
    const TWO_ADICITY: usize = 32;
}

impl Field for Goldilocks {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    const NEG_ONE: Self = Self(Self::P - 1);
    const TWO: Self = Self(2);
    const TWO_INV: Self = Self(0x7fffffff80000001);
    const NUM_BITS: usize = 64;
    const GENERATOR: Self = Self(7);

    fn rand(mut rng: impl rand::RngCore) -> Self {
        loop {
            let e = Self(rng.next_u64());
            if e.is_in_field() {
                return e;
            }
        }
    }

    fn is_zero(&self) -> bool {
        self.0 == 0 || self.0 == Self::P
    }

    fn is_one(&self) -> bool {
        self.0 == 1 || self.0 == Self::P + 1
    }

    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(self.pow([Self::P - 2]))
        }
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let el = Self(u64::from_le_bytes(bytes.try_into().unwrap()));
        el.is_in_field().then_some(el)
    }

    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        // let mut buf = vec![0u8; 16];
        // let off = std::cmp::min(16, bytes.len());
        // buf[..off].copy_from_slice(&bytes[..off]);
        // let wide = u128::from_le_bytes(buf.try_into().unwrap());
        // reduce_wide(wide)
        let mut buf = vec![0u8; 8];
        let off = std::cmp::min(8, bytes.len());
        buf[..off].copy_from_slice(&bytes[..off]);
        let wide = u64::from_le_bytes(buf.try_into().unwrap());
        Self::from_u64(wide)
        // reduce_wide(wide)
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }
}

// impl_add!(Goldilocks);
impl core::ops::Add<Goldilocks> for Goldilocks {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Goldilocks) -> Self {
        let (sum, over) = self.0.overflowing_add(rhs.0);

        let (mut sum, over) = sum.overflowing_add(u64::from(over) * Self::NEGP);

        if over {
            branch_hint();

            debug_assert!(self.0 > Self::P && rhs.0 > Self::P);
            sum += Self::NEGP;
        }
        Self(sum)
    }
}

impl core::ops::AddAssign for Goldilocks {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::Sub<Goldilocks> for Goldilocks {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let (diff, under) = self.0.overflowing_sub(rhs.0);
        let (mut diff, under) = diff.overflowing_sub(u64::from(under) * Self::NEGP);
        if under {
            branch_hint();

            debug_assert!(self.0 < Self::NEGP - 1 && rhs.0 > Self::P);
            diff -= Self::NEGP;
        }
        Self(diff)
    }
}

impl core::ops::SubAssign for Goldilocks {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::Mul<Goldilocks> for Goldilocks {
    type Output = Goldilocks;

    #[inline(always)]
    fn mul(self, rhs: Goldilocks) -> Goldilocks {
        reduce_wide(u128::from(self.0) * u128::from(rhs.0))
    }
}

impl core::ops::MulAssign for Goldilocks {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::Neg for Goldilocks {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self(Self::P - self.u64())
    }
}

impl core::ops::Neg for &Goldilocks {
    type Output = Goldilocks;

    #[inline(always)]
    fn neg(self) -> Goldilocks {
        -*self
    }
}

impl core::iter::Sum for Goldilocks {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let sum = iter.map(|x| x.0 as u128).sum::<u128>();
        reduce_wide(sum)
    }
}

impl<'a> core::iter::Sum<&'a Goldilocks> for Goldilocks {
    fn sum<I: Iterator<Item = &'a Goldilocks>>(iter: I) -> Self {
        let sum = iter.map(|x| x.0 as u128).sum::<u128>();
        reduce_wide(sum)
    }
}

impl core::iter::Product for Goldilocks {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Goldilocks::ONE)
    }
}

impl<'a> core::iter::Product<&'a Goldilocks> for Goldilocks {
    fn product<I: Iterator<Item = &'a Goldilocks>>(iter: I) -> Goldilocks {
        iter.cloned()
            .reduce(|x, y| x * y)
            .unwrap_or(Goldilocks::ONE)
    }
}
