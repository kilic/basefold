use itertools::Itertools;
use rand::{distributions::Standard, prelude::Distribution, RngCore};

pub mod arithmetic;
pub use arithmetic::*;

#[macro_export]
macro_rules! split128 {
    ($x:expr) => {
        ($x as u64, ($x >> 64) as u64)
    };
}

pub fn n_rand<F>(mut rng: impl RngCore, n: usize) -> Vec<F>
where
    Standard: Distribution<F>,
{
    use rand::Rng;
    std::iter::repeat_with(|| rng.gen()).take(n).collect_vec()
}

#[inline(always)]
pub fn log2_strict(n: usize) -> usize {
    let res = n.trailing_zeros();
    debug_assert_eq!(n.wrapping_shr(res), 1);
    res as usize
}

pub(crate) trait TwoAdicSlice<T>: core::ops::Deref<Target = [T]> {
    #[inline(always)]
    fn k(&self) -> usize {
        log2_strict(self.len())
    }
}

impl<V> TwoAdicSlice<V> for Vec<V> {}
impl<V> TwoAdicSlice<V> for &[V] {}
impl<V> TwoAdicSlice<V> for &mut [V] {}

// copy from a16z/jolt
pub(crate) fn unsafe_allocate_zero_vec<F: Default + Sized>(size: usize) -> Vec<F> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    // Check for safety of 0 allocation
    unsafe {
        let value = &F::default();
        let ptr = value as *const F as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F>());
        assert!(bytes.iter().all(|&byte| byte == 0));
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<F>;
    unsafe {
        let layout = std::alloc::Layout::array::<F>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut F;

        if ptr.is_null() {
            panic!("Zero vec allocaiton failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}

// #[derive(Clone, Debug)]
// pub struct LadderVec<F: Default + Copy> {
//     layout: Vec<F>,
// }

// impl<F: Default + Copy> LadderVec<F> {
//     pub fn new(k: usize) -> Self {
//         let layout = unsafe_allocate_zero_vec((1 << (1 + k)) - 1);
//         Self { layout }
//     }

//     pub fn k(&self) -> usize {
//         log2_strict((self.layout.len() + 1) / 2)
//     }

//     pub fn last_row(&self) -> &[F] {
//         self.row(self.k())
//     }

//     pub fn row(&self, i: usize) -> &[F] {
//         let start = (1 << i) - 1;
//         let end = (1 << (i + 1)) - 1;
//         &self.layout[start..end]
//     }

//     pub fn update_row(&mut self, i: usize, row: &[F]) {
//         assert_eq!(row.len(), 1 << i);
//         let start = (1 << i) - 1;
//         let end = (1 << (i + 1)) - 1;
//         self.layout[start..end].copy_from_slice(row);
//     }
// }
