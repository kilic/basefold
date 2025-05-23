use super::TwoAdicSlice;
use crate::utils::transpose::w_transpose_in_place_square;
use crate::utils::{log2_strict, transpose::transpose_in_place_square};

const BIG_T_SIZE: usize = 1 << 14;
const SMALL_ARR_SIZE: usize = 1 << 16;

pub trait BitReverse<V>: TwoAdicSlice<V> {
    fn reverse_bits(self);
    fn reverse_bits_2d(self, w: usize);
}

impl<V> BitReverse<V> for &mut [V] {
    #[tracing::instrument(level = "info", skip_all, fields(k = self.k()))]
    fn reverse_bits(self) {
        reverse_index_bits_in_place(self);
    }

    #[tracing::instrument(level = "info", skip_all, fields(k = self.k()))]
    fn reverse_bits_2d(self, w: usize) {
        w_reverse_index_bits_in_place(self, w);
    }
}

#[rustfmt::skip]
const BIT_REVERSE_6BIT: &[u8] = &[
    0o00, 0o40, 0o20, 0o60, 0o10, 0o50, 0o30, 0o70,
    0o04, 0o44, 0o24, 0o64, 0o14, 0o54, 0o34, 0o74,
    0o02, 0o42, 0o22, 0o62, 0o12, 0o52, 0o32, 0o72,
    0o06, 0o46, 0o26, 0o66, 0o16, 0o56, 0o36, 0o76,
    0o01, 0o41, 0o21, 0o61, 0o11, 0o51, 0o31, 0o71,
    0o05, 0o45, 0o25, 0o65, 0o15, 0o55, 0o35, 0o75,
    0o03, 0o43, 0o23, 0o63, 0o13, 0o53, 0o33, 0o73,
    0o07, 0o47, 0o27, 0o67, 0o17, 0o57, 0o37, 0o77,
];

pub fn reverse_index_bits<T: Copy>(arr: &[T]) -> Vec<T> {
    let n = arr.len();
    let n_power = log2_strict(n);

    if n_power <= 6 {
        reverse_index_bits_small(arr, n_power)
    } else {
        reverse_index_bits_large(arr, n_power)
    }
}

fn reverse_index_bits_small<T: Copy>(arr: &[T], n_power: usize) -> Vec<T> {
    let n = arr.len();
    let mut result = Vec::with_capacity(n);
    let dst_shr_amt = 6 - n_power;

    BIT_REVERSE_6BIT.iter().for_each(|&i| {
        let src = (i as usize) >> dst_shr_amt;
        result.push(arr[src]);
    });
    result
}

fn reverse_index_bits_large<T: Copy>(arr: &[T], n_power: usize) -> Vec<T> {
    let n = arr.len();

    let src_lo_shr_amt = 64 - (n_power - 6);
    let src_hi_shl_amt = n_power - 6;
    let mut result = Vec::with_capacity(n);
    for i_chunk in 0..(n >> 6) {
        let src_lo = i_chunk.reverse_bits() >> src_lo_shr_amt;
        BIT_REVERSE_6BIT.iter().for_each(|&i| {
            let src_hi = (i as usize) << src_hi_shl_amt;
            let src = src_hi + src_lo;
            result.push(arr[src]);
        });
    }
    result
}

unsafe fn reverse_index_bits_in_place_small<T>(arr: &mut [T]) {
    let n = arr.len();
    let k = log2_strict(n);
    for i in 0..arr.len() {
        let j = i.reverse_bits().wrapping_shr(usize::BITS - k as u32);
        if i < j {
            core::ptr::swap(arr.get_unchecked_mut(i), arr.get_unchecked_mut(j));
        }
    }
}

unsafe fn w_reverse_index_bits_in_place_small<T>(arr: &mut [T], w: usize) {
    let n = arr.len() / w;
    let k = log2_strict(n);
    (0..n).for_each(|i| {
        let j = i.reverse_bits().wrapping_shr(usize::BITS - k as u32);
        if i < j {
            unsafe {
                core::ptr::swap_nonoverlapping(
                    arr.get_unchecked_mut(i * w),
                    arr.get_unchecked_mut(j * w),
                    w,
                );
            }
        }
    });
}

unsafe fn reverse_index_bits_in_place_chunks<T>(
    arr: &mut [T],
    lb_num_chunks: usize,
    lb_chunk_size: usize,
) {
    for i in 0..1usize << lb_num_chunks {
        let j = i
            .reverse_bits()
            .wrapping_shr(usize::BITS - lb_num_chunks as u32);
        if i < j {
            core::ptr::swap_nonoverlapping(
                arr.get_unchecked_mut(i << lb_chunk_size),
                arr.get_unchecked_mut(j << lb_chunk_size),
                1 << lb_chunk_size,
            );
        }
    }
}

pub fn reverse_index_bits_in_place<T>(arr: &mut [T]) {
    let n = arr.len();
    let k = log2_strict(n);
    if size_of::<T>() << k <= SMALL_ARR_SIZE || size_of::<T>() >= BIG_T_SIZE {
        unsafe {
            reverse_index_bits_in_place_small(arr);
        }
    } else {
        let lb_num_chunks = k >> 1;
        let lb_chunk_size = k - lb_num_chunks;

        unsafe {
            reverse_index_bits_in_place_chunks(arr, lb_num_chunks, lb_chunk_size);
            transpose_in_place_square(arr, lb_chunk_size, lb_num_chunks, 0);
            if lb_num_chunks != lb_chunk_size {
                let arr_with_offset = &mut arr[1 << lb_num_chunks..];
                transpose_in_place_square(arr_with_offset, lb_chunk_size, lb_num_chunks, 0);
            }
            reverse_index_bits_in_place_chunks(arr, lb_num_chunks, lb_chunk_size);
        }
    }
}

unsafe fn w_reverse_index_bits_in_place_chunks<T>(
    arr: &mut [T],
    lb_num_chunks: usize,
    lb_chunk_size: usize,
    w: usize,
) {
    for i in 0..1usize << lb_num_chunks {
        let j = i
            .reverse_bits()
            .wrapping_shr(usize::BITS - lb_num_chunks as u32);
        if i < j {
            core::ptr::swap_nonoverlapping(
                arr.get_unchecked_mut((i << lb_chunk_size) * w),
                arr.get_unchecked_mut((j << lb_chunk_size) * w),
                (1 << lb_chunk_size) * w,
            );
        }
    }
}

pub fn w_reverse_index_bits_in_place<T>(arr: &mut [T], w: usize) {
    let n = arr.len() / w;
    let k = log2_strict(n);

    if size_of::<T>() << k <= SMALL_ARR_SIZE || size_of::<T>() >= BIG_T_SIZE {
        unsafe {
            w_reverse_index_bits_in_place_small(arr, w);
        }
    } else {
        let lb_num_chunks = k >> 1;
        let lb_chunk_size = k - lb_num_chunks;
        unsafe {
            w_reverse_index_bits_in_place_chunks(arr, lb_num_chunks, lb_chunk_size, w);
            w_transpose_in_place_square(arr, lb_chunk_size, lb_num_chunks, 0, w);
            if lb_num_chunks != lb_chunk_size {
                let arr_with_offset = &mut arr[w * (1 << lb_num_chunks)..];
                w_transpose_in_place_square(arr_with_offset, lb_chunk_size, lb_num_chunks, 0, w);
            }
            w_reverse_index_bits_in_place_chunks(arr, lb_num_chunks, lb_chunk_size, w);
        }
    }
}

#[cfg(test)]
mod test {

    use crate::utils::bitreverse::BitReverse;
    use crate::utils::TwoAdicSlice;
    use crate::{
        data::MatrixOwn,
        utils::bitreverse::{
            reverse_index_bits_in_place_small, w_reverse_index_bits_in_place_small,
        },
    };

    #[cfg(test)]
    #[tracing::instrument(level = "info", skip_all, fields(k = e.k()))]
    fn reverse_bits_naive<T>(e: &mut [T]) {
        unsafe { reverse_index_bits_in_place_small(e) };
    }

    #[cfg(test)]
    #[tracing::instrument(level = "info", skip_all, fields(k = e.k()))]
    fn reverse_bits_2d_naive<T>(e: &mut [T], w: usize) {
        unsafe { w_reverse_index_bits_in_place_small(e, w) };
    }

    #[test]
    fn test_bitreverse() {
        for k in 20..25 {
            let n = 1 << k;
            let a0: Vec<_> = (0..n).collect::<Vec<_>>();
            let mut a0 = a0.clone();
            let mut a1 = a0.clone();
            a0.reverse_bits();
            unsafe { reverse_index_bits_in_place_small(&mut a1) };
            assert_eq!(a0, a1);

            for w in 3..4 {
                let a0: Vec<_> = (0..n * w).collect::<Vec<_>>();
                let mut m0 = MatrixOwn::new(w, a0);
                let mut m1 = m0.clone();
                m0.reverse_bits();
                reverse_bits_2d_naive(&mut m1.storage, w);
                assert_eq!(m0.storage, m1.storage);
            }
        }
    }
}
