use core::ptr::swap;
use std::ptr::swap_nonoverlapping;

const LB_BLOCK_SIZE: usize = 3;

unsafe fn transpose_in_place_square_small<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
) {
    for i in x + 1..x + (1 << lb_size) {
        for j in x..i {
            swap(
                arr.get_unchecked_mut(i + (j << lb_stride)),
                arr.get_unchecked_mut((i << lb_stride) + j),
            );
        }
    }
}

unsafe fn w_transpose_in_place_square_small<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
    w: usize,
) {
    for i in x + 1..x + (1 << lb_size) {
        for j in x..i {
            swap_nonoverlapping(
                arr.get_unchecked_mut(w * (i + (j << lb_stride))),
                arr.get_unchecked_mut(w * ((i << lb_stride) + j)),
                w,
            );
        }
    }
}

unsafe fn transpose_swap_square_small<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
    y: usize,
) {
    for i in x..x + (1 << lb_size) {
        for j in y..y + (1 << lb_size) {
            swap(
                arr.get_unchecked_mut(i + (j << lb_stride)),
                arr.get_unchecked_mut((i << lb_stride) + j),
            );
        }
    }
}

unsafe fn w_transpose_swap_square_small<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
    y: usize,
    w: usize,
) {
    for i in x..x + (1 << lb_size) {
        for j in y..y + (1 << lb_size) {
            swap_nonoverlapping(
                arr.get_unchecked_mut(w * (i + (j << lb_stride))),
                arr.get_unchecked_mut(w * ((i << lb_stride) + j)),
                w,
            );
        }
    }
}

unsafe fn transpose_swap_square<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
    y: usize,
) {
    if lb_size <= LB_BLOCK_SIZE {
        transpose_swap_square_small(arr, lb_stride, lb_size, x, y);
    } else {
        let lb_block_size = lb_size - 1;
        let block_size = 1 << lb_block_size;
        transpose_swap_square(arr, lb_stride, lb_block_size, x, y);
        transpose_swap_square(arr, lb_stride, lb_block_size, x + block_size, y);
        transpose_swap_square(arr, lb_stride, lb_block_size, x, y + block_size);
        transpose_swap_square(
            arr,
            lb_stride,
            lb_block_size,
            x + block_size,
            y + block_size,
        );
    }
}

unsafe fn w_transpose_swap_square<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
    y: usize,
    w: usize,
) {
    if lb_size <= LB_BLOCK_SIZE {
        w_transpose_swap_square_small(arr, lb_stride, lb_size, x, y, w);
    } else {
        let lb_block_size = lb_size - 1;
        let block_size = 1 << lb_block_size;
        w_transpose_swap_square(arr, lb_stride, lb_block_size, x, y, w);
        w_transpose_swap_square(arr, lb_stride, lb_block_size, x + block_size, y, w);
        w_transpose_swap_square(arr, lb_stride, lb_block_size, x, y + block_size, w);
        w_transpose_swap_square(
            arr,
            lb_stride,
            lb_block_size,
            x + block_size,
            y + block_size,
            w,
        );
    }
}

pub(crate) unsafe fn transpose_in_place_square<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
) {
    if lb_size <= LB_BLOCK_SIZE {
        transpose_in_place_square_small(arr, lb_stride, lb_size, x);
    } else {
        let lb_block_size = lb_size - 1;
        let block_size = 1 << lb_block_size;
        transpose_in_place_square(arr, lb_stride, lb_block_size, x);
        transpose_swap_square(arr, lb_stride, lb_block_size, x, x + block_size);
        transpose_in_place_square(arr, lb_stride, lb_block_size, x + block_size);
    }
}

pub(crate) unsafe fn w_transpose_in_place_square<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
    w: usize,
) {
    if lb_size <= LB_BLOCK_SIZE {
        w_transpose_in_place_square_small(arr, lb_stride, lb_size, x, w);
    } else {
        let lb_block_size = lb_size - 1;
        let block_size = 1 << lb_block_size;
        w_transpose_in_place_square(arr, lb_stride, lb_block_size, x, w);
        w_transpose_swap_square(arr, lb_stride, lb_block_size, x, x + block_size, w);
        w_transpose_in_place_square(arr, lb_stride, lb_block_size, x + block_size, w);
    }
}
