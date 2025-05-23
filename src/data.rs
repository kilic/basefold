use crate::utils::bitreverse::BitReverse;
use crate::utils::log2_strict;
use core::fmt::Debug;
use rand::distr::{Distribution, StandardUniform};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

#[allow(clippy::len_without_is_empty)]
pub trait Storage {
    fn len(&self) -> usize;
}

impl<V> Storage for &[V] {
    fn len(&self) -> usize {
        <[V]>::len(self)
    }
}

impl<V> Storage for &mut [V] {
    fn len(&self) -> usize {
        <[V]>::len(self)
    }
}

impl<V> Storage for Vec<V> {
    fn len(&self) -> usize {
        self.len()
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Matrix<S> {
    pub storage: S,
    width: usize,
}

pub type MatrixRef<'a, V> = Matrix<&'a [V]>;
pub type MatrixMut<'a, V> = Matrix<&'a mut [V]>;
pub type MatrixOwn<V> = Matrix<Vec<V>>;

impl<V> Matrix<Vec<V>> {
    pub fn zero(width: usize, height: usize) -> Self
    where
        V: Default + Copy,
    {
        Self::new(width, vec![V::default(); width * height])
    }

    pub fn rand(rng: impl rand::Rng, width: usize, height: usize) -> Self
    where
        StandardUniform: Distribution<V>,
    {
        let values = crate::utils::n_rand(rng, width * height);
        Self::new(width, values)
    }

    pub fn to_vec(&self) -> Vec<V>
    where
        V: Clone,
    {
        assert_eq!(self.width, 1);
        self.storage.clone()
    }

    pub fn as_mut(&mut self) -> Matrix<&mut [V]> {
        Matrix {
            storage: self.storage.as_mut(),
            width: self.width,
        }
    }

    pub fn as_ref(&mut self) -> Matrix<&[V]> {
        Matrix {
            storage: self.storage.as_ref(),
            width: self.width,
        }
    }

    pub fn resize(&mut self, k: usize)
    where
        V: Default + Clone,
    {
        assert!(k >= self.k());
        self.storage.resize((1 << k) * self.width, V::default());
    }

    pub fn truncate(&mut self, k: usize)
    where
        V: Default + Clone,
    {
        assert!(k <= self.k());
        self.storage.truncate((1 << k) * self.width);
    }

    pub fn columns(&self) -> Vec<Vec<V>>
    where
        V: Copy,
    {
        (0..self.width)
            .map(|i| self.iter().map(|row| row[i]).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    }
}

impl<S: Storage> Matrix<S> {
    pub fn new(width: usize, storage: S) -> Self {
        Self { storage, width }
    }

    pub fn owned<V>(&self) -> Matrix<Vec<V>>
    where
        S: AsRef<[V]>,
        V: Clone,
    {
        Matrix::new(self.width, self.storage.as_ref().to_vec())
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        assert!(self.storage.len() % self.width == 0);
        self.storage.len() / self.width
    }

    pub fn k(&self) -> usize {
        log2_strict(self.height())
    }

    pub fn row<V>(&self, i: usize) -> &[V]
    where
        S: AsRef<[V]>,
    {
        let start = i * self.width;
        let end = start + self.width;
        &self.storage.as_ref()[start..end]
    }

    pub fn row_mut<V>(&mut self, i: usize) -> &mut [V]
    where
        S: AsMut<[V]>,
    {
        let start = i * self.width;
        let end = start + self.width;
        &mut self.storage.as_mut()[start..end]
    }

    pub fn iter<'a, V: 'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a [V]>
    where
        S: AsRef<[V]>,
    {
        self.storage.as_ref().chunks_exact(self.width)
    }

    pub fn iter_mut<'a, V: 'a>(&'a mut self) -> impl Iterator<Item = &'a mut [V]>
    where
        S: AsMut<[V]>,
    {
        self.storage.as_mut().chunks_exact_mut(self.width)
    }

    pub fn par_iter_mut<'a, V>(&'a mut self) -> impl IndexedParallelIterator<Item = &'a mut [V]>
    where
        S: AsMut<[V]>,
        V: 'a + Send + Sync,
    {
        self.storage.as_mut().par_chunks_exact_mut(self.width)
    }

    pub fn par_iter<'a, V>(&'a self) -> impl IndexedParallelIterator<Item = &'a [V]>
    where
        S: AsRef<[V]>,
        V: Send + Sync + 'a,
    {
        self.storage.as_ref().par_chunks_exact(self.width)
    }

    pub fn par_pair<'a, V>(&'a self) -> impl IndexedParallelIterator<Item = (&'a [V], &'a [V])>
    where
        S: AsRef<[V]> + Send + Sync,
        V: Send + Sync + 'a,
    {
        self.storage
            .as_ref()
            .par_chunks(2 * self.width)
            .map(|inner| inner.split_at(self.width))
    }

    pub fn chunks<'a, V: 'a>(&'a self, size: usize) -> impl Iterator<Item = Matrix<&'a [V]>>
    where
        S: AsRef<[V]>,
    {
        self.storage
            .as_ref()
            .chunks(size * self.width)
            .map(|inner| Matrix::new(self.width, inner))
    }

    pub fn chunks_mut<'a, V: 'a>(
        &'a mut self,
        size: usize,
    ) -> impl Iterator<Item = Matrix<&'a mut [V]>>
    where
        S: AsMut<[V]>,
    {
        self.storage
            .as_mut()
            .chunks_mut(size * self.width)
            .map(|inner| Matrix::new(self.width, inner))
    }

    pub fn par_chunks<'a, V>(
        &'a self,
        size: usize,
    ) -> impl IndexedParallelIterator<Item = Matrix<&'a [V]>>
    where
        S: AsRef<[V]> + Send + Sync,
        V: Send + Sync + 'a,
    {
        self.storage
            .as_ref()
            .par_chunks(size * self.width)
            .map(|inner| Matrix::new(self.width, inner))
    }

    pub fn par_chunks_mut<'a, V>(
        &'a mut self,
        size: usize,
    ) -> impl IndexedParallelIterator<Item = Matrix<&'a mut [V]>>
    where
        S: AsMut<[V]>,
        V: Send + Sync + 'a,
    {
        self.storage
            .as_mut()
            .par_chunks_mut(size * self.width)
            .map(|inner| Matrix::new(self.width, inner))
    }

    pub fn split_mut<V>(&mut self, k: usize) -> (Matrix<&mut [V]>, Matrix<&mut [V]>)
    where
        S: AsMut<[V]>,
    {
        let width = self.width();
        let (v0, v1) = self.storage.as_mut().split_at_mut((1 << k) * width);
        (Matrix::new(width, v0), Matrix::new(width, v1))
    }

    pub fn split<V>(&self, h: usize) -> (Matrix<&[V]>, Matrix<&[V]>)
    where
        S: AsRef<[V]>,
    {
        let width = self.width();
        let (v0, v1) = self.storage.as_ref().split_at((1 << h) * width);
        (Matrix::new(width, v0), Matrix::new(width, v1))
    }

    pub fn split_half_mut<V>(&mut self) -> (Matrix<&mut [V]>, Matrix<&mut [V]>)
    where
        S: AsMut<[V]>,
    {
        self.split_mut(self.k() - 1)
    }

    pub fn split_half<V>(&self) -> (Matrix<&[V]>, Matrix<&[V]>)
    where
        S: AsRef<[V]>,
    {
        self.split(self.k() - 1)
    }

    pub fn reverse_bits<V>(&mut self)
    where
        S: AsMut<[V]>,
        V: Clone + Send + Sync,
    {
        self.storage.as_mut().reverse_bits_2d(self.width);
    }
}
