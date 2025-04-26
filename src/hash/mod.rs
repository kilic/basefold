use itertools::Itertools;

pub mod rust_crypto;
pub mod transcript;

pub trait Compress<T, const N: usize>: Send + Sync {
    fn compress(&self, input: [T; N]) -> T;
}

pub trait Hasher<Item, Out>: Send + Sync {
    fn hash(&self, input: &[Item]) -> Out;
    fn hash_iter<'a, I>(&self, input: I) -> Out
    where
        I: IntoIterator<Item = &'a Item>,
        Item: 'a + Clone,
    {
        let input = input.into_iter().cloned().collect_vec();
        self.hash(&input)
    }
}
