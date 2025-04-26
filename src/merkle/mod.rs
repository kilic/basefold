use crate::{
    field::Field,
    hash::{Compress, Hasher},
    Error,
};
use std::fmt::Debug;

pub mod matrix;
pub mod vector;

#[allow(clippy::precedence)]
pub(super) fn to_interleaved_index(k: usize, index: usize) -> usize {
    ((index << 1) & (1 << k) - 1) | (index >> (k - 1))
}

pub fn verify_merkle_proof<C, Node>(
    c: &C,
    claim: Node,
    mut index: usize,
    leaf: Node,
    witness: &[Node],
) -> Result<(), Error>
where
    C: Compress<Node, 2>,
    Node: Copy + Clone + Send + Sync + Debug + Eq + PartialEq,
{
    assert!(index < 1 << witness.len());
    let found = witness.iter().fold(leaf, |acc, &w| {
        let acc = c.compress(if index & 1 == 1 { [w, acc] } else { [acc, w] });
        index >>= 1;
        acc
    });
    (claim == found).then_some(()).ok_or(Error::Verify)
}

pub struct MerkleTree<F, Digest, H: Hasher<F, Digest>, C: Compress<Digest, 2>> {
    pub(crate) h: H,
    pub(crate) c: C,
    pub(crate) _phantom: std::marker::PhantomData<(F, Digest)>,
}

impl<F, Digest, H, C> MerkleTree<F, Digest, H, C>
where
    F: Field,
    H: Hasher<F, Digest>,
    C: Compress<Digest, 2>,
{
    pub fn new(h: H, c: C) -> Self {
        Self {
            h,
            c,
            _phantom: std::marker::PhantomData,
        }
    }
}
