use crate::{
    data::MatrixOwn,
    hash::{
        transcript::{Reader, Writer},
        Compress, Hasher,
    },
    merkle::{to_interleaved_index, verify_merkle_proof},
    Error,
};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use std::fmt::Debug;

use super::MerkleTree;

#[derive(Debug)]
pub struct MatrixCommitmentData<T, Digest> {
    pub(crate) data: MatrixOwn<T>,
    pub(crate) layers: Vec<Vec<Digest>>,
}

impl<T, Digest: Clone> MatrixCommitmentData<T, Digest> {
    pub fn k(&self) -> usize {
        self.data.k()
    }

    pub fn row(&self, index: usize) -> &[T] {
        self.data.row(index)
    }

    pub fn root(&self) -> Digest {
        self.layers.last().unwrap().first().unwrap().clone()
    }
}

pub trait MatrixCommitment<F> {
    type Digest: Copy + Debug;
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: MatrixOwn<F>,
    ) -> Result<MatrixCommitmentData<F, Self::Digest>, Error>
    where
        Transcript: Writer<Self::Digest>;

    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        comm: &MatrixCommitmentData<F, Self::Digest>,
    ) -> Result<Vec<F>, Error>
    where
        F: Copy,
        Transcript: Writer<F> + Writer<Self::Digest>;

    fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        comm: Self::Digest,
        index: usize,
        width: usize,
        height: usize,
    ) -> Result<Vec<F>, Error>
    where
        Transcript: Reader<F> + Reader<Self::Digest>;
}

impl<F, Digest, H, C> MatrixCommitment<F> for MerkleTree<F, Digest, H, C>
where
    F: Copy + Clone + Debug + Send + Sync,
    Digest: Copy + Clone + Debug + Send + Sync + Eq + PartialEq,
    H: Hasher<F, Digest>,
    C: Compress<Digest, 2>,
{
    type Digest = Digest;
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: MatrixOwn<F>,
    ) -> Result<MatrixCommitmentData<F, Self::Digest>, Error>
    where
        Transcript: Writer<Self::Digest>,
    {
        let (l, r) = data.split_half();
        let layer0 = l
            .par_iter()
            .zip(r.par_iter())
            .flat_map(|(l, r)| [self.h.hash_iter(l), self.h.hash_iter(r)])
            .collect::<Vec<_>>();

        let mut layers = vec![layer0];
        for _ in 0..data.k() {
            let next_layer = layers
                .last()
                .unwrap()
                .par_chunks(2)
                .map(|chunk| self.c.compress(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            layers.push(next_layer);
        }

        let top = layers.last().unwrap();
        debug_assert_eq!(top.len(), 1);
        transcript.write(top[0])?;
        Ok(MatrixCommitmentData { data, layers })
    }

    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        comm: &MatrixCommitmentData<F, Self::Digest>,
    ) -> Result<Vec<F>, Error>
    where
        Transcript: Writer<F> + Writer<Digest>,
    {
        let k = comm.k();
        let row = comm.data.row(index).to_vec();
        transcript.unsafe_write_many(&row)?;

        let mid = 1 << (k - 1);
        let _sb = comm.data.row(index ^ mid).to_vec();

        let index = to_interleaved_index(comm.k(), index);
        let mut witness = vec![];
        let mut index_asc = index;
        comm.layers
            .iter()
            .take(k)
            .enumerate()
            .try_for_each(|(_i, layer)| {
                let node = layer[index_asc ^ 1];
                #[cfg(debug_assertions)]
                if _i == 0 {
                    assert_eq!(self.h.hash_iter(&_sb), node);
                }
                witness.push(node);
                index_asc >>= 1;
                transcript.unsafe_write(node)
            })?;

        #[cfg(debug_assertions)]
        {
            let leaf = self.h.hash_iter(&row);
            verify_merkle_proof(&self.c, comm.root(), index, leaf, &witness).unwrap();
        }

        Ok(row)
    }

    fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        comm: Self::Digest,
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<F>, Error>
    where
        Transcript: Reader<F> + Reader<Digest>,
    {
        let row: Vec<F> = transcript.unsafe_read_many(width)?;
        let leaf = self.h.hash_iter(&row);
        let index = to_interleaved_index(k, index);
        let witness: Vec<Digest> = transcript.unsafe_read_many(k)?;
        verify_merkle_proof(&self.c, comm, index, leaf, &witness)?;
        Ok(row)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        data::MatrixOwn,
        hash::{
            rust_crypto::{RustCrypto, RustCryptoReader, RustCryptoWriter},
            transcript::Reader,
        },
        merkle::{matrix::MatrixCommitment, MerkleTree},
        utils::n_rand,
    };
    use p3_goldilocks::Goldilocks;

    #[test]
    fn test_mat_com() {
        type F = Goldilocks;
        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;
        type Hasher = RustCrypto<sha3::Keccak256>;
        type Compress = RustCrypto<sha3::Keccak256>;
        let hasher = Hasher::new();
        let compress = Compress::new();

        let mat_comm = MerkleTree::<F, [u8; 32], _, _>::new(hasher, compress);
        let mut transcript = Writer::init("test");

        let k = 3;
        let width = 1;
        let mut rng = crate::test::seed_rng();
        let coeffs = n_rand(&mut rng, 1 << k);
        let data = MatrixOwn::new(width, coeffs);
        let comm_data = mat_comm.commit(&mut transcript, data).unwrap();
        (0..1 << k).for_each(|index| {
            mat_comm.query(&mut transcript, index, &comm_data).unwrap();
        });

        let proof = transcript.finalize();
        let mut transcript = Reader::init(&proof, "test");
        let comm: [u8; 32] = transcript.read().unwrap();

        (0..1 << k).for_each(|index| {
            mat_comm
                .verify(&mut transcript, comm, index, width, k)
                .unwrap();
        });
    }
}
