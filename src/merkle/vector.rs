use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};

use super::{verify_merkle_proof, MerkleTree};
use crate::{
    field::Field,
    hash::{
        transcript::{Reader, Writer},
        Compress, Hasher,
    },
    utils::TwoAdicSlice,
    Error,
};
use std::fmt::Debug;

#[derive(Debug)]
pub struct VectorCommitmentData<T, Digest> {
    pub(crate) data: Vec<T>,
    pub(crate) layers: Vec<Vec<Digest>>,
}

impl<T, Digest: Clone> VectorCommitmentData<T, Digest> {
    pub fn k(&self) -> usize {
        self.data.k()
    }

    pub fn root(&self) -> Digest {
        self.layers.last().unwrap().first().unwrap().clone()
    }
}

pub trait VectorCommitment<F> {
    type Digest: Copy + Debug;
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: Vec<F>,
    ) -> Result<VectorCommitmentData<F, Self::Digest>, Error>
    where
        Transcript: Writer<Self::Digest>;
    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        comm: &VectorCommitmentData<F, Self::Digest>,
    ) -> Result<F, Error>
    where
        F: Copy,
        Transcript: Writer<F> + Writer<Self::Digest>;
    fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        comm: Self::Digest,
        el: F,
        index: usize,
        k: usize,
    ) -> Result<(F, F), Error>
    where
        Transcript: Reader<F> + Reader<Self::Digest>;
}

impl<F, Digest, H, C> VectorCommitment<F> for MerkleTree<F, Digest, H, C>
where
    F: Field,
    Digest: Copy + Clone + Debug + Send + Sync + Eq + PartialEq,
    H: Hasher<F, Digest>,
    C: Compress<Digest, 2>,
{
    type Digest = Digest;
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: Vec<F>,
    ) -> Result<VectorCommitmentData<F, Self::Digest>, Error>
    where
        Transcript: Writer<Self::Digest>,
    {
        let (l, r) = data.split_at(data.len() / 2);
        let layer0 = l
            .par_iter()
            .zip(r.par_iter())
            .map(|(l, r)| self.h.hash_iter([l, r]))
            .collect::<Vec<_>>();

        let mut layers = vec![layer0];
        for _ in 0..data.k() - 1 {
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
        Ok(VectorCommitmentData { data, layers })
    }

    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        comm: &VectorCommitmentData<F, Self::Digest>,
    ) -> Result<F, Error>
    where
        Transcript: Writer<F> + Writer<Digest>,
    {
        let k = comm.k() - 1;
        let el = comm.data[index];
        let mid = 1 << k;

        transcript.write(el)?;

        let mut witness = vec![];
        let mut index_asc = index & (mid - 1);
        comm.layers.iter().take(k).try_for_each(|layer| {
            let node = layer[index_asc ^ 1];
            witness.push(node);
            index_asc >>= 1;
            transcript.unsafe_write(node)
        })?;

        #[cfg(debug_assertions)]
        {
            let sb = comm.data[index ^ mid];
            let pair = if index < mid { [el, sb] } else { [sb, el] };
            let leaf = self.h.hash_iter(&pair);
            verify_merkle_proof(&self.c, comm.root(), index & (mid - 1), leaf, &witness).unwrap();
        }

        Ok(el)
    }

    fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        comm: Self::Digest,
        sb: F,
        index: usize,
        k: usize,
    ) -> Result<(F, F), Error>
    where
        Transcript: Reader<F> + Reader<Digest>,
    {
        let k = k - 1;
        let mid = 1 << k;
        let el = transcript.read()?;

        let pair = if index < mid { [el, sb] } else { [sb, el] };
        let leaf = self.h.hash_iter(&pair);
        let witness: Vec<Digest> = transcript.unsafe_read_many(k)?;
        verify_merkle_proof(&self.c, comm, index & (mid - 1), leaf, &witness)?;
        Ok((pair[0], pair[1]))
    }
}

#[cfg(test)]
mod test {
    use crate::{
        hash::{
            rust_crypto::{RustCrypto, RustCryptoReader, RustCryptoWriter},
            transcript::Reader,
        },
        merkle::{vector::VectorCommitment, MerkleTree},
    };
    use rand::Rng;

    #[test]
    fn test_vec_com() {
        type F = crate::field::goldilocks::Goldilocks;
        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;
        type Hasher = RustCrypto<sha3::Keccak256>;
        type Compress = RustCrypto<sha3::Keccak256>;
        let hasher = Hasher::new();
        let compress = Compress::new();

        let vec_comm = MerkleTree::<F, [u8; 32], _, _>::new(hasher, compress);
        let mut transcript = Writer::init("test");

        let k = 3;
        let mut rng = crate::test::seed_rng();
        let data = (0..1 << k).map(|_| rng.gen()).collect::<Vec<F>>();

        let comm_data = vec_comm.commit(&mut transcript, data).unwrap();
        (0..1 << k).for_each(|index| {
            vec_comm.query(&mut transcript, index, &comm_data).unwrap();
        });

        let proof = transcript.finalize();
        let mut transcript = Reader::init(&proof, "test");
        let comm: [u8; 32] = transcript.read().unwrap();
        let mid = 1 << (k - 1);
        (0..1 << k).for_each(|index| {
            let sb = comm_data.data[index ^ mid];
            vec_comm
                .verify(&mut transcript, comm, sb, index, k)
                .unwrap();
        });
    }
}
