use crate::basefold::code::Basecode;
use crate::basefold::Basefold;
use crate::data::MatrixOwn;
use crate::hash::rust_crypto::{RustCrypto, RustCryptoReader, RustCryptoWriter};
use crate::hash::transcript::{Challenge, Reader};
use crate::merkle::MerkleTree;
use rand::Rng;

#[test]
fn test_pcs() {
    type F = crate::field::goldilocks::Goldilocks;
    type Ext = crate::field::goldilocks::Goldilocks2;

    type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
    type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;
    type Hasher = RustCrypto<sha3::Keccak256>;
    type Compress = RustCrypto<sha3::Keccak256>;
    let hasher = Hasher::new();
    let compress = Compress::new();

    let n_vars = 15;
    let c = 2;
    let k0 = 3;
    let width = 3;
    let n_test = 22;
    let num_points = 10;

    let mut rng = crate::test::seed_rng();
    let code = Basecode::<F>::generate(&mut rng, n_vars, k0, c);

    let mat_comm = MerkleTree::<F, [u8; 32], _, _>::new(hasher.clone(), compress.clone());
    let mat_comm_ext = MerkleTree::<Ext, [u8; 32], _, _>::new(hasher, compress);

    let basefold = Basefold::new(code, mat_comm, mat_comm_ext, n_test);

    let coeffs = (0..width << n_vars).map(|_| rng.gen()).collect::<Vec<F>>();
    let data = MatrixOwn::new(width, coeffs);
    let mut transcript = Writer::init("");
    let comm = basefold.commit::<_>(&mut transcript, &data).unwrap();

    let points = (0..num_points)
        .map(|_| (0..n_vars).map(|_| rng.gen()).collect::<Vec<Ext>>())
        .collect::<Vec<_>>();
    let points = points.iter().map(Vec::as_slice).collect::<Vec<_>>();
    basefold
        .open(&mut transcript, &points, &data, &comm)
        .unwrap();
    let checkpoint0: F = transcript.draw();

    let proof = transcript.finalize();
    let mut transcript = Reader::init(&proof, "");

    let comm: [u8; 32] = transcript.read().unwrap();
    basefold
        .verify(comm, width, &mut transcript, &points)
        .unwrap();
    let checkpoint1: F = transcript.draw();
    assert_eq!(checkpoint0, checkpoint1);
}
