use crate::basefold::code::basecode::Basecode;
use crate::basefold::code::rs::ReedSolomon;
use crate::basefold::sumcheck::{EqSumcheck, Pointcheck, SumcheckProver};
use crate::basefold::Basefold;
use crate::data::MatrixOwn;
use crate::field::FromUniformBytes;
use crate::hash::rust_crypto::{RustCrypto, RustCryptoReader, RustCryptoWriter};
use crate::hash::transcript::{Challenge, Reader};
use crate::merkle::MerkleTree;
use crate::utils::n_rand;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field};
use rand::distr::{Distribution, StandardUniform};
use rand::Rng;

use super::code::RandomFoldableCode;

fn run_basefold<
    F: Field + FromUniformBytes,
    Ext: ExtensionField<F> + FromUniformBytes,
    Code: RandomFoldableCode<F>,
    Sumcheck: SumcheckProver<F, Ext>,
>(
    mut rng: impl rand::RngCore,
    cfg: Sumcheck::Cfg,
    code: Code,
    width: usize,
) where
    StandardUniform: Distribution<F> + Distribution<Ext>,
{
    type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
    type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;
    type Hasher = RustCrypto<sha3::Keccak256>;
    type Compress = RustCrypto<sha3::Keccak256>;
    let hasher = Hasher::new();
    let compress = Compress::new();

    // let n_vars = 15;
    let n_test = 22;

    let mat_comm = MerkleTree::<F, [u8; 32], _, _>::new(hasher.clone(), compress.clone());
    let mat_comm_ext = MerkleTree::<Ext, [u8; 32], _, _>::new(hasher, compress);

    let n_vars = code.n_vars();
    let basefold = Basefold::new(code, mat_comm, mat_comm_ext, n_test);

    let coeffs = (0..width << n_vars)
        .map(|_| rng.random())
        .collect::<Vec<F>>();
    let data = MatrixOwn::new(width, coeffs);
    let mut transcript = Writer::init("");
    let comm = basefold.commit::<_>(&mut transcript, &data).unwrap();

    let zs: Vec<Ext> = n_rand(&mut rng, n_vars);

    let mut sp = Sumcheck::new(&zs, &data, cfg);
    basefold
        .open(&mut transcript, &mut sp, &zs, &data, &comm)
        .unwrap();
    let checkpoint0: F = transcript.draw();

    let proof = transcript.finalize();
    let mut transcript = Reader::init(&proof, "");

    let comm: [u8; 32] = transcript.read().unwrap();
    basefold
        .verify::<_, Sumcheck::Verifier>(&mut transcript, comm, width, &zs)
        .unwrap();
    let checkpoint1: F = transcript.draw();
    assert_eq!(checkpoint0, checkpoint1);
}

#[test]
fn test_pcs() {
    type F = p3_goldilocks::Goldilocks;
    type Ext = BinomialExtensionField<F, 2>;

    let mut rng = crate::test::seed_rng();

    let n_vars = 15;
    let sumcheck_cfg = 2;
    let width = 7;

    {
        let c = 3;
        let k0 = 5;
        let code = Basecode::<F>::generate(&mut rng, n_vars, k0, c);
        run_basefold::<_, _, _, EqSumcheck<_, Ext>>(&mut rng, sumcheck_cfg, code.clone(), width);
        run_basefold::<_, _, _, Pointcheck<_, Ext>>(&mut rng, sumcheck_cfg, code, width);
    }

    {
        let c = 2;
        let code = ReedSolomon::<F>::new(c, n_vars);
        run_basefold::<_, _, _, EqSumcheck<_, Ext>>(&mut rng, sumcheck_cfg, code.clone(), width);
        run_basefold::<_, _, _, Pointcheck<_, Ext>>(&mut rng, sumcheck_cfg, code, width);
    }
}
