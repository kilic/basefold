use p3_field::{ExtensionField, Field};

use crate::{
    data::MatrixOwn,
    hash::transcript::{Challenge, Reader, Writer},
    mle::SplitEqTree,
    utils::{BatchInverse, TwoAdicSlice},
};

use super::{SumcheckProver, SumcheckVerifier};

pub struct Pointcheck<F: Field, Ext: ExtensionField<F>> {
    eq_tree: SplitEqTree<F, Ext>,
    zs_inv: Vec<Ext>,
    round: usize,
    evals: Vec<Ext>,
    _marker: std::marker::PhantomData<F>,
}

pub struct PointcheckVerifier<F: Field, Ext: ExtensionField<F>> {
    rs: Vec<Ext>,
    zs: Vec<Ext>,
    claim: Ext,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> SumcheckVerifier<F, Ext> for PointcheckVerifier<F, Ext> {
    fn new(claim: Ext, zs: &[Ext]) -> Self {
        Self {
            rs: vec![],
            zs: zs.to_vec(),
            claim,
            _phantom: std::marker::PhantomData,
        }
    }

    fn reduce_claim<Transcript>(&mut self, transcript: &mut Transcript) -> Result<Ext, crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<Ext>,
    {
        let round = self.rs.len();
        let a1: Ext = transcript.read()?;
        let z = self.zs[self.zs.len() - round - 1];
        let a0 = self.claim - a1 * z;
        let r: Ext = transcript.draw();
        self.claim = a0 + r * a1;
        self.rs.push(r);
        Ok(r)
    }

    fn verify<Transcript>(
        self,
        reader: &mut Transcript,
    ) -> Result<(Vec<Ext>, Vec<Ext>), crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<Ext>,
    {
        let k = self.zs.len();
        let n = k - self.rs.len();
        let fin: Vec<Ext> = reader.read_many(1 << n).unwrap();
        let (z_fin, _) = self.zs.split_at(n);
        (crate::mle::eval_poly(z_fin, &fin) == self.claim)
            .then_some(())
            .ok_or(crate::Error::Verify)?;
        Ok((fin, self.rs))
    }
}

impl<F: Field, Ext: ExtensionField<F>> SumcheckProver<F, Ext> for Pointcheck<F, Ext> {
    type Cfg = usize;
    type Verifier = PointcheckVerifier<F, Ext>;

    // * Evaluate matrix
    // * Store partial EQ trees in order to reuse them in pointcheck rounds
    fn new(zs: &[Ext], mat: &MatrixOwn<F>, split: usize) -> Self {
        let (mut eq_tree, evals) = tracing::info_span!("eval", s = split).in_scope(|| {
            let eq_tree = crate::mle::SplitEqTree::new(zs, split);
            let evals = eq_tree.eval_mat(mat);
            (eq_tree, evals)
        });

        // Last element of tailing eq tree is not required in pointcheck, so we dump it
        eq_tree.pop_right();

        let mut zs_inv = zs.to_vec();
        zs_inv.inverse();

        Self {
            zs_inv,
            eq_tree,
            round: 0,
            evals,
            _marker: std::marker::PhantomData,
        }
    }

    fn evals(&self) -> &[Ext] {
        &self.evals
    }

    // In the first round `poly` is assumed to be compressed matrix that is used
    // when creating new instance of `EqSumcheck`. Similarly `claim` is the
    // compressed claim.
    fn round<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        claim: &mut Ext,
        poly: &mut Vec<Ext>,
    ) -> Result<Ext, crate::Error>
    where
        Transcript: Writer<Ext> + Challenge<Ext>,
    {
        let k = poly.k() - 1;
        let (p0, _) = poly.split_at_mut(1 << k);
        let a0 = self.eq_tree.dot(p0);
        let a1 = (*claim - a0) * self.zs_inv[k];
        transcript.write(a1)?;
        let r = transcript.draw();
        crate::mle::fix_var(poly, r);
        *claim = a0 + r * a1;
        self.round += 1;
        Ok(r)
    }
}

#[cfg(test)]
mod test {

    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, Field};

    use crate::basefold::sumcheck::{SumcheckProver, SumcheckVerifier};
    use crate::data::MatrixOwn;
    use crate::hash::rust_crypto::{RustCryptoReader, RustCryptoWriter};
    use crate::hash::transcript::{Challenge, Reader, Writer};
    use crate::utils::{n_rand, VecOps};

    impl<F: Field, Ext: ExtensionField<F>> super::Pointcheck<F, Ext> {
        #[tracing::instrument(skip_all)]
        pub fn run_prover<Transcript>(
            &mut self,
            transcript: &mut Transcript,
            d: usize,
            claim: &mut Ext,
            poly: &mut Vec<Ext>,
        ) -> Result<Vec<Ext>, crate::Error>
        where
            Transcript: Writer<Ext> + Challenge<Ext>,
        {
            (0..d)
                .map(|_| self.round(transcript, claim, poly))
                .collect()
        }
    }

    impl<F: Field, Ext: ExtensionField<F>> super::PointcheckVerifier<F, Ext> {
        fn run_verifier<Transcript>(
            &mut self,
            transcript: &mut Transcript,
            d: usize,
        ) -> Result<Vec<Ext>, crate::Error>
        where
            Transcript: Reader<Ext> + Challenge<Ext>,
        {
            (0..d).map(|_| self.reduce_claim(transcript)).collect()
        }
    }

    #[test]
    fn test_pointcheck() {
        type F = p3_goldilocks::Goldilocks;
        type Ext = BinomialExtensionField<F, 2>;
        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;

        // crate::test::init_tracing();
        let mut rng = crate::test::seed_rng();
        let k = 11;
        let width = 2;
        let mat: Vec<F> = n_rand(&mut rng, width * (1 << k));
        let mat = MatrixOwn::new(width, mat);
        let zs = n_rand(&mut rng, k);

        let _evals = crate::mle::SplitEq::new(&zs, 1).eval_mat(&mat);

        for eq_split in 1..7 {
            for d in 0..=k {
                let (proof, checkpoint0) = {
                    let mut writer = Writer::init(b"");
                    let mut sp = super::Pointcheck::new(&zs, &mat, eq_split);
                    assert_eq!(_evals, sp.evals());
                    sp.evals().iter().for_each(|&e| writer.write(e).unwrap());
                    let alpha = writer.draw();
                    let mut claim = sp.evals.horner(alpha);
                    let mut poly = mat.iter().map(|row| row.horner(alpha)).collect::<Vec<_>>();

                    let _rs = sp
                        .run_prover(&mut writer, d, &mut claim, &mut poly)
                        .unwrap();

                    assert_eq!(poly.len(), 1 << (k - d));
                    writer.write_many(&poly).unwrap();

                    let checkpoint: F = writer.draw();
                    (writer.finalize(), checkpoint)
                };

                {
                    let mut reader = Reader::init(&proof, b"");

                    let evals: Vec<Ext> = reader.read_many(mat.width()).unwrap();
                    let alpha = reader.draw();
                    let claim = evals.horner(alpha);

                    let mut sv = super::PointcheckVerifier::<F, Ext>::new(claim, &zs);
                    let _ = sv.run_verifier(&mut reader, d).unwrap();

                    sv.verify(&mut reader).unwrap();

                    let checkpoint1: F = reader.draw();
                    assert_eq!(checkpoint0, checkpoint1);
                }
            }
        }
    }
}
