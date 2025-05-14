use p3_field::{ExtensionField, Field};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    data::MatrixOwn,
    hash::transcript::{Challenge, Reader, Writer},
    utils::{interpolate, TwoAdicSlice, VecOps},
};

use super::{SumcheckProver, SumcheckVerifier};

fn extrapolate<F: Field, EF: ExtensionField<F>>(evals: &[F], target: EF) -> EF {
    let points = (0..evals.len())
        .map(|i| F::from_usize(i))
        .collect::<Vec<_>>();
    interpolate(&points, evals).horner(target)
}

pub struct EqSumcheck<F: Field, Ext: ExtensionField<F>> {
    split_eq: crate::mle::SplitEq<F, Ext>,
    round: usize,
    k: usize,
    evals: Vec<Ext>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> EqSumcheck<F, Ext> {
    fn round_chunked<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        claim: &mut Ext,
        poly: &mut Vec<Ext>,
    ) -> Result<Ext, crate::Error>
    where
        Transcript: Writer<Ext> + Challenge<Ext>,
    {
        let chunk_size = self.split_eq.left.len();
        let mid = poly.len() / 2;
        let (p0, p1) = poly.split_at_mut(mid);
        let evals = p0
            .chunks(chunk_size)
            .zip(p1.chunks(chunk_size))
            .map(|(part0, part1)| {
                part0
                    .par_iter()
                    .zip_eq(part1.par_iter())
                    .zip_eq(self.split_eq.left.par_iter())
                    .map(|((&a0, &a1), &b)| [a0 * b, (a1.double() - a0) * b])
                    .reduce(|| [Ext::ZERO, Ext::ZERO], |a, b| [a[0] + b[0], a[1] + b[1]])
                    .to_vec()
            })
            .collect::<Vec<_>>();

        let mut evals = {
            let coeffs = evals.iter().map(|row| row[0]).collect::<Vec<_>>();
            let v0 = coeffs
                .iter()
                .zip(self.split_eq.right.iter())
                .map(|(&a, &b)| b * a)
                .sum::<Ext>();

            let coeffs = evals.iter().map(|row| row[1]).collect::<Vec<_>>();
            let (zzlo, zzhi) = self.split_eq.right.split_at(self.split_eq.right.len() / 2);
            let v2 = zzlo
                .iter()
                .zip(zzhi.iter())
                .zip(coeffs.iter())
                .map(|((&zzlo, &zzhi), &c)| (zzhi.double() - zzlo) * c)
                .sum::<Ext>();

            vec![v0, v2]
        };

        evals.iter().try_for_each(|&e| transcript.write(e))?;
        evals.insert(1, *claim - evals[0]);
        let r: Ext = transcript.draw();
        *claim = extrapolate(&evals, r);

        crate::mle::fix_var(poly, r);
        crate::mle::fix_var(&mut self.split_eq.right, r);
        Ok(r)
    }

    fn round_nonchunked<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        claim: &mut Ext,
        poly: &mut Vec<Ext>,
    ) -> Result<Ext, crate::Error>
    where
        Transcript: Writer<Ext> + Challenge<Ext>,
    {
        assert_eq!(poly.k(), self.split_eq.left.k());
        let mid = poly.len() / 2;
        let (p0, p1) = poly.split_at_mut(mid);
        let (eq0, eq1) = self.split_eq.left.split_at_mut(mid);
        let mut evals = p0
            .par_iter()
            .zip(p1.par_iter())
            .zip(eq0.par_iter())
            .zip(eq1.par_iter())
            .map(|(((&a0, &a1), &b0), &b1)| {
                let v0 = b0 * a0;
                let dif0 = a1 - a0;
                let dif1 = b1 - b0;
                let u0 = a1 + dif0;
                let u1 = b1 + dif1;
                let v2 = u1 * u0;
                [v0, v2]
            })
            .reduce(|| [Ext::ZERO, Ext::ZERO], |a, b| [a[0] + b[0], a[1] + b[1]])
            .to_vec();

        {
            let eq1 = *self.split_eq.right.first().unwrap();
            evals[0] *= eq1;
            evals[1] *= eq1;
        }

        evals.iter().try_for_each(|&e| transcript.write(e))?;
        evals.insert(1, *claim - evals[0]);
        let r: Ext = transcript.draw();
        *claim = extrapolate(&evals, r);

        crate::mle::fix_var(poly, r);
        crate::mle::fix_var(&mut self.split_eq.left, r);

        Ok(r)
    }
}

impl<F: Field, Ext: ExtensionField<F>> SumcheckProver<F, Ext> for EqSumcheck<F, Ext> {
    type Cfg = usize;
    type Verifier = EqSumcheckVerifier<F, Ext>;

    fn new(zs: &[Ext], mat: &MatrixOwn<F>, eq_split: usize) -> Self {
        let k = mat.k();
        assert_eq!(k, zs.len());

        let (split_eq, evals) = tracing::info_span!("eval", s = eq_split).in_scope(|| {
            let split_eq = crate::mle::SplitEq::new(zs, eq_split);
            let evals = split_eq.eval_mat(mat);
            (split_eq, evals)
        });

        Self {
            split_eq,
            round: 0,
            k,
            evals,
            _phantom: std::marker::PhantomData,
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
        assert!(self.round < self.k);
        self.round += 1;
        if poly.k() > self.split_eq.left.k() {
            self.round_chunked(transcript, claim, poly)
        } else {
            self.round_nonchunked(transcript, claim, poly)
        }
    }
}

pub struct EqSumcheckVerifier<F: Field, Ext: ExtensionField<F>> {
    rs: Vec<Ext>,
    claim: Ext,
    zs: Vec<Ext>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> SumcheckVerifier<F, Ext> for EqSumcheckVerifier<F, Ext> {
    fn new(claim: Ext, zs: &[Ext]) -> Self {
        Self {
            claim,
            zs: zs.to_vec(),
            rs: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    fn reduce_claim<Transcript>(&mut self, reader: &mut Transcript) -> Result<Ext, crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<Ext>,
    {
        let v0: Ext = reader.read()?;
        let v2 = reader.read()?;
        let v = vec![v0, self.claim - v0, v2];
        let r: Ext = reader.draw();
        self.claim = extrapolate(&v, r);
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

        let (z_fin, z_eq) = self.zs.split_at(n);
        let mut rs = self.rs.clone();
        rs.reverse();
        let eq = crate::mle::eval_eq_xy(z_eq, &rs);
        let fe = crate::mle::eval_poly(z_fin, &fin);

        (fe * eq == self.claim)
            .then_some(())
            .ok_or(crate::Error::Verify)?;

        Ok((fin, self.rs))
    }
}

#[cfg(test)]
mod test {
    use crate::{
        basefold::sumcheck::{SumcheckProver, SumcheckVerifier},
        data::MatrixOwn,
        hash::{
            rust_crypto::{RustCryptoReader, RustCryptoWriter},
            transcript::{Challenge, Reader, Writer},
        },
        utils::{n_rand, VecOps},
    };
    use p3_field::{extension::BinomialExtensionField, ExtensionField, Field};

    impl<F: Field, Ext: ExtensionField<F>> super::EqSumcheck<F, Ext> {
        #[tracing::instrument(skip_all)]
        fn run_prover<Transcript>(
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

    impl<F: Field, Ext: ExtensionField<F>> super::EqSumcheckVerifier<F, Ext> {
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
    fn test_eq_sumcheck() {
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
        let evals = crate::mle::SplitEq::new(&zs, 1).eval_mat(&mat);

        for eq_split in 1..7 {
            for d in 0..=k {
                let (proof, checkpoint0) = {
                    let mut writer = Writer::init(b"");
                    let mut sp = super::EqSumcheck::new(&zs, &mat, eq_split);
                    assert_eq!(evals, sp.evals());
                    writer.write_many(sp.evals()).unwrap();
                    let alpha = writer.draw();
                    let mut claim: Ext = evals.horner(alpha);
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

                    let mut sv = super::EqSumcheckVerifier::<F, Ext>::new(claim, &zs);
                    let _ = sv.run_verifier(&mut reader, d).unwrap();
                    sv.verify(&mut reader).unwrap();

                    let checkpoint1: F = reader.draw();
                    assert_eq!(checkpoint0, checkpoint1);
                }
            }
        }
    }
}
