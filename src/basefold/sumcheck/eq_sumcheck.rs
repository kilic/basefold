use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    data::MatrixOwn,
    field::{ExtField, Field},
    hash::transcript::{Challenge, Reader, Writer},
    utils::{interpolate, TwoAdicSlice, VecOps},
};

use super::{SumcheckProver, SumcheckVerifier};

fn extrapolate<F: Field, EF: ExtField<F>>(evals: &[F], target: EF) -> EF {
    let points = (0..evals.len())
        .map(|i| F::from(i as u64))
        .collect::<Vec<_>>();
    interpolate(&points, evals).horner(target)
}

pub struct EqSumcheck<F: Field, Ext: ExtField<F>> {
    eq0: Vec<Ext>,
    eq1: Vec<Ext>,
    round: usize,
    k: usize,
    evals: Vec<Ext>,
    _phantom: std::marker::PhantomData<F>,
}

pub struct EqSumcheckVerifier<F: Field, Ext: ExtField<F>> {
    rs: Vec<Ext>,
    claim: Ext,
    zs: Vec<Ext>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtField<F>> EqSumcheckVerifier<F, Ext> {}

impl<F: Field, Ext: ExtField<F>> SumcheckVerifier<F, Ext> for EqSumcheckVerifier<F, Ext> {
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

impl<F: Field, Ext: ExtField<F>> EqSumcheck<F, Ext> {
    fn round_chunked<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        claim: &mut Ext,
        poly: &mut Vec<Ext>,
    ) -> Result<Ext, crate::Error>
    where
        Transcript: Writer<Ext> + Challenge<Ext>,
    {
        let chunk_size = self.eq0.len();
        let mid = poly.len() / 2;
        let (p0, p1) = poly.split_at_mut(mid);
        let evals = p0
            .chunks(chunk_size)
            .zip(p1.chunks(chunk_size))
            .map(|(part0, part1)| {
                part0
                    .par_iter()
                    .zip_eq(part1.par_iter())
                    .zip_eq(self.eq0.par_iter())
                    .map(|((&a0, &a1), &b)| [a0 * b, (a1.double() - a0) * b])
                    .reduce(|| [Ext::ZERO, Ext::ZERO], |a, b| [a[0] + b[0], a[1] + b[1]])
                    .to_vec()
            })
            .collect::<Vec<_>>();

        let mut evals = {
            let coeffs = evals.iter().map(|row| row[0]).collect::<Vec<_>>();
            let v0 = coeffs
                .iter()
                .zip(self.eq1.iter())
                .map(|(&a, &b)| b * a)
                .sum::<Ext>();

            let coeffs = evals.iter().map(|row| row[1]).collect::<Vec<_>>();
            let (zzlo, zzhi) = self.eq1.split_at(self.eq1.len() / 2);
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
        crate::mle::fix_var(&mut self.eq1, r);
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
        assert_eq!(poly.k(), self.eq0.k());
        let mid = poly.len() / 2;
        let (p0, p1) = poly.split_at_mut(mid);
        let (eq0, eq1) = self.eq0.split_at_mut(mid);
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
            let eq1 = *self.eq1.first().unwrap();
            evals[0] *= eq1;
            evals[1] *= eq1;
        }

        evals.iter().try_for_each(|&e| transcript.write(e))?;
        evals.insert(1, *claim - evals[0]);
        let r: Ext = transcript.draw();
        *claim = extrapolate(&evals, r);

        crate::mle::fix_var(poly, r);
        crate::mle::fix_var(&mut self.eq0, r);

        Ok(r)
    }
}

impl<F: Field, Ext: ExtField<F>> SumcheckProver<F, Ext> for EqSumcheck<F, Ext> {
    type Cfg = usize;
    type Verifier = EqSumcheckVerifier<F, Ext>;

    fn new(zs: &[Ext], mat: &MatrixOwn<F>, sweetness: usize) -> Self {
        let k = mat.k();
        assert_eq!(k, zs.len());

        let (eq0, eq1, evals) = tracing::info_span!("eval", s = sweetness).in_scope(|| {
            let (z0, z1) = zs.split_at(k - sweetness);
            let eq0 = crate::mle::eq(z0);
            let eq1 = crate::mle::eq(z1);

            let evals = (0..mat.width())
                .map(|col| {
                    mat.chunks(eq0.len())
                        .zip_eq(eq1.iter())
                        .map(|(part, &c)| {
                            c * part
                                .par_iter()
                                .zip(eq0.par_iter())
                                .map(|(a, &b)| b * a[col])
                                .sum::<Ext>()
                        })
                        .sum::<Ext>()
                })
                .collect();

            (eq0, eq1, evals)
        });

        Self {
            eq0,
            eq1,
            round: 0,
            k,
            evals,
            _phantom: std::marker::PhantomData,
        }
    }

    fn evals(&self) -> &[Ext] {
        &self.evals
    }

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
        if poly.k() > self.eq0.k() {
            self.round_chunked(transcript, claim, poly)
        } else {
            self.round_nonchunked(transcript, claim, poly)
        }
    }
}

#[cfg(test)]
mod test {

    use crate::{
        basefold::sumcheck::{SumcheckProver, SumcheckVerifier},
        data::MatrixOwn,
        field::{
            goldilocks::{Goldilocks, Goldilocks2},
            ExtField, Field,
        },
        hash::{
            rust_crypto::{RustCryptoReader, RustCryptoWriter},
            transcript::{Challenge, Reader, Writer},
        },
        utils::VecOps,
    };

    impl<F: Field, Ext: ExtField<F>> super::EqSumcheck<F, Ext> {
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

    impl<F: Field, Ext: ExtField<F>> super::EqSumcheckVerifier<F, Ext> {
        #[tracing::instrument(skip_all)]
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
        type F = Goldilocks;
        type Ext = Goldilocks2;
        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;

        // crate::test::init_tracing();
        let mut rng = crate::test::seed_rng();
        let k = 11;
        let width = 2;
        let mat = (0..width * (1 << k))
            .map(|_| F::rand(&mut rng))
            .collect::<Vec<_>>();
        let mat = MatrixOwn::new(width, mat);
        let zs = (0..k).map(|_| Ext::rand(&mut rng)).collect::<Vec<_>>();
        let evals = crate::mle::eval_mat(&zs, &mat, 1);

        for sweetness in 1..7 {
            let (proof, checkpoint0) = {
                let mut writer = Writer::init(b"");
                let mut sp = super::EqSumcheck::new(&zs, &mat, sweetness);
                assert_eq!(evals, sp.evals());
                writer.write_many(sp.evals()).unwrap();
                let alpha = writer.draw();
                let mut claim: Ext = evals.horner(alpha);

                let poly = mat.iter().map(|row| row.horner(alpha)).collect::<Vec<_>>();
                let mut fin = poly.clone();
                let mut _rs = sp.run_prover(&mut writer, k, &mut claim, &mut fin).unwrap();

                assert_eq!(fin.len(), 1);
                writer.write(fin[0]).unwrap();

                let checkpoint: F = writer.draw();
                (writer.finalize(), checkpoint)
            };

            {
                let mut reader = Reader::init(&proof, b"");
                let evals: Vec<Ext> = reader.read_many(mat.width()).unwrap();
                let alpha = reader.draw();
                let claim = evals.horner(alpha);

                let mut sv = super::EqSumcheckVerifier::<F, Ext>::new(claim, &zs);
                let _ = sv.run_verifier(&mut reader, k).unwrap();
                sv.verify(&mut reader).unwrap();

                let checkpoint1: F = reader.draw();
                assert_eq!(checkpoint0, checkpoint1);
            }
        }

        for sweetness in 1..7 {
            for d in 0..=k {
                let (proof, checkpoint0) = {
                    let mut writer = Writer::init(b"");
                    let mut sp = super::EqSumcheck::new(&zs, &mat, sweetness);
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
