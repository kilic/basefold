use crate::{
    arithmetic::VecOps,
    data::MatrixOwn,
    field::{ExtField, Field},
    hash::transcript::{Challenge, Reader, Writer},
    utils::interpolate,
    Error,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

fn extrapolate<F: Field, EF: ExtField<F>>(evals: &[F], target: EF) -> EF {
    let points = (0..evals.len())
        .map(|i| F::from(i as u64))
        .collect::<Vec<_>>();
    interpolate(&points, evals).horner(target)
}

pub fn sumcheck_verifier<F: Field, Transcript>(
    d: usize,
    degree: usize,
    mut sum: F,
    transcript: &mut Transcript,
) -> Result<(F, Vec<F>), Error>
where
    Transcript: Reader<F> + Reader<F> + Challenge<F>,
{
    let rs = (0..d)
        .map(|_k| sumcheck_verifier_round(degree, &mut sum, transcript))
        .collect::<Result<Vec<_>, Error>>()?;
    Ok((sum, rs))
}

pub fn sumcheck_verifier_round<F: ExtField<F>, Transcript>(
    degree: usize,
    sum: &mut F,
    transcript: &mut Transcript,
) -> Result<F, Error>
where
    Transcript: Reader<F> + Reader<F> + Challenge<F>,
{
    let v0: F = transcript.read()?;
    let v_rest = (0..degree - 1)
        .map(|_| transcript.read())
        .collect::<Result<Vec<_>, crate::Error>>()?;
    let v = [v0, *sum - v0]
        .into_iter()
        .chain(v_rest)
        .collect::<Vec<_>>();
    let r: F = transcript.draw();
    *sum = extrapolate(&v, r);
    Ok(r)
}

pub fn sumcheck_prover<F: Field, Transcript>(
    transcript: &mut Transcript,
    d: usize,
    mut sum: F,
    point: &[F],
    poly: &mut Vec<F>,
) -> Result<(F, Vec<F>), Error>
where
    Transcript: Writer<F> + Challenge<F>,
{
    assert!(point.len() >= d);
    let mut eq = crate::mle::eq(point);
    let rs = (0..d)
        .map(|_| sumcheck_prover_round(transcript, &mut sum, &mut eq, poly))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((sum, rs))
}

pub fn sumcheck_prover_round<F: Field, Transcript>(
    transcript: &mut Transcript,
    sum: &mut F,
    eq: &mut Vec<F>,
    poly: &mut Vec<F>,
) -> Result<F, Error>
where
    Transcript: Writer<F> + Challenge<F>,
{
    let mid = poly.len() / 2;
    let (p0, p1) = poly.split_at_mut(mid);
    let (eq0, eq1) = eq.split_at_mut(mid);
    let mut evals = p0
        .par_iter()
        .zip(p1.par_iter())
        .zip(eq0.par_iter())
        .zip(eq1.par_iter())
        .map(|(((&a0, &a1), &b0), &b1)| {
            let v0 = a0 * b0;
            let dif0 = a1 - a0;
            let dif1 = b1 - b0;
            let u0 = a1 + dif0;
            let u1 = b1 + dif1;
            let v2 = u0 * u1;
            [v0, v2]
        })
        .reduce(|| [F::ZERO, F::ZERO], |a, b| [a[0] + b[0], a[1] + b[1]])
        .to_vec();

    evals.iter().try_for_each(|&e| transcript.write(e))?;
    evals.insert(1, *sum - evals[0]);

    let r: F = transcript.draw();
    *sum = extrapolate(&evals, r);

    p0.par_iter_mut()
        .zip(p1.par_iter())
        .for_each(|(a0, a1)| *a0 += r * (*a1 - *a0));
    poly.truncate(mid);

    eq0.par_iter_mut()
        .zip(eq1.par_iter())
        .for_each(|(a0, a1)| *a0 += r * (*a1 - *a0));
    eq.truncate(mid);

    Ok(r)
}

pub fn batch_sumcheck_verifier<F: Field, Transcript>(
    d: usize,
    degree: usize,
    sums: &[F],
    transcript: &mut Transcript,
) -> Result<(F, Vec<F>), Error>
where
    Transcript: Reader<F> + Reader<F> + Challenge<F>,
{
    let beta: F = transcript.draw();
    let mut sum = sums.horner(beta);
    let rs = (0..d)
        .map(|_k| sumcheck_verifier_round(degree, &mut sum, transcript))
        .collect::<Result<Vec<_>, Error>>()?;
    Ok((sum, rs))
}

pub fn batch_sumcheck_prover<F: Field, Transcript>(
    transcript: &mut Transcript,
    d: usize,
    sums: &[F],
    points: &[&[F]],
    poly: &mut Vec<F>,
) -> Result<(F, Vec<F>, F), Error>
where
    Transcript: Writer<F> + Challenge<F>,
{
    assert!(!points.is_empty());
    assert_eq!(sums.len(), points.len());

    let beta: F = transcript.draw();
    let mut sum = sums.horner(beta);

    let mut eqs = {
        let eqs = points
            .iter()
            .flat_map(|p| crate::mle::eq(p))
            .collect::<Vec<_>>();
        let num_vars = points.first().unwrap().len();
        let height = 1 << num_vars;
        let width = points.len();
        let mut output = vec![F::ZERO; height * width];
        transpose::transpose(&eqs, &mut output, height, width);
        MatrixOwn::new(width, output)
    };

    let rs = (0..d)
        .map(|_| batch_sumcheck_prover_round(transcript, &mut sum, &mut eqs, poly, beta))
        .collect::<Result<Vec<_>, _>>()?;

    Ok((sum, rs, beta))
}

pub fn batch_sumcheck_prover_round<F: Field, Transcript>(
    transcript: &mut Transcript,
    sum: &mut F,
    eqs: &mut MatrixOwn<F>,
    poly: &mut Vec<F>,
    beta: F,
) -> Result<F, Error>
where
    Transcript: Writer<F> + Challenge<F>,
{
    let mid = poly.k() - 1;
    let (p0, p1) = poly.split_at_mut(1 << mid);
    use crate::utils::TwoAdicSlice;
    let (mut eq0, eq1) = eqs.split_mut(mid);
    let mut evals = p0
        .par_iter()
        .zip(p1.par_iter())
        .zip(eq0.par_iter())
        .zip(eq1.par_iter())
        .map(|(((&a0, &a1), eq0), eq1)| {
            let v0 = a0 * eq0.horner(beta);

            let dif0 = a1 - a0;
            let difeq = eq0
                .iter()
                .zip(eq1.iter())
                .map(|(&e0, &e1)| e1 - e0)
                .collect::<Vec<_>>();

            let u0 = a1 + dif0;
            let u1 = eq1
                .iter()
                .zip(difeq.iter())
                .map(|(&e1, &dif)| e1 + dif)
                .collect::<Vec<_>>();

            let v2 = u0 * u1.horner(beta);
            [v0, v2]

            //
        })
        .reduce(|| [F::ZERO, F::ZERO], |a, b| [a[0] + b[0], a[1] + b[1]])
        .to_vec();

    evals.iter().try_for_each(|&e| transcript.write(e))?;
    evals.insert(1, *sum - evals[0]);

    let r: F = transcript.draw();
    *sum = extrapolate(&evals, r);

    p0.par_iter_mut()
        .zip(p1.par_iter())
        .for_each(|(a0, a1)| *a0 += r * (*a1 - *a0));
    poly.truncate(1 << mid);

    eq0.par_iter_mut().zip(eq1.par_iter()).for_each(|(a0, a1)| {
        a0.par_iter_mut()
            .zip(a1.par_iter())
            .for_each(|(e0, e1)| *e0 += r * (*e1 - *e0))
    });
    eqs.truncate(mid);

    Ok(r)
}

#[cfg(test)]
mod test {

    use crate::{
        arithmetic::VecOps,
        basefold::sumcheck::{
            batch_sumcheck_prover, batch_sumcheck_verifier, sumcheck_prover, sumcheck_verifier,
        },
        field::{goldilocks::Goldilocks, Field},
        hash::{
            rust_crypto::{RustCryptoReader, RustCryptoWriter},
            transcript::Challenge,
        },
    };

    #[test]
    fn test_sumcheck_single() {
        type F = Goldilocks;
        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;

        let mut rng = crate::test::seed_rng();
        let k = 8;
        let poly = (0..1 << k).map(|_| F::rand(&mut rng)).collect::<Vec<_>>();
        let point = (0..k).map(|_| F::rand(&mut rng)).collect::<Vec<_>>();
        let sum = crate::mle::eval(&poly, &point);

        {
            let mut writer = Writer::init(b"");
            let mut fin = poly.clone();
            let (red0, mut rs0) = sumcheck_prover(&mut writer, k, sum, &point, &mut fin).unwrap();

            assert_eq!(fin.len(), 1);

            rs0.reverse();
            let e0 = crate::mle::eval(&poly, &rs0);
            let e1 = crate::mle::eval_eq_xy(&point, &rs0);

            assert_eq!(e0, *fin.first().unwrap());
            assert_eq!(e0 * e1, red0);

            let checkpoint0: F = writer.draw();
            let proof = writer.finalize();

            let mut reader = Reader::init(&proof, b"");
            let (red1, mut rs1) = sumcheck_verifier(k, 2, sum, &mut reader).unwrap();
            rs1.reverse();
            assert_eq!(red0, red1);
            assert_eq!(rs0, rs1);

            let checkpoint1: F = reader.draw();
            assert_eq!(checkpoint0, checkpoint1);
        }

        for d in 0..8 {
            let mut writer = Writer::init(b"");
            let mut fin = poly.clone();
            let (red0, mut rs0) = sumcheck_prover(&mut writer, d, sum, &point, &mut fin).unwrap();

            assert_eq!(fin.len(), 1 << (k - d));
            let (z_fin, z_eq) = point.split_at(k - d);

            rs0.reverse();
            let e1 = crate::mle::eval_eq_xy(&rs0, z_eq);
            let e0 = crate::mle::eval(&fin, z_fin);
            assert_eq!(e0 * e1, red0);

            let checkpoint0: F = writer.draw();
            let proof = writer.finalize();

            let mut reader = Reader::init(&proof, b"");
            let (red1, mut rs1) = sumcheck_verifier(d, 2, sum, &mut reader).unwrap();
            rs1.reverse();
            assert_eq!(red0, red1);
            assert_eq!(rs0, rs1);

            let checkpoint1: F = reader.draw();
            assert_eq!(checkpoint0, checkpoint1);
        }
    }

    #[test]
    fn test_sumcheck_batch() {
        type F = Goldilocks;
        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;

        let mut rng = crate::test::seed_rng();
        let k = 8;
        let poly = (0..1 << k).map(|_| F::rand(&mut rng)).collect::<Vec<_>>();

        {
            for num_points in 1..5 {
                let points = (0..num_points)
                    .map(|_| (0..k).map(|_| F::rand(&mut rng)).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                let points = points.iter().map(|p| p.as_slice()).collect::<Vec<_>>();

                let sums = points
                    .iter()
                    .map(|point| crate::mle::eval(&poly, point))
                    .collect::<Vec<_>>();

                {
                    let mut writer = Writer::init(b"");
                    let mut fin = poly.clone();
                    let (red0, mut rs0, beta) =
                        batch_sumcheck_prover(&mut writer, k, &sums, &points, &mut fin).unwrap();
                    assert_eq!(fin.len(), 1);

                    rs0.reverse();
                    let eqs = points
                        .iter()
                        .map(|z| crate::mle::eval_eq_xy(z, &rs0))
                        .collect::<Vec<_>>();

                    let e0 = crate::mle::eval(&poly, &rs0);
                    let e1 = eqs.horner(beta);
                    assert_eq!(e0 * e1, red0);

                    let checkpoint0: F = writer.draw();
                    let proof = writer.finalize();
                    let mut reader = Reader::init(&proof, b"");
                    let (red1, mut rs1) =
                        batch_sumcheck_verifier(k, 2, &sums, &mut reader).unwrap();
                    rs1.reverse();
                    assert_eq!(red0, red1);
                    assert_eq!(rs0, rs1);

                    let checkpoint1: F = reader.draw();
                    assert_eq!(checkpoint0, checkpoint1);
                }

                for d in 0..=k {
                    let mut writer = Writer::init(b"");
                    let mut fin = poly.clone();
                    let (red0, mut rs0, beta) =
                        batch_sumcheck_prover(&mut writer, d, &sums, &points, &mut fin).unwrap();
                    assert_eq!(fin.len(), 1 << (k - d));

                    rs0.reverse();
                    let e = points
                        .iter()
                        .map(|point| {
                            let (z_fin, z_eq) = point.split_at(k - d);
                            let e1 = crate::mle::eval_eq_xy(&rs0, z_eq);
                            let e0 = crate::mle::eval(&fin, z_fin);
                            e0 * e1
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(e.horner(beta), red0);

                    let checkpoint0: F = writer.draw();
                    let proof = writer.finalize();
                    let mut reader = Reader::init(&proof, b"");
                    let (red1, mut rs1) =
                        batch_sumcheck_verifier(d, 2, &sums, &mut reader).unwrap();
                    rs1.reverse();
                    assert_eq!(red0, red1);
                    assert_eq!(rs0, rs1);

                    let checkpoint1: F = reader.draw();
                    assert_eq!(checkpoint0, checkpoint1);
                }
            }
        }
    }
}
