use crate::arithmetic::VecOps;
use crate::data::MatrixOwn;
use crate::data::MatrixRef;
use crate::field::ExtField;
use crate::field::Field;
use crate::hash::transcript::Challenge;
use crate::hash::transcript::ChallengeBits;
use crate::hash::transcript::Reader;
use crate::hash::transcript::Writer;
use crate::merkle::matrix::MatrixCommitment;
use crate::merkle::matrix::MatrixCommitmentData;
use crate::merkle::vector::VectorCommitment;
use crate::merkle::vector::VectorCommitmentData;
use crate::Error;
use code::RandomFoldableCode;
use itertools::Itertools;
use std::marker::PhantomData;
use sumcheck::batch_sumcheck_prover_round;
use sumcheck::sumcheck_verifier_round;

pub mod code;
pub mod sumcheck;
#[cfg(test)]
pub mod test;

pub struct Basefold<
    F: Field,
    Ext: ExtField<F>,
    MCom: MatrixCommitment<F>,
    VCom: VectorCommitment<Ext>,
    Code: RandomFoldableCode<F>,
> {
    code: Code,
    n_test: usize,
    mat_comm: MCom,
    vec_comm: VCom,
    _phantom: PhantomData<(F, Ext)>,
}

impl<
        F: Field,
        Ext: ExtField<F>,
        MCom: MatrixCommitment<F>,
        VCom: VectorCommitment<Ext>,
        Code: RandomFoldableCode<F>,
    > Basefold<F, Ext, MCom, VCom, Code>
{
    pub fn new(code: Code, mat_comm: MCom, vec_comm: VCom, n_test: usize) -> Self {
        Self {
            code,
            n_test,
            mat_comm,
            vec_comm,
            _phantom: PhantomData,
        }
    }

    pub fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: &MatrixOwn<F>,
    ) -> Result<MatrixCommitmentData<F, MCom::Digest>, Error>
    where
        Transcript: Writer<F> + Writer<MCom::Digest>,
    {
        // commit to the encoded data
        let cw = self.code.encode(data);
        self.mat_comm.commit(transcript, cw)
    }

    fn fold<Transcript>(
        &self,
        transcript: &mut Transcript,
        evals: &[Ext],
        points: &[&[Ext]],
        data: &mut Vec<Ext>,
        cw: &mut Vec<Ext>,
    ) -> Result<Vec<VectorCommitmentData<Ext, VCom::Digest>>, Error>
    where
        Transcript: Writer<VCom::Digest> + Writer<Ext> + Challenge<Ext>,
    {
        let beta: Ext = transcript.draw();
        let mut sum = evals.horner(beta);

        let mut eqs = {
            let eqs = points
                .iter()
                .flat_map(|p| crate::mle::eq(p))
                .collect::<Vec<_>>();
            let num_vars = points.first().unwrap().len();
            let height = 1 << num_vars;
            let width = points.len();
            let mut output = vec![Ext::ZERO; height * width]; // TODO: use cheaper alloc
            transpose::transpose(&eqs, &mut output, height, width);
            MatrixOwn::new(width, output)
        };

        (0..self.code.d())
            .map(|_| {
                let r = batch_sumcheck_prover_round(transcript, &mut sum, &mut eqs, data, beta)?;
                let comm: VectorCommitmentData<Ext, VCom::Digest> =
                    self.vec_comm.commit(transcript, cw.to_vec())?;
                self.code.fold(cw, r);
                Ok(comm)
            })
            .collect()
    }

    pub fn open<Transcript>(
        &self,
        transcript: &mut Transcript,
        points: &[&[Ext]],
        data: &MatrixOwn<F>,
        comm: &MatrixCommitmentData<F, MCom::Digest>,
    ) -> Result<(), Error>
    where
        Transcript: Writer<F>
            + Writer<Ext>
            + Writer<MCom::Digest>
            + Writer<VCom::Digest>
            + Challenge<Ext>
            + ChallengeBits,
    {
        assert_eq!(self.code.n_vars(), data.k());
        assert_eq!(self.code.n_vars() + self.code.c(), comm.k());
        assert_eq!(self.code.n_vars(), points.first().unwrap().len());
        assert!(!points.is_empty());

        // evaluate the data at the point
        let evals = points
            .iter()
            .map(|point| {
                let evals = crate::mle::eval_mat(data, point);
                evals.iter().try_for_each(|&e| transcript.write(e))?;
                Ok(evals)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // compress with alpha
        let alpha: Ext = Challenge::draw(transcript);
        let mut compressed_data = data.iter().map(|row| row.horner(alpha)).collect::<Vec<_>>();
        let mut compressed_cw = comm
            .data
            .iter()
            .map(|row| row.horner(alpha))
            .collect::<Vec<_>>();
        let compressed_eval = evals
            .iter()
            .map(|row| row.horner(alpha))
            .collect::<Vec<_>>();

        // run sumcheck and fold the data
        let comms = self.fold(
            transcript,
            &compressed_eval,
            points,
            &mut compressed_data,
            &mut compressed_cw,
        )?;

        debug_assert_eq!(
            self.code.encode_base(MatrixRef::new(1, &compressed_data)),
            compressed_cw
        );
        // write final poly to the transcript
        transcript.write_many(&compressed_data)?;

        // draw query indexes
        let k = comm.k();
        let indicies: Vec<usize> = (0..self.n_test)
            .map(|_| ChallengeBits::draw(transcript, k))
            .collect::<Vec<_>>();

        // open queries
        for mut index in indicies.clone().into_iter() {
            self.mat_comm.query(transcript, index, comm).unwrap();
            for (round, comm) in comms.iter().enumerate() {
                let mid = 1 << (k - round - 1);
                index ^= mid; // flip the index to the sibling
                self.vec_comm.query(transcript, index, comm)?;
                index &= mid - 1; // half the size
            }
        }

        Ok(())
    }

    pub fn verify<Transcript>(
        &self,
        comm: MCom::Digest,
        width: usize,
        transcript: &mut Transcript,
        points: &[&[Ext]],
    ) -> Result<(), Error>
    where
        Transcript: Reader<F>
            + Reader<Ext>
            + Reader<MCom::Digest>
            + Reader<VCom::Digest>
            + Challenge<Ext>
            + ChallengeBits,
    {
        // read and compress evaluations
        let num_points = points.len();
        let evals: Vec<Vec<Ext>> = (0..num_points)
            .map(|_| transcript.read_many(width))
            .collect::<Result<Vec<_>, _>>()?;
        let alpha: Ext = Challenge::draw(transcript);
        let k = self.code.n_vars() + self.code.c();
        let compressed_eval = evals
            .iter()
            .map(|row| row.horner(alpha))
            .collect::<Vec<_>>();

        // run sumcheck and read round commitments
        // let mut red = compressed_evals;

        let mut comms: Vec<VCom::Digest> = vec![];
        let mut rs = vec![];
        let beta: Ext = Challenge::draw(transcript);
        let mut red = compressed_eval.horner(beta);
        for _ in 0..self.code.d() {
            rs.push(sumcheck_verifier_round(2, &mut red, transcript)?);
            comms.push(transcript.read()?);
        }
        // read final polynomial
        let poly: Vec<Ext> = transcript.read_many(1 << self.code.k0())?;

        // make sure final poly is correct
        {
            // let (z_poly, z_eq) = point.split_at(self.code.k0());
            // let u0 = crate::mle::eval_eq_xy(&rs, z_eq);
            // let u1 = crate::mle::eval(&poly, z_poly);
            // (u0 * u1 == red).then_some(()).ok_or(Error::Verify)?;

            rs.reverse();
            let e = points
                .iter()
                .map(|point| {
                    let (z_poly, z_eq) = point.split_at(self.code.k0());
                    let e1 = crate::mle::eval_eq_xy(&rs, z_eq);
                    let e0 = crate::mle::eval(&poly, z_poly);
                    e0 * e1
                })
                .collect::<Vec<_>>();
            rs.reverse();
            (e.horner(beta) == red).then_some(()).ok_or(Error::Verify)?;
        }

        // find the base codeword
        let cw0 = self.code.encode_base(MatrixRef::new(1, &poly));

        // draw query indexes
        let indicies: Vec<usize> = (0..self.n_test)
            .map(|_| ChallengeBits::draw(transcript, self.code.n_vars() + self.code.c()))
            .collect_vec();

        // open queries
        indicies
            .into_iter()
            .map(|index| {
                let row = self
                    .mat_comm
                    .verify(transcript, comm, index, width, k)
                    .unwrap();
                let mut acc = row.horner(alpha);

                let mut index_acc = index;
                for (round, (comm, r)) in comms.iter().zip(rs.iter()).enumerate() {
                    let mid = 1 << (k - round - 1);
                    index_acc ^= mid; // flip the index to the sibling

                    let (u0, u1) = self
                        .vec_comm
                        .verify(transcript, *comm, acc, index_acc, k - round)
                        .unwrap();

                    index_acc &= mid - 1; // half the size

                    // ascend a fri folding step
                    acc = self
                        .code
                        .fold_single(self.code.d() - round - 1, index_acc, &u0, &u1, r);
                }

                (acc == cw0[index % (1 << self.code.n0())])
                    .then_some(())
                    .ok_or(Error::Verify)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }
}
