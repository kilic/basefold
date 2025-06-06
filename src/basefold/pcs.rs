use p3_field::ExtensionField;
use p3_field::Field;

use super::code::RandomFoldableCode;
use super::sumcheck::SumcheckProver;
use super::sumcheck::SumcheckVerifier;
use crate::data::MatrixOwn;
use crate::data::MatrixRef;
use crate::hash::transcript::Challenge;
use crate::hash::transcript::ChallengeBits;
use crate::hash::transcript::Reader;
use crate::hash::transcript::Writer;
use crate::merkle::matrix::MatrixCommitment;
use crate::merkle::matrix::MatrixCommitmentData;
use crate::merkle::vector::VectorCommitment;
use crate::merkle::vector::VectorCommitmentData;
use crate::utils::VecOps;
use crate::Error;
use std::marker::PhantomData;

pub struct Basefold<
    F: Field,
    Ext: ExtensionField<F>,
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
        Ext: ExtensionField<F>,
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

    fn fold<Transcript, S>(
        &self,
        transcript: &mut Transcript,
        sumcheck: &mut S,
        claim: &mut Ext,
        data: &mut Vec<Ext>,
        cw: &mut Vec<Ext>,
    ) -> Result<Vec<VectorCommitmentData<Ext, VCom::Digest>>, Error>
    where
        Transcript: Writer<VCom::Digest> + Writer<Ext> + Challenge<Ext>,
        S: SumcheckProver<F, Ext>,
    {
        (0..self.code.d())
            .map(|_| {
                let r = sumcheck.round(transcript, claim, data)?;
                let comm: VectorCommitmentData<Ext, VCom::Digest> =
                    self.vec_comm.commit(transcript, cw.to_vec())?;
                self.code.fold(cw, r);
                Ok(comm)
            })
            .collect()
    }

    pub fn open<Transcript, Sumcheck>(
        &self,
        transcript: &mut Transcript,
        sumcheck: &mut Sumcheck,
        zs: &[Ext],
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
        Sumcheck: SumcheckProver<F, Ext>,
    {
        assert_eq!(self.code.n_vars(), data.k());
        assert_eq!(self.code.n_vars() + self.code.c(), comm.k());
        assert_eq!(self.code.n_vars(), zs.len());

        // write matrix evaluations
        transcript.write_many(sumcheck.evals())?;

        // compress with alpha
        let alpha: Ext = Challenge::draw(transcript);
        let mut compressed_data = data.iter().map(|row| row.horner(alpha)).collect::<Vec<_>>();
        let mut compressed_cw = comm
            .data
            .iter()
            .map(|row| row.horner(alpha))
            .collect::<Vec<_>>();
        let mut claim = sumcheck.evals().horner(alpha);

        // run sumcheck and fold the data
        let comms = self.fold(
            transcript,
            sumcheck,
            &mut claim,
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

    pub fn verify<Transcript, Sumcheck>(
        &self,
        transcript: &mut Transcript,
        comm: MCom::Digest,
        width: usize,
        zs: &[Ext],
    ) -> Result<(), Error>
    where
        Transcript: Reader<F>
            + Reader<Ext>
            + Reader<MCom::Digest>
            + Reader<VCom::Digest>
            + Challenge<Ext>
            + ChallengeBits,
        Sumcheck: SumcheckVerifier<F, Ext>,
    {
        let k = self.code.n_vars() + self.code.c();

        // read matrix evaluations
        let evals: Vec<Ext> = transcript.read_many(width)?;

        // compress with alpha
        let alpha: Ext = Challenge::draw(transcript);
        let claim = evals.horner(alpha);

        let mut comms: Vec<VCom::Digest> = vec![];

        let mut sv = Sumcheck::new(claim, zs);
        for _ in 0..self.code.d() {
            sv.reduce_claim(transcript)?;
            comms.push(transcript.read()?);
        }

        let (fin, rs) = sv.verify(transcript)?;

        // find the base codeword
        let cw0 = self.code.encode_base(MatrixRef::new(1, &fin));

        // draw query indexes
        let indicies: Vec<usize> = (0..self.n_test)
            .map(|_| ChallengeBits::draw(transcript, self.code.n_vars() + self.code.c()))
            .collect::<Vec<_>>();

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
