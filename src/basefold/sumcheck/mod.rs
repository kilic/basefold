pub mod eq_sumcheck;
pub mod pointcheck;
pub use eq_sumcheck::*;
pub use pointcheck::*;

use crate::{
    data::MatrixOwn,
    field::{ExtField, Field},
    hash::transcript::{Challenge, Reader, Writer},
};

pub trait SumcheckVerifier<F: Field, Ext: ExtField<F>> {
    fn new(claim: Ext, zs: &[Ext]) -> Self;
    fn reduce_claim<Transcript>(
        &mut self,
        transcript: &mut Transcript,
    ) -> Result<Ext, crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<Ext>;
    fn verify<Transcript>(
        self,
        reader: &mut Transcript,
    ) -> Result<(Vec<Ext>, Vec<Ext>), crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<Ext>;
}

pub trait SumcheckProver<F: Field, Ext: ExtField<F>>: Sized {
    type Cfg;
    type Verifier: SumcheckVerifier<F, Ext>;
    fn new(zs: &[Ext], mat: &MatrixOwn<F>, cfg: Self::Cfg) -> Self;
    fn round<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        claim: &mut Ext,
        poly: &mut Vec<Ext>,
    ) -> Result<Ext, crate::Error>
    where
        Transcript: Writer<Ext> + Challenge<Ext>;
    fn evals(&self) -> &[Ext];
}
