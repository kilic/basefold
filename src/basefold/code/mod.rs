use crate::{
    data::{MatrixOwn, MatrixRef},
    utils::TwoAdicSlice,
};
use p3_field::{ExtensionField, Field};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub mod basecode;
pub mod rs;

pub trait RandomFoldableCode<F: Field> {
    fn n_vars(&self) -> usize;
    fn c(&self) -> usize;

    // Remove this
    fn n0(&self) -> usize {
        self.k0() + self.c()
    }

    // Remove this
    fn k0(&self) -> usize {
        0
    }

    // Remove this
    fn d(&self) -> usize {
        self.n_vars() - self.k0()
    }

    fn encode_base<'a, Ext: ExtensionField<F>>(&self, m: MatrixRef<'a, Ext>) -> Vec<Ext>;
    fn encode(&self, m: &MatrixOwn<F>) -> MatrixOwn<F>;
    fn ts_invs(&self) -> &[Vec<F>];

    fn fold<Ext: ExtensionField<F>>(&self, cw: &mut Vec<Ext>, x: Ext) {
        let t = &self.ts_invs()[cw.k() - self.n0() - 1];
        let mid = cw.len() / 2;
        let (l, r) = cw.split_at_mut(mid);
        t.par_iter().zip_eq(l).zip_eq(r).for_each(|((&t, l), r)| {
            let u0 = (*l + *r).halve();
            let u1 = (*l - *r) * t;
            *l = u0 + (u1 - u0) * x
        });
        cw.truncate(mid);
    }

    fn fold_single<Ext: ExtensionField<F>>(
        &self,
        height: usize,
        index: usize,
        l: &Ext,
        r: &Ext,
        x: &Ext,
    ) -> Ext
    where
        F: Field,
    {
        let t = self.ts_invs()[height][index];
        let u0 = (*l + *r).halve();
        let u1 = (*l - *r) * t;
        u0 + (u1 - u0) * *x
    }
}
