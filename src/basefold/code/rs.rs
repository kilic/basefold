use crate::data::{MatrixOwn, MatrixRef};
use p3_field::{ExtensionField, TwoAdicField};

#[derive(Clone, Debug)]
pub struct ReedSolomon<F> {
    c: usize,
    n_vars: usize,
    ts: Vec<Vec<F>>,
}

impl<F: TwoAdicField> ReedSolomon<F> {
    pub fn new(c: usize, n_vars: usize) -> Self {
        let n = 1 << c;
        let ts = (0..n_vars)
            .map(|i| {
                let mut t = F::two_adic_generator(c + i + 1)
                    .powers()
                    .take(n << i)
                    .map(|x| x.double())
                    .collect::<Vec<_>>();
                use crate::utils::arithmetic::BatchInverse;
                t.inverse();
                t
            })
            .collect::<Vec<_>>();

        Self { c, n_vars, ts }
    }
}

fn encode<F: TwoAdicField, Ext: ExtensionField<F>>(m: &MatrixOwn<Ext>, c: usize) -> MatrixOwn<Ext> {
    let mut m = m.owned();
    let k = m.k();

    m.reverse_bits();
    m.resize(k + c);

    let omega = F::two_adic_generator(k + c);
    let omega = Ext::from(omega);
    crate::utils::arithmetic::fft(&mut m, omega);
    m
}

impl<F: TwoAdicField> super::RandomFoldableCode<F> for ReedSolomon<F> {
    fn n_vars(&self) -> usize {
        self.n_vars
    }

    fn c(&self) -> usize {
        self.c
    }

    fn encode_base<'a, Ext: ExtensionField<F>>(&self, m: MatrixRef<'a, Ext>) -> Vec<Ext> {
        assert_eq!(m.width(), 1);

        // effectively,
        // vec![m.row(0).first().unwrap().clone(); 1 << self.c()];
        let m = m.owned();
        encode(&m, self.c).to_vec()
    }

    fn encode(&self, m: &MatrixOwn<F>) -> MatrixOwn<F> {
        assert_eq!(m.height(), 1 << self.n_vars);
        assert_eq!(m.k(), self.n_vars);
        encode(m, self.c)
    }

    fn ts_invs(&self) -> &[Vec<F>] {
        &self.ts
    }
}
