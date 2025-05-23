use crate::data::{MatrixOwn, MatrixRef};
use crate::utils::BatchInverse;
use p3_field::{ExtensionField, Field};
use rand::{distr::StandardUniform, prelude::Distribution, Rng};
use rayon::iter::ParallelIterator;

#[derive(Clone, Debug)]
pub struct Basecode<F> {
    g0: Vec<Vec<F>>,
    c: usize,
    k0: usize,
    n_vars: usize,
    ts: Vec<Vec<F>>,
    ts_inv: Vec<Vec<F>>,
}

impl<F: Field> Basecode<F> {
    pub fn generate(mut rng: impl rand::RngCore, n_vars: usize, k0: usize, c: usize) -> Basecode<F>
    where
        StandardUniform: Distribution<F>,
    {
        let d = n_vars.checked_sub(k0).unwrap(); // is this distance?
        let n0 = k0 + c;

        let g0 = (0..1 << n0)
            .map(|i| F::from_usize(i).powers().take(1 << k0).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let (ts, ts_inv): (Vec<Vec<F>>, Vec<Vec<F>>) = (0..d)
            .map(|i| {
                let ts = (0..(1 << n0) << i)
                    .map(|_| rng.random())
                    .collect::<Vec<F>>();
                let mut ts_inv = ts.iter().map(|t_i| t_i.double()).collect::<Vec<_>>();
                ts_inv.inverse();
                (ts, ts_inv)
            })
            .unzip();

        Basecode {
            n_vars,
            g0,
            c,
            k0,
            ts,
            ts_inv,
        }
    }
}

impl<F: Field> super::RandomFoldableCode<F> for Basecode<F> {
    fn n_vars(&self) -> usize {
        self.n_vars
    }

    fn k0(&self) -> usize {
        self.k0
    }

    fn c(&self) -> usize {
        self.c
    }

    fn encode_base<'a, Ext: ExtensionField<F>>(&self, m: MatrixRef<'a, Ext>) -> Vec<Ext> {
        assert_eq!(m.height(), 1 << self.k0());
        self.g0
            .iter()
            .flat_map(|g_col| {
                let mut c_row = vec![Ext::ZERO; m.width()];
                g_col.iter().zip(m.iter()).for_each(|(&g_e, m_row)| {
                    m_row
                        .iter()
                        .zip(c_row.iter_mut())
                        .for_each(|(&m_i, c_i)| *c_i += m_i * g_e)
                });
                c_row
            })
            .collect::<Vec<_>>()
    }

    fn encode(&self, m: &MatrixOwn<F>) -> MatrixOwn<F> {
        assert_eq!(m.height(), 1 << self.n_vars);

        let cw = m
            .par_chunks(1 << self.k0())
            .flat_map(|chunk| self.encode_base(chunk))
            .collect::<Vec<_>>();

        let mut cw = MatrixOwn::new(m.width(), cw);
        (1..=self.d()).for_each(|i| {
            let ni = 1 << (self.n0() + i);
            cw.par_chunks_mut(ni).for_each(|mut chunk| {
                let (mut l, mut r) = chunk.split_half_mut();
                l.iter_mut()
                    .zip(r.iter_mut())
                    .zip(self.ts[i - 1].iter())
                    .for_each(|((l, r), &t)| {
                        l.iter_mut().zip(r.iter_mut()).for_each(|(a, b)| {
                            let x = *b * t;
                            *b = *a - x;
                            *a += x;
                        });
                    });
            });
        });

        cw
    }

    fn ts_invs(&self) -> &[Vec<F>] {
        &self.ts_inv
    }
}
