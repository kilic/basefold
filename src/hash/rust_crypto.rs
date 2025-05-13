use crate::field::FromUniformBytes;
use crate::Error;
use digest::{Digest, FixedOutputReset, Output};
use p3_field::Field;
use std::io::{Read, Write};

use super::transcript::{Challenge, ChallengeBits, Reader, Writer};
use super::{Compress, Hasher};

#[derive(Debug, Clone, Default)]
pub struct RustCrypto<D: Digest> {
    _0: std::marker::PhantomData<D>,
}

impl<D: Digest> RustCrypto<D> {
    pub fn new() -> Self {
        Self {
            _0: Default::default(),
        }
    }

    pub(crate) fn result_to_field<F: Field + FromUniformBytes>(out: &Output<D>) -> F {
        let out = out.iter().rev().cloned().collect::<Vec<_>>();
        F::from_bytes(&out)
    }

    pub(crate) fn result_to_bits(out: &Output<D>, bit_size: usize) -> usize {
        let ret = usize::from_le_bytes(out[0..usize::BITS as usize / 8].try_into().unwrap());
        ret & ((1 << bit_size) - 1)
    }
}

impl<D: Digest + FixedOutputReset> RustCrypto<D> {
    fn cycle(h: &mut D) -> Output<D> {
        let ret = h.finalize_reset();
        Digest::update(h, &ret);
        ret
    }

    pub(crate) fn draw_field_element<F: FromUniformBytes>(h: &mut D) -> F {
        let ret = Self::cycle(h);
        RustCrypto::<D>::result_to_field(&ret)
    }

    pub(crate) fn draw_bits(h: &mut D, bit_size: usize) -> usize {
        let ret = Self::cycle(h);
        RustCrypto::<D>::result_to_bits(&ret, bit_size)
    }
}

impl<D: Digest + FixedOutputReset + Send + Sync, F: FromUniformBytes> Hasher<F, [u8; 32]>
    for RustCrypto<D>
{
    fn hash(&self, input: &[F]) -> [u8; 32] {
        let mut h = D::new();
        input
            .iter()
            .for_each(|el| Digest::update(&mut h, el.to_bytes()));
        Digest::finalize(h).to_vec().try_into().unwrap()
    }
}

impl<D: Digest + FixedOutputReset + Send + Sync, const N: usize> Compress<[u8; 32], N>
    for RustCrypto<D>
{
    fn compress(&self, input: [[u8; 32]; N]) -> [u8; 32] {
        let mut h = D::new();
        input.iter().for_each(|e| Digest::update(&mut h, e));
        Digest::finalize(h).to_vec().try_into().unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct RustCryptoWriter<W: Write, D: Digest + FixedOutputReset> {
    h: D,
    pub(crate) writer: W,
}

impl<W: Write + Default, D: Digest + FixedOutputReset> RustCryptoWriter<W, D> {
    pub fn init(prefix: impl AsRef<[u8]>) -> Self {
        RustCryptoWriter {
            h: D::new_with_prefix(prefix),
            writer: W::default(),
        }
    }
}

impl<W: Write, D: Digest + FixedOutputReset> RustCryptoWriter<W, D> {
    pub fn finalize(self) -> W {
        self.writer
    }

    pub fn update(&mut self, data: impl AsRef<[u8]>) {
        Digest::update(&mut self.h, data);
    }
}

impl<W: Write, D: Digest + FixedOutputReset, F: FromUniformBytes> Writer<F>
    for RustCryptoWriter<W, D>
{
    fn unsafe_write(&mut self, e: F) -> Result<(), Error> {
        let data = e.to_bytes();
        self.writer
            .write_all(data.as_ref())
            .map_err(|_| Error::Transcript)?;
        Ok(())
    }

    fn write(&mut self, e: F) -> Result<(), Error> {
        self.unsafe_write(e)?;
        self.update(e.to_bytes());
        Ok(())
    }
}

impl<W: Write, D: Digest + FixedOutputReset> Writer<[u8; 32]> for RustCryptoWriter<W, D> {
    fn unsafe_write(&mut self, e: [u8; 32]) -> Result<(), Error> {
        self.writer
            .write_all(e.as_ref())
            .map_err(|_| Error::Transcript)?;
        Ok(())
    }

    fn write(&mut self, e: [u8; 32]) -> Result<(), Error> {
        self.unsafe_write(e)?;
        self.update(e);
        Ok(())
    }
}

impl<W: Write, D: Digest + FixedOutputReset, F: FromUniformBytes> Challenge<F>
    for RustCryptoWriter<W, D>
{
    fn draw(&mut self) -> F {
        RustCrypto::draw_field_element(&mut self.h)
    }
}

impl<W: Write, D: Digest + FixedOutputReset> ChallengeBits for RustCryptoWriter<W, D> {
    fn draw(&mut self, bit_size: usize) -> usize {
        RustCrypto::draw_bits(&mut self.h, bit_size)
    }
}

#[derive(Debug, Clone)]
pub struct RustCryptoReader<R: Read, D: Digest + FixedOutputReset> {
    h: D,
    reader: R,
}

impl<R: Read, D: Digest + FixedOutputReset> RustCryptoReader<R, D> {
    pub fn init(reader: R, prefix: impl AsRef<[u8]>) -> Self {
        RustCryptoReader {
            h: D::new_with_prefix(prefix),
            reader,
        }
    }

    fn update(&mut self, data: impl AsRef<[u8]>) {
        Digest::update(&mut self.h, data);
    }
}

impl<R: Read, D: Digest + FixedOutputReset, F: FromUniformBytes> Challenge<F>
    for RustCryptoReader<R, D>
{
    fn draw(&mut self) -> F {
        RustCrypto::draw_field_element(&mut self.h)
    }
}

impl<R: Read, D: Digest + FixedOutputReset> ChallengeBits for RustCryptoReader<R, D> {
    fn draw(&mut self, bit_size: usize) -> usize {
        RustCrypto::draw_bits(&mut self.h, bit_size)
    }
}

impl<R: Read, D: Digest + FixedOutputReset, F: FromUniformBytes> Reader<F>
    for RustCryptoReader<R, D>
{
    fn unsafe_read(&mut self) -> Result<F, Error> {
        let mut data = vec![0u8; F::NUM_BYTES];
        self.reader
            .read_exact(data.as_mut())
            .map_err(|_| Error::Transcript)?;
        Ok(F::from_bytes(&data))
    }

    fn read(&mut self) -> Result<F, Error> {
        let e: F = self.unsafe_read()?;
        self.update(e.to_bytes());
        Ok(e)
    }
}

impl<R: Read, D: Digest + FixedOutputReset> Reader<[u8; 32]> for RustCryptoReader<R, D> {
    fn unsafe_read(&mut self) -> Result<[u8; 32], Error> {
        let mut data = [0u8; 32];
        self.reader
            .read_exact(data.as_mut())
            .map_err(|_| Error::Transcript)?;
        Ok(data)
    }

    fn read(&mut self) -> Result<[u8; 32], Error> {
        let e: [u8; 32] = self.unsafe_read()?;
        self.update(e);
        Ok(e)
    }
}

#[cfg(test)]
fn transcript_test<F: Field + FromUniformBytes, D: Digest + FixedOutputReset>()
where
    rand::distr::StandardUniform: rand::distr::Distribution<F>,
{
    use rand::Rng;

    let mut rng = crate::test::seed_rng();
    let a0: F = rng.random();
    let b0: F = rng.random();
    let c0: F = rng.random();
    let mut w = RustCryptoWriter::<Vec<u8>, D>::init("");

    w.write(a0).unwrap();
    w.write(b0).unwrap();
    let _: F = Challenge::<F>::draw(&mut w);
    w.write(c0).unwrap();
    let u0: F = Challenge::<F>::draw(&mut w);
    w.write(a0).unwrap();
    let i0 = ChallengeBits::draw(&mut w, 8);

    let stream = w.finalize();
    let mut r = RustCryptoReader::<&[u8], D>::init(&stream, "");
    let _: F = r.read().unwrap();
    let _: F = r.read().unwrap();
    let _: F = Challenge::<F>::draw(&mut r);
    let _: F = r.read().unwrap();
    let u1: F = Challenge::<F>::draw(&mut r);
    let a1: F = r.read().unwrap();
    let i1 = ChallengeBits::draw(&mut r, 8);

    assert_eq!(u0, u1);
    assert_eq!(i0, i1);
    assert_eq!(a0, a1);
}

#[test]
fn test_transcript() {
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;
    transcript_test::<Goldilocks, sha2::Sha256>();
    transcript_test::<Goldilocks, sha3::Keccak256>();
    transcript_test::<BinomialExtensionField<Goldilocks, 2>, sha2::Sha256>();
    transcript_test::<BinomialExtensionField<Goldilocks, 2>, sha3::Keccak256>();
}
