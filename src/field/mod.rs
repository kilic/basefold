use p3_field::Field;

pub trait FromUniformBytes: Field {
    fn from_bytes(bytes: &[u8]) -> Self;
    fn to_bytes(&self) -> Vec<u8>;
}

// This implementation is temporary and introduces modulus bias.
// Two alternative might be considered:
// * Repeat hashing until come across a slice that is in the field.
// * Take `2*F::NUM_BYTES` bytes and reduce it to the field.
macro_rules! impl_from_uniform_bytes {
    ($field:ty) => {
        impl FromUniformBytes for $field {
            fn from_bytes(bytes: &[u8]) -> Self {
                bincode::deserialize(bytes).unwrap()
            }

            fn to_bytes(&self) -> Vec<u8> {
                bincode::serialize(&self).unwrap()
            }
        }
    };
}

impl_from_uniform_bytes!(p3_goldilocks::Goldilocks);
impl_from_uniform_bytes!(p3_field::extension::BinomialExtensionField<p3_goldilocks::Goldilocks, 2>);
