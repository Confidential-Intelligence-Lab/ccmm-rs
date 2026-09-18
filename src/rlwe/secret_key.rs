use rand::rngs::OsRng;
use rand::{CryptoRng, Rng, RngCore};

use crate::ring::Polynomial;

use super::RlweParameters;

/// Ternary RLWE secret key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretKey {
    polynomial: Polynomial,
}

impl SecretKey {
    /// Generates a secret key using operating-system randomness.
    pub fn generate(params: RlweParameters) -> Self {
        let mut rng = OsRng;
        Self::generate_with_rng(params, &mut rng)
    }

    /// Generates a secret key from an explicit CSPRNG.
    ///
    /// This entry point exists primarily for deterministic tests.
    pub fn generate_with_rng<R>(params: RlweParameters, rng: &mut R) -> Self
    where
        R: RngCore + CryptoRng,
    {
        loop {
            let coefficients: Vec<u64> = (0..params.degree())
                .map(|_| match rng.gen_range(0_u8..3) {
                    0 => 0,
                    1 => 1,
                    _ => params.modulus().value() - 1,
                })
                .collect();

            if coefficients.iter().any(|&coefficient| coefficient != 0) {
                return Self {
                    polynomial: Polynomial::new(params.modulus(), coefficients),
                };
            }
        }
    }

    pub(crate) fn from_polynomial(polynomial: Polynomial) -> Self {
        Self { polynomial }
    }

    pub fn polynomial(&self) -> &Polynomial {
        &self.polynomial
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ring::Modulus;

    use super::*;

    #[test]
    fn generated_secret_is_ternary_and_nonzero() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 1);

        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let key = SecretKey::generate_with_rng(params, &mut rng);

        let q_minus_one = params.modulus().value() - 1;

        assert!(key
            .polynomial()
            .coefficients()
            .iter()
            .all(|&value| matches!(value, 0 | 1) || value == q_minus_one));

        assert!(key
            .polynomial()
            .coefficients()
            .iter()
            .any(|&value| value != 0));
    }

    #[test]
    fn seeded_key_generation_is_deterministic() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 1);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(42);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(42);

        assert_eq!(
            SecretKey::generate_with_rng(params, &mut lhs_rng),
            SecretKey::generate_with_rng(params, &mut rhs_rng)
        );
    }
}
