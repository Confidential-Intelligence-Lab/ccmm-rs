use crate::ring::{Modulus, Polynomial};

/// Parameters for the minimal coefficient-message RLWE layer.
///
/// This is the correctness-oriented RLWE substrate used before CKKS
/// scale/level semantics are introduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RlweParameters {
    degree: usize,
    modulus: Modulus,
    plaintext_modulus: u64,
    noise_bound: i64,
}

impl RlweParameters {
    /// Constructs RLWE parameters.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - the degree is zero or not a power of two;
    /// - the plaintext modulus is invalid;
    /// - the noise bound is negative;
    /// - the coefficient spacing is too small for the configured noise.
    pub fn new(degree: usize, modulus: Modulus, plaintext_modulus: u64, noise_bound: i64) -> Self {
        assert!(
            degree > 0 && degree.is_power_of_two(),
            "RLWE degree must be a positive power of two"
        );

        assert!(
            plaintext_modulus >= 2 && plaintext_modulus < modulus.value(),
            "plaintext modulus must satisfy 2 <= t < q"
        );

        assert!(noise_bound >= 0, "noise bound must be nonnegative");

        let delta = modulus.value() / plaintext_modulus;

        assert!(
            delta > (2 * noise_bound + 1) as u64,
            "coefficient spacing is too small for the noise bound"
        );

        Self {
            degree,
            modulus,
            plaintext_modulus,
            noise_bound,
        }
    }

    pub fn degree(self) -> usize {
        self.degree
    }

    pub fn modulus(self) -> Modulus {
        self.modulus
    }

    pub fn plaintext_modulus(self) -> u64 {
        self.plaintext_modulus
    }

    pub fn noise_bound(self) -> i64 {
        self.noise_bound
    }

    pub fn delta(self) -> u64 {
        self.modulus.value() / self.plaintext_modulus
    }

    pub(crate) fn encode_coeff(self, value: u64) -> u64 {
        let message = value % self.plaintext_modulus;
        self.modulus.mul(message, self.delta())
    }

    pub(crate) fn decode_coeff(self, value: u64) -> u64 {
        let value = self.modulus.reduce(value);
        let delta = self.delta();

        let rounded = (u128::from(value) + u128::from(delta) / 2) / u128::from(delta);

        (rounded as u64) % self.plaintext_modulus
    }

    pub(crate) fn encode_message(self, values: &[u64]) -> Polynomial {
        assert_eq!(
            values.len(),
            self.degree,
            "plaintext length must match RLWE degree"
        );

        Polynomial::new(
            self.modulus,
            values
                .iter()
                .map(|&value| self.encode_coeff(value))
                .collect(),
        )
    }

    pub(crate) fn decode_message(self, polynomial: &Polynomial) -> Vec<u64> {
        assert_eq!(
            polynomial.modulus(),
            self.modulus,
            "plaintext polynomial modulus must match RLWE modulus"
        );

        assert_eq!(
            polynomial.degree(),
            self.degree,
            "plaintext polynomial degree must match RLWE degree"
        );

        polynomial
            .coefficients()
            .iter()
            .map(|&value| self.decode_coeff(value))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> RlweParameters {
        RlweParameters::new(8, Modulus::new(12_289), 16, 1)
    }

    #[test]
    fn parameters_preserve_configuration() {
        let params = params();

        assert_eq!(params.degree(), 8);
        assert_eq!(params.modulus(), Modulus::new(12_289));
        assert_eq!(params.plaintext_modulus(), 16);
        assert_eq!(params.noise_bound(), 1);
        assert_eq!(params.delta(), 768);
    }

    #[test]
    fn coefficient_encoding_roundtrips() {
        let params = params();

        for value in 0..params.plaintext_modulus() {
            assert_eq!(params.decode_coeff(params.encode_coeff(value)), value);
        }
    }

    #[test]
    fn decoding_tolerates_small_positive_and_negative_noise() {
        let params = params();
        let q = params.modulus();

        for value in 0..params.plaintext_modulus() {
            let encoded = params.encode_coeff(value);

            let plus_one = q.add(encoded, 1);
            let minus_one = q.sub(encoded, 1);

            assert_eq!(params.decode_coeff(plus_one), value);
            assert_eq!(params.decode_coeff(minus_one), value);
        }
    }
}
