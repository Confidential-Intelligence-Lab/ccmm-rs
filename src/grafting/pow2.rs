/// Polynomial over
///
/// `Z_(2^k)[X] / (X^N + 1)`.
///
/// This type is intentionally separate from the prime-modulus
/// `Polynomial` type. Power-of-two sprouts are not fields, so APIs that
/// rely on prime inverses or Fermat's little theorem must not apply here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pow2Polynomial {
    bits: u32,
    degree: usize,
    coefficients: Vec<u64>,
}

impl Pow2Polynomial {
    pub fn new(bits: u32, coefficients: Vec<u64>) -> Self {
        assert!(
            bits > 0 && bits <= 63,
            "power-of-two modulus bits must be in 1..=63"
        );

        assert!(
            !coefficients.is_empty(),
            "power-of-two polynomial degree must be positive"
        );

        let mask = mask(bits);

        let coefficients: Vec<u64> = coefficients.into_iter().map(|value| value & mask).collect();

        Self {
            bits,
            degree: coefficients.len(),
            coefficients,
        }
    }

    pub fn zero(bits: u32, degree: usize) -> Self {
        assert!(
            degree > 0,
            "power-of-two polynomial degree must be positive"
        );

        Self::new(bits, vec![0; degree])
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn modulus(&self) -> u128 {
        1_u128 << self.bits
    }

    pub fn coefficients(&self) -> &[u64] {
        &self.coefficients
    }

    pub fn add(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        let mask = mask(self.bits);

        Self::new(
            self.bits,
            self.coefficients
                .iter()
                .zip(&rhs.coefficients)
                .map(|(&lhs, &rhs)| lhs.wrapping_add(rhs) & mask)
                .collect(),
        )
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        let mask = mask(self.bits);

        Self::new(
            self.bits,
            self.coefficients
                .iter()
                .zip(&rhs.coefficients)
                .map(|(&lhs, &rhs)| lhs.wrapping_sub(rhs) & mask)
                .collect(),
        )
    }

    pub fn neg(&self) -> Self {
        let mask = mask(self.bits);

        Self::new(
            self.bits,
            self.coefficients
                .iter()
                .map(|&value| 0_u64.wrapping_sub(value) & mask)
                .collect(),
        )
    }

    /// Correctness-oriented negacyclic multiplication modulo `2^k`.
    ///
    /// Multiplication is computed in `u128` so that all intermediate
    /// products of two at-most-63-bit residues are exact.
    pub fn negacyclic_mul(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        let modulus = 1_u128 << self.bits;
        let mut output = vec![0_u128; self.degree];

        for (i, &lhs_value) in self.coefficients.iter().enumerate() {
            for (j, &rhs_value) in rhs.coefficients.iter().enumerate() {
                let product = (u128::from(lhs_value) * u128::from(rhs_value)) % modulus;

                let raw_index = i + j;

                if raw_index < self.degree {
                    output[raw_index] = (output[raw_index] + product) % modulus;
                } else {
                    let index = raw_index - self.degree;

                    output[index] = (output[index] + modulus - product) % modulus;
                }
            }
        }

        Self::new(
            self.bits,
            output.into_iter().map(|value| value as u64).collect(),
        )
    }

    fn assert_compatible(&self, rhs: &Self) {
        assert_eq!(
            self.bits, rhs.bits,
            "power-of-two polynomial moduli must match"
        );

        assert_eq!(
            self.degree, rhs.degree,
            "power-of-two polynomial degrees must match"
        );
    }
}

fn mask(bits: u32) -> u64 {
    (1_u64 << bits) - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_reduces_mod_power_of_two() {
        let polynomial = Pow2Polynomial::new(5, vec![0, 1, 31, 32, 33, 63]);

        assert_eq!(polynomial.coefficients(), &[0, 1, 31, 0, 1, 31]);
    }

    #[test]
    fn addition_wraps_mod_power_of_two() {
        let lhs = Pow2Polynomial::new(4, vec![15, 7, 1, 8]);

        let rhs = Pow2Polynomial::new(4, vec![1, 10, 15, 8]);

        assert_eq!(lhs.add(&rhs).coefficients(), &[0, 1, 0, 0]);
    }

    #[test]
    fn subtraction_wraps_mod_power_of_two() {
        let lhs = Pow2Polynomial::new(4, vec![0, 1, 2, 3]);

        let rhs = Pow2Polynomial::new(4, vec![1, 2, 3, 4]);

        assert_eq!(lhs.sub(&rhs).coefficients(), &[15, 15, 15, 15]);
    }

    #[test]
    fn negation_is_additive_inverse() {
        let polynomial = Pow2Polynomial::new(8, vec![1, 7, 42, 255]);

        assert_eq!(
            polynomial.add(&polynomial.neg()).coefficients(),
            &[0, 0, 0, 0]
        );
    }

    #[test]
    fn negacyclic_wrap_changes_sign() {
        // X^3 * X^2 = X^5 = -X mod (X^4 + 1).
        let lhs = Pow2Polynomial::new(8, vec![0, 0, 0, 1]);

        let rhs = Pow2Polynomial::new(8, vec![0, 0, 1, 0]);

        assert_eq!(lhs.negacyclic_mul(&rhs).coefficients(), &[0, 255, 0, 0]);
    }

    #[test]
    fn multiplication_matches_independent_reference() {
        let bits = 16;
        let degree = 16;

        let lhs = Pow2Polynomial::new(bits, (0..degree).map(|i| 3 + 7 * i as u64).collect());

        let rhs = Pow2Polynomial::new(bits, (0..degree).map(|i| 11 + 13 * i as u64).collect());

        let modulus = 1_u128 << bits;

        let mut expected = vec![0_u128; degree];

        for i in 0..degree {
            for j in 0..degree {
                let product = (u128::from(lhs.coefficients()[i])
                    * u128::from(rhs.coefficients()[j]))
                    % modulus;

                if i + j < degree {
                    expected[i + j] = (expected[i + j] + product) % modulus;
                } else {
                    let index = i + j - degree;

                    expected[index] = (expected[index] + modulus - product) % modulus;
                }
            }
        }

        assert_eq!(
            lhs.negacyclic_mul(&rhs).coefficients(),
            expected
                .iter()
                .map(|&value| value as u64)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn multiplication_campaign_across_bit_sizes() {
        for bits in [1_u32, 2, 4, 8, 16, 24, 32, 48, 63] {
            let degree = 16;

            for seed in 0_u64..16 {
                let lhs = Pow2Polynomial::new(
                    bits,
                    (0..degree)
                        .map(|i| {
                            seed.wrapping_mul(17)
                                .wrapping_add(31 * i as u64)
                                .wrapping_add(7)
                        })
                        .collect(),
                );

                let rhs = Pow2Polynomial::new(
                    bits,
                    (0..degree)
                        .map(|i| {
                            seed.wrapping_mul(29)
                                .wrapping_add(13 * i as u64)
                                .wrapping_add(11)
                        })
                        .collect(),
                );

                let product = lhs.negacyclic_mul(&rhs);

                assert_eq!(product.bits(), bits);

                assert_eq!(product.degree(), degree);

                assert!(product
                    .coefficients()
                    .iter()
                    .all(|&value| { u128::from(value) < (1_u128 << bits) }));
            }
        }
    }

    #[test]
    fn universal_sprout_power_of_two_matches_polynomial_modulus() {
        use crate::grafting::UniversalSproutPolicy;

        let policy = UniversalSproutPolicy::new(1 << 15, 63);

        for bits in 1_u32..=32 {
            let sprout = policy.power_of_two(bits);

            let polynomial = Pow2Polynomial::zero(bits, 8);

            assert_eq!(polynomial.modulus(), sprout.modulus());
        }
    }

    #[test]
    #[should_panic(expected = "1..=63")]
    fn rejects_zero_bits() {
        let _ = Pow2Polynomial::zero(0, 8);
    }

    #[test]
    #[should_panic(expected = "1..=63")]
    fn rejects_more_than_63_bits() {
        let _ = Pow2Polynomial::zero(64, 8);
    }
}
