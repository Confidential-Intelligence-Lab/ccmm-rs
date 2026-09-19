use super::PhysicalLimbArithmetic;

/// Polynomial over one physical modular limb.
///
/// This type is intentionally below the RNS and CKKS layers. It provides a
/// width-parametric correctness surface for 32-, 64-, and 128-bit physical
/// words before width-parametric NTT integration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimbPolynomial<A>
where
    A: PhysicalLimbArithmetic + Clone + PartialEq + Eq,
{
    arithmetic: A,
    coefficients: Vec<A::Word>,
}

impl<A> LimbPolynomial<A>
where
    A: PhysicalLimbArithmetic + Clone + PartialEq + Eq,
{
    pub fn new(arithmetic: A, coefficients: Vec<A::Word>) -> Self {
        assert!(
            !coefficients.is_empty(),
            "physical-limb polynomial degree must be positive"
        );

        let coefficients = coefficients
            .into_iter()
            .map(|value| arithmetic.canonicalize(value))
            .collect();

        Self {
            arithmetic,
            coefficients,
        }
    }

    pub fn arithmetic(&self) -> &A {
        &self.arithmetic
    }

    pub fn coefficients(&self) -> &[A::Word] {
        &self.coefficients
    }

    pub fn degree(&self) -> usize {
        self.coefficients.len()
    }

    pub fn add(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        let coefficients = self
            .coefficients
            .iter()
            .copied()
            .zip(rhs.coefficients.iter().copied())
            .map(|(lhs, rhs)| self.arithmetic.add_mod(lhs, rhs))
            .collect();

        Self::new(self.arithmetic.clone(), coefficients)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        let coefficients = self
            .coefficients
            .iter()
            .copied()
            .zip(rhs.coefficients.iter().copied())
            .map(|(lhs, rhs)| self.arithmetic.sub_mod(lhs, rhs))
            .collect();

        Self::new(self.arithmetic.clone(), coefficients)
    }

    pub fn neg(&self) -> Self {
        let coefficients = self
            .coefficients
            .iter()
            .copied()
            .map(|value| self.arithmetic.neg_mod(value))
            .collect();

        Self::new(self.arithmetic.clone(), coefficients)
    }

    /// Exact reference negacyclic multiplication modulo `x^N + 1`.
    ///
    /// This is deliberately quadratic. It is the width-parametric polynomial
    /// oracle that the later NTT backends must match exactly.
    pub fn negacyclic_mul(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        let degree = self.degree();
        let zero = self.arithmetic.canonicalize(self.arithmetic.modulus());
        let mut output = vec![zero; degree];

        for (lhs_index, lhs) in self.coefficients.iter().copied().enumerate() {
            for (rhs_index, rhs) in rhs.coefficients.iter().copied().enumerate() {
                let product = self.arithmetic.mul_mod(lhs, rhs);
                let raw_index = lhs_index + rhs_index;

                if raw_index < degree {
                    output[raw_index] = self.arithmetic.add_mod(output[raw_index], product);
                } else {
                    output[raw_index - degree] =
                        self.arithmetic.sub_mod(output[raw_index - degree], product);
                }
            }
        }

        Self::new(self.arithmetic.clone(), output)
    }

    fn assert_compatible(&self, rhs: &Self) {
        assert!(
            self.arithmetic == rhs.arithmetic,
            "physical-limb polynomial arithmetic backends must match"
        );

        assert_eq!(
            self.degree(),
            rhs.degree(),
            "physical-limb polynomial degrees must match"
        );
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use num_traits::ToPrimitive;

    use crate::ring::{Limb128, Limb32, Limb64, PhysicalLimbArithmetic};

    use super::LimbPolynomial;

    fn oracle_negacyclic(lhs: &[u128], rhs: &[u128], modulus: u128) -> Vec<u128> {
        assert_eq!(lhs.len(), rhs.len());

        let degree = lhs.len();
        let q = BigUint::from(modulus);
        let mut output = vec![BigUint::from(0_u8); degree];

        for (lhs_index, &lhs_value) in lhs.iter().enumerate() {
            for (rhs_index, &rhs_value) in rhs.iter().enumerate() {
                let product = (BigUint::from(lhs_value) * BigUint::from(rhs_value)) % &q;

                let raw_index = lhs_index + rhs_index;

                if raw_index < degree {
                    output[raw_index] = (&output[raw_index] + &product) % &q;
                } else {
                    let index = raw_index - degree;

                    output[index] = if output[index] >= product {
                        &output[index] - &product
                    } else {
                        &q - (&product - &output[index])
                    };
                }
            }
        }

        output
            .into_iter()
            .map(|value| value.to_u128().unwrap())
            .collect()
    }

    #[test]
    fn identical_small_modulus_is_exact_across_32_64_128_bit_words() {
        let modulus = 12_289_u128;

        let lhs = [0_u128, 1, 2, 17, 42, 1_000, 4_096, 12_288];
        let rhs = [7_u128, 11, 19, 23, 29, 31, 37, 41];

        let p32 = LimbPolynomial::new(
            Limb32::new(modulus as u32),
            lhs.iter().map(|&value| value as u32).collect(),
        );
        let q32 = LimbPolynomial::new(
            Limb32::new(modulus as u32),
            rhs.iter().map(|&value| value as u32).collect(),
        );

        let p64 = LimbPolynomial::new(
            Limb64::new(modulus as u64),
            lhs.iter().map(|&value| value as u64).collect(),
        );
        let q64 = LimbPolynomial::new(
            Limb64::new(modulus as u64),
            rhs.iter().map(|&value| value as u64).collect(),
        );

        let p128 = LimbPolynomial::new(Limb128::new(modulus), lhs.to_vec());
        let q128 = LimbPolynomial::new(Limb128::new(modulus), rhs.to_vec());

        let product32: Vec<u128> = p32
            .negacyclic_mul(&q32)
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect();

        let product64: Vec<u128> = p64
            .negacyclic_mul(&q64)
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect();

        let product128 = p128.negacyclic_mul(&q128).coefficients().to_vec();

        let oracle = oracle_negacyclic(&lhs, &rhs, modulus);

        assert_eq!(product32, oracle);
        assert_eq!(product64, oracle);
        assert_eq!(product128, oracle);
    }

    #[test]
    fn near_width_32_bit_polynomial_matches_biguint_oracle() {
        let modulus = 4_294_967_291_u32;
        let lhs = [
            0_u32,
            1,
            2,
            modulus / 2,
            modulus - 2,
            modulus - 1,
            0xDEAD_BEEF,
            17,
        ];
        let rhs = [3_u32, 5, 7, 11, 13, 17, 19, 23];

        let p = LimbPolynomial::new(Limb32::new(modulus), lhs.to_vec());
        let q = LimbPolynomial::new(Limb32::new(modulus), rhs.to_vec());

        let actual: Vec<u128> = p
            .negacyclic_mul(&q)
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect();

        let lhs_u128: Vec<u128> = lhs.iter().map(|&value| u128::from(value)).collect();
        let rhs_u128: Vec<u128> = rhs.iter().map(|&value| u128::from(value)).collect();

        assert_eq!(
            actual,
            oracle_negacyclic(&lhs_u128, &rhs_u128, u128::from(modulus))
        );
    }

    #[test]
    fn near_width_64_bit_polynomial_matches_biguint_oracle() {
        let modulus = 18_446_744_073_709_551_557_u64;
        let lhs = [
            0_u64,
            1,
            2,
            modulus / 2,
            modulus - 2,
            modulus - 1,
            0xDEAD_BEEF_CAFE_BABE,
            17,
        ];
        let rhs = [3_u64, 5, 7, 11, 13, 17, 19, 23];

        let p = LimbPolynomial::new(Limb64::new(modulus), lhs.to_vec());
        let q = LimbPolynomial::new(Limb64::new(modulus), rhs.to_vec());

        let actual: Vec<u128> = p
            .negacyclic_mul(&q)
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect();

        let lhs_u128: Vec<u128> = lhs.iter().map(|&value| u128::from(value)).collect();
        let rhs_u128: Vec<u128> = rhs.iter().map(|&value| u128::from(value)).collect();

        assert_eq!(
            actual,
            oracle_negacyclic(&lhs_u128, &rhs_u128, u128::from(modulus))
        );
    }

    #[test]
    fn near_width_128_bit_polynomial_matches_biguint_oracle() {
        let modulus = (1_u128 << 127) - 1;
        let lhs = [
            0_u128,
            1,
            2,
            modulus / 2,
            modulus - 2,
            modulus - 1,
            (1_u128 << 126) + 0x1234_5678_9ABC_DEF0,
            17,
        ];
        let rhs = [3_u128, 5, 7, 11, 13, 17, 19, 23];

        let p = LimbPolynomial::new(Limb128::new(modulus), lhs.to_vec());
        let q = LimbPolynomial::new(Limb128::new(modulus), rhs.to_vec());

        let actual = p.negacyclic_mul(&q).coefficients().to_vec();

        assert_eq!(actual, oracle_negacyclic(&lhs, &rhs, modulus));
    }

    #[test]
    fn add_sub_neg_are_exact_across_widths_for_shared_modulus() {
        let modulus = 65_537_u128;
        let lhs = [0_u128, 1, 17, 65_536];
        let rhs = [65_536_u128, 17, 1, 2];

        let p32 = LimbPolynomial::new(
            Limb32::new(modulus as u32),
            lhs.iter().map(|&value| value as u32).collect(),
        );
        let q32 = LimbPolynomial::new(
            Limb32::new(modulus as u32),
            rhs.iter().map(|&value| value as u32).collect(),
        );

        let p64 = LimbPolynomial::new(
            Limb64::new(modulus as u64),
            lhs.iter().map(|&value| value as u64).collect(),
        );
        let q64 = LimbPolynomial::new(
            Limb64::new(modulus as u64),
            rhs.iter().map(|&value| value as u64).collect(),
        );

        let p128 = LimbPolynomial::new(Limb128::new(modulus), lhs.to_vec());
        let q128 = LimbPolynomial::new(Limb128::new(modulus), rhs.to_vec());

        let add32: Vec<u128> = p32
            .add(&q32)
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect();
        let add64: Vec<u128> = p64
            .add(&q64)
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect();
        let add128 = p128.add(&q128).coefficients().to_vec();

        assert_eq!(add32, add64);
        assert_eq!(add64, add128);

        let sub32: Vec<u128> = p32
            .sub(&q32)
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect();
        let sub64: Vec<u128> = p64
            .sub(&q64)
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect();
        let sub128 = p128.sub(&q128).coefficients().to_vec();

        assert_eq!(sub32, sub64);
        assert_eq!(sub64, sub128);

        let neg32: Vec<u128> = p32
            .neg()
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect();
        let neg64: Vec<u128> = p64
            .neg()
            .coefficients()
            .iter()
            .map(|&value| u128::from(value))
            .collect();
        let neg128 = p128.neg().coefficients().to_vec();

        assert_eq!(neg32, neg64);
        assert_eq!(neg64, neg128);
    }

    #[test]
    fn backend_reports_expected_word_widths() {
        assert_eq!(Limb32::WORD_BITS, 32);
        assert_eq!(Limb64::WORD_BITS, 64);
        assert_eq!(Limb128::WORD_BITS, 128);
    }
}
