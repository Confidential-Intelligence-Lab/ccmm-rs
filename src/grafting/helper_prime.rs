use crate::ring::{make_ntt_plan, Modulus, NttPlan, NttPolynomial, Polynomial};

use super::Pow2Polynomial;

/// NTT multiplication backend for a power-of-two sprout.
///
/// Coefficients modulo `2^k` are lifted canonically into a helper-prime
/// polynomial, multiplied with the negacyclic NTT, recovered as signed
/// integer convolution coefficients, and finally reduced modulo `2^k`.
///
/// Exactness requires the helper prime to exceed twice the largest
/// possible absolute signed convolution coefficient.
pub struct HelperPrimeNttPlan {
    bits: u32,
    degree: usize,
    helper_modulus: Modulus,
    ntt_plan: NttPlan,
    signed_convolution_bound: u128,
}

impl HelperPrimeNttPlan {
    pub fn new(bits: u32, degree: usize, helper_modulus: Modulus) -> Self {
        assert!(
            bits > 0 && bits <= 63,
            "power-of-two sprout bits must be in 1..=63"
        );

        assert!(
            degree > 0 && degree.is_power_of_two(),
            "ring degree must be a positive power of two"
        );

        let max_coefficient = (1_u128 << bits) - 1;

        let signed_convolution_bound = (degree as u128)
            .checked_mul(max_coefficient)
            .and_then(|value| value.checked_mul(max_coefficient))
            .expect("helper-prime convolution bound exceeds u128");

        let required = signed_convolution_bound
            .checked_mul(2)
            .expect("helper-prime exactness bound exceeds u128");

        assert!(
            u128::from(helper_modulus.value()) > required,
            "helper prime is too small for exact signed convolution recovery"
        );

        let ntt_plan = make_ntt_plan(helper_modulus, degree);

        Self {
            bits,
            degree,
            helper_modulus,
            ntt_plan,
            signed_convolution_bound,
        }
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn helper_modulus(&self) -> Modulus {
        self.helper_modulus
    }

    pub fn signed_convolution_bound(&self) -> u128 {
        self.signed_convolution_bound
    }

    /// Multiplies two power-of-two sprout polynomials through the helper
    /// prime NTT path.
    pub fn negacyclic_mul(&self, lhs: &Pow2Polynomial, rhs: &Pow2Polynomial) -> Pow2Polynomial {
        self.assert_compatible(lhs);
        self.assert_compatible(rhs);

        let lhs_helper = self.lift(lhs);

        let rhs_helper = self.lift(rhs);

        let product_helper = self.ntt_plan.negacyclic_mul(&lhs_helper, &rhs_helper);

        self.project(&product_helper)
    }

    fn lift(&self, polynomial: &Pow2Polynomial) -> Polynomial {
        Polynomial::new(self.helper_modulus, polynomial.coefficients().to_vec())
    }

    fn project(&self, polynomial: &Polynomial) -> Pow2Polynomial {
        assert_eq!(
            polynomial.modulus(),
            self.helper_modulus,
            "helper polynomial modulus mismatch"
        );

        assert_eq!(
            polynomial.degree(),
            self.degree,
            "helper polynomial degree mismatch"
        );

        let helper_q = u128::from(self.helper_modulus.value());

        let pow2_q = 1_u128 << self.bits;

        let coefficients = polynomial
            .coefficients()
            .iter()
            .map(|&residue| {
                let residue = u128::from(residue);

                let (negative, magnitude) = if residue <= helper_q / 2 {
                    (false, residue)
                } else {
                    (true, helper_q - residue)
                };

                assert!(
                    magnitude <= self.signed_convolution_bound,
                    "helper-prime result exceeds exact recovery bound"
                );

                let reduced = magnitude % pow2_q;

                if negative && reduced != 0 {
                    (pow2_q - reduced) as u64
                } else {
                    reduced as u64
                }
            })
            .collect();

        Pow2Polynomial::new(self.bits, coefficients)
    }

    fn assert_compatible(&self, polynomial: &Pow2Polynomial) {
        assert_eq!(
            polynomial.bits(),
            self.bits,
            "power-of-two sprout bit size must match helper plan"
        );

        assert_eq!(
            polynomial.degree(),
            self.degree,
            "power-of-two polynomial degree must match helper plan"
        );
    }
}

/// Power-of-two polynomial cached in the helper-prime NTT domain.
///
/// The coefficients have been canonically lifted from `Z_(2^k)` into
/// the helper-prime ring and transformed once. This representation is
/// intended for operands such as evaluation-key polynomials that are
/// reused across many multiplications.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelperPrimeNttPolynomial {
    bits: u32,
    helper_modulus: Modulus,
    transformed: NttPolynomial,
}

impl HelperPrimeNttPolynomial {
    pub fn bits(&self) -> u32 {
        self.bits
    }

    pub fn degree(&self) -> usize {
        self.transformed.degree()
    }

    pub fn helper_modulus(&self) -> Modulus {
        self.helper_modulus
    }

    pub fn transformed(&self) -> &NttPolynomial {
        &self.transformed
    }
}

impl HelperPrimeNttPlan {
    /// Lifts and transforms a power-of-two polynomial once for reuse.
    pub fn prepare(&self, polynomial: &Pow2Polynomial) -> HelperPrimeNttPolynomial {
        self.assert_compatible(polynomial);

        let helper = self.lift(polynomial);

        HelperPrimeNttPolynomial {
            bits: self.bits,
            helper_modulus: self.helper_modulus,
            transformed: NttPolynomial::from_polynomial(&self.ntt_plan, &helper),
        }
    }

    /// Multiplies a coefficient-domain power-of-two polynomial by a
    /// reusable helper-prime NTT-domain operand.
    pub fn negacyclic_mul_prepared(
        &self,
        lhs: &Pow2Polynomial,
        rhs: &HelperPrimeNttPolynomial,
    ) -> Pow2Polynomial {
        self.assert_compatible(lhs);

        assert_eq!(
            rhs.bits, self.bits,
            "prepared helper operand sprout size must match plan"
        );

        assert_eq!(
            rhs.helper_modulus, self.helper_modulus,
            "prepared helper operand modulus must match plan"
        );

        assert_eq!(
            rhs.degree(),
            self.degree,
            "prepared helper operand degree must match plan"
        );

        let lhs_helper = self.lift(lhs);

        let lhs_ntt = NttPolynomial::from_polynomial(&self.ntt_plan, &lhs_helper);

        let product_ntt = lhs_ntt.pointwise_mul(&rhs.transformed);

        let product = product_ntt.to_polynomial(&self.ntt_plan);

        self.project(&product)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2013265921 = 15 * 2^27 + 1.
    //
    // This is a convenient NTT-friendly prime for correctness testing.
    const HELPER_PRIME: u64 = 2_013_265_921;

    fn polynomial(bits: u32, degree: usize, seed: u64, multiplier: u64) -> Pow2Polynomial {
        Pow2Polynomial::new(
            bits,
            (0..degree)
                .map(|index| {
                    seed.wrapping_mul(multiplier)
                        .wrapping_add(17 * index as u64)
                        .wrapping_add(3 * (index as u64).pow(2))
                        .wrapping_add(11)
                })
                .collect(),
        )
    }

    #[test]
    fn helper_plan_preserves_configuration() {
        let plan = HelperPrimeNttPlan::new(8, 64, Modulus::new(HELPER_PRIME));

        assert_eq!(plan.bits(), 8);
        assert_eq!(plan.degree(), 64);

        assert_eq!(plan.helper_modulus(), Modulus::new(HELPER_PRIME));
    }

    #[test]
    fn helper_path_matches_direct_power_of_two_product() {
        let bits = 8;
        let degree = 64;

        let plan = HelperPrimeNttPlan::new(bits, degree, Modulus::new(HELPER_PRIME));

        let lhs = polynomial(bits, degree, 3, 19);

        let rhs = polynomial(bits, degree, 7, 29);

        assert_eq!(plan.negacyclic_mul(&lhs, &rhs,), lhs.negacyclic_mul(&rhs));
    }

    #[test]
    fn helper_path_handles_negative_negacyclic_wrap() {
        let bits = 8;
        let degree = 16;

        let plan = HelperPrimeNttPlan::new(bits, degree, Modulus::new(HELPER_PRIME));

        let mut lhs = vec![0_u64; degree];

        let mut rhs = vec![0_u64; degree];

        lhs[degree - 1] = 1;
        rhs[2] = 1;

        let lhs = Pow2Polynomial::new(bits, lhs);

        let rhs = Pow2Polynomial::new(bits, rhs);

        assert_eq!(plan.negacyclic_mul(&lhs, &rhs,), lhs.negacyclic_mul(&rhs));
    }

    #[test]
    fn helper_path_matches_reference_campaign() {
        for bits in [2_u32, 4, 8, 10] {
            for degree in [16_usize, 64, 256] {
                let plan = HelperPrimeNttPlan::new(bits, degree, Modulus::new(HELPER_PRIME));

                for seed in 0_u64..8 {
                    let lhs = polynomial(bits, degree, seed + 1, 17);

                    let rhs = polynomial(bits, degree, seed + 101, 31);

                    assert_eq!(
                        plan.negacyclic_mul(&lhs, &rhs,),
                        lhs.negacyclic_mul(&rhs,),
                        "helper-prime mismatch: bits={bits}, degree={degree}, seed={seed}"
                    );
                }
            }
        }
    }

    #[test]
    fn helper_path_matches_reference_at_degree_1024() {
        let bits = 8;
        let degree = 1024;

        let plan = HelperPrimeNttPlan::new(bits, degree, Modulus::new(HELPER_PRIME));

        let lhs = polynomial(bits, degree, 13, 17);

        let rhs = polynomial(bits, degree, 71, 31);

        assert_eq!(plan.negacyclic_mul(&lhs, &rhs,), lhs.negacyclic_mul(&rhs));
    }

    #[test]
    #[should_panic(expected = "too small")]
    fn rejects_helper_prime_when_exact_recovery_is_impossible() {
        // For k=16 and N=256 the signed convolution bound is much
        // larger than the approximately 31-bit helper prime.
        let _ = HelperPrimeNttPlan::new(16, 256, Modulus::new(HELPER_PRIME));
    }

    #[test]
    #[should_panic(expected = "bit size")]
    fn rejects_mismatched_sprout_bits() {
        let plan = HelperPrimeNttPlan::new(8, 64, Modulus::new(HELPER_PRIME));

        let wrong = Pow2Polynomial::zero(7, 64);

        let rhs = Pow2Polynomial::zero(8, 64);

        let _ = plan.negacyclic_mul(&wrong, &rhs);
    }

    #[test]
    #[should_panic(expected = "degree")]
    fn rejects_mismatched_degree() {
        let plan = HelperPrimeNttPlan::new(8, 64, Modulus::new(HELPER_PRIME));

        let wrong = Pow2Polynomial::zero(8, 32);

        let rhs = Pow2Polynomial::zero(8, 64);

        let _ = plan.negacyclic_mul(&wrong, &rhs);
    }

    #[test]
    fn prepared_helper_operand_matches_uncached_path() {
        for bits in [2_u32, 4, 8, 10] {
            for degree in [16_usize, 64, 256] {
                let plan = HelperPrimeNttPlan::new(bits, degree, Modulus::new(HELPER_PRIME));

                for seed in 0_u64..16 {
                    let lhs = polynomial(bits, degree, seed + 1, 17);

                    let rhs = polynomial(bits, degree, seed + 101, 31);

                    let prepared = plan.prepare(&rhs);

                    assert_eq!(
                        plan.negacyclic_mul_prepared(&lhs, &prepared,),
                        plan.negacyclic_mul(&lhs, &rhs,),
                        "prepared helper mismatch: \
                         bits={bits}, \
                         degree={degree}, \
                         seed={seed}"
                    );
                }
            }
        }
    }
}
