use super::{Modulus, NttPlan, Polynomial, PreparedNttPlan};

/// Polynomial represented in the negacyclic NTT domain.
///
/// This type intentionally prevents coefficient-domain and NTT-domain
/// polynomials from being mixed accidentally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NttPolynomial {
    modulus: Modulus,
    degree: usize,
    values: Vec<u64>,
}

impl NttPolynomial {
    /// Transforms a coefficient-domain polynomial into the NTT domain.
    pub fn from_polynomial(plan: &NttPlan, polynomial: &Polynomial) -> Self {
        assert_eq!(
            polynomial.modulus(),
            plan.modulus(),
            "polynomial modulus must match NTT plan"
        );

        assert_eq!(
            polynomial.degree(),
            plan.degree(),
            "polynomial degree must match NTT plan"
        );

        Self {
            modulus: plan.modulus(),
            degree: plan.degree(),
            values: plan.forward_radix2(polynomial),
        }
    }

    /// Constructs the additive identity directly in the NTT domain.
    ///
    /// No transform or modular reduction is required because zero is
    /// represented identically in both coefficient and NTT domains.
    pub(crate) fn zero(plan: &NttPlan) -> Self {
        Self {
            modulus: plan.modulus(),
            degree: plan.degree(),
            values: vec![0_u64; plan.degree()],
        }
    }

    /// Constructs an NTT-domain polynomial from validated raw values.
    pub fn from_values(plan: &NttPlan, values: Vec<u64>) -> Self {
        assert_eq!(
            values.len(),
            plan.degree(),
            "NTT value count must match plan degree"
        );

        Self {
            modulus: plan.modulus(),
            degree: plan.degree(),
            values: values
                .into_iter()
                .map(|value| plan.modulus().reduce(value))
                .collect(),
        }
    }

    /// Constructs an NTT-domain polynomial from already-canonical values.
    ///
    /// Every value must already lie in `[0, q)`. Unlike `from_values`,
    /// this constructor does not perform another modular-reduction pass.
    pub(crate) fn from_canonical_values(plan: &NttPlan, values: Vec<u64>) -> Self {
        assert_eq!(
            values.len(),
            plan.degree(),
            "NTT value count must match plan degree"
        );

        debug_assert!(
            values.iter().all(|&value| value < plan.modulus().value()),
            "canonical NTT values must lie below the modulus"
        );

        Self {
            modulus: plan.modulus(),
            degree: plan.degree(),
            values,
        }
    }

    /// Constructs an NTT-domain polynomial from values produced by a
    /// prepared execution plan.
    pub fn from_prepared_values(plan: &PreparedNttPlan, values: Vec<u64>) -> Self {
        assert_eq!(
            values.len(),
            plan.degree(),
            "NTT value count must match prepared NTT plan degree"
        );

        Self {
            modulus: plan.modulus(),
            degree: plan.degree(),
            values: values
                .into_iter()
                .map(|value| plan.modulus().reduce(value))
                .collect(),
        }
    }

    pub fn modulus(&self) -> Modulus {
        self.modulus
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn values(&self) -> &[u64] {
        &self.values
    }

    /// Multiplies by `X^exponent` directly in the negacyclic NTT domain.
    ///
    /// The exponent is reduced modulo `2N`, matching `X^N = -1`.
    /// Negative exponents are therefore supported naturally.
    ///
    /// For NTT evaluation point
    ///
    /// `r_k = psi^(2k + 1)`,
    ///
    /// multiplication by `X^e` is pointwise multiplication by `r_k^e`.
    ///
    /// The phases are generated as a geometric progression, so this
    /// requires two modular exponentiations total and O(N) multiplications.
    pub fn mul_monomial_signed(&self, plan: &NttPlan, exponent: i64) -> Self {
        assert_eq!(
            self.modulus,
            plan.modulus(),
            "NTT polynomial modulus must match plan"
        );

        assert_eq!(
            self.degree,
            plan.degree(),
            "NTT polynomial degree must match plan"
        );

        let period = 2_i128 * self.degree as i128;

        let normalized = i128::from(exponent).rem_euclid(period) as u64;

        if normalized == 0 {
            return self.clone();
        }

        let psi = plan.psi();

        let omega = self.modulus.mul_canonical(psi, psi);

        // r_k^e =
        //
        //   (psi * omega^k)^e
        // = psi^e * (omega^e)^k.
        let mut phase = self.modulus.pow(psi, normalized);

        let phase_step = self.modulus.pow(omega, normalized);

        let values = self
            .values
            .iter()
            .map(|&value| {
                let result = self.modulus.mul_canonical(value, phase);

                phase = self.modulus.mul_canonical(phase, phase_step);

                result
            })
            .collect();

        Self {
            modulus: self.modulus,
            degree: self.degree,
            values,
        }
    }

    /// Pointwise multiplication in the NTT domain.
    pub fn pointwise_mul(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        let values = self
            .values
            .iter()
            .zip(&rhs.values)
            .map(|(&lhs, &rhs)| self.modulus.mul_canonical(lhs, rhs))
            .collect();

        Self {
            modulus: self.modulus,
            degree: self.degree,
            values,
        }
    }

    /// Accumulates a pointwise NTT-domain product into this polynomial.
    ///
    /// This is exactly equivalent to:
    ///
    /// `*self = self.add(&lhs.pointwise_mul(rhs));`
    ///
    /// but updates the accumulator in place and avoids materializing both
    /// the temporary product and the temporary sum.
    pub fn pointwise_mul_add_assign(&mut self, lhs: &Self, rhs: &Self) {
        self.assert_compatible(lhs);
        self.assert_compatible(rhs);

        self.pointwise_mul_add_assign_prevalidated(lhs, rhs);
    }

    /// Accumulates a pointwise product assuming compatibility was already
    /// validated by the caller.
    ///
    /// This is an internal hot-path primitive for RNS matrix kernels.
    pub(crate) fn pointwise_mul_add_assign_prevalidated(&mut self, lhs: &Self, rhs: &Self) {
        debug_assert_eq!(self.modulus, lhs.modulus);
        debug_assert_eq!(self.modulus, rhs.modulus);
        debug_assert_eq!(self.degree, lhs.degree);
        debug_assert_eq!(self.degree, rhs.degree);

        for ((acc, &lhs_value), &rhs_value) in
            self.values.iter_mut().zip(&lhs.values).zip(&rhs.values)
        {
            *acc = self
                .modulus
                .add_canonical(*acc, self.modulus.mul_canonical(lhs_value, rhs_value));
        }
    }

    /// Pointwise addition in the NTT domain.
    pub fn add(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        let values = self
            .values
            .iter()
            .zip(&rhs.values)
            .map(|(&lhs, &rhs)| self.modulus.add_canonical(lhs, rhs))
            .collect();

        Self {
            modulus: self.modulus,
            degree: self.degree,
            values,
        }
    }

    /// Converts the NTT representation back to coefficient form.
    pub fn to_polynomial(&self, plan: &NttPlan) -> Polynomial {
        assert_eq!(
            self.modulus,
            plan.modulus(),
            "NTT polynomial modulus must match plan"
        );

        assert_eq!(
            self.degree,
            plan.degree(),
            "NTT polynomial degree must match plan"
        );

        plan.inverse_radix2(&self.values)
    }

    fn assert_compatible(&self, rhs: &Self) {
        assert_eq!(
            self.modulus, rhs.modulus,
            "NTT polynomial moduli must match"
        );

        assert_eq!(self.degree, rhs.degree, "NTT polynomial degrees must match");
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn canonical_values_constructor_matches_reducing_constructor() {
        let plan = crate::ring::make_ntt_plan(Modulus::new(12_289), 8);

        let values = vec![0, 1, 7, 31, 127, 511, 4095, 12_288];

        let reduced = NttPolynomial::from_values(&plan, values.clone());

        let canonical = NttPolynomial::from_canonical_values(&plan, values);

        assert_eq!(canonical, reduced);
    }
    use super::*;
    use crate::ring::make_ntt_plan;

    fn plan() -> NttPlan {
        make_ntt_plan(Modulus::new(12_289), 64)
    }

    fn polynomial(offset: u64) -> Polynomial {
        let plan = plan();

        Polynomial::new(
            plan.modulus(),
            (0..plan.degree())
                .map(|index| {
                    (offset + 11 * index as u64 + 5 * (index as u64).pow(2))
                        % plan.modulus().value()
                })
                .collect(),
        )
    }

    #[test]
    fn transform_roundtrip_preserves_polynomial() {
        let plan = plan();
        let original = polynomial(7);

        let transformed = NttPolynomial::from_polynomial(&plan, &original);

        assert_eq!(transformed.to_polynomial(&plan), original);
    }

    #[test]
    fn stored_values_match_radix2_transform() {
        let plan = plan();
        let original = polynomial(9);

        let transformed = NttPolynomial::from_polynomial(&plan, &original);

        assert_eq!(transformed.values(), plan.forward_radix2(&original));
    }

    #[test]
    fn ntt_monomial_matches_coefficient_domain_exactly() {
        let plan = plan();

        let exponents = [
            -257_i64, -129, -128, -127, -65, -64, -63, -17, -2, -1, 0, 1, 2, 17, 63, 64, 65, 127,
            128, 129, 257,
        ];

        for offset in [1_u64, 7, 19, 83, 211] {
            let polynomial = polynomial(offset);

            let transformed = NttPolynomial::from_polynomial(&plan, &polynomial);

            for exponent in exponents {
                let degree = polynomial.degree();
                let period = 2_i128 * degree as i128;

                let normalized = i128::from(exponent).rem_euclid(period) as usize;

                let shift = normalized % degree;

                let sign = if normalized >= degree { -1_i8 } else { 1_i8 };

                let reference = polynomial.mul_monomial_signed(shift, sign);

                let actual = transformed
                    .mul_monomial_signed(&plan, exponent)
                    .to_polynomial(&plan);

                assert_eq!(
                    actual, reference,
                    "NTT monomial mismatch: \
                     offset={offset}, exponent={exponent}"
                );
            }
        }

        println!("NTT_SIGNED_MONOMIAL_EQUIVALENCE=PASS");
    }

    #[test]
    fn pointwise_mul_add_assign_matches_materialized_reference_exactly() {
        let plan = plan();

        for offsets in [(3_u64, 17_u64, 29_u64), (11, 31, 47), (101, 203, 307)] {
            let accumulator = polynomial(offsets.0);
            let lhs = polynomial(offsets.1);
            let rhs = polynomial(offsets.2);

            let accumulator_ntt = NttPolynomial::from_polynomial(&plan, &accumulator);
            let lhs_ntt = NttPolynomial::from_polynomial(&plan, &lhs);
            let rhs_ntt = NttPolynomial::from_polynomial(&plan, &rhs);

            let reference = accumulator_ntt.add(&lhs_ntt.pointwise_mul(&rhs_ntt));

            let mut fused = accumulator_ntt;
            fused.pointwise_mul_add_assign(&lhs_ntt, &rhs_ntt);

            assert_eq!(fused, reference);
        }

        println!("NTT_POINTWISE_MUL_ADD_ASSIGN_EQUIVALENCE=PASS");
    }

    #[test]
    fn pointwise_product_matches_naive_negacyclic_product() {
        let plan = plan();
        let lhs = polynomial(3);
        let rhs = polynomial(17);

        let lhs_ntt = NttPolynomial::from_polynomial(&plan, &lhs);

        let rhs_ntt = NttPolynomial::from_polynomial(&plan, &rhs);

        let product = lhs_ntt.pointwise_mul(&rhs_ntt).to_polynomial(&plan);

        assert_eq!(product, lhs.negacyclic_mul(&rhs));
    }

    #[test]
    fn optimized_multiplier_matches_reference_across_degrees() {
        let modulus = Modulus::new(12_289);

        for degree in [16_usize, 32, 64, 128, 256] {
            let plan = crate::ring::make_ntt_plan(modulus, degree);

            let lhs = Polynomial::new(
                modulus,
                (0..degree)
                    .map(|i| (7 + 13 * i as u64 + 5 * (i as u64).pow(2)) % modulus.value())
                    .collect(),
            );

            let rhs = Polynomial::new(
                modulus,
                (0..degree)
                    .map(|i| (19 + 17 * i as u64 + 3 * (i as u64).pow(2)) % modulus.value())
                    .collect(),
            );

            assert_eq!(
                plan.negacyclic_mul(&lhs, &rhs),
                lhs.negacyclic_mul(&rhs),
                "optimized/reference mismatch at degree {degree}"
            );
        }
    }

    #[test]
    fn raw_values_are_reduced_mod_q() {
        let plan = plan();
        let q = plan.modulus().value();

        let transformed = NttPolynomial::from_values(&plan, vec![q + 1; 64]);

        assert!(transformed.values().iter().all(|&value| value == 1));
    }
}
