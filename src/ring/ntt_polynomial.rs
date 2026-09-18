use super::{Modulus, NttPlan, Polynomial};

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

    pub fn modulus(&self) -> Modulus {
        self.modulus
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn values(&self) -> &[u64] {
        &self.values
    }

    /// Pointwise multiplication in the NTT domain.
    pub fn pointwise_mul(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        let values = self
            .values
            .iter()
            .zip(&rhs.values)
            .map(|(&lhs, &rhs)| self.modulus.mul(lhs, rhs))
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
