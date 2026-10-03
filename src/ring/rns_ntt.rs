use super::{make_ntt_plan, Modulus, NttPlan, NttPolynomial, RnsPolynomial};

/// Collection of one negacyclic NTT plan per RNS modulus.
pub struct RnsNttPlan {
    degree: usize,
    moduli: Vec<Modulus>,
    plans: Vec<NttPlan>,
}

impl RnsNttPlan {
    pub fn new(moduli: Vec<Modulus>, degree: usize) -> Self {
        assert!(
            !moduli.is_empty(),
            "RNS NTT plan requires at least one modulus"
        );

        assert!(
            degree > 0 && degree.is_power_of_two(),
            "RNS NTT degree must be a positive power of two"
        );

        let plans = moduli
            .iter()
            .copied()
            .map(|modulus| make_ntt_plan(modulus, degree))
            .collect();

        Self {
            degree,
            moduli,
            plans,
        }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn moduli(&self) -> &[Modulus] {
        &self.moduli
    }

    pub fn plan(&self, index: usize) -> &NttPlan {
        &self.plans[index]
    }

    pub fn forward(&self, polynomial: &RnsPolynomial) -> RnsNttPolynomial {
        assert_eq!(
            polynomial.degree(),
            self.degree,
            "RNS polynomial degree must match RNS NTT plan"
        );

        assert_eq!(
            polynomial.moduli(),
            self.moduli,
            "RNS polynomial basis must match RNS NTT plan"
        );

        let residues = polynomial
            .residues()
            .iter()
            .zip(&self.plans)
            .map(|(residue, plan)| NttPolynomial::from_polynomial(plan, residue))
            .collect();

        RnsNttPolynomial {
            degree: self.degree,
            moduli: self.moduli.clone(),
            residues,
        }
    }

    pub fn inverse(&self, polynomial: &RnsNttPolynomial) -> RnsPolynomial {
        assert_eq!(
            polynomial.degree, self.degree,
            "RNS NTT polynomial degree must match plan"
        );

        assert_eq!(
            polynomial.moduli, self.moduli,
            "RNS NTT polynomial basis must match plan"
        );

        let residues = polynomial
            .residues
            .iter()
            .zip(&self.plans)
            .map(|(residue, plan)| residue.to_polynomial(plan))
            .collect();

        RnsPolynomial::from_residues(residues)
    }

    /// Negacyclic multiplication in the full RNS basis.
    pub fn negacyclic_mul(&self, lhs: &RnsPolynomial, rhs: &RnsPolynomial) -> RnsPolynomial {
        let lhs_ntt = self.forward(lhs);
        let rhs_ntt = self.forward(rhs);

        self.inverse(&lhs_ntt.pointwise_mul(&rhs_ntt))
    }
}

/// RNS polynomial represented in the NTT domain in every residue limb.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsNttPolynomial {
    degree: usize,
    moduli: Vec<Modulus>,
    residues: Vec<NttPolynomial>,
}

impl RnsNttPolynomial {
    /// Constructs an RNS NTT polynomial from already-separated NTT limbs.
    pub fn from_residues(residues: Vec<NttPolynomial>) -> Self {
        assert!(
            !residues.is_empty(),
            "RNS NTT polynomial must contain at least one residue limb"
        );

        let degree = residues[0].degree();
        let moduli: Vec<_> = residues.iter().map(NttPolynomial::modulus).collect();

        for residue in &residues {
            assert_eq!(
                residue.degree(),
                degree,
                "all RNS NTT residue polynomials must have the same degree"
            );
        }

        Self {
            degree,
            moduli,
            residues,
        }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn moduli(&self) -> &[Modulus] {
        &self.moduli
    }

    pub fn residues(&self) -> &[NttPolynomial] {
        &self.residues
    }

    pub fn residue(&self, index: usize) -> &NttPolynomial {
        &self.residues[index]
    }

    /// Addition in the NTT domain across the full RNS basis.
    pub fn add(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.degree, rhs.degree,
            "RNS NTT polynomial degrees must match"
        );

        assert_eq!(
            self.moduli, rhs.moduli,
            "RNS NTT polynomial bases must match"
        );

        let residues = self
            .residues
            .iter()
            .zip(&rhs.residues)
            .map(|(lhs, rhs)| lhs.add(rhs))
            .collect();

        Self {
            degree: self.degree,
            moduli: self.moduli.clone(),
            residues,
        }
    }

    /// Subtraction in the NTT domain across the full RNS basis.
    pub fn sub(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.degree, rhs.degree,
            "RNS NTT polynomial degrees must match"
        );

        assert_eq!(
            self.moduli, rhs.moduli,
            "RNS NTT polynomial bases must match"
        );

        let residues = self
            .residues
            .iter()
            .zip(&rhs.residues)
            .map(|(lhs, rhs)| {
                let values = lhs
                    .values()
                    .iter()
                    .zip(rhs.values())
                    .map(|(&a, &b)| lhs.modulus().sub_canonical(a, b))
                    .collect();

                NttPolynomial::from_canonical_values(
                    &make_ntt_plan(lhs.modulus(), lhs.degree()),
                    values,
                )
            })
            .collect();

        Self::from_residues(residues)
    }

    /// Multiplies by `X^exponent` directly in every RNS NTT limb.
    pub fn mul_monomial_signed(&self, plan: &RnsNttPlan, exponent: i64) -> Self {
        assert_eq!(
            self.degree,
            plan.degree(),
            "RNS NTT polynomial degree must match plan"
        );

        assert_eq!(
            self.moduli,
            plan.moduli(),
            "RNS NTT polynomial basis must match plan"
        );

        let residues = self
            .residues
            .iter()
            .enumerate()
            .map(|(index, residue)| residue.mul_monomial_signed(plan.plan(index), exponent))
            .collect();

        Self::from_residues(residues)
    }

    /// Accumulates a pointwise RNS NTT-domain product into this polynomial.
    ///
    /// This is exactly equivalent to:
    ///
    /// `*self = self.add(&lhs.pointwise_mul(rhs));`
    ///
    /// but delegates to the in-place fused limb operation and avoids
    /// materializing temporary RNS products and sums.
    pub fn pointwise_mul_add_assign(&mut self, lhs: &Self, rhs: &Self) {
        assert_eq!(
            self.degree, lhs.degree,
            "RNS NTT polynomial degrees must match"
        );
        assert_eq!(
            self.degree, rhs.degree,
            "RNS NTT polynomial degrees must match"
        );

        assert_eq!(
            self.moduli, lhs.moduli,
            "RNS NTT polynomial bases must match"
        );
        assert_eq!(
            self.moduli, rhs.moduli,
            "RNS NTT polynomial bases must match"
        );

        for ((acc, lhs_residue), rhs_residue) in self
            .residues
            .iter_mut()
            .zip(&lhs.residues)
            .zip(&rhs.residues)
        {
            acc.pointwise_mul_add_assign_prevalidated(lhs_residue, rhs_residue);
        }
    }

    pub fn pointwise_mul(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.degree, rhs.degree,
            "RNS NTT polynomial degrees must match"
        );

        assert_eq!(
            self.moduli, rhs.moduli,
            "RNS NTT polynomial bases must match"
        );

        let residues = self
            .residues
            .iter()
            .zip(&rhs.residues)
            .map(|(lhs, rhs)| lhs.pointwise_mul(rhs))
            .collect();

        Self {
            degree: self.degree,
            moduli: self.moduli.clone(),
            residues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basis() -> Vec<Modulus> {
        vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]
    }

    fn deterministic_coefficients(degree: usize, offset: u128, modulus: u128) -> Vec<u128> {
        (0..degree)
            .map(|index| {
                let i = index as u128;

                (offset + 17 * i + 5 * i * i + 3 * i * i * i) % modulus
            })
            .collect()
    }

    fn reference_negacyclic_product(lhs: &[u128], rhs: &[u128], modulus: u128) -> Vec<u128> {
        assert_eq!(lhs.len(), rhs.len());

        let degree = lhs.len();
        let mut out = vec![0_u128; degree];

        for (i, &lhs_value) in lhs.iter().enumerate() {
            for (j, &rhs_value) in rhs.iter().enumerate() {
                let product = (lhs_value * rhs_value) % modulus;

                let raw_index = i + j;

                if raw_index < degree {
                    out[raw_index] = (out[raw_index] + product) % modulus;
                } else {
                    let index = raw_index - degree;

                    out[index] = (out[index] + modulus - product) % modulus;
                }
            }
        }

        out
    }

    #[test]
    fn rns_ntt_sub_matches_coefficient_domain_exactly() {
        let degree = 64;
        let plan = RnsNttPlan::new(basis(), degree);

        let lhs = RnsPolynomial::from_coefficients(
            basis(),
            &(0..degree)
                .map(|i| 17_u128 + 31 * i as u128)
                .collect::<Vec<_>>(),
        );

        let rhs = RnsPolynomial::from_coefficients(
            basis(),
            &(0..degree)
                .map(|i| 7_u128 + 13 * i as u128)
                .collect::<Vec<_>>(),
        );

        let lhs_ntt = plan.forward(&lhs);
        let rhs_ntt = plan.forward(&rhs);

        let actual = plan.inverse(&lhs_ntt.sub(&rhs_ntt));

        let expected = RnsPolynomial::from_residues(
            lhs.residues()
                .iter()
                .zip(rhs.residues())
                .map(|(a, b)| a.sub(b))
                .collect(),
        );

        assert_eq!(actual, expected);

        println!("RNS_NTT_SUB_EQUIVALENCE=PASS");
    }

    #[test]
    fn rns_ntt_monomial_matches_coefficient_domain_exactly() {
        let degree = 64;
        let plan = RnsNttPlan::new(basis(), degree);

        let polynomial = RnsPolynomial::from_coefficients(
            basis(),
            &(0..degree)
                .map(|i| {
                    let x = i as u128;
                    5 + 17 * x + 3 * x * x
                })
                .collect::<Vec<_>>(),
        );

        let transformed = plan.forward(&polynomial);

        for exponent in [
            -257_i64, -129, -128, -65, -64, -1, 0, 1, 63, 64, 65, 127, 128, 129, 257,
        ] {
            let actual = plan.inverse(&transformed.mul_monomial_signed(&plan, exponent));

            let expected = RnsPolynomial::from_residues(
                polynomial
                    .residues()
                    .iter()
                    .map(|residue| {
                        let period = 2_i128 * degree as i128;

                        let normalized = i128::from(exponent).rem_euclid(period) as usize;

                        let shift = normalized % degree;

                        let sign = if normalized >= degree { -1_i8 } else { 1_i8 };

                        residue.mul_monomial_signed(shift, sign)
                    })
                    .collect(),
            );

            assert_eq!(
                actual, expected,
                "RNS NTT monomial mismatch for exponent={exponent}"
            );
        }

        println!("RNS_NTT_SIGNED_MONOMIAL_EQUIVALENCE=PASS");
    }

    #[test]
    fn plan_constructs_one_ntt_per_rns_limb() {
        let degree = 64;
        let plan = RnsNttPlan::new(basis(), degree);

        assert_eq!(plan.degree(), degree);
        assert_eq!(plan.moduli(), basis());

        for limb in 0..basis().len() {
            let modulus = basis()[limb];
            let scalar_plan = plan.plan(limb);
            let psi = scalar_plan.psi();

            assert_eq!(modulus.pow(psi, (2 * degree) as u64,), 1);

            assert_eq!(modulus.pow(psi, degree as u64,), modulus.value() - 1);
        }
    }

    #[test]
    fn multi_prime_ntt_roundtrip_is_exact() {
        let degree = 64;
        let plan = RnsNttPlan::new(basis(), degree);

        let zero = RnsPolynomial::zero(basis(), degree);

        let composite = zero.composite_modulus();

        let coefficients = deterministic_coefficients(degree, 7, composite);

        let polynomial = RnsPolynomial::from_coefficients(basis(), &coefficients);

        let transformed = plan.forward(&polynomial);

        let recovered = plan.inverse(&transformed);

        assert_eq!(recovered, polynomial);

        assert_eq!(recovered.reconstruct_coefficients(), coefficients);
    }

    #[test]
    fn each_ntt_limb_matches_scalar_ntt() {
        let degree = 64;
        let plan = RnsNttPlan::new(basis(), degree);

        let polynomial = RnsPolynomial::from_coefficients(
            basis(),
            &(0..degree)
                .map(|index| 11_u128 + 13 * index as u128)
                .collect::<Vec<_>>(),
        );

        let transformed = plan.forward(&polynomial);

        for limb in 0..basis().len() {
            assert_eq!(
                transformed.residue(limb).values(),
                plan.plan(limb).forward_radix2(polynomial.residue(limb))
            );
        }
    }

    #[test]
    fn multi_prime_ntt_product_matches_limbwise_reference() {
        let degree = 64;
        let plan = RnsNttPlan::new(basis(), degree);

        let lhs = RnsPolynomial::from_coefficients(
            basis(),
            &(0..degree)
                .map(|index| 5_u128 + 7 * index as u128)
                .collect::<Vec<_>>(),
        );

        let rhs = RnsPolynomial::from_coefficients(
            basis(),
            &(0..degree)
                .map(|index| 17_u128 + 11 * index as u128)
                .collect::<Vec<_>>(),
        );

        let product = plan.negacyclic_mul(&lhs, &rhs);

        for limb in 0..basis().len() {
            assert_eq!(
                product.residue(limb),
                &lhs.residue(limb).negacyclic_mul(rhs.residue(limb))
            );
        }
    }

    #[test]
    fn multi_prime_ntt_product_matches_composite_crt_reference() {
        let degree = 64;

        let plan = RnsNttPlan::new(basis(), degree);

        let zero = RnsPolynomial::zero(basis(), degree);

        let composite = zero.composite_modulus();

        let lhs_coefficients = deterministic_coefficients(degree, 3, composite);

        let rhs_coefficients = deterministic_coefficients(degree, 29, composite);

        let lhs = RnsPolynomial::from_coefficients(basis(), &lhs_coefficients);

        let rhs = RnsPolynomial::from_coefficients(basis(), &rhs_coefficients);

        let actual = plan.negacyclic_mul(&lhs, &rhs).reconstruct_coefficients();

        let expected =
            reference_negacyclic_product(&lhs_coefficients, &rhs_coefficients, composite);

        assert_eq!(actual, expected);
    }

    #[test]
    fn multi_prime_ntt_matches_composite_reference_across_degrees() {
        for degree in [16_usize, 32, 64, 128, 256] {
            let plan = RnsNttPlan::new(basis(), degree);

            let zero = RnsPolynomial::zero(basis(), degree);

            let composite = zero.composite_modulus();

            let lhs_coefficients = deterministic_coefficients(degree, 13, composite);

            let rhs_coefficients = deterministic_coefficients(degree, 71, composite);

            let lhs = RnsPolynomial::from_coefficients(basis(), &lhs_coefficients);

            let rhs = RnsPolynomial::from_coefficients(basis(), &rhs_coefficients);

            let actual = plan.negacyclic_mul(&lhs, &rhs).reconstruct_coefficients();

            let expected =
                reference_negacyclic_product(&lhs_coefficients, &rhs_coefficients, composite);

            assert_eq!(
                actual, expected,
                "multi-prime NTT mismatch at degree {degree}"
            );
        }
    }
}
