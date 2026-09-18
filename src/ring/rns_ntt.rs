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
