use super::{Modulus, NttPolynomial, PreparedNttPlan, RnsNttPlan, RnsNttPolynomial, RnsPolynomial};

/// Prepared radix-2 execution representation of an `RnsNttPlan`.
///
/// The canonical `RnsNttPlan` remains the mathematical plan. This type
/// prepares one reusable scalar NTT execution schedule per RNS modulus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedRnsNttPlan {
    degree: usize,
    moduli: Vec<Modulus>,
    plans: Vec<PreparedNttPlan>,
}

impl PreparedRnsNttPlan {
    pub fn new(plan: &RnsNttPlan) -> Self {
        let plans = (0..plan.moduli().len())
            .map(|index| PreparedNttPlan::new(plan.plan(index)))
            .collect();

        Self {
            degree: plan.degree(),
            moduli: plan.moduli().to_vec(),
            plans,
        }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn moduli(&self) -> &[Modulus] {
        &self.moduli
    }

    pub fn plan(&self, index: usize) -> &PreparedNttPlan {
        &self.plans[index]
    }

    pub fn forward(&self, polynomial: &RnsPolynomial) -> RnsNttPolynomial {
        assert_eq!(
            polynomial.degree(),
            self.degree,
            "RNS polynomial degree must match prepared RNS NTT plan"
        );

        assert_eq!(
            polynomial.moduli(),
            self.moduli,
            "RNS polynomial basis must match prepared RNS NTT plan"
        );

        let residues = polynomial
            .residues()
            .iter()
            .zip(&self.plans)
            .map(|(residue, plan)| NttPolynomial::from_prepared_values(plan, plan.forward(residue)))
            .collect();

        RnsNttPolynomial::from_residues(residues)
    }

    pub fn inverse(&self, polynomial: &RnsNttPolynomial) -> RnsPolynomial {
        assert_eq!(
            polynomial.degree(),
            self.degree,
            "RNS NTT polynomial degree must match prepared RNS NTT plan"
        );

        assert_eq!(
            polynomial.moduli(),
            self.moduli,
            "RNS NTT polynomial basis must match prepared RNS NTT plan"
        );

        let residues = polynomial
            .residues()
            .iter()
            .zip(&self.plans)
            .map(|(residue, plan)| plan.inverse(residue.values()))
            .collect();

        RnsPolynomial::from_residues(residues)
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

    fn polynomial(degree: usize) -> RnsPolynomial {
        let coefficients: Vec<u128> = (0..degree)
            .map(|index| {
                let i = index as u128;
                17 + 11 * i + 7 * i * i + 3 * i * i * i
            })
            .collect();

        RnsPolynomial::from_coefficients(basis(), &coefficients)
    }

    #[test]
    fn prepared_rns_forward_matches_canonical_exactly() {
        let plan = RnsNttPlan::new(basis(), 8);
        let prepared = PreparedRnsNttPlan::new(&plan);
        let polynomial = polynomial(8);

        let canonical = plan.forward(&polynomial);
        let actual = prepared.forward(&polynomial);

        assert_eq!(actual, canonical);

        println!("PREPARED_RNS_NTT_FORWARD_EQUIVALENCE=PASS");
    }

    #[test]
    fn prepared_rns_inverse_matches_canonical_exactly() {
        let plan = RnsNttPlan::new(basis(), 8);
        let prepared = PreparedRnsNttPlan::new(&plan);
        let polynomial = polynomial(8);

        let transformed = plan.forward(&polynomial);

        assert_eq!(prepared.inverse(&transformed), plan.inverse(&transformed));

        println!("PREPARED_RNS_NTT_INVERSE_EQUIVALENCE=PASS");
    }

    #[test]
    fn prepared_rns_roundtrip_recovers_polynomial() {
        let plan = RnsNttPlan::new(basis(), 8);
        let prepared = PreparedRnsNttPlan::new(&plan);
        let polynomial = polynomial(8);

        let transformed = prepared.forward(&polynomial);
        let recovered = prepared.inverse(&transformed);

        assert_eq!(recovered, polynomial);

        println!("PREPARED_RNS_NTT_ROUNDTRIP=PASS");
    }
}
