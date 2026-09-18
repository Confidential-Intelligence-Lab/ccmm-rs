use super::{Modulus, Polynomial};

/// Polynomial represented in residue-number-system form.
///
/// One logical polynomial is represented simultaneously modulo several
/// pairwise-coprime moduli:
///
/// `R_Q ~= R_q0 x R_q1 x ... x R_qL`,
///
/// where `Q = product(q_i)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnsPolynomial {
    degree: usize,
    moduli: Vec<Modulus>,
    residues: Vec<Polynomial>,
}

impl RnsPolynomial {
    /// Constructs an RNS polynomial from integer coefficients.
    ///
    /// Each coefficient is reduced independently modulo every RNS limb.
    pub fn from_coefficients(moduli: Vec<Modulus>, coefficients: &[u128]) -> Self {
        validate_moduli(&moduli);

        assert!(
            !coefficients.is_empty(),
            "RNS polynomial degree must be positive"
        );

        let degree = coefficients.len();

        let residues = moduli
            .iter()
            .copied()
            .map(|modulus| {
                Polynomial::new(
                    modulus,
                    coefficients
                        .iter()
                        .map(|&value| (value % u128::from(modulus.value())) as u64)
                        .collect(),
                )
            })
            .collect();

        Self {
            degree,
            moduli,
            residues,
        }
    }

    /// Constructs an RNS polynomial from already-separated residue limbs.
    pub fn from_residues(residues: Vec<Polynomial>) -> Self {
        assert!(
            !residues.is_empty(),
            "RNS polynomial must contain at least one residue limb"
        );

        let degree = residues[0].degree();

        let moduli: Vec<_> = residues.iter().map(Polynomial::modulus).collect();

        validate_moduli(&moduli);

        for residue in &residues {
            assert_eq!(
                residue.degree(),
                degree,
                "all RNS residue polynomials must have the same degree"
            );
        }

        Self {
            degree,
            moduli,
            residues,
        }
    }

    pub fn zero(moduli: Vec<Modulus>, degree: usize) -> Self {
        validate_moduli(&moduli);

        assert!(degree > 0, "RNS polynomial degree must be positive");

        let residues = moduli
            .iter()
            .copied()
            .map(|modulus| Polynomial::zero(modulus, degree))
            .collect();

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

    pub fn residues(&self) -> &[Polynomial] {
        &self.residues
    }

    pub fn residue(&self, index: usize) -> &Polynomial {
        &self.residues[index]
    }

    /// Product of all RNS moduli.
    pub fn composite_modulus(&self) -> u128 {
        self.moduli.iter().fold(1_u128, |product, modulus| {
            product
                .checked_mul(u128::from(modulus.value()))
                .expect("RNS composite modulus exceeds u128")
        })
    }

    pub fn add(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        Self::from_residues(
            self.residues
                .iter()
                .zip(&rhs.residues)
                .map(|(lhs, rhs)| lhs.add(rhs))
                .collect(),
        )
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        Self::from_residues(
            self.residues
                .iter()
                .zip(&rhs.residues)
                .map(|(lhs, rhs)| lhs.sub(rhs))
                .collect(),
        )
    }

    pub fn neg(&self) -> Self {
        Self::from_residues(self.residues.iter().map(Polynomial::neg).collect())
    }

    /// Reconstructs canonical coefficients modulo the composite modulus
    /// using the Chinese Remainder Theorem.
    pub fn reconstruct_coefficients(&self) -> Vec<u128> {
        let composite = self.composite_modulus();

        (0..self.degree)
            .map(|coefficient_index| {
                let mut reconstructed = 0_u128;

                for (limb_index, modulus) in self.moduli.iter().copied().enumerate() {
                    let qi = u128::from(modulus.value());
                    let partial = composite / qi;

                    let partial_mod_qi = (partial % qi) as u64;

                    let inverse = modulus.inverse_prime(partial_mod_qi);

                    let residue =
                        u128::from(self.residues[limb_index].coefficients()[coefficient_index]);

                    let term = residue * partial * u128::from(inverse);

                    reconstructed = (reconstructed + term % composite) % composite;
                }

                reconstructed
            })
            .collect()
    }

    fn assert_compatible(&self, rhs: &Self) {
        assert_eq!(self.degree, rhs.degree, "RNS polynomial degrees must match");

        assert_eq!(
            self.moduli, rhs.moduli,
            "RNS polynomial modulus bases must match"
        );
    }
}

fn validate_moduli(moduli: &[Modulus]) {
    assert!(
        !moduli.is_empty(),
        "RNS basis must contain at least one modulus"
    );

    for (index, lhs) in moduli.iter().enumerate() {
        for rhs in &moduli[index + 1..] {
            assert_eq!(
                gcd(lhs.value(), rhs.value()),
                1,
                "RNS moduli must be pairwise coprime"
            );
        }
    }
}

fn gcd(mut lhs: u64, mut rhs: u64) -> u64 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }

    lhs
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

    #[test]
    fn composite_modulus_is_product_of_limbs() {
        let polynomial = RnsPolynomial::zero(basis(), 8);

        assert_eq!(
            polynomial.composite_modulus(),
            12_289_u128 * 40_961_u128 * 65_537_u128
        );
    }

    #[test]
    fn coefficients_roundtrip_through_rns_and_crt() {
        let moduli = basis();

        let composite = moduli.iter().fold(1_u128, |accumulator, modulus| {
            accumulator * u128::from(modulus.value())
        });

        let coefficients = vec![
            0,
            1,
            42,
            12_288,
            65_536,
            1_000_000,
            composite / 2,
            composite - 1,
        ];

        let polynomial = RnsPolynomial::from_coefficients(moduli, &coefficients);

        assert_eq!(polynomial.reconstruct_coefficients(), coefficients);
    }

    #[test]
    fn coefficients_are_canonical_mod_composite() {
        let polynomial = RnsPolynomial::zero(basis(), 4);

        let composite = polynomial.composite_modulus();

        let input = vec![
            composite,
            composite + 1,
            2 * composite + 17,
            7 * composite + 1234,
        ];

        let polynomial = RnsPolynomial::from_coefficients(basis(), &input);

        assert_eq!(polynomial.reconstruct_coefficients(), vec![0, 1, 17, 1234]);
    }

    #[test]
    fn each_limb_contains_expected_residues() {
        let coefficients = vec![1_u128, 12_290, 40_962, 65_538];

        let polynomial = RnsPolynomial::from_coefficients(basis(), &coefficients);

        for (limb_index, modulus) in polynomial.moduli().iter().enumerate() {
            for (coefficient_index, &coefficient) in coefficients.iter().enumerate() {
                assert_eq!(
                    polynomial.residue(limb_index).coefficients()[coefficient_index],
                    (coefficient % u128::from(modulus.value())) as u64
                );
            }
        }
    }

    #[test]
    fn addition_matches_composite_ring_arithmetic() {
        let lhs = RnsPolynomial::from_coefficients(basis(), &[1, 2, 3, 4, 5, 6, 7, 8]);

        let rhs = RnsPolynomial::from_coefficients(basis(), &[8, 7, 6, 5, 4, 3, 2, 1]);

        assert_eq!(lhs.add(&rhs).reconstruct_coefficients(), vec![9; 8]);
    }

    #[test]
    fn subtraction_wraps_mod_composite() {
        let lhs = RnsPolynomial::from_coefficients(basis(), &[1, 2, 3, 4]);

        let rhs = RnsPolynomial::from_coefficients(basis(), &[2, 3, 4, 5]);

        let composite = lhs.composite_modulus();

        assert_eq!(
            lhs.sub(&rhs).reconstruct_coefficients(),
            vec![composite - 1; 4]
        );
    }

    #[test]
    fn negation_is_additive_inverse() {
        let polynomial = RnsPolynomial::from_coefficients(basis(), &[7, 11, 13, 17]);

        let zero = polynomial.add(&polynomial.neg());

        assert_eq!(zero.reconstruct_coefficients(), vec![0; 4]);
    }

    #[test]
    #[should_panic(expected = "pairwise coprime")]
    fn rejects_non_coprime_basis() {
        let _ = RnsPolynomial::zero(vec![Modulus::new(15), Modulus::new(21)], 8);
    }

    #[test]
    #[should_panic(expected = "same degree")]
    fn rejects_mismatched_residue_degrees() {
        let _ = RnsPolynomial::from_residues(vec![
            Polynomial::zero(Modulus::new(12_289), 8),
            Polynomial::zero(Modulus::new(40_961), 16),
        ]);
    }
}
