use super::Modulus;

/// Polynomial in `Z_q[X]` with fixed degree bound `N`.
///
/// Coefficients are stored in ascending order:
///
/// `coefficients[i]` is the coefficient of `X^i`.
///
/// All coefficients are maintained canonically in `[0, q)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polynomial {
    modulus: Modulus,
    coefficients: Vec<u64>,
}

impl Polynomial {
    /// Constructs a polynomial and reduces every coefficient modulo `q`.
    ///
    /// # Panics
    ///
    /// Panics if the coefficient vector is empty.
    pub fn new(modulus: Modulus, coefficients: Vec<u64>) -> Self {
        assert!(
            !coefficients.is_empty(),
            "polynomial must contain at least one coefficient"
        );

        let coefficients = coefficients
            .into_iter()
            .map(|coefficient| modulus.reduce(coefficient))
            .collect();

        Self {
            modulus,
            coefficients,
        }
    }

    /// Constructs the zero polynomial of ring degree `degree`.
    ///
    /// # Panics
    ///
    /// Panics if `degree == 0`.
    pub fn zero(modulus: Modulus, degree: usize) -> Self {
        assert!(degree > 0, "polynomial degree must be positive");

        Self {
            modulus,
            coefficients: vec![0; degree],
        }
    }

    pub fn modulus(&self) -> Modulus {
        self.modulus
    }

    /// Number of stored ring coefficients.
    pub fn degree(&self) -> usize {
        self.coefficients.len()
    }

    pub fn coefficients(&self) -> &[u64] {
        &self.coefficients
    }

    pub fn coefficient(&self, index: usize) -> u64 {
        self.coefficients[index]
    }

    /// Coefficient-wise addition in `Z_q[X]`.
    ///
    /// # Panics
    ///
    /// Panics if the polynomials have different moduli or degrees.
    pub fn add(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        let coefficients = self
            .coefficients
            .iter()
            .zip(&rhs.coefficients)
            .map(|(&lhs, &rhs)| self.modulus.add(lhs, rhs))
            .collect();

        Self {
            modulus: self.modulus,
            coefficients,
        }
    }

    /// Coefficient-wise subtraction in `Z_q[X]`.
    pub fn sub(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        let coefficients = self
            .coefficients
            .iter()
            .zip(&rhs.coefficients)
            .map(|(&lhs, &rhs)| self.modulus.sub(lhs, rhs))
            .collect();

        Self {
            modulus: self.modulus,
            coefficients,
        }
    }

    /// Additive inverse in `Z_q[X]`.
    pub fn neg(&self) -> Self {
        let coefficients = self
            .coefficients
            .iter()
            .map(|&coefficient| self.modulus.neg(coefficient))
            .collect();

        Self {
            modulus: self.modulus,
            coefficients,
        }
    }

    /// Scalar multiplication in `Z_q[X]`.
    pub fn scalar_mul(&self, scalar: u64) -> Self {
        let coefficients = self
            .coefficients
            .iter()
            .map(|&coefficient| self.modulus.mul(coefficient, scalar))
            .collect();

        Self {
            modulus: self.modulus,
            coefficients,
        }
    }

    /// Multiplies by `sign * X^exponent` in `Z_q[X] / (X^N + 1)`.
    ///
    /// This is a signed coefficient permutation and therefore runs in
    /// O(N), rather than using general negacyclic polynomial multiplication.
    ///
    /// `sign` must be either `1` or `-1`. The exponent is reduced modulo
    /// `2N`, using `X^N = -1` and hence `X^(2N) = 1`.
    pub fn mul_monomial_signed(&self, exponent: usize, sign: i8) -> Self {
        assert!(
            sign == 1 || sign == -1,
            "monomial sign must be either 1 or -1"
        );

        let n = self.degree();
        let exponent = exponent % (2 * n);
        let shift = exponent % n;
        let exponent_negates = exponent >= n;
        let external_negates = sign == -1;

        let mut coefficients = vec![0_u64; n];

        for (source, &coefficient) in self.coefficients.iter().enumerate() {
            let sum = source + shift;
            let (destination, wrap_negates) = if sum >= n {
                (sum - n, true)
            } else {
                (sum, false)
            };

            let negate = exponent_negates ^ external_negates ^ wrap_negates;

            coefficients[destination] = if negate {
                self.modulus.neg(coefficient)
            } else {
                coefficient
            };
        }

        Self {
            modulus: self.modulus,
            coefficients,
        }
    }

    /// Naive negacyclic multiplication in
    ///
    /// `Z_q[X] / (X^N + 1)`.
    ///
    /// Terms with degree `>= N` wrap with a sign change because
    /// `X^N = -1`.
    ///
    /// # Panics
    ///
    /// Panics if the polynomials have different moduli or degrees.
    pub fn negacyclic_mul(&self, rhs: &Self) -> Self {
        self.assert_compatible(rhs);

        let n = self.degree();
        let mut coefficients = vec![0_u64; n];

        for i in 0..n {
            for j in 0..n {
                let product = self.modulus.mul(self.coefficients[i], rhs.coefficients[j]);

                let degree = i + j;

                if degree < n {
                    coefficients[degree] = self.modulus.add(coefficients[degree], product);
                } else {
                    let wrapped = degree - n;
                    coefficients[wrapped] = self.modulus.sub(coefficients[wrapped], product);
                }
            }
        }

        Self {
            modulus: self.modulus,
            coefficients,
        }
    }

    fn assert_compatible(&self, rhs: &Self) {
        assert_eq!(self.modulus, rhs.modulus, "polynomial moduli must match");
        assert_eq!(self.degree(), rhs.degree(), "polynomial degrees must match");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_reduces_coefficients() {
        let modulus = Modulus::new(17);
        let polynomial = Polynomial::new(modulus, vec![0, 16, 17, 18, 35]);

        assert_eq!(polynomial.coefficients(), &[0, 16, 0, 1, 1]);
        assert_eq!(polynomial.modulus(), modulus);
        assert_eq!(polynomial.degree(), 5);
    }

    #[test]
    fn zero_constructs_fixed_degree_polynomial() {
        let polynomial = Polynomial::zero(Modulus::new(17), 4);

        assert_eq!(polynomial.coefficients(), &[0, 0, 0, 0]);
        assert_eq!(polynomial.degree(), 4);
    }

    #[test]
    #[should_panic(expected = "at least one coefficient")]
    fn rejects_empty_polynomial() {
        let _ = Polynomial::new(Modulus::new(17), vec![]);
    }

    #[test]
    #[should_panic(expected = "degree must be positive")]
    fn zero_rejects_zero_degree() {
        let _ = Polynomial::zero(Modulus::new(17), 0);
    }

    #[test]
    fn addition_is_coefficient_wise_mod_q() {
        let q = Modulus::new(17);

        let lhs = Polynomial::new(q, vec![16, 2, 10, 7]);
        let rhs = Polynomial::new(q, vec![4, 16, 10, 12]);

        assert_eq!(lhs.add(&rhs).coefficients(), &[3, 1, 3, 2]);
    }

    #[test]
    fn subtraction_wraps_mod_q() {
        let q = Modulus::new(17);

        let lhs = Polynomial::new(q, vec![1, 2, 3, 4]);
        let rhs = Polynomial::new(q, vec![2, 2, 5, 1]);

        assert_eq!(lhs.sub(&rhs).coefficients(), &[16, 0, 15, 3]);
    }

    #[test]
    fn negation_produces_additive_inverse() {
        let q = Modulus::new(17);
        let polynomial = Polynomial::new(q, vec![0, 1, 16, 8]);

        assert_eq!(polynomial.neg().coefficients(), &[0, 16, 1, 9]);

        assert_eq!(
            polynomial.add(&polynomial.neg()).coefficients(),
            &[0, 0, 0, 0]
        );
    }

    #[test]
    fn scalar_multiplication_reduces_mod_q() {
        let q = Modulus::new(17);
        let polynomial = Polynomial::new(q, vec![1, 2, 3, 4]);

        assert_eq!(polynomial.scalar_mul(6).coefficients(), &[6, 12, 1, 7]);
    }

    #[test]
    #[should_panic(expected = "polynomial moduli must match")]
    fn rejects_mismatched_moduli() {
        let lhs = Polynomial::new(Modulus::new(17), vec![1, 2]);
        let rhs = Polynomial::new(Modulus::new(19), vec![1, 2]);

        let _ = lhs.add(&rhs);
    }

    #[test]
    #[should_panic(expected = "polynomial degrees must match")]
    fn rejects_mismatched_degrees() {
        let q = Modulus::new(17);

        let lhs = Polynomial::new(q, vec![1, 2]);
        let rhs = Polynomial::new(q, vec![1, 2, 3]);

        let _ = lhs.add(&rhs);
    }

    #[test]
    fn signed_monomial_multiplication_matches_naive_negacyclic_product() {
        const DEGREE: usize = 8;

        let modulus = Modulus::new(97);

        let polynomial = Polynomial::new(
            modulus,
            (0..DEGREE)
                .map(|index| ((index * 17 + 11) % 97) as u64)
                .collect(),
        );

        for sign in [-1_i8, 1_i8] {
            for exponent in 0..(4 * DEGREE) {
                let reduced = exponent % (2 * DEGREE);
                let shift = reduced % DEGREE;

                let mut monomial = vec![0_u64; DEGREE];

                let mut coefficient = 1_u64;

                if reduced >= DEGREE {
                    coefficient = modulus.neg(coefficient);
                }

                if sign == -1 {
                    coefficient = modulus.neg(coefficient);
                }

                monomial[shift] = coefficient;

                let reference = polynomial.negacyclic_mul(&Polynomial::new(modulus, monomial));

                let optimized = polynomial.mul_monomial_signed(exponent, sign);

                assert_eq!(
                    optimized, reference,
                    "signed monomial mismatch for exponent={exponent}, sign={sign}"
                );
            }
        }

        println!("SIGNED_MONOMIAL_EQUIVALENCE=PASS");
    }

    #[test]
    fn negacyclic_multiplication_without_wrap_matches_convolution() {
        let q = Modulus::new(97);

        let lhs = Polynomial::new(q, vec![1, 2, 0, 0]);
        let rhs = Polynomial::new(q, vec![3, 4, 0, 0]);

        assert_eq!(lhs.negacyclic_mul(&rhs).coefficients(), &[3, 10, 8, 0]);
    }

    #[test]
    fn negacyclic_multiplication_wraps_with_sign_change() {
        let q = Modulus::new(17);

        let lhs = Polynomial::new(q, vec![0, 0, 0, 1]);
        let rhs = Polynomial::new(q, vec![0, 1, 0, 0]);

        assert_eq!(lhs.negacyclic_mul(&rhs).coefficients(), &[16, 0, 0, 0]);
    }

    #[test]
    fn negacyclic_multiplication_handles_multiple_wraps() {
        let q = Modulus::new(17);

        let lhs = Polynomial::new(q, vec![1, 2, 3, 4]);
        let rhs = Polynomial::new(q, vec![5, 6, 7, 8]);

        assert_eq!(lhs.negacyclic_mul(&rhs).coefficients(), &[12, 15, 2, 9]);
    }

    #[test]
    fn negacyclic_identity_holds() {
        let q = Modulus::new(97);

        let a = Polynomial::new(q, vec![11, 22, 33, 44]);
        let one = Polynomial::new(q, vec![1, 0, 0, 0]);

        assert_eq!(a.negacyclic_mul(&one), a);
        assert_eq!(one.negacyclic_mul(&a), a);
    }

    #[test]
    fn negacyclic_zero_annihilates() {
        let q = Modulus::new(97);

        let a = Polynomial::new(q, vec![11, 22, 33, 44]);
        let zero = Polynomial::zero(q, 4);

        assert_eq!(a.negacyclic_mul(&zero), zero);
        assert_eq!(zero.negacyclic_mul(&a), zero);
    }

    #[test]
    fn negacyclic_multiplication_is_commutative() {
        let q = Modulus::new(97);

        let a = Polynomial::new(q, vec![1, 8, 23, 42]);
        let b = Polynomial::new(q, vec![5, 7, 11, 13]);

        assert_eq!(a.negacyclic_mul(&b), b.negacyclic_mul(&a));
    }

    #[test]
    fn negacyclic_multiplication_distributes_over_addition() {
        let q = Modulus::new(97);

        let a = Polynomial::new(q, vec![1, 2, 3, 4]);
        let b = Polynomial::new(q, vec![5, 6, 7, 8]);
        let c = Polynomial::new(q, vec![9, 10, 11, 12]);

        let lhs = a.negacyclic_mul(&b.add(&c));
        let rhs = a.negacyclic_mul(&b).add(&a.negacyclic_mul(&c));

        assert_eq!(lhs, rhs);
    }

    #[test]
    fn ring_identities_hold() {
        let q = Modulus::new(97);
        let a = Polynomial::new(q, vec![1, 20, 96, 45]);
        let zero = Polynomial::zero(q, 4);

        assert_eq!(a.add(&zero), a);
        assert_eq!(a.sub(&a), zero);
        assert_eq!(a.add(&a.neg()), zero);
        assert_eq!(a.scalar_mul(1), a);
        assert_eq!(a.scalar_mul(0), zero);
    }
}
