use super::{Modulus, Polynomial};

/// Reference negacyclic NTT plan for
///
/// `Z_q[X] / (X^N + 1)`.
///
/// The implementation deliberately uses an `O(N^2)` evaluation/interpolation
/// algorithm. It is intended as a correctness oracle for a later optimized
/// radix-2 NTT.
///
/// `psi` must be a primitive `2N`-th root of unity modulo `q`, so:
///
/// `psi^(2N) = 1`
///
/// and
///
/// `psi^N = -1 mod q`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NttPlan {
    modulus: Modulus,
    degree: usize,
    psi: u64,
    psi_inverse: u64,
    degree_inverse: u64,
}

impl NttPlan {
    /// Constructs a reference negacyclic NTT plan.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - `degree` is zero or not a power of two;
    /// - `2 * degree` does not divide `q - 1`;
    /// - `psi` does not have the required primitive `2N`-th-root property.
    ///
    /// The modulus is assumed prime because inverses use Fermat's theorem.
    pub fn new(modulus: Modulus, degree: usize, psi: u64) -> Self {
        assert!(
            degree > 0 && degree.is_power_of_two(),
            "NTT degree must be a positive power of two"
        );

        let two_n = degree.checked_mul(2).expect("NTT degree is too large");

        assert_eq!(
            (modulus.value() - 1) % two_n as u64,
            0,
            "2N must divide q - 1"
        );

        let psi = modulus.reduce(psi);

        assert_eq!(
            modulus.pow(psi, two_n as u64),
            1,
            "psi must satisfy psi^(2N) = 1"
        );

        assert_eq!(
            modulus.pow(psi, degree as u64),
            modulus.value() - 1,
            "psi must satisfy psi^N = -1"
        );

        let psi_inverse = modulus.inverse_prime(psi);
        let degree_inverse = modulus.inverse_prime(degree as u64);

        Self {
            modulus,
            degree,
            psi,
            psi_inverse,
            degree_inverse,
        }
    }

    pub fn modulus(self) -> Modulus {
        self.modulus
    }

    pub fn degree(self) -> usize {
        self.degree
    }

    pub fn psi(self) -> u64 {
        self.psi
    }

    /// Reference forward negacyclic transform.
    ///
    /// Output entry `k` is the polynomial evaluated at
    ///
    /// `psi^(2k + 1)`.
    pub fn forward_reference(&self, polynomial: &Polynomial) -> Vec<u64> {
        self.assert_polynomial_compatible(polynomial);

        let mut output = vec![0_u64; self.degree];

        for (k, output_value) in output.iter_mut().enumerate() {
            let evaluation_point = self.modulus.pow(self.psi, (2 * k + 1) as u64);

            let mut power = 1_u64;
            let mut sum = 0_u64;

            for &coefficient in polynomial.coefficients() {
                sum = self.modulus.add(sum, self.modulus.mul(coefficient, power));

                power = self.modulus.mul(power, evaluation_point);
            }

            *output_value = sum;
        }

        output
    }

    /// Reference inverse negacyclic transform.
    ///
    /// This interpolates the polynomial from evaluations at the odd powers
    /// of `psi`.
    pub fn inverse_reference(&self, values: &[u64]) -> Polynomial {
        assert_eq!(
            values.len(),
            self.degree,
            "NTT value count must match plan degree"
        );

        let mut coefficients = vec![0_u64; self.degree];

        for (j, coefficient) in coefficients.iter_mut().enumerate() {
            let mut sum = 0_u64;

            for (k, &value) in values.iter().enumerate() {
                let exponent = ((2 * k + 1) * j) as u64;

                let inverse_power = self.modulus.pow(self.psi_inverse, exponent);

                sum = self
                    .modulus
                    .add(sum, self.modulus.mul(value, inverse_power));
            }

            *coefficient = self.modulus.mul(sum, self.degree_inverse);
        }

        Polynomial::new(self.modulus, coefficients)
    }

    /// Pointwise multiplication of two NTT-domain vectors.
    pub fn pointwise_mul(&self, lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
        assert_eq!(
            lhs.len(),
            self.degree,
            "left NTT value count must match plan degree"
        );
        assert_eq!(
            rhs.len(),
            self.degree,
            "right NTT value count must match plan degree"
        );

        lhs.iter()
            .zip(rhs)
            .map(|(&lhs, &rhs)| self.modulus.mul(lhs, rhs))
            .collect()
    }

    fn assert_polynomial_compatible(&self, polynomial: &Polynomial) {
        assert_eq!(
            polynomial.modulus(),
            self.modulus,
            "polynomial modulus must match NTT modulus"
        );

        assert_eq!(
            polynomial.degree(),
            self.degree,
            "polynomial degree must match NTT degree"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> NttPlan {
        // N = 8, q = 97.
        //
        // 2N = 16 divides q - 1 = 96.
        // psi = 8 has exact order 16 modulo 97.
        NttPlan::new(Modulus::new(97), 8, 8)
    }

    #[test]
    fn plan_preserves_parameters() {
        let plan = plan();

        assert_eq!(plan.modulus(), Modulus::new(97));
        assert_eq!(plan.degree(), 8);
        assert_eq!(plan.psi(), 8);
    }

    #[test]
    fn root_has_required_negacyclic_order() {
        let plan = plan();
        let q = plan.modulus();

        assert_eq!(q.pow(plan.psi(), 16), 1);
        assert_eq!(q.pow(plan.psi(), 8), 96);
    }

    #[test]
    fn reference_roundtrip_recovers_polynomial() {
        let plan = plan();

        let polynomial = Polynomial::new(Modulus::new(97), vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let transformed = plan.forward_reference(&polynomial);
        let recovered = plan.inverse_reference(&transformed);

        assert_eq!(recovered, polynomial);
    }

    #[test]
    fn reference_roundtrip_recovers_zero() {
        let plan = plan();
        let polynomial = Polynomial::zero(Modulus::new(97), 8);

        let transformed = plan.forward_reference(&polynomial);
        let recovered = plan.inverse_reference(&transformed);

        assert_eq!(recovered, polynomial);
    }

    #[test]
    fn reference_roundtrip_recovers_constant_one() {
        let plan = plan();

        let polynomial = Polynomial::new(Modulus::new(97), vec![1, 0, 0, 0, 0, 0, 0, 0]);

        let transformed = plan.forward_reference(&polynomial);

        assert_eq!(transformed, vec![1; 8]);
        assert_eq!(plan.inverse_reference(&transformed), polynomial);
    }

    #[test]
    fn reference_ntt_product_matches_naive_negacyclic_product() {
        let plan = plan();
        let q = Modulus::new(97);

        let lhs = Polynomial::new(q, vec![1, 8, 23, 42, 7, 11, 19, 31]);

        let rhs = Polynomial::new(q, vec![5, 7, 11, 13, 17, 29, 37, 41]);

        let lhs_ntt = plan.forward_reference(&lhs);
        let rhs_ntt = plan.forward_reference(&rhs);

        let product_ntt = plan.pointwise_mul(&lhs_ntt, &rhs_ntt);
        let product = plan.inverse_reference(&product_ntt);

        assert_eq!(product, lhs.negacyclic_mul(&rhs));
    }

    #[test]
    fn reference_ntt_matches_naive_product_for_multiple_vectors() {
        let plan = plan();
        let q = Modulus::new(97);

        let vectors = [
            (vec![1, 2, 3, 4, 5, 6, 7, 8], vec![8, 7, 6, 5, 4, 3, 2, 1]),
            (vec![0, 1, 0, 1, 0, 1, 0, 1], vec![1, 0, 1, 0, 1, 0, 1, 0]),
            (
                vec![96, 95, 94, 93, 92, 91, 90, 89],
                vec![9, 10, 11, 12, 13, 14, 15, 16],
            ),
        ];

        for (lhs, rhs) in vectors {
            let lhs = Polynomial::new(q, lhs);
            let rhs = Polynomial::new(q, rhs);

            let transformed_product =
                plan.pointwise_mul(&plan.forward_reference(&lhs), &plan.forward_reference(&rhs));

            let actual = plan.inverse_reference(&transformed_product);

            let expected = lhs.negacyclic_mul(&rhs);

            assert_eq!(actual, expected);
        }
    }

    #[test]
    #[should_panic(expected = "positive power of two")]
    fn rejects_non_power_of_two_degree() {
        let _ = NttPlan::new(Modulus::new(97), 6, 8);
    }

    #[test]
    #[should_panic(expected = "2N must divide q - 1")]
    fn rejects_incompatible_modulus_and_degree() {
        let _ = NttPlan::new(Modulus::new(97), 32, 8);
    }

    #[test]
    #[should_panic(expected = "psi^N = -1")]
    fn rejects_wrong_root_order() {
        let _ = NttPlan::new(Modulus::new(97), 8, 1);
    }
}
