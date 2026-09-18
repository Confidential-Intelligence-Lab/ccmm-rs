use super::{Modulus, NttPlan, Polynomial};

impl NttPlan {
    /// Optimized radix-2 forward negacyclic NTT.
    ///
    /// The negacyclic transform is implemented by twisting coefficient
    /// `a_j` by `psi^j`, followed by an ordinary cyclic NTT using
    /// `omega = psi^2`, where `psi` is a primitive `2N`-th root.
    pub fn forward_radix2(&self, polynomial: &Polynomial) -> Vec<u64> {
        assert_eq!(
            polynomial.modulus(),
            self.modulus(),
            "polynomial modulus must match NTT modulus"
        );

        assert_eq!(
            polynomial.degree(),
            self.degree(),
            "polynomial degree must match NTT degree"
        );

        let modulus = self.modulus();
        let psi = self.psi();

        let mut values = polynomial.coefficients().to_vec();

        let mut twist = 1_u64;

        for value in &mut values {
            *value = modulus.mul(*value, twist);
            twist = modulus.mul(twist, psi);
        }

        let omega = modulus.mul(psi, psi);

        cyclic_ntt_radix2(&mut values, modulus, omega);

        values
    }

    /// Multiplies two coefficient-domain polynomials using
    /// the radix-2 negacyclic NTT.
    ///
    /// This is the optimized counterpart to
    /// `Polynomial::negacyclic_mul`, which remains the correctness
    /// reference implementation.
    pub fn negacyclic_mul(&self, lhs: &Polynomial, rhs: &Polynomial) -> Polynomial {
        assert_eq!(
            lhs.modulus(),
            self.modulus(),
            "left polynomial modulus must match NTT plan"
        );
        assert_eq!(
            rhs.modulus(),
            self.modulus(),
            "right polynomial modulus must match NTT plan"
        );
        assert_eq!(
            lhs.degree(),
            self.degree(),
            "left polynomial degree must match NTT plan"
        );
        assert_eq!(
            rhs.degree(),
            self.degree(),
            "right polynomial degree must match NTT plan"
        );

        let lhs_ntt = self.forward_radix2(lhs);
        let rhs_ntt = self.forward_radix2(rhs);

        let product_ntt = self.pointwise_mul(&lhs_ntt, &rhs_ntt);

        self.inverse_radix2(&product_ntt)
    }

    /// Optimized radix-2 inverse negacyclic NTT.
    pub fn inverse_radix2(&self, values: &[u64]) -> Polynomial {
        assert_eq!(
            values.len(),
            self.degree(),
            "NTT value count must match plan degree"
        );

        let modulus = self.modulus();
        let psi = self.psi();

        let omega = modulus.mul(psi, psi);
        let omega_inverse = modulus.inverse_prime(omega);

        let mut coefficients = values.to_vec();

        cyclic_ntt_radix2(&mut coefficients, modulus, omega_inverse);

        let degree_inverse = modulus.inverse_prime(self.degree() as u64);

        for coefficient in &mut coefficients {
            *coefficient = modulus.mul(*coefficient, degree_inverse);
        }

        let psi_inverse = modulus.inverse_prime(psi);
        let mut untwist = 1_u64;

        for coefficient in &mut coefficients {
            *coefficient = modulus.mul(*coefficient, untwist);

            untwist = modulus.mul(untwist, psi_inverse);
        }

        Polynomial::new(modulus, coefficients)
    }
}

/// In-place iterative radix-2 cyclic NTT.
///
/// `root` must have exact order equal to `values.len()`.
fn cyclic_ntt_radix2(values: &mut [u64], modulus: Modulus, root: u64) {
    let n = values.len();

    assert!(
        n > 0 && n.is_power_of_two(),
        "radix-2 NTT length must be a positive power of two"
    );

    bit_reverse_permute(values);

    let mut len = 2_usize;

    while len <= n {
        let root_step = modulus.pow(root, (n / len) as u64);

        for start in (0..n).step_by(len) {
            let mut twiddle = 1_u64;
            let half = len / 2;

            for offset in 0..half {
                let even = values[start + offset];

                let odd = modulus.mul(values[start + offset + half], twiddle);

                values[start + offset] = modulus.add(even, odd);

                values[start + offset + half] = modulus.sub(even, odd);

                twiddle = modulus.mul(twiddle, root_step);
            }
        }

        len *= 2;
    }
}

fn bit_reverse_permute(values: &mut [u64]) {
    let n = values.len();
    let mut j = 0_usize;

    for i in 1..n {
        let mut bit = n >> 1;

        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if i < j {
            values.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> NttPlan {
        NttPlan::new(Modulus::new(97), 8, 8)
    }

    #[test]
    fn radix2_forward_matches_reference() {
        let plan = plan();
        let q = Modulus::new(97);

        let polynomial = Polynomial::new(q, vec![1, 8, 23, 42, 7, 11, 19, 31]);

        assert_eq!(
            plan.forward_radix2(&polynomial),
            plan.forward_reference(&polynomial)
        );
    }

    #[test]
    fn radix2_roundtrip_recovers_polynomial() {
        let plan = plan();
        let q = Modulus::new(97);

        let polynomial = Polynomial::new(q, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let transformed = plan.forward_radix2(&polynomial);

        let recovered = plan.inverse_radix2(&transformed);

        assert_eq!(recovered, polynomial);
    }

    #[test]
    fn radix2_matches_reference_for_multiple_vectors() {
        let plan = plan();
        let q = Modulus::new(97);

        let vectors = [
            vec![0, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![96, 95, 94, 93, 92, 91, 90, 89],
            vec![11, 22, 33, 44, 55, 66, 77, 88],
        ];

        for vector in vectors {
            let polynomial = Polynomial::new(q, vector);

            assert_eq!(
                plan.forward_radix2(&polynomial),
                plan.forward_reference(&polynomial)
            );

            let transformed = plan.forward_radix2(&polynomial);

            assert_eq!(plan.inverse_radix2(&transformed), polynomial);
        }
    }

    #[test]
    fn radix2_product_matches_naive_negacyclic_product() {
        let plan = plan();
        let q = Modulus::new(97);

        let lhs = Polynomial::new(q, vec![1, 8, 23, 42, 7, 11, 19, 31]);

        let rhs = Polynomial::new(q, vec![5, 7, 11, 13, 17, 29, 37, 41]);

        let lhs_ntt = plan.forward_radix2(&lhs);

        let rhs_ntt = plan.forward_radix2(&rhs);

        let product_ntt = plan.pointwise_mul(&lhs_ntt, &rhs_ntt);

        let actual = plan.inverse_radix2(&product_ntt);

        let expected = lhs.negacyclic_mul(&rhs);

        assert_eq!(actual, expected);
    }

    #[test]
    fn radix2_and_reference_products_match() {
        let plan = plan();
        let q = Modulus::new(97);

        let lhs = Polynomial::new(q, vec![9, 17, 25, 33, 41, 49, 57, 65]);

        let rhs = Polynomial::new(q, vec![3, 6, 9, 12, 15, 18, 21, 24]);

        let radix2 = plan.inverse_radix2(
            &plan.pointwise_mul(&plan.forward_radix2(&lhs), &plan.forward_radix2(&rhs)),
        );

        let reference = plan.inverse_reference(
            &plan.pointwise_mul(&plan.forward_reference(&lhs), &plan.forward_reference(&rhs)),
        );

        assert_eq!(radix2, reference);
    }
}
