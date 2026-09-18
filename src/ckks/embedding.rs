use std::f64::consts::PI;

use num_complex::Complex64;

/// Reference implementation of the canonical CKKS embedding.
///
/// The ring is
///
/// ```text
/// R = R[X] / (X^N + 1)
/// ```
///
/// for power-of-two `N`.
///
/// Let
///
/// ```text
/// zeta = exp(pi * i / N).
/// ```
///
/// The `N` complex embeddings evaluate a polynomial at
///
/// ```text
/// xi_j = zeta^(2j + 1),  j = 0, ..., N-1.
/// ```
///
/// Since
///
/// ```text
/// conjugate(xi_j) = xi_(N - 1 - j),
/// ```
///
/// a real polynomial has conjugate-paired evaluations. Therefore only
/// `N/2` complex values are independent and may be treated as CKKS slots.
///
/// This implementation intentionally uses direct O(N^2) evaluation and
/// interpolation. It is the correctness/reference path for R2.6 and can
/// later serve as the oracle for optimized FFT-based transforms.
#[derive(Debug, Clone)]
pub struct CkksCanonicalEmbedding {
    degree: usize,
    roots: Vec<Complex64>,
}

impl CkksCanonicalEmbedding {
    /// Constructs the canonical embedding for `R[X]/(X^N + 1)`.
    pub fn new(degree: usize) -> Self {
        assert!(
            degree >= 2 && degree.is_power_of_two(),
            "CKKS embedding degree must be a power of two at least 2"
        );

        let roots = (0..degree)
            .map(|index| {
                let angle = PI * (2 * index + 1) as f64 / degree as f64;

                Complex64::from_polar(1.0, angle)
            })
            .collect();

        Self { degree, roots }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn slot_count(&self) -> usize {
        self.degree / 2
    }

    /// Ordered odd 2N-th roots used by the complete canonical embedding.
    pub fn roots(&self) -> &[Complex64] {
        &self.roots
    }

    /// Evaluates a real polynomial at all canonical embedding points.
    ///
    /// Returns `N` complex evaluations. For real coefficients they obey
    ///
    /// ```text
    /// values[N - 1 - j] = conjugate(values[j]).
    /// ```
    pub fn evaluate_all(&self, coefficients: &[f64]) -> Vec<Complex64> {
        assert_eq!(
            coefficients.len(),
            self.degree,
            "CKKS coefficient count must equal ring degree"
        );

        coefficients.iter().for_each(|value| {
            assert!(
                value.is_finite(),
                "CKKS embedding coefficients must be finite"
            );
        });

        self.roots
            .iter()
            .copied()
            .map(|root| {
                /*
                 * Horner evaluation is both simpler and numerically
                 * preferable to repeatedly computing root powers.
                 */
                coefficients
                    .iter()
                    .rev()
                    .fold(Complex64::new(0.0, 0.0), |accumulator, &coefficient| {
                        accumulator * root + coefficient
                    })
            })
            .collect()
    }

    /// Maps real polynomial coefficients to the `N/2` independent CKKS
    /// slots according to this module's deterministic slot ordering.
    pub fn coefficients_to_slots(&self, coefficients: &[f64]) -> Vec<Complex64> {
        self.evaluate_all(coefficients)[..self.slot_count()].to_vec()
    }

    /// Expands `N/2` slots into the full conjugate-symmetric canonical
    /// embedding vector.
    pub fn expand_slots(&self, slots: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(
            slots.len(),
            self.slot_count(),
            "CKKS slot count must equal N/2"
        );

        slots.iter().for_each(|value| {
            assert!(
                value.re.is_finite() && value.im.is_finite(),
                "CKKS slots must be finite"
            );
        });

        let mut evaluations = vec![Complex64::new(0.0, 0.0); self.degree];

        for (index, &slot) in slots.iter().enumerate() {
            let conjugate_index = self.degree - 1 - index;

            evaluations[index] = slot;

            evaluations[conjugate_index] = slot.conj();
        }

        evaluations
    }

    /// Interpolates the real polynomial whose canonical evaluations encode
    /// the supplied `N/2` CKKS slots.
    ///
    /// If
    ///
    /// ```text
    /// v_j = p(xi_j),
    /// ```
    ///
    /// then
    ///
    /// ```text
    /// p_k = (1/N) * sum_j v_j * xi_j^(-k).
    /// ```
    ///
    /// Conjugate symmetry guarantees real coefficients in exact arithmetic.
    pub fn slots_to_coefficients(&self, slots: &[Complex64]) -> Vec<f64> {
        let evaluations = self.expand_slots(slots);

        let normalization = 1.0 / self.degree as f64;

        (0..self.degree)
            .map(|coefficient_index| {
                let coefficient = evaluations
                    .iter()
                    .zip(&self.roots)
                    .map(|(&evaluation, &root)| {
                        evaluation * root.conj().powu(coefficient_index as u32)
                    })
                    .sum::<Complex64>()
                    * normalization;

                /*
                 * The imaginary part is numerical residue only: the
                 * conjugate-symmetric embedding corresponds to a real
                 * polynomial.
                 */
                assert!(
                    coefficient.im.abs() <= 1.0e-10 * (1.0 + coefficient.re.abs()),
                    "canonical inverse embedding produced non-real coefficient: \
                     index={coefficient_index}, coefficient={coefficient}"
                );

                coefficient.re
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
        assert!(
            (actual - expected).norm() <= tolerance,
            "actual={actual:?}, expected={expected:?}, error={}",
            (actual - expected).norm()
        );
    }

    fn assert_real_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, error={}",
            (actual - expected).abs()
        );
    }

    #[test]
    fn embedding_has_expected_dimensions() {
        let embedding = CkksCanonicalEmbedding::new(8);

        assert_eq!(embedding.degree(), 8);

        assert_eq!(embedding.slot_count(), 4);

        assert_eq!(embedding.roots().len(), 8);
    }

    #[test]
    fn roots_are_roots_of_xn_plus_one() {
        for degree in [2_usize, 4, 8, 16, 32] {
            let embedding = CkksCanonicalEmbedding::new(degree);

            for &root in embedding.roots() {
                assert_complex_close(root.powu(degree as u32), Complex64::new(-1.0, 0.0), 1.0e-12);

                assert_complex_close(
                    root.powu((2 * degree) as u32),
                    Complex64::new(1.0, 0.0),
                    1.0e-12,
                );
            }
        }
    }

    #[test]
    fn root_order_has_expected_conjugate_pairs() {
        let embedding = CkksCanonicalEmbedding::new(16);

        for index in 0..embedding.degree() {
            let conjugate_index = embedding.degree() - 1 - index;

            assert_complex_close(
                embedding.roots()[conjugate_index],
                embedding.roots()[index].conj(),
                1.0e-12,
            );
        }
    }

    #[test]
    fn real_polynomial_evaluations_have_conjugate_symmetry() {
        let embedding = CkksCanonicalEmbedding::new(8);

        let coefficients = [1.0, -0.5, 0.25, 2.0, -1.0, 0.125, 0.75, -0.25];

        let values = embedding.evaluate_all(&coefficients);

        for index in 0..8 {
            let conjugate_index = 7 - index;

            assert_complex_close(values[conjugate_index], values[index].conj(), 1.0e-11);
        }
    }

    #[test]
    fn slot_expansion_has_exact_conjugate_layout() {
        let embedding = CkksCanonicalEmbedding::new(8);

        let slots = [
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 0.25),
            Complex64::new(0.125, -0.75),
            Complex64::new(3.0, 1.5),
        ];

        let expanded = embedding.expand_slots(&slots);

        for (index, &slot) in slots.iter().enumerate() {
            assert_eq!(expanded[index], slot);

            assert_eq!(expanded[7 - index], slot.conj());
        }
    }

    #[test]
    fn complex_slots_roundtrip_through_canonical_embedding() {
        let embedding = CkksCanonicalEmbedding::new(8);

        let slots = [
            Complex64::new(1.25, 0.5),
            Complex64::new(-0.75, 1.125),
            Complex64::new(0.25, -0.375),
            Complex64::new(2.0, -1.0),
        ];

        let coefficients = embedding.slots_to_coefficients(&slots);

        let recovered = embedding.coefficients_to_slots(&coefficients);

        for (actual, expected) in recovered.iter().copied().zip(slots) {
            assert_complex_close(actual, expected, 1.0e-11);
        }
    }

    #[test]
    fn real_slots_roundtrip_through_canonical_embedding() {
        let embedding = CkksCanonicalEmbedding::new(16);

        let slots: Vec<_> = [1.0, -0.5, 0.25, 2.0, -1.25, 0.125, 0.0, 3.5]
            .into_iter()
            .map(|value| Complex64::new(value, 0.0))
            .collect();

        let coefficients = embedding.slots_to_coefficients(&slots);

        let recovered = embedding.coefficients_to_slots(&coefficients);

        for (actual, expected) in recovered.iter().zip(&slots) {
            assert_complex_close(*actual, *expected, 1.0e-11);
        }
    }

    #[test]
    fn coefficients_roundtrip_through_all_embeddings() {
        let embedding = CkksCanonicalEmbedding::new(8);

        let coefficients = [0.5, -1.25, 0.375, 2.0, -0.75, 0.125, 1.5, -0.25];

        let slots = embedding.coefficients_to_slots(&coefficients);

        let recovered = embedding.slots_to_coefficients(&slots);

        for (actual, expected) in recovered.iter().zip(coefficients) {
            assert_real_close(*actual, expected, 1.0e-11);
        }
    }

    #[test]
    fn basis_vectors_roundtrip() {
        let embedding = CkksCanonicalEmbedding::new(16);

        for index in 0..16 {
            let mut coefficients = vec![0.0; 16];

            coefficients[index] = 1.0;

            let slots = embedding.coefficients_to_slots(&coefficients);

            let recovered = embedding.slots_to_coefficients(&slots);

            for (coefficient_index, value) in recovered.iter().enumerate() {
                let expected = if coefficient_index == index { 1.0 } else { 0.0 };

                assert_real_close(*value, expected, 1.0e-11);
            }
        }
    }

    #[test]
    #[should_panic(expected = "CKKS embedding degree must be a power of two at least 2")]
    fn rejects_non_power_of_two_degree() {
        let _ = CkksCanonicalEmbedding::new(12);
    }

    #[test]
    #[should_panic(expected = "CKKS slot count must equal N/2")]
    fn rejects_wrong_slot_count() {
        let embedding = CkksCanonicalEmbedding::new(8);

        let _ = embedding.slots_to_coefficients(&[Complex64::new(1.0, 0.0)]);
    }
}
