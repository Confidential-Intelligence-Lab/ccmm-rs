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
    slot_root_indices: Vec<usize>,
}

/// In-place radix-2 complex DFT.
///
/// `sign = -1.0` computes
///
/// ```text
/// X[k] = sum_j x[j] exp(-2*pi*i*j*k/N)
/// ```
///
/// while `sign = +1.0` computes the corresponding positive-sign transform.
///
/// No normalization is applied.
fn radix2_dft_in_place(values: &mut [Complex64], sign: f64) {
    let n = values.len();

    assert!(
        n > 0 && n.is_power_of_two(),
        "radix-2 DFT length must be a positive power of two"
    );
    assert!(
        sign == -1.0 || sign == 1.0,
        "radix-2 DFT sign must be -1 or +1"
    );

    /*
     * Bit-reversal permutation.
     */
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

    /*
     * Iterative radix-2 Cooley-Tukey butterflies.
     */
    let mut length = 2_usize;

    while length <= n {
        let angle = sign * 2.0 * PI / length as f64;
        let root = Complex64::from_polar(1.0, angle);
        let half = length / 2;

        for start in (0..n).step_by(length) {
            let mut twiddle = Complex64::new(1.0, 0.0);

            for offset in 0..half {
                let even = values[start + offset];
                let odd = values[start + offset + half] * twiddle;

                values[start + offset] = even + odd;
                values[start + offset + half] = even - odd;

                twiddle *= root;
            }
        }

        length *= 2;
    }
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

        /*
         * Logical CKKS slots use the Galois-compatible orbit
         *
         *     1, 5, 5^2, ..., 5^(N/2-1)  (mod 2N).
         *
         * Each exponent is an odd 2N-th root exponent. Since the full
         * canonical root array stores exponent 2j+1 at index j, the
         * corresponding root index is (exponent - 1) / 2.
         *
         * This ordering makes sigma_5 act cyclically on logical slots.
         */
        let two_n = 2 * degree;
        let mut exponent = 1_usize;
        let mut slot_root_indices = Vec::with_capacity(degree / 2);

        for _ in 0..degree / 2 {
            slot_root_indices.push((exponent - 1) / 2);
            exponent = (exponent * 5) % two_n;
        }

        Self {
            degree,
            roots,
            slot_root_indices,
        }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn slot_count(&self) -> usize {
        self.degree / 2
    }

    /// Full canonical-root indices used by the logical CKKS slots.
    ///
    /// Logical slot `j` corresponds to root exponent
    /// `5^j mod 2N`.
    pub fn slot_root_indices(&self) -> &[usize] {
        &self.slot_root_indices
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
    ///
    /// The production implementation uses the identity
    ///
    /// ```text
    /// xi_j = zeta * omega^j,
    /// zeta = exp(pi*i/N),
    /// omega = exp(2*pi*i/N),
    /// ```
    ///
    /// so canonical evaluation is one coefficient-wise phase twist followed
    /// by a radix-2 positive-sign DFT.
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

        let zeta = Complex64::from_polar(1.0, PI / self.degree as f64);
        let mut twist = Complex64::new(1.0, 0.0);

        let mut evaluations = Vec::with_capacity(self.degree);

        for &coefficient in coefficients {
            evaluations.push(Complex64::new(coefficient, 0.0) * twist);
            twist *= zeta;
        }

        radix2_dft_in_place(&mut evaluations, 1.0);

        evaluations
    }

    /// Direct O(N^2) canonical evaluation retained as a mathematical oracle
    /// for regression testing of the optimized embedding.
    #[cfg(test)]
    fn evaluate_all_reference(&self, coefficients: &[f64]) -> Vec<Complex64> {
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
        let evaluations = self.evaluate_all(coefficients);

        self.slot_root_indices
            .iter()
            .map(|&root_index| evaluations[root_index])
            .collect()
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

        for (&root_index, &slot) in self.slot_root_indices.iter().zip(slots) {
            let conjugate_index = self.degree - 1 - root_index;

            evaluations[root_index] = slot;
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
    /// p_k = zeta^(-k) * (1/N) * sum_j v_j * omega^(-j*k).
    /// ```
    ///
    /// The production path therefore uses one radix-2 negative-sign DFT,
    /// followed by normalization and a coefficient-wise inverse phase twist.
    ///
    /// Conjugate symmetry guarantees real coefficients in exact arithmetic.
    pub fn slots_to_coefficients(&self, slots: &[Complex64]) -> Vec<f64> {
        let mut evaluations = self.expand_slots(slots);

        radix2_dft_in_place(&mut evaluations, -1.0);

        let normalization = 1.0 / self.degree as f64;
        let inverse_zeta = Complex64::from_polar(1.0, -PI / self.degree as f64);
        let mut twist = Complex64::new(1.0, 0.0);

        evaluations
            .into_iter()
            .enumerate()
            .map(|(coefficient_index, transformed)| {
                let coefficient = transformed * normalization * twist;
                twist *= inverse_zeta;

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

    /// Direct O(N^2) canonical interpolation retained as a mathematical oracle
    /// for regression testing of the optimized embedding.
    #[cfg(test)]
    fn slots_to_coefficients_reference(&self, slots: &[Complex64]) -> Vec<f64> {
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

                assert!(
                    coefficient.im.abs() <= 1.0e-10 * (1.0 + coefficient.re.abs()),
                    "reference canonical inverse embedding produced non-real coefficient: \
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
    fn optimized_evaluation_matches_direct_reference() {
        for degree in [2_usize, 4, 8, 16, 32, 64, 128] {
            let embedding = CkksCanonicalEmbedding::new(degree);

            let coefficients: Vec<f64> = (0..degree)
                .map(|index| {
                    let centered = ((index * 17 + 5) % 31) as f64 - 15.0;
                    centered / 32.0
                })
                .collect();

            let optimized = embedding.evaluate_all(&coefficients);
            let reference = embedding.evaluate_all_reference(&coefficients);

            for (actual, expected) in optimized.into_iter().zip(reference) {
                assert_complex_close(actual, expected, 1.0e-10);
            }
        }
    }

    #[test]
    fn optimized_interpolation_matches_direct_reference() {
        for degree in [2_usize, 4, 8, 16, 32, 64, 128] {
            let embedding = CkksCanonicalEmbedding::new(degree);

            let slots: Vec<Complex64> = (0..embedding.slot_count())
                .map(|index| {
                    let re = ((index * 11 + 3) % 23) as f64 - 11.0;
                    let im = ((index * 7 + 1) % 19) as f64 - 9.0;

                    Complex64::new(re / 16.0, im / 16.0)
                })
                .collect();

            let optimized = embedding.slots_to_coefficients(&slots);
            let reference = embedding.slots_to_coefficients_reference(&slots);

            for (actual, expected) in optimized.into_iter().zip(reference) {
                assert_real_close(actual, expected, 1.0e-10);
            }
        }
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
    fn logical_slot_roots_follow_power_of_five_orbit() {
        let embedding = CkksCanonicalEmbedding::new(8);

        assert_eq!(embedding.slot_root_indices(), &[0, 2, 4, 6]);
    }

    #[test]
    fn logical_slot_root_orbit_has_full_capacity() {
        for degree in [2_usize, 4, 8, 16, 32] {
            let embedding = CkksCanonicalEmbedding::new(degree);

            let mut indices = embedding.slot_root_indices().to_vec();

            indices.sort_unstable();
            indices.dedup();

            assert_eq!(indices.len(), degree / 2, "degree={degree}");

            for &index in embedding.slot_root_indices() {
                let conjugate_index = degree - 1 - index;

                assert!(
                    !embedding.slot_root_indices().contains(&conjugate_index,),
                    "slot orbit contains both members of a conjugate pair: \
                     degree={degree}, index={index}"
                );
            }
        }
    }

    #[test]
    fn slot_expansion_has_galois_compatible_conjugate_layout() {
        let embedding = CkksCanonicalEmbedding::new(8);

        let slots = [
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 0.25),
            Complex64::new(0.125, -0.75),
            Complex64::new(3.0, 1.5),
        ];

        let expanded = embedding.expand_slots(&slots);

        for (&root_index, &slot) in embedding.slot_root_indices().iter().zip(&slots) {
            let conjugate_index = embedding.degree() - 1 - root_index;

            assert_eq!(expanded[root_index], slot);

            assert_eq!(expanded[conjugate_index], slot.conj());
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
