use num_complex::Complex64;

use crate::ring::{Modulus, Polynomial};

use super::CkksCanonicalEmbedding;

/// Canonical CKKS slot encoder.
///
/// This layer performs:
///
/// ```text
/// slots
///   -> inverse canonical embedding
///   -> real polynomial coefficients
///   -> scale by Delta
///   -> nearest-integer quantization
///   -> reduction modulo Q
/// ```
///
/// It does not perform encryption.
#[derive(Debug, Clone)]
pub struct CkksSlotEncoder {
    embedding: CkksCanonicalEmbedding,
    modulus: Modulus,
    scale: f64,
}

impl CkksSlotEncoder {
    pub fn new(degree: usize, modulus: Modulus, scale: f64) -> Self {
        assert!(
            scale.is_finite() && scale > 0.0,
            "CKKS slot scale must be positive and finite"
        );

        Self {
            embedding: CkksCanonicalEmbedding::new(degree),
            modulus,
            scale,
        }
    }

    pub fn degree(&self) -> usize {
        self.embedding.degree()
    }

    pub fn slot_count(&self) -> usize {
        self.embedding.slot_count()
    }

    pub fn modulus(&self) -> Modulus {
        self.modulus
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn embedding(&self) -> &CkksCanonicalEmbedding {
        &self.embedding
    }

    /// Encodes `N/2` complex CKKS slots into one polynomial in `Z_Q[X]`.
    pub fn encode_slots(&self, slots: &[Complex64]) -> Polynomial {
        assert_eq!(
            slots.len(),
            self.slot_count(),
            "CKKS slot count must equal N/2"
        );

        let coefficients = self.embedding.slots_to_coefficients(slots);

        let encoded = coefficients
            .iter()
            .map(|&coefficient| {
                assert!(
                    coefficient.is_finite(),
                    "CKKS encoded coefficient must be finite"
                );

                let scaled = (coefficient * self.scale).round();

                assert!(
                    scaled >= i128::MIN as f64 && scaled <= i128::MAX as f64,
                    "scaled CKKS slot coefficient exceeds i128 range"
                );

                signed_to_mod(scaled as i128, self.modulus.value())
            })
            .collect();

        Polynomial::new(self.modulus, encoded)
    }

    /// Convenience path for real-valued slots.
    pub fn encode_real_slots(&self, slots: &[f64]) -> Polynomial {
        assert_eq!(
            slots.len(),
            self.slot_count(),
            "CKKS slot count must equal N/2"
        );

        let complex_slots: Vec<_> = slots
            .iter()
            .map(|&value| {
                assert!(value.is_finite(), "CKKS slots must be finite");

                Complex64::new(value, 0.0)
            })
            .collect();

        self.encode_slots(&complex_slots)
    }

    /// Decodes a polynomial in `Z_Q[X]` into canonical CKKS complex slots.
    ///
    /// This performs:
    ///
    /// ```text
    /// modular coefficients
    ///   -> centered representatives
    ///   -> divide by Delta
    ///   -> canonical embedding
    ///   -> N/2 complex slots
    /// ```
    pub fn decode_slots(&self, polynomial: &Polynomial) -> Vec<Complex64> {
        assert_eq!(
            polynomial.degree(),
            self.degree(),
            "CKKS decoded polynomial degree must match encoder degree"
        );

        assert_eq!(
            polynomial.modulus(),
            self.modulus,
            "CKKS decoded polynomial modulus must match encoder modulus"
        );

        let coefficients: Vec<f64> = polynomial
            .coefficients()
            .iter()
            .map(|&coefficient| centered(coefficient, self.modulus.value()) as f64 / self.scale)
            .collect();

        self.embedding.coefficients_to_slots(&coefficients)
    }

    /// Decodes canonical CKKS slots and requires that their imaginary
    /// components are numerically negligible.
    pub fn decode_real_slots(&self, polynomial: &Polynomial) -> Vec<f64> {
        self.decode_slots(polynomial)
            .into_iter()
            .map(|slot| {
                assert!(
                    slot.im.abs() <= 1.0e-10 * (1.0 + slot.re.abs()),
                    "decoded CKKS slot is not real: {slot:?}"
                );

                slot.re
            })
            .collect()
    }

    /// Returns the unquantized inverse-embedding coefficients.
    ///
    /// This is useful for validation and characterization of quantization.
    pub fn reference_coefficients(&self, slots: &[Complex64]) -> Vec<f64> {
        self.embedding.slots_to_coefficients(slots)
    }
}

fn centered(value: u64, modulus: u64) -> i128 {
    let value = i128::from(value);

    let modulus = i128::from(modulus);

    if value > modulus / 2 {
        value - modulus
    } else {
        value
    }
}

fn signed_to_mod(value: i128, modulus: u64) -> u64 {
    let modulus = i128::from(modulus);

    let reduced = (value % modulus + modulus) % modulus;

    reduced as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder() -> CkksSlotEncoder {
        CkksSlotEncoder::new(8, Modulus::new(2_147_483_647), 65_536.0)
    }

    fn centered(value: u64, modulus: u64) -> i128 {
        let value = i128::from(value);

        let modulus = i128::from(modulus);

        if value > modulus / 2 {
            value - modulus
        } else {
            value
        }
    }

    #[test]
    fn encoder_preserves_configuration() {
        let encoder = encoder();

        assert_eq!(encoder.degree(), 8);

        assert_eq!(encoder.slot_count(), 4);

        assert_eq!(encoder.modulus(), Modulus::new(2_147_483_647,));

        assert_eq!(encoder.scale(), 65_536.0);
    }

    #[test]
    fn real_slots_encode_to_ring_polynomial() {
        let encoder = encoder();

        let polynomial = encoder.encode_real_slots(&[1.0, -0.5, 0.25, 2.0]);

        assert_eq!(polynomial.degree(), 8);

        assert_eq!(polynomial.modulus(), encoder.modulus());
    }

    #[test]
    fn complex_slots_encode_to_ring_polynomial() {
        let encoder = encoder();

        let polynomial = encoder.encode_slots(&[
            Complex64::new(1.0, 0.5),
            Complex64::new(-0.75, 0.25),
            Complex64::new(0.125, -0.5),
            Complex64::new(2.0, 1.0),
        ]);

        assert_eq!(polynomial.degree(), 8);

        assert_eq!(polynomial.modulus(), encoder.modulus());
    }

    #[test]
    fn quantization_error_is_at_most_half_scale_unit() {
        let encoder = encoder();

        let slots = [
            Complex64::new(1.25, 0.5),
            Complex64::new(-0.75, 1.125),
            Complex64::new(0.25, -0.375),
            Complex64::new(2.0, -1.0),
        ];

        let reference = encoder.reference_coefficients(&slots);

        let encoded = encoder.encode_slots(&slots);

        for (index, (&raw, &coefficient)) in
            reference.iter().zip(encoded.coefficients()).enumerate()
        {
            let quantized =
                centered(coefficient, encoder.modulus().value()) as f64 / encoder.scale();

            let error = (quantized - raw).abs();

            assert!(
                error <= (0.5 / encoder.scale()) + 1.0e-15,
                "coefficient {index}: \
                 raw={raw}, \
                 quantized={quantized}, \
                 error={error}"
            );
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let encoder = encoder();

        let slots = [
            Complex64::new(0.25, 0.5),
            Complex64::new(-1.0, 0.125),
            Complex64::new(0.0, -0.75),
            Complex64::new(2.0, 0.0),
        ];

        assert_eq!(encoder.encode_slots(&slots,), encoder.encode_slots(&slots,));
    }

    #[test]
    fn negative_embedding_coefficients_map_to_upper_half_modulus() {
        let encoder = encoder();

        let slots = [
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let reference = encoder.reference_coefficients(&slots);

        let encoded = encoder.encode_slots(&slots);

        let mut found_negative = false;

        for (raw, encoded_value) in reference.iter().zip(encoded.coefficients()) {
            if *raw < 0.0 {
                found_negative = true;

                assert!(*encoded_value > encoder.modulus().value() / 2);
            }
        }

        assert!(
            found_negative,
            "test vector produced no negative embedding coefficients"
        );
    }

    #[test]
    #[should_panic(expected = "CKKS slot count must equal N/2")]
    fn rejects_wrong_number_of_slots() {
        let encoder = encoder();

        let _ = encoder.encode_slots(&[Complex64::new(1.0, 0.0)]);
    }

    #[test]
    #[should_panic(expected = "CKKS slot scale must be positive and finite")]
    fn rejects_zero_scale() {
        let _ = CkksSlotEncoder::new(8, Modulus::new(65_537), 0.0);
    }

    #[test]
    fn complex_slots_roundtrip_through_quantized_encoding() {
        let encoder = encoder();

        let slots = [
            Complex64::new(1.25, 0.5),
            Complex64::new(-0.75, 1.125),
            Complex64::new(0.25, -0.375),
            Complex64::new(2.0, -1.0),
        ];

        let encoded = encoder.encode_slots(&slots);

        let decoded = encoder.decode_slots(&encoded);

        for (index, (&actual, &expected)) in decoded.iter().zip(&slots).enumerate() {
            let error = (actual - expected).norm();

            /*
             * Each polynomial coefficient is quantized with error
             * at most 1/(2 Delta). The canonical embedding sums N
             * such coefficient errors with unit-magnitude roots,
             * giving the conservative bound N/(2 Delta).
             */
            let tolerance = encoder.degree() as f64 / (2.0 * encoder.scale()) + 1.0e-12;

            assert!(
                error <= tolerance,
                "slot {index}: \
                 actual={actual:?}, \
                 expected={expected:?}, \
                 error={error}, \
                 tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn real_slots_roundtrip_through_quantized_encoding() {
        let encoder = encoder();

        let slots = [1.0, -0.5, 0.25, 2.0];

        let encoded = encoder.encode_real_slots(&slots);

        let decoded = encoder.decode_real_slots(&encoded);

        let tolerance = encoder.degree() as f64 / (2.0 * encoder.scale()) + 1.0e-12;

        for (index, (&actual, &expected)) in decoded.iter().zip(&slots).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "slot {index}: \
                 actual={actual}, \
                 expected={expected}, \
                 error={}, \
                 tolerance={tolerance}",
                (actual - expected).abs()
            );
        }
    }

    #[test]
    fn zero_slots_roundtrip_exactly() {
        let encoder = encoder();

        let slots = [Complex64::new(0.0, 0.0); 4];

        let encoded = encoder.encode_slots(&slots);

        let decoded = encoder.decode_slots(&encoded);

        for value in decoded {
            assert!(value.norm() <= 1.0e-15);
        }
    }

    #[test]
    fn basis_slot_campaign_roundtrips_with_quantization_bound() {
        let encoder = encoder();

        let tolerance = encoder.degree() as f64 / (2.0 * encoder.scale()) + 1.0e-12;

        for slot_index in 0..encoder.slot_count() {
            for value in [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)] {
                let mut slots = vec![Complex64::new(0.0, 0.0,); encoder.slot_count()];

                slots[slot_index] = value;

                let encoded = encoder.encode_slots(&slots);

                let decoded = encoder.decode_slots(&encoded);

                for (index, (&actual, &expected)) in decoded.iter().zip(&slots).enumerate() {
                    assert!(
                        (actual - expected).norm() <= tolerance,
                        "basis slot={slot_index}, \
                         component={index}, \
                         actual={actual:?}, \
                         expected={expected:?}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "CKKS decoded polynomial modulus must match encoder modulus")]
    fn decoder_rejects_wrong_modulus() {
        let encoder = encoder();

        let polynomial = Polynomial::new(Modulus::new(65_537), vec![0; 8]);

        let _ = encoder.decode_slots(&polynomial);
    }

    #[test]
    #[should_panic(expected = "CKKS decoded polynomial degree must match encoder degree")]
    fn decoder_rejects_wrong_degree() {
        let encoder = encoder();

        let polynomial = Polynomial::new(encoder.modulus(), vec![0; 4]);

        let _ = encoder.decode_slots(&polynomial);
    }

    #[test]
    fn randomized_slot_roundtrip_campaign() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        for degree in [8_usize, 16, 32] {
            for scale in [4_096.0, 65_536.0, 1_048_576.0] {
                let encoder = CkksSlotEncoder::new(degree, Modulus::new(2_147_483_647), scale);

                let tolerance = degree as f64 / (2.0 * scale) + 1.0e-10;

                for seed in 0_u64..32 {
                    let mut rng = ChaCha20Rng::seed_from_u64(
                        seed ^ ((degree as u64) << 32) ^ scale.to_bits(),
                    );

                    let slots: Vec<_> = (0..encoder.slot_count())
                        .map(|_| Complex64::new(rng.gen_range(-2.0..2.0), rng.gen_range(-2.0..2.0)))
                        .collect();

                    let encoded = encoder.encode_slots(&slots);

                    let decoded = encoder.decode_slots(&encoded);

                    for (index, (&actual, &expected)) in decoded.iter().zip(&slots).enumerate() {
                        let error = (actual - expected).norm();

                        assert!(
                            error <= tolerance,
                            "degree={degree}, \
                             scale={scale}, \
                             seed={seed}, \
                             slot={index}, \
                             actual={actual:?}, \
                             expected={expected:?}, \
                             error={error}, \
                             tolerance={tolerance}"
                        );
                    }
                }
            }
        }
    }
}
