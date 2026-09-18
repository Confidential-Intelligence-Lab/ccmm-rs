mod automorphism;
mod chain;
mod embedding;
mod evaluation_keys;
mod evaluator;
mod galois_key;
mod hybrid_ciphertext;
mod hybrid_multiply;
mod key_accounting;
mod parameters;
mod rns_automorphism;
mod rns_ciphertext;
mod rns_galois_key;
mod rns_multiply;
mod rns_rescale;
mod slot_encoder;
mod slot_layout;

pub use automorphism::{
    apply_automorphism, canonical_slot_automorphism, conjugation_exponent,
    inverse_automorphism_exponent, rotation_exponent_left, rotation_exponent_right,
    CkksSlotAutomorphism,
};
pub use chain::CkksChainState;
pub use embedding::CkksCanonicalEmbedding;
pub use evaluation_keys::{
    conjugate_with_evaluation_keys, multiply_with_evaluation_keys,
    rotate_left_with_evaluation_keys, rotate_right_with_evaluation_keys, RnsCkksEvaluationKeys,
    RnsCkksHybridLevelKeys, RnsCkksLevelKeys, RnsCkksMultiplicationBackend,
    RnsCkksMultiplicationPolicy,
};
pub use evaluator::RnsCkksEvaluator;
pub use galois_key::{
    apply_galois_automorphism, conjugate_slots, rotate_left, rotate_right, GaloisKey,
};
pub use hybrid_ciphertext::HybridCkksCiphertext;
pub use hybrid_multiply::{multiply_transition_relinearize_hybrid_ckks, HybridCkksMultiplyConfig};
pub use key_accounting::{
    evaluation_key_storage, prepared_hybrid_cache_bytes, LogicalPayloadBytes,
    RnsCkksEvaluationKeyStorage, RnsCkksLevelKeyStorage,
};
pub use parameters::{
    correctness_profile_8, research_profile_16384, research_profile_4096, research_profile_8192,
    CkksParameterClass, CkksParameterProfile,
};
pub use rns_automorphism::apply_rns_automorphism;
pub use rns_ciphertext::RnsCkksCiphertext;
pub use rns_galois_key::{
    apply_rns_ckks_galois_automorphism, apply_rns_galois_automorphism, conjugate_rns_ckks,
    rotate_left_rns_ckks, rotate_right_rns_ckks, RnsGaloisKey,
};
pub use rns_multiply::{
    evaluate_rns_ckks_product_chain, multiply_relinearize_rescale_rns_ckks,
    multiply_relinearize_rescale_rns_ckks_with_ntt,
};
pub use rns_rescale::rescale_rns_ckks_to_next;
pub use slot_encoder::CkksSlotEncoder;
pub use slot_layout::CkksSlotLayout;

use rand::{CryptoRng, RngCore};

use crate::eval::{relinearize, MultiplicationKey};
use crate::ring::{Modulus, Polynomial};
use crate::rlwe::{
    decrypt_raw, encrypt_raw_with_rng, tensor, RlweCiphertext, RlweParameters, SecretKey,
};

/// Two-level correctness-oriented CKKS parameter set.
///
/// The active modulus chain is:
///
/// `Q = p * q  ->  q`.
///
/// Multiplication raises the scale from `Delta` to `Delta^2`.
/// Rescaling divides ciphertext coefficients by `p`, drops modulus
/// `Q -> q`, and updates the scale to `Delta^2 / p`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CkksParameters {
    degree: usize,
    high_modulus: Modulus,
    low_modulus: Modulus,
    rescale_prime: u64,
    initial_scale: f64,
    noise_bound: i64,
}

impl CkksParameters {
    pub fn new(
        degree: usize,
        low_modulus: u64,
        rescale_prime: u64,
        initial_scale: f64,
        noise_bound: i64,
    ) -> Self {
        assert!(
            degree > 0 && degree.is_power_of_two(),
            "CKKS degree must be a positive power of two"
        );
        assert!(low_modulus >= 3, "low CKKS modulus must be at least 3");
        assert!(rescale_prime >= 2, "rescale prime must be at least 2");
        assert!(
            initial_scale.is_finite() && initial_scale > 0.0,
            "CKKS scale must be positive and finite"
        );
        assert!(noise_bound >= 0, "noise bound must be nonnegative");

        let high_modulus = low_modulus
            .checked_mul(rescale_prime)
            .expect("CKKS modulus product must fit in u64");

        Self {
            degree,
            high_modulus: Modulus::new(high_modulus),
            low_modulus: Modulus::new(low_modulus),
            rescale_prime,
            initial_scale,
            noise_bound,
        }
    }

    pub fn degree(self) -> usize {
        self.degree
    }

    pub fn high_modulus(self) -> Modulus {
        self.high_modulus
    }

    pub fn low_modulus(self) -> Modulus {
        self.low_modulus
    }

    pub fn rescale_prime(self) -> u64 {
        self.rescale_prime
    }

    pub fn initial_scale(self) -> f64 {
        self.initial_scale
    }

    pub fn high_rlwe(self) -> RlweParameters {
        // The plaintext modulus is unused by raw CKKS encryption.
        RlweParameters::new(self.degree, self.high_modulus, 2, self.noise_bound)
    }

    pub fn low_rlwe(self) -> RlweParameters {
        RlweParameters::new(self.degree, self.low_modulus, 2, self.noise_bound)
    }
}

/// CKKS ciphertext with cryptographically meaningful level and scale state.
#[derive(Debug, Clone, PartialEq)]
pub struct CkksCiphertext {
    inner: RlweCiphertext,
    level: usize,
    scale: f64,
}

impl CkksCiphertext {
    pub(crate) fn new(inner: RlweCiphertext, level: usize, scale: f64) -> Self {
        assert!(scale.is_finite() && scale > 0.0);
        Self {
            inner,
            level,
            scale,
        }
    }

    pub fn rlwe(&self) -> &RlweCiphertext {
        &self.inner
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn modulus(&self) -> Modulus {
        self.inner.b().modulus()
    }
}

/// Encodes real coefficient values at a fixed CKKS scale.
pub fn encode(params: CkksParameters, values: &[f64]) -> Polynomial {
    assert_eq!(
        values.len(),
        params.degree(),
        "CKKS plaintext length must match ring degree"
    );

    let coefficients = values
        .iter()
        .map(|&value| {
            assert!(value.is_finite(), "CKKS inputs must be finite");

            let scaled = (value * params.initial_scale()).round();

            assert!(
                scaled >= i64::MIN as f64 && scaled <= i64::MAX as f64,
                "scaled CKKS coefficient exceeds i64 range"
            );

            signed_to_mod(scaled as i64, params.high_modulus().value())
        })
        .collect();

    Polynomial::new(params.high_modulus(), coefficients)
}

/// Encrypts CKKS coefficient data at the highest level.
pub fn encrypt_with_rng<R>(
    params: CkksParameters,
    secret_key: &SecretKey,
    values: &[f64],
    rng: &mut R,
) -> CkksCiphertext
where
    R: RngCore + CryptoRng,
{
    let encoded = encode(params, values);

    let inner = encrypt_raw_with_rng(params.high_rlwe(), secret_key, &encoded, rng);

    CkksCiphertext::new(inner, 1, params.initial_scale())
}

/// Multiplies and relinearizes two top-level CKKS ciphertexts.
///
/// The resulting scale is `lhs.scale * rhs.scale`.
pub fn multiply_relinearize(
    params: CkksParameters,
    lhs: &CkksCiphertext,
    rhs: &CkksCiphertext,
    multiplication_key: &MultiplicationKey,
) -> CkksCiphertext {
    assert_eq!(lhs.level(), 1, "left ciphertext must be at level 1");
    assert_eq!(rhs.level(), 1, "right ciphertext must be at level 1");
    assert_eq!(
        lhs.modulus(),
        params.high_modulus(),
        "left ciphertext must use the high modulus"
    );
    assert_eq!(
        rhs.modulus(),
        params.high_modulus(),
        "right ciphertext must use the high modulus"
    );

    let quadratic = tensor(lhs.rlwe(), rhs.rlwe());

    let inner = relinearize(params.high_rlwe(), &quadratic, multiplication_key);

    CkksCiphertext::new(inner, 1, lhs.scale() * rhs.scale())
}

/// Drops the ciphertext from `Q = p*q` to `q` and divides its scale by `p`.
///
/// Each polynomial coefficient is interpreted as a centered representative,
/// divided by `p`, rounded to the nearest integer, and mapped into `Z_q`.
pub fn rescale_to_next(params: CkksParameters, ciphertext: &CkksCiphertext) -> CkksCiphertext {
    assert_eq!(
        ciphertext.level(),
        1,
        "rescale requires a level-1 ciphertext"
    );
    assert_eq!(
        ciphertext.modulus(),
        params.high_modulus(),
        "rescale requires the high modulus"
    );

    let b = rescale_polynomial(params, ciphertext.rlwe().b());
    let a = rescale_polynomial(params, ciphertext.rlwe().a());

    CkksCiphertext::new(
        RlweCiphertext::new(b, a),
        0,
        ciphertext.scale() / params.rescale_prime() as f64,
    )
}

/// Projects the small secret key from `Z_Q` into `Z_q`.
pub fn project_secret_key_to_low(params: CkksParameters, secret_key: &SecretKey) -> SecretKey {
    assert_eq!(
        secret_key.polynomial().modulus(),
        params.high_modulus(),
        "secret key must use the high modulus"
    );

    let coefficients = secret_key
        .polynomial()
        .coefficients()
        .iter()
        .map(|&value| {
            let centered = centered_i128(value, params.high_modulus().value());

            signed_i128_to_mod(centered, params.low_modulus().value())
        })
        .collect();

    SecretKey::from_polynomial(Polynomial::new(params.low_modulus(), coefficients))
}

/// Decrypts and decodes a CKKS ciphertext according to its active scale.
pub fn decrypt_decode(
    params: CkksParameters,
    secret_key: &SecretKey,
    ciphertext: &CkksCiphertext,
) -> Vec<f64> {
    let rlwe_params = match ciphertext.level() {
        1 => params.high_rlwe(),
        0 => params.low_rlwe(),
        _ => panic!("unsupported CKKS level"),
    };

    let polynomial = decrypt_raw(rlwe_params, secret_key, ciphertext.rlwe());

    decode_polynomial(&polynomial, ciphertext.scale())
}

fn decode_polynomial(polynomial: &Polynomial, scale: f64) -> Vec<f64> {
    polynomial
        .coefficients()
        .iter()
        .map(|&coefficient| centered_i128(coefficient, polynomial.modulus().value()) as f64 / scale)
        .collect()
}

fn rescale_polynomial(params: CkksParameters, polynomial: &Polynomial) -> Polynomial {
    assert_eq!(
        polynomial.modulus(),
        params.high_modulus(),
        "rescaled polynomial must use the high modulus"
    );

    let divisor = i128::from(params.rescale_prime());

    let coefficients = polynomial
        .coefficients()
        .iter()
        .map(|&coefficient| {
            let centered = centered_i128(coefficient, params.high_modulus().value());

            let rounded = div_round_nearest(centered, divisor);

            signed_i128_to_mod(rounded, params.low_modulus().value())
        })
        .collect();

    Polynomial::new(params.low_modulus(), coefficients)
}

fn centered_i128(value: u64, modulus: u64) -> i128 {
    let value = i128::from(value);
    let modulus = i128::from(modulus);
    let half = modulus / 2;

    if value > half {
        value - modulus
    } else {
        value
    }
}

fn signed_to_mod(value: i64, modulus: u64) -> u64 {
    signed_i128_to_mod(i128::from(value), modulus)
}

fn signed_i128_to_mod(value: i128, modulus: u64) -> u64 {
    let modulus = i128::from(modulus);
    let reduced = ((value % modulus) + modulus) % modulus;

    reduced as u64
}

fn div_round_nearest(value: i128, divisor: i128) -> i128 {
    assert!(divisor > 0);

    if value >= 0 {
        (value + divisor / 2) / divisor
    } else {
        -((-value + divisor / 2) / divisor)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use super::*;

    fn params() -> CkksParameters {
        CkksParameters::new(8, 2_147_483_647, 65_537, 65_537.0, 1)
    }

    fn reference_negacyclic_mul(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
        let n = lhs.len();
        let mut out = vec![0.0; n];

        for i in 0..n {
            for j in 0..n {
                let product = lhs[i] * rhs[j];

                if i + j < n {
                    out[i + j] += product;
                } else {
                    out[i + j - n] -= product;
                }
            }
        }

        out
    }

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());

        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "coefficient {index}: actual={actual}, expected={expected}, \
                 error={}, tolerance={tolerance}",
                (actual - expected).abs()
            );
        }
    }

    #[test]
    fn parameter_chain_is_exact() {
        let params = params();

        assert_eq!(
            params.high_modulus().value(),
            params.low_modulus().value() * params.rescale_prime()
        );
    }

    #[test]
    fn encode_decode_at_high_level_is_approximate_identity() {
        let params = params();

        let values = [1.25, -0.5, 2.0, 0.125, -1.75, 3.5, 0.0, 0.25];

        let encoded = encode(params, &values);
        let decoded = decode_polynomial(&encoded, params.initial_scale());

        assert_close(&decoded, &values, 1.0 / params.initial_scale());
    }

    #[test]
    fn secret_projection_preserves_ternary_values() {
        let params = params();

        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let high_secret = SecretKey::generate_with_rng(params.high_rlwe(), &mut rng);

        let low_secret = project_secret_key_to_low(params, &high_secret);

        let high_q = params.high_modulus().value();
        let low_q = params.low_modulus().value();

        for (&high, &low) in high_secret
            .polynomial()
            .coefficients()
            .iter()
            .zip(low_secret.polynomial().coefficients())
        {
            let expected = match high {
                0 => 0,
                1 => 1,
                value if value == high_q - 1 => low_q - 1,
                _ => panic!("secret was not ternary"),
            };

            assert_eq!(low, expected);
        }
    }

    #[test]
    fn encrypted_ckks_values_roundtrip_before_multiplication() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(10);
        let secret = SecretKey::generate_with_rng(params.high_rlwe(), &mut key_rng);

        let values = [1.25, -0.5, 2.0, 0.125, -1.75, 3.5, 0.0, 0.25];

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(11);

        let ciphertext = encrypt_with_rng(params, &secret, &values, &mut encryption_rng);

        let decoded = decrypt_decode(params, &secret, &ciphertext);

        assert_close(&decoded, &values, 5.0e-5);
    }

    #[test]
    fn multiply_relinearize_rescale_preserves_ckks_product() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(20);
        let high_secret = SecretKey::generate_with_rng(params.high_rlwe(), &mut key_rng);

        let low_secret = project_secret_key_to_low(params, &high_secret);

        let lhs = [1.25, -0.5, 0.75, 0.0, 0.25, -0.125, 0.0, 0.0];

        let rhs = [2.0, 0.75, -0.25, 0.5, 0.0, 0.0, 0.0, 0.0];

        let expected = reference_negacyclic_mul(&lhs, &rhs);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(21);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(22);

        let lhs_ct = encrypt_with_rng(params, &high_secret, &lhs, &mut lhs_rng);

        let rhs_ct = encrypt_with_rng(params, &high_secret, &rhs, &mut rhs_rng);

        let mut eval_rng = ChaCha20Rng::seed_from_u64(23);

        let multiplication_key = MultiplicationKey::generate_with_rng(
            params.high_rlwe(),
            &high_secret,
            65_536,
            &mut eval_rng,
        );

        let multiplied = multiply_relinearize(params, &lhs_ct, &rhs_ct, &multiplication_key);

        assert_eq!(multiplied.level(), 1);

        let expected_product_scale = params.initial_scale() * params.initial_scale();

        assert!((multiplied.scale() - expected_product_scale).abs() < f64::EPSILON);

        let rescaled = rescale_to_next(params, &multiplied);

        assert_eq!(rescaled.level(), 0);
        assert_eq!(rescaled.modulus(), params.low_modulus());

        let expected_rescaled_scale = expected_product_scale / params.rescale_prime() as f64;

        assert!((rescaled.scale() - expected_rescaled_scale).abs() < 1.0e-9);

        let actual = decrypt_decode(params, &low_secret, &rescaled);

        assert_close(&actual, &expected, 2.5e-3);
    }

    #[test]
    fn multiply_rescale_campaign_is_stable() {
        let params = params();

        let lhs = [0.5, -0.25, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0];

        let rhs = [1.5, 0.25, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0];

        let expected = reference_negacyclic_mul(&lhs, &rhs);

        for seed in 0_u64..16 {
            let mut key_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1000);

            let high_secret = SecretKey::generate_with_rng(params.high_rlwe(), &mut key_rng);

            let low_secret = project_secret_key_to_low(params, &high_secret);

            let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x2000);

            let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x3000);

            let lhs_ct = encrypt_with_rng(params, &high_secret, &lhs, &mut lhs_rng);

            let rhs_ct = encrypt_with_rng(params, &high_secret, &rhs, &mut rhs_rng);

            let mut eval_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x4000);

            let multiplication_key = MultiplicationKey::generate_with_rng(
                params.high_rlwe(),
                &high_secret,
                65_536,
                &mut eval_rng,
            );

            let rescaled = rescale_to_next(
                params,
                &multiply_relinearize(params, &lhs_ct, &rhs_ct, &multiplication_key),
            );

            let actual = decrypt_decode(params, &low_secret, &rescaled);

            assert_close(&actual, &expected, 2.5e-3);
        }
    }
}
