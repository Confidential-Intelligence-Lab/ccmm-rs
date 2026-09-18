use rand::rngs::OsRng;
use rand::{CryptoRng, RngCore};

use crate::eval::KeySwitchKey;
use crate::rlwe::{RlweParameters, SecretKey};

use super::apply_automorphism;

/// Galois evaluation key for one CKKS ring automorphism.
///
/// For exponent `k`, applying `sigma_k` to an RLWE ciphertext changes
/// its effective secret from
///
/// `s`
///
/// to
///
/// `sigma_k(s)`.
///
/// The contained switching key maps
///
/// `sigma_k(s) -> s`.
///
/// The transformed secret itself is used only during key generation
/// and is not retained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaloisKey {
    exponent: usize,
    key_switch_key: KeySwitchKey,
}

impl GaloisKey {
    /// Generates a Galois key using operating-system randomness.
    pub fn generate(
        params: RlweParameters,
        secret_key: &SecretKey,
        exponent: usize,
        base: u64,
    ) -> Self {
        let mut rng = OsRng;

        Self::generate_with_rng(params, secret_key, exponent, base, &mut rng)
    }

    /// Generates a Galois key from an explicit CSPRNG.
    pub fn generate_with_rng<R>(
        params: RlweParameters,
        secret_key: &SecretKey,
        exponent: usize,
        base: u64,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        assert_eq!(
            secret_key.polynomial().modulus(),
            params.modulus(),
            "secret key modulus must match RLWE parameters"
        );

        assert_eq!(
            secret_key.polynomial().degree(),
            params.degree(),
            "secret key degree must match RLWE parameters"
        );

        let two_n = 2 * params.degree();

        let exponent = exponent % two_n;

        assert!(
            exponent % 2 == 1,
            "CKKS Galois exponent must be odd modulo 2N"
        );

        let transformed_secret =
            SecretKey::from_polynomial(apply_automorphism(secret_key.polynomial(), exponent));

        let key_switch_key =
            KeySwitchKey::generate_with_rng(params, &transformed_secret, secret_key, base, rng);

        Self {
            exponent,
            key_switch_key,
        }
    }

    pub fn exponent(&self) -> usize {
        self.exponent
    }

    pub fn key_switch_key(&self) -> &KeySwitchKey {
        &self.key_switch_key
    }

    pub fn base(&self) -> u64 {
        self.key_switch_key.base()
    }

    pub fn digit_count(&self) -> usize {
        self.key_switch_key.digit_count()
    }
}

/// Applies a CKKS Galois automorphism to an RLWE ciphertext and
/// switches the transformed ciphertext back to the original secret.
///
/// If the input decrypts under `s` as
///
/// `m = b + a*s`,
///
/// applying `sigma_k` component-wise gives
///
/// `sigma_k(m) = sigma_k(b) + sigma_k(a)*sigma_k(s)`.
///
/// The intermediate ciphertext therefore decrypts under
/// `sigma_k(s)`. The Galois evaluation key switches it back to `s`.
pub fn apply_galois_automorphism(
    params: RlweParameters,
    ciphertext: &crate::rlwe::RlweCiphertext,
    galois_key: &GaloisKey,
) -> crate::rlwe::RlweCiphertext {
    assert_eq!(
        ciphertext.b().modulus(),
        params.modulus(),
        "ciphertext modulus must match RLWE parameters"
    );

    assert_eq!(
        ciphertext.b().degree(),
        params.degree(),
        "ciphertext degree must match RLWE parameters"
    );

    let transformed = crate::rlwe::RlweCiphertext::new(
        apply_automorphism(ciphertext.b(), galois_key.exponent()),
        apply_automorphism(ciphertext.a(), galois_key.exponent()),
    );

    crate::eval::key_switch(params, &transformed, galois_key.key_switch_key())
}

/// Rotates CKKS logical slots left by `steps`.
///
/// The supplied Galois key must correspond to the exponent returned by
/// `rotation_exponent_left(params.degree(), steps)`.
pub fn rotate_left(
    params: RlweParameters,
    ciphertext: &crate::rlwe::RlweCiphertext,
    steps: usize,
    galois_key: &GaloisKey,
) -> crate::rlwe::RlweCiphertext {
    let expected_exponent = crate::ckks::rotation_exponent_left(params.degree(), steps);

    assert_eq!(
        galois_key.exponent(),
        expected_exponent,
        "Galois key exponent does not match requested left rotation"
    );

    apply_galois_automorphism(params, ciphertext, galois_key)
}

/// Rotates CKKS logical slots right by `steps`.
///
/// The supplied Galois key must correspond to the exponent returned by
/// `rotation_exponent_right(params.degree(), steps)`.
pub fn rotate_right(
    params: RlweParameters,
    ciphertext: &crate::rlwe::RlweCiphertext,
    steps: usize,
    galois_key: &GaloisKey,
) -> crate::rlwe::RlweCiphertext {
    let expected_exponent = crate::ckks::rotation_exponent_right(params.degree(), steps);

    assert_eq!(
        galois_key.exponent(),
        expected_exponent,
        "Galois key exponent does not match requested right rotation"
    );

    apply_galois_automorphism(params, ciphertext, galois_key)
}

/// Applies logical CKKS complex conjugation to an encrypted slot vector.
///
/// The supplied Galois key must correspond to exponent
///
/// `2N - 1`.
pub fn conjugate_slots(
    params: RlweParameters,
    ciphertext: &crate::rlwe::RlweCiphertext,
    galois_key: &GaloisKey,
) -> crate::rlwe::RlweCiphertext {
    let expected_exponent = crate::ckks::conjugation_exponent(params.degree());

    assert_eq!(
        galois_key.exponent(),
        expected_exponent,
        "Galois key exponent does not match CKKS complex conjugation"
    );

    apply_galois_automorphism(params, ciphertext, galois_key)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::eval::key_switch;
    use crate::ring::{Modulus, Polynomial};
    use crate::rlwe::decrypt_raw;

    use super::*;

    fn params() -> RlweParameters {
        RlweParameters::new(8, Modulus::new(12_289), 16, 0)
    }

    fn secret(params: RlweParameters) -> SecretKey {
        SecretKey::from_polynomial(Polynomial::new(
            params.modulus(),
            vec![1, 0, params.modulus().value() - 1, 1, 0, 1, 0, 0],
        ))
    }

    #[test]
    fn galois_key_preserves_normalized_exponent() {
        let params = params();

        let secret = secret(params);

        let mut rng = ChaCha20Rng::seed_from_u64(1);

        let key = GaloisKey::generate_with_rng(params, &secret, 19, 16, &mut rng);

        assert_eq!(key.exponent(), 3);

        assert_eq!(key.base(), 16);
    }

    #[test]
    fn galois_key_switches_transformed_secret_back_to_original_secret() {
        let params = params();

        let secret = secret(params);

        let exponent = 3;

        let transformed_secret =
            SecretKey::from_polynomial(apply_automorphism(secret.polynomial(), exponent));

        /*
         * Construct a ciphertext whose raw decryption under
         * sigma_k(s) is known exactly:
         *
         *     b + a * sigma_k(s).
         */
        let b = Polynomial::new(params.modulus(), vec![2, 7, 1, 8, 2, 8, 1, 8]);

        let a = Polynomial::new(params.modulus(), vec![1, 3, 3, 7, 0, 2, 4, 1]);

        let ciphertext = crate::rlwe::RlweCiphertext::new(b, a);

        let expected = decrypt_raw(params, &transformed_secret, &ciphertext);

        let mut rng = ChaCha20Rng::seed_from_u64(2);

        let galois_key = GaloisKey::generate_with_rng(params, &secret, exponent, 16, &mut rng);

        let switched = key_switch(params, &ciphertext, galois_key.key_switch_key());

        let observed = decrypt_raw(params, &secret, &switched);

        assert_eq!(observed, expected);
    }

    #[test]
    fn deterministic_galois_key_generation_is_reproducible() {
        let params = params();

        let secret = secret(params);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(42);

        let mut rhs_rng = ChaCha20Rng::seed_from_u64(42);

        assert_eq!(
            GaloisKey::generate_with_rng(params, &secret, 5, 16, &mut lhs_rng,),
            GaloisKey::generate_with_rng(params, &secret, 5, 16, &mut rhs_rng,)
        );
    }

    #[test]
    #[should_panic(expected = "CKKS Galois exponent must be odd modulo 2N")]
    fn rejects_even_galois_exponent() {
        let params = params();

        let secret = secret(params);

        let mut rng = ChaCha20Rng::seed_from_u64(3);

        let _ = GaloisKey::generate_with_rng(params, &secret, 2, 16, &mut rng);
    }

    #[test]
    fn encrypted_automorphism_matches_plaintext_automorphism_exactly() {
        use crate::rlwe::encrypt_raw_with_rng;

        let params = params();

        let secret = secret(params);

        let message = Polynomial::new(params.modulus(), vec![3, 1, 4, 1, 5, 9, 2, 6]);

        for exponent in [1_usize, 3, 5, 7, 9, 11, 13, 15] {
            let mut encryption_rng = ChaCha20Rng::seed_from_u64(100 + exponent as u64);

            let ciphertext = encrypt_raw_with_rng(params, &secret, &message, &mut encryption_rng);

            let mut key_rng = ChaCha20Rng::seed_from_u64(200 + exponent as u64);

            let galois_key =
                GaloisKey::generate_with_rng(params, &secret, exponent, 16, &mut key_rng);

            let transformed = apply_galois_automorphism(params, &ciphertext, &galois_key);

            let observed = decrypt_raw(params, &secret, &transformed);

            let expected = apply_automorphism(&message, exponent);

            assert_eq!(observed, expected, "exponent={exponent}");
        }
    }

    #[test]
    fn encrypted_automorphism_composition_matches_exponent_product() {
        use crate::rlwe::encrypt_raw_with_rng;

        let params = params();

        let secret = secret(params);

        let message = Polynomial::new(params.modulus(), vec![2, 7, 1, 8, 2, 8, 1, 8]);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(300);

        let ciphertext = encrypt_raw_with_rng(params, &secret, &message, &mut encryption_rng);

        let mut key_rng_3 = ChaCha20Rng::seed_from_u64(301);

        let key_3 = GaloisKey::generate_with_rng(params, &secret, 3, 16, &mut key_rng_3);

        let mut key_rng_5 = ChaCha20Rng::seed_from_u64(302);

        let key_5 = GaloisKey::generate_with_rng(params, &secret, 5, 16, &mut key_rng_5);

        let first = apply_galois_automorphism(params, &ciphertext, &key_3);

        let composed = apply_galois_automorphism(params, &first, &key_5);

        let observed = decrypt_raw(params, &secret, &composed);

        let expected = apply_automorphism(&message, (3 * 5) % (2 * params.degree()));

        assert_eq!(observed, expected);
    }

    #[test]
    fn inverse_encrypted_automorphism_recovers_message() {
        use crate::rlwe::encrypt_raw_with_rng;

        let params = params();

        let secret = secret(params);

        let message = Polynomial::new(params.modulus(), vec![1, 4, 1, 4, 2, 1, 3, 5]);

        let exponent = 3;

        let inverse = crate::ckks::inverse_automorphism_exponent(params.degree(), exponent);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(400);

        let ciphertext = encrypt_raw_with_rng(params, &secret, &message, &mut encryption_rng);

        let mut forward_rng = ChaCha20Rng::seed_from_u64(401);

        let forward_key =
            GaloisKey::generate_with_rng(params, &secret, exponent, 16, &mut forward_rng);

        let mut inverse_rng = ChaCha20Rng::seed_from_u64(402);

        let inverse_key =
            GaloisKey::generate_with_rng(params, &secret, inverse, 16, &mut inverse_rng);

        let transformed = apply_galois_automorphism(params, &ciphertext, &forward_key);

        let recovered = apply_galois_automorphism(params, &transformed, &inverse_key);

        assert_eq!(decrypt_raw(params, &secret, &recovered,), message);
    }

    #[test]
    fn encrypted_left_rotation_matches_plaintext_automorphism() {
        use crate::rlwe::encrypt_raw_with_rng;

        let params = params();

        let secret = secret(params);

        let message = Polynomial::new(params.modulus(), vec![3, 1, 4, 1, 5, 9, 2, 6]);

        let steps = 1;

        let exponent = crate::ckks::rotation_exponent_left(params.degree(), steps);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(500);

        let ciphertext = encrypt_raw_with_rng(params, &secret, &message, &mut encryption_rng);

        let mut key_rng = ChaCha20Rng::seed_from_u64(501);

        let key = GaloisKey::generate_with_rng(params, &secret, exponent, 16, &mut key_rng);

        let rotated = rotate_left(params, &ciphertext, steps, &key);

        let observed = decrypt_raw(params, &secret, &rotated);

        let expected = apply_automorphism(&message, exponent);

        assert_eq!(observed, expected);
    }

    #[test]
    fn encrypted_right_rotation_matches_plaintext_automorphism() {
        use crate::rlwe::encrypt_raw_with_rng;

        let params = params();

        let secret = secret(params);

        let message = Polynomial::new(params.modulus(), vec![2, 7, 1, 8, 2, 8, 1, 8]);

        let steps = 1;

        let exponent = crate::ckks::rotation_exponent_right(params.degree(), steps);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(510);

        let ciphertext = encrypt_raw_with_rng(params, &secret, &message, &mut encryption_rng);

        let mut key_rng = ChaCha20Rng::seed_from_u64(511);

        let key = GaloisKey::generate_with_rng(params, &secret, exponent, 16, &mut key_rng);

        let rotated = rotate_right(params, &ciphertext, steps, &key);

        let observed = decrypt_raw(params, &secret, &rotated);

        let expected = apply_automorphism(&message, exponent);

        assert_eq!(observed, expected);
    }

    #[test]
    fn encrypted_left_then_right_rotation_recovers_message() {
        use crate::rlwe::encrypt_raw_with_rng;

        let params = params();

        let secret = secret(params);

        let message = Polynomial::new(params.modulus(), vec![1, 4, 1, 4, 2, 1, 3, 5]);

        let steps = 2;

        let left_exponent = crate::ckks::rotation_exponent_left(params.degree(), steps);

        let right_exponent = crate::ckks::rotation_exponent_right(params.degree(), steps);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(520);

        let ciphertext = encrypt_raw_with_rng(params, &secret, &message, &mut encryption_rng);

        let mut left_rng = ChaCha20Rng::seed_from_u64(521);

        let left_key =
            GaloisKey::generate_with_rng(params, &secret, left_exponent, 16, &mut left_rng);

        let mut right_rng = ChaCha20Rng::seed_from_u64(522);

        let right_key =
            GaloisKey::generate_with_rng(params, &secret, right_exponent, 16, &mut right_rng);

        let left = rotate_left(params, &ciphertext, steps, &left_key);

        let recovered = rotate_right(params, &left, steps, &right_key);

        assert_eq!(decrypt_raw(params, &secret, &recovered,), message);
    }

    #[test]
    #[should_panic(expected = "Galois key exponent does not match requested left rotation")]
    fn left_rotation_rejects_wrong_galois_key() {
        let params = params();

        let secret = secret(params);

        let ciphertext = crate::rlwe::RlweCiphertext::new(
            Polynomial::zero(params.modulus(), params.degree()),
            Polynomial::zero(params.modulus(), params.degree()),
        );

        let wrong_exponent = crate::ckks::rotation_exponent_left(params.degree(), 2);

        let mut rng = ChaCha20Rng::seed_from_u64(530);

        let wrong_key = GaloisKey::generate_with_rng(params, &secret, wrong_exponent, 16, &mut rng);

        let _ = rotate_left(params, &ciphertext, 1, &wrong_key);
    }

    #[test]
    fn encrypted_conjugation_matches_plaintext_automorphism() {
        use crate::rlwe::encrypt_raw_with_rng;

        let params = params();

        let secret = secret(params);

        let message = Polynomial::new(params.modulus(), vec![3, 1, 4, 1, 5, 9, 2, 6]);

        let exponent = crate::ckks::conjugation_exponent(params.degree());

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(600);

        let ciphertext = encrypt_raw_with_rng(params, &secret, &message, &mut encryption_rng);

        let mut key_rng = ChaCha20Rng::seed_from_u64(601);

        let key = GaloisKey::generate_with_rng(params, &secret, exponent, 16, &mut key_rng);

        let conjugated = conjugate_slots(params, &ciphertext, &key);

        let observed = decrypt_raw(params, &secret, &conjugated);

        let expected = apply_automorphism(&message, exponent);

        assert_eq!(observed, expected);
    }

    #[test]
    fn encrypted_double_conjugation_recovers_message() {
        use crate::rlwe::encrypt_raw_with_rng;

        let params = params();

        let secret = secret(params);

        let message = Polynomial::new(params.modulus(), vec![2, 7, 1, 8, 2, 8, 1, 8]);

        let exponent = crate::ckks::conjugation_exponent(params.degree());

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(610);

        let ciphertext = encrypt_raw_with_rng(params, &secret, &message, &mut encryption_rng);

        let mut key_rng = ChaCha20Rng::seed_from_u64(611);

        let key = GaloisKey::generate_with_rng(params, &secret, exponent, 16, &mut key_rng);

        let once = conjugate_slots(params, &ciphertext, &key);

        let twice = conjugate_slots(params, &once, &key);

        assert_eq!(decrypt_raw(params, &secret, &twice,), message);
    }

    #[test]
    #[should_panic(expected = "Galois key exponent does not match CKKS complex conjugation")]
    fn conjugation_rejects_wrong_galois_key() {
        let params = params();

        let secret = secret(params);

        let ciphertext = crate::rlwe::RlweCiphertext::new(
            Polynomial::zero(params.modulus(), params.degree()),
            Polynomial::zero(params.modulus(), params.degree()),
        );

        let wrong_exponent = crate::ckks::rotation_exponent_left(params.degree(), 1);

        let mut rng = ChaCha20Rng::seed_from_u64(620);

        let wrong_key = GaloisKey::generate_with_rng(params, &secret, wrong_exponent, 16, &mut rng);

        let _ = conjugate_slots(params, &ciphertext, &wrong_key);
    }
}
