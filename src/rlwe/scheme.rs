use rand::rngs::OsRng;
use rand::{CryptoRng, Rng, RngCore};

use crate::ring::Polynomial;

use super::{RlweCiphertext, RlweParameters, RlwePlaintext, SecretKey};

fn signed_to_mod(value: i64, modulus: u64) -> u64 {
    if value >= 0 {
        (value as u64) % modulus
    } else {
        let magnitude = value.unsigned_abs() % modulus;

        if magnitude == 0 {
            0
        } else {
            modulus - magnitude
        }
    }
}

fn sample_error<R>(params: RlweParameters, rng: &mut R) -> Polynomial
where
    R: RngCore + CryptoRng,
{
    let bound = params.noise_bound();

    let coefficients = if bound == 0 {
        vec![0; params.degree()]
    } else {
        (0..params.degree())
            .map(|_| {
                let error = rng.gen_range(-bound..=bound);
                signed_to_mod(error, params.modulus().value())
            })
            .collect()
    };

    Polynomial::new(params.modulus(), coefficients)
}

fn sample_uniform<R>(params: RlweParameters, rng: &mut R) -> Polynomial
where
    R: RngCore + CryptoRng,
{
    let q = params.modulus().value();

    Polynomial::new(
        params.modulus(),
        (0..params.degree()).map(|_| rng.gen_range(0..q)).collect(),
    )
}

/// Encrypts a coefficient plaintext with operating-system randomness.
pub fn encrypt(
    params: RlweParameters,
    secret_key: &SecretKey,
    plaintext: &RlwePlaintext,
) -> RlweCiphertext {
    let mut rng = OsRng;

    encrypt_with_rng(params, secret_key, plaintext, &mut rng)
}

/// Encrypts using an explicit cryptographically secure RNG.
///
/// This is useful for deterministic testing and reproducibility.
///
/// Encryption uses:
///
/// `b = m + e - a*s`.
pub fn encrypt_with_rng<R>(
    params: RlweParameters,
    secret_key: &SecretKey,
    plaintext: &RlwePlaintext,
    rng: &mut R,
) -> RlweCiphertext
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

    let encoded = params.encode_message(plaintext.coefficients());
    let a = sample_uniform(params, rng);
    let error = sample_error(params, rng);

    let a_times_s = a.negacyclic_mul(secret_key.polynomial());

    let b = encoded.add(&error).sub(&a_times_s);

    RlweCiphertext::new(b, a)
}

/// Encrypts an already encoded polynomial.
///
/// This is the raw RLWE primitive used by higher-level encodings such as
/// CKKS. It does not apply the coefficient-message encoding from
/// `RlwePlaintext`.
///
/// Encryption computes:
///
/// `b = m + e - a*s`.
pub fn encrypt_raw_with_rng<R>(
    params: RlweParameters,
    secret_key: &SecretKey,
    message: &Polynomial,
    rng: &mut R,
) -> RlweCiphertext
where
    R: RngCore + CryptoRng,
{
    assert_eq!(
        message.modulus(),
        params.modulus(),
        "raw plaintext modulus must match RLWE parameters"
    );

    assert_eq!(
        message.degree(),
        params.degree(),
        "raw plaintext degree must match RLWE parameters"
    );

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

    let a = sample_uniform(params, rng);
    let error = sample_error(params, rng);

    let a_times_s = a.negacyclic_mul(secret_key.polynomial());

    let b = message.add(&error).sub(&a_times_s);

    RlweCiphertext::new(b, a)
}

/// Returns the noisy encoded plaintext polynomial:
///
/// `b + a*s = m + e`.
pub fn decrypt_raw(
    params: RlweParameters,
    secret_key: &SecretKey,
    ciphertext: &RlweCiphertext,
) -> Polynomial {
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

    let a_times_s = ciphertext.a().negacyclic_mul(secret_key.polynomial());

    ciphertext.b().add(&a_times_s)
}

/// Decrypts and decodes a coefficient plaintext.
pub fn decrypt(
    params: RlweParameters,
    secret_key: &SecretKey,
    ciphertext: &RlweCiphertext,
) -> RlwePlaintext {
    let noisy = decrypt_raw(params, secret_key, ciphertext);

    RlwePlaintext::new(params, params.decode_message(&noisy))
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    use crate::ring::Modulus;

    use super::*;

    fn params() -> RlweParameters {
        RlweParameters::new(8, Modulus::new(12_289), 16, 1)
    }

    #[test]
    fn deterministic_rlwe_roundtrip_works() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(1);
        let key = SecretKey::generate_with_rng(params, &mut key_rng);

        let plaintext = RlwePlaintext::new(params, vec![0, 1, 2, 3, 4, 5, 6, 7]);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(2);

        let ciphertext = encrypt_with_rng(params, &key, &plaintext, &mut encryption_rng);

        assert_eq!(decrypt(params, &key, &ciphertext), plaintext);
    }

    #[test]
    fn multiple_seeded_roundtrips_work() {
        let params = params();

        for seed in 0_u64..64 {
            let mut key_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1357_2468);

            let key = SecretKey::generate_with_rng(params, &mut key_rng);

            let mut message_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xABCD_EF01);

            let message: Vec<u64> = (0..params.degree())
                .map(|_| message_rng.gen_range(0..params.plaintext_modulus()))
                .collect();

            let plaintext = RlwePlaintext::new(params, message);

            let mut encryption_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xDEAD_BEEF);

            let ciphertext = encrypt_with_rng(params, &key, &plaintext, &mut encryption_rng);

            assert_eq!(
                decrypt(params, &key, &ciphertext),
                plaintext,
                "roundtrip failed for seed {seed}"
            );
        }
    }

    #[test]
    fn encryption_is_randomized() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(10);
        let key = SecretKey::generate_with_rng(params, &mut key_rng);

        let plaintext = RlwePlaintext::new(params, vec![1; 8]);

        let mut first_rng = ChaCha20Rng::seed_from_u64(11);
        let mut second_rng = ChaCha20Rng::seed_from_u64(12);

        let first = encrypt_with_rng(params, &key, &plaintext, &mut first_rng);

        let second = encrypt_with_rng(params, &key, &plaintext, &mut second_rng);

        assert_ne!(first, second);

        assert_eq!(decrypt(params, &key, &first), plaintext);
        assert_eq!(decrypt(params, &key, &second), plaintext);
    }

    #[test]
    fn raw_decryption_contains_only_small_encoding_error() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(100);
        let key = SecretKey::generate_with_rng(params, &mut key_rng);

        let plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let encoded = params.encode_message(plaintext.coefficients());

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(101);

        let ciphertext = encrypt_with_rng(params, &key, &plaintext, &mut encryption_rng);

        let noisy = decrypt_raw(params, &key, &ciphertext);

        let q = params.modulus().value();

        for (&expected, &actual) in encoded.coefficients().iter().zip(noisy.coefficients()) {
            let forward = (actual + q - expected) % q;
            let backward = (expected + q - actual) % q;
            let distance = forward.min(backward);

            assert!(
                distance <= params.noise_bound() as u64,
                "noise exceeded configured bound"
            );
        }
    }
}
