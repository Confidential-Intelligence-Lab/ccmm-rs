use rand::rngs::OsRng;
use rand::{CryptoRng, Rng, RngCore};

use crate::ring::Polynomial;
use crate::rlwe::{RlweCiphertext, RlweParameters, RlweQuadraticCiphertext, SecretKey};

/// Multiplication/evaluation key for relinearizing a degree-2 RLWE
/// ciphertext.
///
/// For gadget base `B`, entry `i` is an RLWE encryption of
///
/// `B^i * s^2`
///
/// under the original secret key `s`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiplicationKey {
    base: u64,
    digits: Vec<RlweCiphertext>,
}

impl MultiplicationKey {
    /// Generates a multiplication key with operating-system randomness.
    pub fn generate(params: RlweParameters, secret_key: &SecretKey, base: u64) -> Self {
        let mut rng = OsRng;
        Self::generate_with_rng(params, secret_key, base, &mut rng)
    }

    /// Generates a multiplication key from an explicit CSPRNG.
    ///
    /// This is useful for deterministic tests and reproducibility.
    pub fn generate_with_rng<R>(
        params: RlweParameters,
        secret_key: &SecretKey,
        base: u64,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        assert!(base >= 2, "gadget base must be at least 2");

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

        let digit_count = required_digit_count(params.modulus().value(), base);

        let secret_squared = secret_key
            .polynomial()
            .negacyclic_mul(secret_key.polynomial());

        let mut digits = Vec::with_capacity(digit_count);
        let mut base_power = 1_u64;

        for _ in 0..digit_count {
            let target = secret_squared.scalar_mul(base_power);
            let a = sample_uniform(params, rng);
            let error = sample_error(params, rng);

            let a_times_s = a.negacyclic_mul(secret_key.polynomial());

            // b + a*s = B^i*s^2 + e
            let b = target.add(&error).sub(&a_times_s);

            digits.push(RlweCiphertext::new(b, a));

            base_power = params.modulus().mul(base_power, base);
        }

        Self { base, digits }
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    pub fn digit_count(&self) -> usize {
        self.digits.len()
    }

    pub fn digits(&self) -> &[RlweCiphertext] {
        &self.digits
    }
}

/// Relinearizes a degree-2 ciphertext back to rank 1.
///
/// If
///
/// `ct = c0 + c1*s + c2*s^2`,
///
/// coefficient-wise gadget decomposition gives
///
/// `c2 = sum_i d_i * B^i`.
///
/// Each `B^i*s^2` is then replaced by its RLWE evaluation-key
/// encryption.
pub fn relinearize(
    params: RlweParameters,
    product: &RlweQuadraticCiphertext,
    multiplication_key: &MultiplicationKey,
) -> RlweCiphertext {
    assert_eq!(
        product.c0().modulus(),
        params.modulus(),
        "quadratic ciphertext modulus must match RLWE parameters"
    );

    assert_eq!(
        product.c0().degree(),
        params.degree(),
        "quadratic ciphertext degree must match RLWE parameters"
    );

    let digits = gadget_decompose(
        product.c2(),
        multiplication_key.base(),
        multiplication_key.digit_count(),
    );

    let mut b = product.c0().clone();
    let mut a = product.c1().clone();

    for (digit, evaluation_key) in digits.iter().zip(multiplication_key.digits()) {
        b = b.add(&digit.negacyclic_mul(evaluation_key.b()));
        a = a.add(&digit.negacyclic_mul(evaluation_key.a()));
    }

    RlweCiphertext::new(b, a)
}

fn required_digit_count(modulus: u64, base: u64) -> usize {
    let mut capacity = 1_u128;
    let modulus = u128::from(modulus);
    let base = u128::from(base);
    let mut digits = 0_usize;

    while capacity < modulus {
        capacity *= base;
        digits += 1;
    }

    digits.max(1)
}

fn gadget_decompose(polynomial: &Polynomial, base: u64, digit_count: usize) -> Vec<Polynomial> {
    assert!(base >= 2, "gadget base must be at least 2");
    assert!(digit_count > 0, "gadget digit count must be positive");

    let degree = polynomial.degree();
    let modulus = polynomial.modulus();

    let mut digit_coefficients = vec![vec![0_u64; degree]; digit_count];

    for (coefficient_index, &coefficient) in polynomial.coefficients().iter().enumerate() {
        let mut value = coefficient;

        for digit in digit_coefficients.iter_mut() {
            digit[coefficient_index] = value % base;
            value /= base;
        }

        assert_eq!(value, 0, "gadget decomposition has insufficient digits");
    }

    digit_coefficients
        .into_iter()
        .map(|coefficients| Polynomial::new(modulus, coefficients))
        .collect()
}

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
            .map(|_| signed_to_mod(rng.gen_range(-bound..=bound), params.modulus().value()))
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

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ring::Modulus;
    use crate::rlwe::{
        decrypt_quadratic_raw, decrypt_raw, encrypt_with_rng, tensor, RlwePlaintext,
    };

    use super::*;

    fn params() -> RlweParameters {
        RlweParameters::new(8, Modulus::new(12_289), 16, 1)
    }

    fn circular_distance(modulus: u64, lhs: u64, rhs: u64) -> u64 {
        let forward = (lhs + modulus - rhs) % modulus;
        let backward = (rhs + modulus - lhs) % modulus;

        forward.min(backward)
    }

    #[test]
    fn gadget_decomposition_reconstructs_polynomial() {
        let q = Modulus::new(12_289);

        let polynomial = Polynomial::new(q, vec![0, 1, 15, 16, 255, 256, 4095, 12_288]);

        let base = 16;
        let digit_count = required_digit_count(q.value(), base);

        let digits = gadget_decompose(&polynomial, base, digit_count);

        let mut reconstructed = Polynomial::zero(q, polynomial.degree());

        let mut base_power = 1_u64;

        for digit in &digits {
            reconstructed = reconstructed.add(&digit.scalar_mul(base_power));

            base_power = q.mul(base_power, base);
        }

        assert_eq!(reconstructed, polynomial);
    }

    #[test]
    fn multiplication_key_has_expected_digit_count() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(1);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let mut eval_rng = ChaCha20Rng::seed_from_u64(2);

        let key = MultiplicationKey::generate_with_rng(params, &secret, 16, &mut eval_rng);

        // 16^3 < 12289 <= 16^4.
        assert_eq!(key.digit_count(), 4);
        assert_eq!(key.base(), 16);
    }

    #[test]
    fn evaluation_key_entries_encrypt_scaled_s_squared() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(10);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let mut eval_rng = ChaCha20Rng::seed_from_u64(11);

        let key = MultiplicationKey::generate_with_rng(params, &secret, 16, &mut eval_rng);

        let s_squared = secret.polynomial().negacyclic_mul(secret.polynomial());

        let q = params.modulus();
        let mut base_power = 1_u64;

        for evaluation_key in key.digits() {
            let decrypted = decrypt_raw(params, &secret, evaluation_key);

            let target = s_squared.scalar_mul(base_power);

            for (&actual, &expected) in decrypted.coefficients().iter().zip(target.coefficients()) {
                assert!(
                    circular_distance(q.value(), actual, expected,) <= params.noise_bound() as u64
                );
            }

            base_power = q.mul(base_power, key.base());
        }
    }

    #[test]
    fn relinearization_noise_matches_evaluation_key_noise_exactly() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(20);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let lhs_plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let rhs_plaintext = RlwePlaintext::new(params, vec![8, 7, 6, 5, 4, 3, 2, 1]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(21);

        let mut rhs_rng = ChaCha20Rng::seed_from_u64(22);

        let lhs = encrypt_with_rng(params, &secret, &lhs_plaintext, &mut lhs_rng);

        let rhs = encrypt_with_rng(params, &secret, &rhs_plaintext, &mut rhs_rng);

        let quadratic = tensor(&lhs, &rhs);

        let mut eval_rng = ChaCha20Rng::seed_from_u64(23);

        let multiplication_key =
            MultiplicationKey::generate_with_rng(params, &secret, 16, &mut eval_rng);

        let relinearized = relinearize(params, &quadratic, &multiplication_key);

        let quadratic_raw = decrypt_quadratic_raw(params, &secret, &quadratic);

        let actual = decrypt_raw(params, &secret, &relinearized);

        let digits = gadget_decompose(
            quadratic.c2(),
            multiplication_key.base(),
            multiplication_key.digit_count(),
        );

        let s_squared = secret.polynomial().negacyclic_mul(secret.polynomial());

        let mut expected = quadratic_raw.clone();

        let mut base_power = 1_u64;

        for (digit, evaluation_key) in digits.iter().zip(multiplication_key.digits()) {
            let decrypted_key = decrypt_raw(params, &secret, evaluation_key);

            let target = s_squared.scalar_mul(base_power);

            let evaluation_noise = decrypted_key.sub(&target);

            expected = expected.add(&digit.negacyclic_mul(&evaluation_noise));

            base_power = params.modulus().mul(base_power, multiplication_key.base());
        }

        assert_eq!(actual, expected);
    }

    #[test]
    fn relinearization_adds_bounded_noise() {
        let params = params();
        let q = params.modulus().value();

        for seed in 0_u64..32 {
            let mut key_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1111);

            let secret = SecretKey::generate_with_rng(params, &mut key_rng);

            let plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

            let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x2222);

            let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x3333);

            let lhs = encrypt_with_rng(params, &secret, &plaintext, &mut lhs_rng);

            let rhs = encrypt_with_rng(params, &secret, &plaintext, &mut rhs_rng);

            let quadratic = tensor(&lhs, &rhs);

            let mut eval_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x4444);

            let multiplication_key =
                MultiplicationKey::generate_with_rng(params, &secret, 16, &mut eval_rng);

            let relinearized = relinearize(params, &quadratic, &multiplication_key);

            let before = decrypt_quadratic_raw(params, &secret, &quadratic);

            let after = decrypt_raw(params, &secret, &relinearized);

            // Conservative coefficient bound:
            //
            // digits * N * (B - 1) * e_bound.
            let noise_bound = multiplication_key.digit_count() as u64
                * params.degree() as u64
                * (multiplication_key.base() - 1)
                * params.noise_bound() as u64;

            for (&lhs, &rhs) in after.coefficients().iter().zip(before.coefficients()) {
                assert!(
                    circular_distance(q, lhs, rhs) <= noise_bound,
                    "relinearization noise exceeded bound for seed {seed}"
                );
            }
        }
    }
}
