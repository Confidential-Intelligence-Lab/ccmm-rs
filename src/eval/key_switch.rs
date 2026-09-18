use rand::rngs::OsRng;
use rand::{CryptoRng, Rng, RngCore};

use crate::ring::Polynomial;
use crate::rlwe::{RlweCiphertext, RlweParameters, SecretKey};

/// Evaluation key for switching a rank-1 RLWE ciphertext from one
/// secret key to another.
///
/// For gadget base B, entry i encrypts
///
/// `B^i * s_source`
///
/// under `s_target`.
///
/// If an input ciphertext decrypts as
///
/// `b + a * s_source`,
///
/// decomposing `a` and replacing its gadget multiples with these
/// evaluation-key ciphertexts yields a ciphertext under `s_target`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeySwitchKey {
    base: u64,
    digits: Vec<RlweCiphertext>,
}

impl KeySwitchKey {
    pub fn generate(
        params: RlweParameters,
        source_secret: &SecretKey,
        target_secret: &SecretKey,
        base: u64,
    ) -> Self {
        let mut rng = OsRng;

        Self::generate_with_rng(params, source_secret, target_secret, base, &mut rng)
    }

    pub fn generate_with_rng<R>(
        params: RlweParameters,
        source_secret: &SecretKey,
        target_secret: &SecretKey,
        base: u64,
        rng: &mut R,
    ) -> Self
    where
        R: RngCore + CryptoRng,
    {
        assert!(base >= 2, "gadget base must be at least 2");

        for (name, secret) in [("source", source_secret), ("target", target_secret)] {
            assert_eq!(
                secret.polynomial().modulus(),
                params.modulus(),
                "{name} secret key modulus must match RLWE parameters"
            );

            assert_eq!(
                secret.polynomial().degree(),
                params.degree(),
                "{name} secret key degree must match RLWE parameters"
            );
        }

        let digit_count = required_digit_count(params.modulus().value(), base);

        let mut digits = Vec::with_capacity(digit_count);

        let mut base_power = 1_u64;

        for _ in 0..digit_count {
            let target = source_secret.polynomial().scalar_mul(base_power);

            let a = sample_uniform(params, rng);

            let error = sample_error(params, rng);

            let a_times_target_secret = a.negacyclic_mul(target_secret.polynomial());

            // b + a*s_target = B^i*s_source + e
            let b = target.add(&error).sub(&a_times_target_secret);

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

/// Switches a rank-1 RLWE ciphertext from the source secret encoded by
/// `key_switch_key` to its target secret.
///
/// The input is interpreted as
///
/// `b + a * s_source`.
///
/// The returned ciphertext decrypts under `s_target` to the same
/// message, up to evaluation-key noise.
pub fn key_switch(
    params: RlweParameters,
    ciphertext: &RlweCiphertext,
    key_switch_key: &KeySwitchKey,
) -> RlweCiphertext {
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

    let digits = gadget_decompose(
        ciphertext.a(),
        key_switch_key.base(),
        key_switch_key.digit_count(),
    );

    let mut b = ciphertext.b().clone();

    let mut a = Polynomial::zero(params.modulus(), params.degree());

    for (digit, evaluation_key) in digits.iter().zip(key_switch_key.digits()) {
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

        for digit in &mut digit_coefficients {
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
    use crate::rlwe::{decrypt_raw, encrypt_raw_with_rng};

    use super::*;

    fn params(noise_bound: i64) -> RlweParameters {
        RlweParameters::new(8, Modulus::new(12_289), 16, noise_bound)
    }

    fn source_secret(params: RlweParameters) -> SecretKey {
        SecretKey::from_polynomial(Polynomial::new(
            params.modulus(),
            vec![1, 0, params.modulus().value() - 1, 1, 0, 1, 0, 0],
        ))
    }

    fn target_secret(params: RlweParameters) -> SecretKey {
        SecretKey::from_polynomial(Polynomial::new(
            params.modulus(),
            vec![0, 1, 1, 0, params.modulus().value() - 1, 0, 1, 0],
        ))
    }

    #[test]
    fn key_entries_encrypt_scaled_source_secret_under_target_secret() {
        let params = params(0);

        let source = source_secret(params);

        let target = target_secret(params);

        let mut rng = ChaCha20Rng::seed_from_u64(1);

        let key = KeySwitchKey::generate_with_rng(params, &source, &target, 16, &mut rng);

        let mut base_power = 1_u64;

        for entry in key.digits() {
            let decrypted = decrypt_raw(params, &target, entry);

            let expected = source.polynomial().scalar_mul(base_power);

            assert_eq!(decrypted, expected);

            base_power = params.modulus().mul(base_power, key.base());
        }
    }

    #[test]
    fn noise_free_key_switch_preserves_raw_decryption_exactly() {
        let params = params(0);

        let source = source_secret(params);

        let target = target_secret(params);

        let message = Polynomial::new(params.modulus(), vec![3, 1, 4, 1, 5, 9, 2, 6]);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(2);

        let ciphertext = encrypt_raw_with_rng(params, &source, &message, &mut encryption_rng);

        let before = decrypt_raw(params, &source, &ciphertext);

        let mut key_rng = ChaCha20Rng::seed_from_u64(3);

        let key = KeySwitchKey::generate_with_rng(params, &source, &target, 16, &mut key_rng);

        let switched = key_switch(params, &ciphertext, &key);

        let after = decrypt_raw(params, &target, &switched);

        assert_eq!(after, before);
    }

    #[test]
    fn deterministic_key_generation_is_reproducible() {
        let params = params(0);

        let source = source_secret(params);

        let target = target_secret(params);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(42);

        let mut rhs_rng = ChaCha20Rng::seed_from_u64(42);

        assert_eq!(
            KeySwitchKey::generate_with_rng(params, &source, &target, 16, &mut lhs_rng,),
            KeySwitchKey::generate_with_rng(params, &source, &target, 16, &mut rhs_rng,)
        );
    }
}
