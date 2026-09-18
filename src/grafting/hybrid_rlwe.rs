use rand::{CryptoRng, RngCore};

use crate::ring::RnsPolynomial;

use super::{Pow2Polynomial, Pow2RnsPolynomial, RnsRlweCiphertext};

/// RLWE ciphertext over a power-of-two coefficient ring.
///
/// Arithmetic is over
///
/// `Z_(2^k)[X] / (X^N + 1)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pow2RlweCiphertext {
    b: Pow2Polynomial,
    a: Pow2Polynomial,
}

impl Pow2RlweCiphertext {
    pub fn new(b: Pow2Polynomial, a: Pow2Polynomial) -> Self {
        assert_eq!(
            b.bits(),
            a.bits(),
            "power-of-two RLWE component moduli must match"
        );

        assert_eq!(
            b.degree(),
            a.degree(),
            "power-of-two RLWE component degrees must match"
        );

        Self { b, a }
    }

    pub fn b(&self) -> &Pow2Polynomial {
        &self.b
    }

    pub fn a(&self) -> &Pow2Polynomial {
        &self.a
    }

    pub fn bits(&self) -> u32 {
        self.b.bits()
    }

    pub fn degree(&self) -> usize {
        self.b.degree()
    }
}

/// One logical RLWE ciphertext represented simultaneously over:
///
/// - an odd-prime RNS basis;
/// - one power-of-two sprout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridRlweCiphertext {
    ordinary: RnsRlweCiphertext,
    sprout: Pow2RlweCiphertext,
}

impl HybridRlweCiphertext {
    pub fn new(ordinary: RnsRlweCiphertext, sprout: Pow2RlweCiphertext) -> Self {
        assert_eq!(
            ordinary.degree(),
            sprout.degree(),
            "ordinary and sprout RLWE degrees must match"
        );

        Self { ordinary, sprout }
    }

    pub fn ordinary(&self) -> &RnsRlweCiphertext {
        &self.ordinary
    }

    pub fn sprout(&self) -> &Pow2RlweCiphertext {
        &self.sprout
    }

    pub fn degree(&self) -> usize {
        self.ordinary.degree()
    }
}

/// Projects one logical ternary secret into `Z_(2^k)`.
pub fn project_ternary_secret_pow2(bits: u32, coefficients: &[i8]) -> Pow2Polynomial {
    assert!(
        bits > 0 && bits <= 63,
        "power-of-two secret bits must be in 1..=63"
    );

    assert!(
        coefficients.iter().all(|&value| matches!(value, -1..=1)),
        "power-of-two secret coefficients must be ternary"
    );

    let modulus = 1_u64 << bits;

    Pow2Polynomial::new(
        bits,
        coefficients
            .iter()
            .map(|&value| match value {
                -1 => modulus - 1,
                0 => 0,
                1 => 1,
                _ => unreachable!("secret coefficients validated as ternary"),
            })
            .collect(),
    )
}

/// Noise-free raw RLWE encryption over a power-of-two sprout.
///
/// This is deliberately the algebraic correctness primitive. Noise is
/// introduced only after exact hybrid key-switch semantics are green.
pub fn encrypt_pow2_raw_with_rng<R>(
    secret: &Pow2Polynomial,
    message: &Pow2Polynomial,
    rng: &mut R,
) -> Pow2RlweCiphertext
where
    R: RngCore + CryptoRng,
{
    assert_eq!(
        secret.bits(),
        message.bits(),
        "power-of-two secret/message moduli must match"
    );

    assert_eq!(
        secret.degree(),
        message.degree(),
        "power-of-two secret/message degrees must match"
    );

    let bits = secret.bits();

    let mask = (1_u64 << bits) - 1;

    let a = Pow2Polynomial::new(
        bits,
        (0..secret.degree())
            .map(|_| rng.next_u64() & mask)
            .collect(),
    );

    let a_times_s = a.negacyclic_mul(secret);

    let b = message.sub(&a_times_s);

    Pow2RlweCiphertext::new(b, a)
}

/// Raw power-of-two RLWE decryption.
pub fn decrypt_pow2_raw(
    secret: &Pow2Polynomial,
    ciphertext: &Pow2RlweCiphertext,
) -> Pow2Polynomial {
    assert_eq!(
        secret.bits(),
        ciphertext.bits(),
        "power-of-two secret/ciphertext moduli must match"
    );

    assert_eq!(
        secret.degree(),
        ciphertext.degree(),
        "power-of-two secret/ciphertext degrees must match"
    );

    ciphertext.b().add(&ciphertext.a().negacyclic_mul(secret))
}

/// Reconstructs one logical plaintext polynomial from the odd-RNS and
/// power-of-two decryptions.
pub fn reconstruct_hybrid_plaintext(
    ordinary: &RnsPolynomial,
    sprout: &Pow2Polynomial,
) -> Pow2RnsPolynomial {
    Pow2RnsPolynomial::from_parts(ordinary.clone(), sprout.clone())
}

/// Raw RLWE encryption over a power-of-two sprout with bounded
/// coefficient error.
///
/// Encryption follows
///
/// ```text
/// b = m + e - a*s
/// ```
///
/// where every error coefficient lies in
/// `[-noise_bound, noise_bound]`.
pub fn encrypt_pow2_raw_with_noise_rng<R>(
    secret: &Pow2Polynomial,
    message: &Pow2Polynomial,
    noise_bound: i64,
    rng: &mut R,
) -> Pow2RlweCiphertext
where
    R: RngCore + CryptoRng,
{
    use rand::Rng;

    assert!(
        noise_bound >= 0,
        "power-of-two RLWE noise bound must be nonnegative"
    );

    assert_eq!(
        secret.bits(),
        message.bits(),
        "power-of-two secret/message moduli must match"
    );

    assert_eq!(
        secret.degree(),
        message.degree(),
        "power-of-two secret/message degrees must match"
    );

    let bits = secret.bits();
    let modulus = 1_u64 << bits;
    let mask = modulus - 1;

    let a = Pow2Polynomial::new(
        bits,
        (0..secret.degree())
            .map(|_| rng.next_u64() & mask)
            .collect(),
    );

    let error = Pow2Polynomial::new(
        bits,
        (0..secret.degree())
            .map(|_| {
                let value = rng.gen_range(-noise_bound..=noise_bound);

                if value >= 0 {
                    value as u64
                } else {
                    modulus - value.unsigned_abs()
                }
            })
            .collect(),
    );

    let a_times_s = a.negacyclic_mul(secret);

    let b = message.add(&error).sub(&a_times_s);

    Pow2RlweCiphertext::new(b, a)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use super::*;

    #[test]
    fn ternary_secret_projection_preserves_signed_values() {
        let coefficients = [-1_i8, 0, 1, -1, 1, 0, 1, -1];

        for bits in [4_u32, 8, 16, 32, 63] {
            let secret = project_ternary_secret_pow2(bits, &coefficients);

            let modulus = 1_u64 << bits;

            let expected: Vec<u64> = coefficients
                .iter()
                .map(|&value| match value {
                    -1 => modulus - 1,
                    0 => 0,
                    1 => 1,
                    _ => unreachable!(),
                })
                .collect();

            assert_eq!(secret.coefficients(), expected);
        }
    }

    #[test]
    fn noise_free_pow2_rlwe_roundtrip_is_exact() {
        let secret = project_ternary_secret_pow2(12, &[-1, 0, 1, 1, 0, -1, 1, 0]);

        let message = Pow2Polynomial::new(12, vec![0, 1, 17, 42, 1_001, 2_047, 4_095, 77]);

        let mut rng = ChaCha20Rng::seed_from_u64(1);

        let ciphertext = encrypt_pow2_raw_with_rng(&secret, &message, &mut rng);

        assert_eq!(decrypt_pow2_raw(&secret, &ciphertext,), message);
    }

    #[test]
    fn encryption_is_randomized_but_decryption_is_identical() {
        let secret = project_ternary_secret_pow2(16, &[-1, 0, 1, 1, -1, 0, 1, 0]);

        let message = Pow2Polynomial::new(16, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let mut first_rng = ChaCha20Rng::seed_from_u64(10);

        let mut second_rng = ChaCha20Rng::seed_from_u64(11);

        let first = encrypt_pow2_raw_with_rng(&secret, &message, &mut first_rng);

        let second = encrypt_pow2_raw_with_rng(&secret, &message, &mut second_rng);

        assert_ne!(first, second);

        assert_eq!(decrypt_pow2_raw(&secret, &first,), message);

        assert_eq!(decrypt_pow2_raw(&secret, &second,), message);
    }

    #[test]
    fn roundtrip_campaign_across_sprout_sizes() {
        let ternary = [-1_i8, 0, 1, 1, -1, 0, 1, 0];

        for bits in [4_u32, 8, 12, 16, 24, 32] {
            let secret = project_ternary_secret_pow2(bits, &ternary);

            let modulus = 1_u64 << bits;

            for seed in 0_u64..32 {
                let message = Pow2Polynomial::new(
                    bits,
                    (0..8_u64)
                        .map(|index| {
                            (97 * seed + 31 * index + 7 * index * index + 11) & (modulus - 1)
                        })
                        .collect(),
                );

                let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xDEAD_BEEF);

                let ciphertext = encrypt_pow2_raw_with_rng(&secret, &message, &mut rng);

                assert_eq!(
                    decrypt_pow2_raw(&secret, &ciphertext,),
                    message,
                    "power-of-two RLWE roundtrip failed: bits={bits}, seed={seed}"
                );
            }
        }
    }
}
