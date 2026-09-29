//! Park-style encrypted matrix multiplication primitives.
//!
//! This module implements the reduction machinery underlying the
//! ciphertext-matrix algorithms of Park et al.
//!
//! The first implemented primitive is `Tweak`, Algorithm 3 in the
//! ciphertext-matrix-transpose construction.
//!
//! This module is intentionally separate from the existing reference/native
//! CPMM and CCMM implementations.

use crate::ring::Polynomial;
use crate::rlwe::RlweCiphertext;

/// Adds two RLWE ciphertexts componentwise.
fn ciphertext_add(lhs: &RlweCiphertext, rhs: &RlweCiphertext) -> RlweCiphertext {
    RlweCiphertext::new(lhs.b().add(rhs.b()), lhs.a().add(rhs.a()))
}

/// Subtracts two RLWE ciphertexts componentwise.
fn ciphertext_sub(lhs: &RlweCiphertext, rhs: &RlweCiphertext) -> RlweCiphertext {
    RlweCiphertext::new(lhs.b().sub(rhs.b()), lhs.a().sub(rhs.a()))
}

/// Returns `X^exponent * polynomial` in `Z_q[X]/(X^N + 1)`.
///
/// The exponent is interpreted modulo `2N`, using `X^N = -1`.
fn polynomial_mul_monomial(polynomial: &Polynomial, exponent: usize) -> Polynomial {
    let degree = polynomial.degree();
    let modulus = polynomial.modulus();

    let exponent = exponent % (2 * degree);
    let negative = exponent >= degree;
    let shift = exponent % degree;

    let mut monomial = vec![0_u64; degree];

    monomial[shift] = if negative { modulus.value() - 1 } else { 1 };

    polynomial.negacyclic_mul(&Polynomial::new(modulus, monomial))
}

/// Returns `X^exponent * ciphertext`.
fn ciphertext_mul_monomial(ciphertext: &RlweCiphertext, exponent: usize) -> RlweCiphertext {
    RlweCiphertext::new(
        polynomial_mul_monomial(ciphertext.b(), exponent),
        polynomial_mul_monomial(ciphertext.a(), exponent),
    )
}

/// Park Algorithm 3: Tweak.
///
/// For `n` ciphertexts over ring degree `N`, where both `n` and `N` are
/// powers of two and `n` divides `N`, returns:
///
/// ```text
/// out[j] = sum_i X^(2*i*j*N/n) * input[i].
/// ```
///
/// This is the structured divide-and-conquer transform used twice by the
/// Park ciphertext-matrix-transpose algorithm.
pub fn tweak(ciphertexts: &[RlweCiphertext]) -> Vec<RlweCiphertext> {
    assert!(
        !ciphertexts.is_empty(),
        "Tweak requires at least one ciphertext"
    );

    let n = ciphertexts.len();

    assert!(
        n.is_power_of_two(),
        "Tweak ciphertext count must be a power of two"
    );

    let degree = ciphertexts[0].b().degree();

    assert!(
        degree.is_power_of_two(),
        "Tweak ring degree must be a power of two"
    );

    assert!(
        n <= degree && degree % n == 0,
        "Tweak ciphertext count must divide ring degree"
    );

    let modulus = ciphertexts[0].b().modulus();

    for ciphertext in ciphertexts {
        assert_eq!(
            ciphertext.b().degree(),
            degree,
            "Tweak ciphertext degrees must match"
        );
        assert_eq!(
            ciphertext.b().modulus(),
            modulus,
            "Tweak ciphertext moduli must match"
        );
        assert_eq!(ciphertext.a().degree(), degree);
        assert_eq!(ciphertext.a().modulus(), modulus);
    }

    if n == 1 {
        return vec![ciphertexts[0].clone()];
    }

    // Algorithm 3 progressively doubles the solved subproblem.
    let mut output = vec![ciphertexts[0].clone(); n];
    output[0] = ciphertexts[0].clone();

    let log_n = n.trailing_zeros() as usize;

    for ell in 0..log_n {
        let width = 1usize << ell;
        let stride = n >> (ell + 1);

        let odd_inputs: Vec<RlweCiphertext> = (0..width)
            .map(|j| ciphertexts[(2 * j + 1) * stride].clone())
            .collect();

        let aux = tweak(&odd_inputs);

        // Preserve the previous stage because both branches depend on it.
        let previous: Vec<RlweCiphertext> = output[..width].to_vec();

        for j in 0..width {
            let exponent = j * degree / width;
            let rotated = ciphertext_mul_monomial(&aux[j], exponent);

            output[j] = ciphertext_add(&previous[j], &rotated);
            output[j + width] = ciphertext_sub(&previous[j], &rotated);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ring::{Modulus, Polynomial};
    use crate::rlwe::{decrypt_raw, encrypt_raw_with_rng, RlweParameters, SecretKey};

    use super::*;

    fn direct_tweak(ciphertexts: &[RlweCiphertext]) -> Vec<RlweCiphertext> {
        let n = ciphertexts.len();
        let degree = ciphertexts[0].b().degree();
        let modulus = ciphertexts[0].b().modulus();

        (0..n)
            .map(|j| {
                let mut b = Polynomial::zero(modulus, degree);
                let mut a = Polynomial::zero(modulus, degree);

                for (i, ciphertext) in ciphertexts.iter().enumerate() {
                    let exponent = 2 * i * j * degree / n;

                    b = b.add(&polynomial_mul_monomial(ciphertext.b(), exponent));
                    a = a.add(&polynomial_mul_monomial(ciphertext.a(), exponent));
                }

                RlweCiphertext::new(b, a)
            })
            .collect()
    }

    fn deterministic_ciphertext(modulus: Modulus, degree: usize, seed: u64) -> RlweCiphertext {
        let b = Polynomial::new(
            modulus,
            (0..degree)
                .map(|i| (seed + 3 * i as u64) % modulus.value())
                .collect(),
        );

        let a = Polynomial::new(
            modulus,
            (0..degree)
                .map(|i| (2 * seed + 5 * i as u64) % modulus.value())
                .collect(),
        );

        RlweCiphertext::new(b, a)
    }

    #[test]
    fn monomial_multiplication_respects_negacyclic_wrap() {
        let q = Modulus::new(97);
        let p = Polynomial::new(q, vec![1, 2, 3, 4]);

        assert_eq!(
            polynomial_mul_monomial(&p, 1).coefficients(),
            &[93, 1, 2, 3]
        );

        assert_eq!(polynomial_mul_monomial(&p, 4), p.neg());

        assert_eq!(polynomial_mul_monomial(&p, 8), p);
    }

    #[test]
    fn tweak_matches_direct_definition_for_n1_n2_n4_n8() {
        let q = Modulus::new(12_289);
        let degree = 8;

        for n in [1_usize, 2, 4, 8] {
            let input: Vec<RlweCiphertext> = (0..n)
                .map(|i| deterministic_ciphertext(q, degree, 17 + i as u64))
                .collect();

            let expected = direct_tweak(&input);
            let actual = tweak(&input);

            assert_eq!(
                actual, expected,
                "Park Tweak diverged from direct definition for n={n}"
            );
        }
    }

    #[test]
    fn tweak_preserves_rlwe_decryption_linearity() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 1);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x5041_524b);
        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x0054_5745_414b);

        let ciphertexts: Vec<RlweCiphertext> = (0..8)
            .map(|row| {
                let message = Polynomial::new(
                    params.modulus(),
                    (0..params.degree())
                        .map(|col| {
                            (13 + 7 * row as u64 + 11 * col as u64) % params.modulus().value()
                        })
                        .collect(),
                );

                encrypt_raw_with_rng(params, &secret, &message, &mut encryption_rng)
            })
            .collect();

        let transformed = tweak(&ciphertexts);

        let decrypted_inputs: Vec<Polynomial> = ciphertexts
            .iter()
            .map(|ct| decrypt_raw(params, &secret, ct))
            .collect();

        for (j, ciphertext) in transformed.iter().enumerate() {
            let mut expected = Polynomial::zero(params.modulus(), params.degree());

            for (i, plaintext) in decrypted_inputs.iter().enumerate() {
                let exponent = 2 * i * j * params.degree() / ciphertexts.len();

                expected = expected.add(&polynomial_mul_monomial(plaintext, exponent));
            }

            let actual = decrypt_raw(params, &secret, ciphertext);

            assert_eq!(
                actual, expected,
                "decrypted Park Tweak output diverged at index {j}"
            );
        }
    }
}
