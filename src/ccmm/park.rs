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

/// Multiplicative inverse modulo `modulus`.
///
/// This is used by Park C-MT both for:
///
/// - `N^{-1} mod q`; and
/// - `(2j+1)^{-1} mod 2N`.
fn modular_inverse(value: u64, modulus: u64) -> u64 {
    assert!(modulus > 1, "modulus must exceed one");

    let mut t: i128 = 0;
    let mut new_t: i128 = 1;
    let mut r: i128 = modulus as i128;
    let mut new_r: i128 = (value % modulus) as i128;

    while new_r != 0 {
        let quotient = r / new_r;

        let next_t = t - quotient * new_t;
        t = new_t;
        new_t = next_t;

        let next_r = r - quotient * new_r;
        r = new_r;
        new_r = next_r;
    }

    assert_eq!(r, 1, "value must be invertible modulo modulus");

    let modulus_i128 = modulus as i128;
    let inverse = ((t % modulus_i128) + modulus_i128) % modulus_i128;

    inverse as u64
}

/// Multiplies both RLWE components by a scalar modulo q.
fn ciphertext_scalar_mul(ciphertext: &RlweCiphertext, scalar: u64) -> RlweCiphertext {
    RlweCiphertext::new(
        ciphertext.b().scalar_mul(scalar),
        ciphertext.a().scalar_mul(scalar),
    )
}

/// Returns the Galois key for an odd exponent.
///
/// Exponent one is the identity automorphism and therefore needs no key.
fn galois_key_for_exponent(
    keys: &[crate::ckks::GaloisKey],
    exponent: usize,
) -> &crate::ckks::GaloisKey {
    keys.iter()
        .find(|key| key.exponent() == exponent)
        .unwrap_or_else(|| panic!("missing Park C-MT Galois key for exponent {exponent}"))
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

/// Park Algorithm 4: ciphertext matrix transpose (C-MT).
///
/// Input consists of `N` RLWE ciphertexts over a ring of degree `N`.
/// Ciphertext `i` encrypts one row:
///
/// ```text
/// m_i(X) = sum_j M[i,j] X^j.
/// ```
///
/// The output consists of `N` ciphertexts encrypting the columns:
///
/// ```text
/// m'_j(X) = sum_i M[i,j] X^i.
/// ```
///
/// This implementation follows Algorithm 4 directly:
///
/// 1. `Tweak(X^i * ct_i)`;
/// 2. reindex by `(2j+1)^{-1} mod 2N` and multiply by `N^{-1} mod q`;
/// 3. apply `sigma_(2j+1)` and switch back to the original secret;
/// 4. apply `Tweak` again;
/// 5. apply the final Park monomial/sign correction.
///
/// The identity automorphism (`j = 0`, exponent `1`) does not require
/// a switching key.
pub fn transpose(
    params: crate::rlwe::RlweParameters,
    ciphertexts: &[RlweCiphertext],
    galois_keys: &[crate::ckks::GaloisKey],
) -> Vec<RlweCiphertext> {
    assert!(!ciphertexts.is_empty(), "C-MT requires ciphertexts");

    let degree = params.degree();
    let modulus = params.modulus();

    assert_eq!(
        ciphertexts.len(),
        degree,
        "Park Algorithm 4 requires N ciphertexts for ring degree N"
    );

    assert!(
        degree.is_power_of_two(),
        "Park C-MT requires power-of-two ring degree"
    );

    assert_eq!(
        modulus.value() % 2,
        1,
        "Park C-MT requires N invertible modulo q; CKKS q must be odd"
    );

    for ciphertext in ciphertexts {
        assert_eq!(
            ciphertext.b().degree(),
            degree,
            "C-MT ciphertext degree must match RLWE degree"
        );
        assert_eq!(
            ciphertext.b().modulus(),
            modulus,
            "C-MT ciphertext modulus must match RLWE modulus"
        );
        assert_eq!(ciphertext.a().degree(), degree);
        assert_eq!(ciphertext.a().modulus(), modulus);
    }

    let two_n = 2 * degree;

    // Algorithm 4, Step 1:
    //
    // aux <- Tweak((X^i * ct_i)_i)
    let shifted_inputs: Vec<RlweCiphertext> = ciphertexts
        .iter()
        .enumerate()
        .map(|(i, ciphertext)| ciphertext_mul_monomial(ciphertext, i))
        .collect();

    let aux = tweak(&shifted_inputs);

    // Algorithm 4, Steps 2--5.
    let n_inverse = modular_inverse(degree as u64, modulus.value());

    let mut transformed = Vec::with_capacity(degree);

    for j in 0..degree {
        let exponent = 2 * j + 1;

        // index =
        //   (-1 + (2j+1)^(-1) mod 2N) / 2
        let inverse_exponent = modular_inverse(exponent as u64, two_n as u64) as usize;

        assert_eq!(
            inverse_exponent % 2,
            1,
            "inverse of odd C-MT exponent must remain odd"
        );

        let source_index = (inverse_exponent - 1) / 2;

        let normalized = ciphertext_scalar_mul(&aux[source_index], n_inverse);

        let automorphed = if exponent == 1 {
            normalized
        } else {
            let key = galois_key_for_exponent(galois_keys, exponent);

            crate::ckks::apply_galois_automorphism(params, &normalized, key)
        };

        transformed.push(automorphed);
    }

    // Algorithm 4, Step 6.
    let second_tweak = tweak(&transformed);

    // Algorithm 4, Steps 7--9:
    //
    // ct'_(j mod N) =
    //   -X^(N-j) * ct''_((N-j) mod N)
    //
    // for j = 1..N.
    let zero = RlweCiphertext::new(
        Polynomial::zero(modulus, degree),
        Polynomial::zero(modulus, degree),
    );

    let mut output = vec![zero; degree];

    // Handle the constant-coefficient column explicitly.
    output[0] = second_tweak[0].clone();

    // Apply the negacyclic correction to the remaining columns.
    for (output_index, output_slot) in output.iter_mut().enumerate().take(degree).skip(1) {
        let source_index = degree - output_index;
        let exponent = degree - output_index;

        let corrected = ciphertext_mul_monomial(&second_tweak[source_index], exponent);

        *output_slot = ciphertext_scalar_mul(&corrected, modulus.value() - 1);
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

    fn park_galois_keys(
        params: RlweParameters,
        secret: &SecretKey,
        seed: u64,
    ) -> Vec<crate::ckks::GaloisKey> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        (1..params.degree())
            .map(|j| {
                let exponent = 2 * j + 1;

                crate::ckks::GaloisKey::generate_with_rng(params, secret, exponent, 16, &mut rng)
            })
            .collect()
    }

    fn encrypt_coefficient_rows(
        params: RlweParameters,
        secret: &SecretKey,
        rows: &[Vec<u64>],
        seed: u64,
    ) -> Vec<RlweCiphertext> {
        assert_eq!(rows.len(), params.degree());

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        rows.iter()
            .map(|row| {
                assert_eq!(row.len(), params.degree());

                let message = Polynomial::new(params.modulus(), row.clone());

                encrypt_raw_with_rng(params, secret, &message, &mut rng)
            })
            .collect()
    }

    fn decrypt_coefficient_rows(
        params: RlweParameters,
        secret: &SecretKey,
        ciphertexts: &[RlweCiphertext],
    ) -> Vec<Vec<u64>> {
        ciphertexts
            .iter()
            .map(|ciphertext| {
                decrypt_raw(params, secret, ciphertext)
                    .coefficients()
                    .to_vec()
            })
            .collect()
    }

    fn transpose_cleartext(matrix: &[Vec<u64>]) -> Vec<Vec<u64>> {
        let n = matrix.len();

        assert!(matrix.iter().all(|row| row.len() == n));

        (0..n)
            .map(|col| (0..n).map(|row| matrix[row][col]).collect())
            .collect()
    }

    #[test]
    fn modular_inverse_matches_expected_values() {
        assert_eq!(modular_inverse(8, 12_289), 10_753);

        for exponent in [1_u64, 3, 5, 7, 9, 11, 13, 15] {
            let inverse = modular_inverse(exponent, 16);

            assert_eq!((exponent * inverse) % 16, 1);
        }
    }

    /// Builds the ordinary N x N coefficient matrix for multiplication by
    /// `polynomial` in Z_q[X]/(X^N + 1).
    ///
    /// Column j is the coefficient vector of polynomial * X^j.
    fn negacyclic_multiplication_matrix(polynomial: &Polynomial) -> Vec<Vec<u64>> {
        let degree = polynomial.degree();
        let modulus = polynomial.modulus();

        let mut matrix = vec![vec![0_u64; degree]; degree];

        for column in 0..degree {
            let mut monomial_coefficients = vec![0_u64; degree];
            monomial_coefficients[column] = 1;

            let monomial = Polynomial::new(modulus, monomial_coefficients);
            let product = polynomial.negacyclic_mul(&monomial);

            for (row, &coefficient) in product.coefficients().iter().enumerate() {
                matrix[row][column] = coefficient;
            }
        }

        matrix
    }

    fn modular_matrix_mul(lhs: &[Vec<u64>], rhs: &[Vec<u64>], modulus: u64) -> Vec<Vec<u64>> {
        let rows = lhs.len();
        let inner = lhs[0].len();
        let cols = rhs[0].len();

        assert_eq!(rhs.len(), inner);

        let mut result = vec![vec![0_u64; cols]; rows];

        for row in 0..rows {
            for col in 0..cols {
                let mut accumulator = 0_u128;

                for k in 0..inner {
                    accumulator += (lhs[row][k] as u128) * (rhs[k][col] as u128);
                    accumulator %= modulus as u128;
                }

                result[row][col] = accumulator as u64;
            }
        }

        result
    }

    fn modular_matrix_add(lhs: &[Vec<u64>], rhs: &[Vec<u64>], modulus: u64) -> Vec<Vec<u64>> {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs[0].len(), rhs[0].len());

        lhs.iter()
            .zip(rhs)
            .map(|(lhs_row, rhs_row)| {
                lhs_row
                    .iter()
                    .zip(rhs_row)
                    .map(|(&lhs_value, &rhs_value)| {
                        ((lhs_value as u128 + rhs_value as u128) % modulus as u128) as u64
                    })
                    .collect()
            })
            .collect()
    }

    fn modular_matrix_transpose(matrix: &[Vec<u64>]) -> Vec<Vec<u64>> {
        let rows = matrix.len();
        let cols = matrix[0].len();

        let mut result = vec![vec![0_u64; rows]; cols];

        for (row, matrix_row) in matrix.iter().enumerate() {
            for (col, &value) in matrix_row.iter().enumerate() {
                result[col][row] = value;
            }
        }

        result
    }

    fn ciphertext_component_matrix(
        ciphertexts: &[RlweCiphertext],
        select_a: bool,
    ) -> Vec<Vec<u64>> {
        let degree = ciphertexts.len();
        let mut matrix = vec![vec![0_u64; degree]; degree];

        // Ciphertext j is interpreted as coefficient column j.
        for (column, ciphertext) in ciphertexts.iter().enumerate() {
            let polynomial = if select_a {
                ciphertext.a()
            } else {
                ciphertext.b()
            };

            for (row, &coefficient) in polynomial.coefficients().iter().enumerate() {
                matrix[row][column] = coefficient;
            }
        }

        matrix
    }

    fn matrix_from_ciphertext_columns(
        ciphertexts: &[RlweCiphertext],
        select_a: bool,
    ) -> Vec<Vec<u64>> {
        ciphertext_component_matrix(ciphertexts, select_a)
    }

    fn matrix_from_ciphertext_rows(
        ciphertexts: &[RlweCiphertext],
        select_a: bool,
    ) -> Vec<Vec<u64>> {
        modular_matrix_transpose(&ciphertext_component_matrix(ciphertexts, select_a))
    }

    fn synthetic_a_bundle_from_rows(matrix: &[Vec<u64>], modulus: Modulus) -> Vec<RlweCiphertext> {
        let degree = matrix.len();
        assert_eq!(matrix[0].len(), degree);

        (0..degree)
            .map(|row| {
                let a = Polynomial::new(modulus, matrix[row].clone());
                let b = Polynomial::zero(modulus, degree);
                RlweCiphertext::new(b, a)
            })
            .collect()
    }

    #[test]
    fn park_cmt_component_identity_for_synthetic_bundle() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x434d_545f_434f_4d50);
        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let galois_keys = park_galois_keys(params, &secret, 0x434d_545f_4b45_5953);

        let degree = params.degree();
        let modulus = params.modulus();
        let q = modulus.value();

        // Arbitrary, asymmetric ordinary matrix C.
        let c: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        (7 + 11 * row as u64 + 17 * col as u64 + 5 * row as u64 * col as u64) % q
                    })
                    .collect()
            })
            .collect();

        // Construct Park's synthetic pair (C, 0).
        //
        // Park notation is (A,B), whereas FHE-rs stores (b,a).
        // Therefore each ciphertext column is (b=0, a=C[:,j]).
        let synthetic: Vec<RlweCiphertext> = (0..degree)
            .map(|column| {
                let a = Polynomial::new(modulus, (0..degree).map(|row| c[row][column]).collect());

                let b = Polynomial::zero(modulus, degree);

                RlweCiphertext::new(b, a)
            })
            .collect();

        let transformed = transpose(params, &synthetic, &galois_keys);

        // FHE-rs component matrices after C-MT.
        let d0 = ciphertext_component_matrix(&transformed, true);
        let d1 = ciphertext_component_matrix(&transformed, false);

        // S* is defined operationally: the matrix representing
        // multiplication by s(X) in the negacyclic ring.
        let s_star = negacyclic_multiplication_matrix(secret.polynomial());
        let s_star_transpose = modular_matrix_transpose(&s_star);

        // Park Algorithm 8 requires:
        //
        // Because C is packed by columns, C-MT gives:
        //
        //     C^T S*^T = S* D0 + D1.
        //
        let lhs = modular_matrix_mul(&modular_matrix_transpose(&c), &s_star_transpose, q);

        let rhs = modular_matrix_add(&modular_matrix_mul(&s_star, &d0, q), &d1, q);

        assert_eq!(
            lhs, rhs,
            "C-MT component identity C^T*S*^T = S*D0 + D1 failed"
        );
    }

    #[test]
    fn park_algorithm8_representation_flow_matches_matrix_algebra() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5041_524b_5033_4101);
        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let galois_keys = park_galois_keys(params, &secret, 0x5041_524b_5033_4102);

        let degree = params.degree();
        let modulus = params.modulus();
        let q = modulus.value();

        let lhs_plain: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        (3 + 7 * row as u64 + 11 * col as u64 + 2 * row as u64 * col as u64) % q
                    })
                    .collect()
            })
            .collect();

        let rhs_plain: Vec<Vec<u64>> = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|col| {
                        (5 + 13 * row as u64 + 17 * col as u64 + 3 * row as u64 * col as u64) % q
                    })
                    .collect()
            })
            .collect();

        let lhs = encrypt_coefficient_rows(params, &secret, &lhs_plain, 0x5041_524b_5033_4103);

        let rhs = encrypt_coefficient_rows(params, &secret, &rhs_plain, 0x5041_524b_5033_4104);

        // Existing C-MT contract:
        // row-packed RHS -> column-packed RHS^T.
        let rhs_cmt = transpose(params, &rhs, &galois_keys);

        // LHS input is row-packed, so its raw component matrices are read
        // by rows. The C-MT output is column-packed, so its components are
        // read by columns.
        let a = matrix_from_ciphertext_rows(&lhs, true);
        let b = matrix_from_ciphertext_rows(&lhs, false);

        let a_tilde = matrix_from_ciphertext_columns(&rhs_cmt, true);
        let b_tilde = matrix_from_ciphertext_columns(&rhs_cmt, false);

        let c00 = modular_matrix_mul(&a, &a_tilde, q);
        let c01 = modular_matrix_mul(&a, &b_tilde, q);
        let c10 = modular_matrix_mul(&b, &a_tilde, q);
        let c11 = modular_matrix_mul(&b, &b_tilde, q);

        // Algorithm 8 Steps 3-4 require C00 and C10 to enter C-MT
        // by rows, not by columns.
        let c00_bundle = synthetic_a_bundle_from_rows(&c00, modulus);
        let c10_bundle = synthetic_a_bundle_from_rows(&c10, modulus);

        let d01 = transpose(params, &c00_bundle, &galois_keys);
        let d23 = transpose(params, &c10_bundle, &galois_keys);

        let d0 = matrix_from_ciphertext_columns(&d01, true);
        let d1 = matrix_from_ciphertext_columns(&d01, false);
        let d2 = matrix_from_ciphertext_columns(&d23, true);
        let d3 = matrix_from_ciphertext_columns(&d23, false);

        let s_star = negacyclic_multiplication_matrix(secret.polynomial());
        let s_star_t = modular_matrix_transpose(&s_star);

        // These are precisely the two component identities required by
        // Algorithm 8 after its second and third C-MTs.
        assert_eq!(
            modular_matrix_mul(&c00, &s_star_t, q),
            modular_matrix_add(&modular_matrix_mul(&s_star, &d0, q), &d1, q,),
            "C00 row-packed C-MT identity failed",
        );

        assert_eq!(
            modular_matrix_mul(&c10, &s_star_t, q),
            modular_matrix_add(&modular_matrix_mul(&s_star, &d2, q), &d3, q,),
            "C10 row-packed C-MT identity failed",
        );

        // Keep the four PP-MM outputs live and dimension-checked.
        assert_eq!(c01.len(), degree);
        assert_eq!(c11.len(), degree);
        assert!(c01.iter().all(|row| row.len() == degree));
        assert!(c11.iter().all(|row| row.len() == degree));
    }
    #[test]
    fn park_cmt_transposes_asymmetric_matrix_exactly() {
        // Noise-free parameters isolate Park Algorithm 4 algebra from
        // approximation/noise behavior.
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x5041_524b_434d_5401);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let keys = park_galois_keys(params, &secret, 0x5041_524b_434d_5402);

        let matrix: Vec<Vec<u64>> = (0..8)
            .map(|row| (0..8).map(|col| 1 + 10 * row as u64 + col as u64).collect())
            .collect();

        let ciphertexts = encrypt_coefficient_rows(params, &secret, &matrix, 0x5041_524b_434d_5403);

        let transposed = transpose(params, &ciphertexts, &keys);

        let actual = decrypt_coefficient_rows(params, &secret, &transposed);

        let expected = transpose_cleartext(&matrix);

        assert_eq!(
            actual, expected,
            "Park C-MT failed asymmetric 8x8 transpose"
        );
    }

    #[test]
    fn park_cmt_maps_basis_matrix_eij_to_eji() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x5041_524b_434d_5411);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let keys = park_galois_keys(params, &secret, 0x5041_524b_434d_5412);

        for source_row in 0..8 {
            for source_col in 0..8 {
                let mut matrix = vec![vec![0_u64; 8]; 8];
                matrix[source_row][source_col] = 1;

                let ciphertexts = encrypt_coefficient_rows(
                    params,
                    &secret,
                    &matrix,
                    0x5041_524b_434d_5500 + (8 * source_row + source_col) as u64,
                );

                let transposed = transpose(params, &ciphertexts, &keys);

                let actual = decrypt_coefficient_rows(params, &secret, &transposed);

                let expected = transpose_cleartext(&matrix);

                assert_eq!(
                    actual, expected,
                    "Park C-MT failed basis E_{{{source_row},{source_col}}}"
                );
            }
        }
    }

    #[test]
    fn park_cmt_is_involution_on_encrypted_matrix() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x5041_524b_434d_5421);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let keys = park_galois_keys(params, &secret, 0x5041_524b_434d_5422);

        let matrix: Vec<Vec<u64>> = (0..8)
            .map(|row| {
                (0..8)
                    .map(|col| {
                        (37 + 19 * row as u64 + 23 * col as u64 + 3 * row as u64 * col as u64)
                            % params.modulus().value()
                    })
                    .collect()
            })
            .collect();

        let ciphertexts = encrypt_coefficient_rows(params, &secret, &matrix, 0x5041_524b_434d_5423);

        let once = transpose(params, &ciphertexts, &keys);

        let twice = transpose(params, &once, &keys);

        let actual = decrypt_coefficient_rows(params, &secret, &twice);

        assert_eq!(actual, matrix, "Park C-MT involution failed");
    }
}
