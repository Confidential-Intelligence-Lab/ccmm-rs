use rand::{CryptoRng, RngCore};

use crate::matrix::PolynomialMatrix;
use crate::rlwe::{decrypt_raw, encrypt_raw_with_rng, RlweParameters, SecretKey};

/// RLWE-encrypted matrix represented component-wise as `(B, A)`.
///
/// Each logical matrix entry is an RLWE ciphertext:
///
/// `B_ij + A_ij * s`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlweMatrixCiphertext {
    b: PolynomialMatrix,
    a: PolynomialMatrix,
}

impl RlweMatrixCiphertext {
    pub fn new(b: PolynomialMatrix, a: PolynomialMatrix) -> Self {
        assert_eq!(
            b.rows(),
            a.rows(),
            "ciphertext matrix components must have matching row counts"
        );
        assert_eq!(
            b.cols(),
            a.cols(),
            "ciphertext matrix components must have matching column counts"
        );
        assert_eq!(
            b.modulus(),
            a.modulus(),
            "ciphertext matrix components must have matching moduli"
        );
        assert_eq!(
            b.ring_degree(),
            a.ring_degree(),
            "ciphertext matrix components must have matching ring degrees"
        );

        Self { b, a }
    }

    pub fn b(&self) -> &PolynomialMatrix {
        &self.b
    }

    pub fn a(&self) -> &PolynomialMatrix {
        &self.a
    }

    pub fn rows(&self) -> usize {
        self.b.rows()
    }

    pub fn cols(&self) -> usize {
        self.b.cols()
    }
}

/// Encrypts every polynomial matrix entry under the same RLWE secret key.
pub fn encrypt_matrix_with_rng<R>(
    params: RlweParameters,
    secret_key: &SecretKey,
    plaintext: &PolynomialMatrix,
    rng: &mut R,
) -> RlweMatrixCiphertext
where
    R: RngCore + CryptoRng,
{
    assert_eq!(
        plaintext.modulus(),
        params.modulus(),
        "plaintext matrix modulus must match RLWE parameters"
    );
    assert_eq!(
        plaintext.ring_degree(),
        params.degree(),
        "plaintext matrix ring degree must match RLWE parameters"
    );

    let mut b = PolynomialMatrix::new(
        plaintext.rows(),
        plaintext.cols(),
        params.modulus(),
        params.degree(),
    );

    let mut a = PolynomialMatrix::new(
        plaintext.rows(),
        plaintext.cols(),
        params.modulus(),
        params.degree(),
    );

    for col in 0..plaintext.cols() {
        for row in 0..plaintext.rows() {
            let ciphertext = encrypt_raw_with_rng(params, secret_key, plaintext.get(row, col), rng);

            b.set(row, col, ciphertext.b().clone());
            a.set(row, col, ciphertext.a().clone());
        }
    }

    RlweMatrixCiphertext::new(b, a)
}

/// Decrypts an RLWE matrix ciphertext without higher-level decoding.
pub fn decrypt_matrix_raw(
    params: RlweParameters,
    secret_key: &SecretKey,
    ciphertext: &RlweMatrixCiphertext,
) -> PolynomialMatrix {
    assert_eq!(
        ciphertext.b().modulus(),
        params.modulus(),
        "ciphertext matrix modulus must match RLWE parameters"
    );
    assert_eq!(
        ciphertext.b().ring_degree(),
        params.degree(),
        "ciphertext matrix ring degree must match RLWE parameters"
    );

    let mut plaintext = PolynomialMatrix::new(
        ciphertext.rows(),
        ciphertext.cols(),
        params.modulus(),
        params.degree(),
    );

    for col in 0..ciphertext.cols() {
        for row in 0..ciphertext.rows() {
            let scalar_ciphertext = crate::rlwe::RlweCiphertext::new(
                ciphertext.b().get(row, col).clone(),
                ciphertext.a().get(row, col).clone(),
            );

            plaintext.set(
                row,
                col,
                decrypt_raw(params, secret_key, &scalar_ciphertext),
            );
        }
    }

    plaintext
}

/// Native ciphertext-plaintext matrix multiplication.
///
/// For an encrypted matrix `(B, A)` and plaintext polynomial matrix `U`,
///
/// `(B, A) U = (BU, AU)`.
pub fn cpmm(
    ciphertext: &RlweMatrixCiphertext,
    plaintext: &PolynomialMatrix,
) -> RlweMatrixCiphertext {
    assert_eq!(
        ciphertext.cols(),
        plaintext.rows(),
        "ciphertext/plaintext matrix dimensions must be compatible"
    );
    assert_eq!(
        ciphertext.b().modulus(),
        plaintext.modulus(),
        "ciphertext/plaintext matrix moduli must match"
    );
    assert_eq!(
        ciphertext.b().ring_degree(),
        plaintext.ring_degree(),
        "ciphertext/plaintext matrix ring degrees must match"
    );

    RlweMatrixCiphertext::new(
        ciphertext.b().matmul(plaintext),
        ciphertext.a().matmul(plaintext),
    )
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ring::{Modulus, Polynomial};

    use super::*;

    fn constant(modulus: Modulus, degree: usize, value: u64) -> Polynomial {
        let mut coefficients = vec![0; degree];
        coefficients[0] = value;
        Polynomial::new(modulus, coefficients)
    }

    fn matrix_2x2(
        modulus: Modulus,
        degree: usize,
        values_column_major: [u64; 4],
    ) -> PolynomialMatrix {
        PolynomialMatrix::from_vec_column_major(
            2,
            2,
            values_column_major
                .into_iter()
                .map(|value| constant(modulus, degree, value))
                .collect(),
        )
    }

    #[test]
    fn encrypted_matrix_roundtrip_matches_scalar_rlwe_semantics() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 1);

        let mut key_rng = ChaCha20Rng::seed_from_u64(1);
        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let plaintext = matrix_2x2(params.modulus(), params.degree(), [1, 3, 2, 4]);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(2);

        let ciphertext = encrypt_matrix_with_rng(params, &secret, &plaintext, &mut encryption_rng);

        let decrypted = decrypt_matrix_raw(params, &secret, &ciphertext);

        assert_eq!(decrypted.rows(), 2);
        assert_eq!(decrypted.cols(), 2);

        // Encryption noise may perturb each constant coefficient by at
        // most one, but the matrix shape and ring representation remain
        // intact.
        for col in 0..2 {
            for row in 0..2 {
                let expected = plaintext.get(row, col).coefficient(0);
                let actual = decrypted.get(row, col).coefficient(0);
                let q = params.modulus().value();

                let forward = (actual + q - expected) % q;
                let backward = (expected + q - actual) % q;

                assert!(forward.min(backward) <= 1);
            }
        }
    }

    #[test]
    fn cpmm_decryption_matches_decrypted_ciphertext_times_plaintext_exactly() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 1);

        let mut key_rng = ChaCha20Rng::seed_from_u64(10);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let message = matrix_2x2(params.modulus(), params.degree(), [1, 3, 2, 4]);

        let plaintext_multiplier = matrix_2x2(params.modulus(), params.degree(), [5, 7, 6, 8]);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(11);

        let ciphertext = encrypt_matrix_with_rng(params, &secret, &message, &mut encryption_rng);

        let decrypted_before = decrypt_matrix_raw(params, &secret, &ciphertext);

        let expected = decrypted_before.matmul(&plaintext_multiplier);

        let result = cpmm(&ciphertext, &plaintext_multiplier);

        let actual = decrypt_matrix_raw(params, &secret, &result);

        assert_eq!(actual, expected);
    }

    #[test]
    fn deterministic_noise_free_cpmm_matches_matrix_product() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 0);

        let mut key_rng = ChaCha20Rng::seed_from_u64(20);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        // M = [[1, 2],
        //      [3, 4]]
        let message = matrix_2x2(params.modulus(), params.degree(), [1, 3, 2, 4]);

        // U = [[5, 6],
        //      [7, 8]]
        let multiplier = matrix_2x2(params.modulus(), params.degree(), [5, 7, 6, 8]);

        let expected = matrix_2x2(params.modulus(), params.degree(), [19, 43, 22, 50]);

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(21);

        let ciphertext = encrypt_matrix_with_rng(params, &secret, &message, &mut encryption_rng);

        let result = cpmm(&ciphertext, &multiplier);

        let actual = decrypt_matrix_raw(params, &secret, &result);

        assert_eq!(actual, expected);
    }

    #[test]
    fn cpmm_seeded_campaign_preserves_homomorphic_identity() {
        let params = RlweParameters::new(8, Modulus::new(12_289), 16, 1);

        let message = matrix_2x2(params.modulus(), params.degree(), [1, 3, 2, 4]);

        let multiplier = matrix_2x2(params.modulus(), params.degree(), [5, 7, 6, 8]);

        for seed in 0_u64..32 {
            let mut key_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1000);

            let secret = SecretKey::generate_with_rng(params, &mut key_rng);

            let mut encryption_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x2000);

            let ciphertext =
                encrypt_matrix_with_rng(params, &secret, &message, &mut encryption_rng);

            let expected = decrypt_matrix_raw(params, &secret, &ciphertext).matmul(&multiplier);

            let actual = decrypt_matrix_raw(params, &secret, &cpmm(&ciphertext, &multiplier));

            assert_eq!(actual, expected, "CPMM identity failed for seed {seed}");
        }
    }
}
