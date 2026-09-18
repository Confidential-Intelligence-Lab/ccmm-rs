use rand::{CryptoRng, RngCore};

use crate::ckks::{
    decrypt_decode, encrypt_with_rng, rescale_to_next, CkksCiphertext, CkksParameters,
};
use crate::eval::{relinearize, MultiplicationKey};
use crate::matrix::{BatchMatrix, PolynomialMatrix};
use crate::rlwe::{RlweCiphertext, RlweQuadraticCiphertext, SecretKey};

/// Matrix ciphertext represented by the two RLWE components `(B, A)`.
#[derive(Debug, Clone, PartialEq)]
pub struct CkksMatrixCiphertext {
    b: PolynomialMatrix,
    a: PolynomialMatrix,
    level: usize,
    scale: f64,
}

impl CkksMatrixCiphertext {
    fn new(b: PolynomialMatrix, a: PolynomialMatrix, level: usize, scale: f64) -> Self {
        assert_eq!(b.rows(), a.rows());
        assert_eq!(b.cols(), a.cols());
        assert_eq!(b.modulus(), a.modulus());
        assert_eq!(b.ring_degree(), a.ring_degree());
        assert!(scale.is_finite() && scale > 0.0);

        Self { b, a, level, scale }
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

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

/// Matrix-valued degree-2 product before relinearization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CkksMatrixQuadraticProduct {
    c0: PolynomialMatrix,
    c1: PolynomialMatrix,
    c2: PolynomialMatrix,
}

impl CkksMatrixQuadraticProduct {
    pub fn c0(&self) -> &PolynomialMatrix {
        &self.c0
    }

    pub fn c1(&self) -> &PolynomialMatrix {
        &self.c1
    }

    pub fn c2(&self) -> &PolynomialMatrix {
        &self.c2
    }
}

/// Encrypts a single-batch real matrix.
///
/// Each logical matrix entry is encoded as the constant coefficient of
/// its own CKKS plaintext polynomial.
pub fn encrypt_ckks_matrix_with_rng<R>(
    params: CkksParameters,
    secret_key: &SecretKey,
    matrix: &BatchMatrix<f64>,
    rng: &mut R,
) -> CkksMatrixCiphertext
where
    R: RngCore + CryptoRng,
{
    assert_eq!(
        matrix.batches(),
        1,
        "native CKKS matrix encryption currently supports one batch"
    );

    let mut b = PolynomialMatrix::new(
        matrix.rows(),
        matrix.cols(),
        params.high_modulus(),
        params.degree(),
    );

    let mut a = PolynomialMatrix::new(
        matrix.rows(),
        matrix.cols(),
        params.high_modulus(),
        params.degree(),
    );

    for col in 0..matrix.cols() {
        for row in 0..matrix.rows() {
            let mut coefficients = vec![0.0; params.degree()];
            coefficients[0] = *matrix.get(0, row, col);

            let ciphertext = encrypt_with_rng(params, secret_key, &coefficients, rng);

            b.set(row, col, ciphertext.rlwe().b().clone());
            a.set(row, col, ciphertext.rlwe().a().clone());
        }
    }

    CkksMatrixCiphertext::new(b, a, 1, params.initial_scale())
}

/// Forms the native matrix degree-2 CCMM product:
///
/// `c0 = BD`
///
/// `c1 = BC + AD`
///
/// `c2 = AC`.
pub fn ccmm_quadratic(
    lhs: &CkksMatrixCiphertext,
    rhs: &CkksMatrixCiphertext,
) -> CkksMatrixQuadraticProduct {
    assert_eq!(lhs.level(), rhs.level(), "ciphertext levels must match");
    assert_eq!(lhs.level(), 1, "CCMM multiplication requires level 1");
    assert_eq!(
        lhs.cols(),
        rhs.rows(),
        "ciphertext matrix dimensions must be compatible"
    );
    assert_eq!(
        lhs.b().modulus(),
        rhs.b().modulus(),
        "ciphertext matrix moduli must match"
    );
    assert_eq!(
        lhs.b().ring_degree(),
        rhs.b().ring_degree(),
        "ciphertext matrix ring degrees must match"
    );

    let bd = lhs.b().matmul(rhs.b());
    let bc = lhs.b().matmul(rhs.a());
    let ad = lhs.a().matmul(rhs.b());
    let ac = lhs.a().matmul(rhs.a());

    CkksMatrixQuadraticProduct {
        c0: bd,
        c1: bc.add(&ad),
        c2: ac,
    }
}

/// Native CKKS ciphertext-ciphertext matrix multiplication.
///
/// The operation performs:
///
/// 1. four polynomial-matrix products;
/// 2. degree-2 combination `(BD, BC + AD, AC)`;
/// 3. entrywise RLWE relinearization;
/// 4. entrywise `Q -> q` CKKS rescaling.
pub fn ccmm(
    params: CkksParameters,
    lhs: &CkksMatrixCiphertext,
    rhs: &CkksMatrixCiphertext,
    multiplication_key: &MultiplicationKey,
) -> CkksMatrixCiphertext {
    let quadratic = ccmm_quadratic(lhs, rhs);

    let rows = quadratic.c0().rows();
    let cols = quadratic.c0().cols();

    let mut b = PolynomialMatrix::new(rows, cols, params.low_modulus(), params.degree());

    let mut a = PolynomialMatrix::new(rows, cols, params.low_modulus(), params.degree());

    let product_scale = lhs.scale() * rhs.scale();

    for col in 0..cols {
        for row in 0..rows {
            let degree_two = RlweQuadraticCiphertext::new(
                quadratic.c0().get(row, col).clone(),
                quadratic.c1().get(row, col).clone(),
                quadratic.c2().get(row, col).clone(),
            );

            let relinearized = relinearize(params.high_rlwe(), &degree_two, multiplication_key);

            let wrapped = CkksCiphertext::new(relinearized, 1, product_scale);

            let rescaled = rescale_to_next(params, &wrapped);

            b.set(row, col, rescaled.rlwe().b().clone());
            a.set(row, col, rescaled.rlwe().a().clone());
        }
    }

    CkksMatrixCiphertext::new(b, a, 0, product_scale / params.rescale_prime() as f64)
}

/// Decrypts a native CKKS matrix ciphertext into one real scalar per
/// matrix entry.
///
/// Since matrix entries are encoded as constant polynomials, coefficient
/// zero is the logical scalar result.
pub fn decrypt_ckks_matrix(
    params: CkksParameters,
    secret_key: &SecretKey,
    ciphertext: &CkksMatrixCiphertext,
) -> BatchMatrix<f64> {
    let mut output = BatchMatrix::<f64>::new(ciphertext.rows(), ciphertext.cols(), 1);

    for col in 0..ciphertext.cols() {
        for row in 0..ciphertext.rows() {
            let rlwe = RlweCiphertext::new(
                ciphertext.b().get(row, col).clone(),
                ciphertext.a().get(row, col).clone(),
            );

            let scalar = CkksCiphertext::new(rlwe, ciphertext.level(), ciphertext.scale());

            let decoded = decrypt_decode(params, secret_key, &scalar);

            output.set(0, row, col, decoded[0]);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ckks::project_secret_key_to_low;

    use super::*;

    fn params() -> CkksParameters {
        CkksParameters::new(8, 2_147_483_647, 65_537, 65_537.0, 1)
    }

    fn matrix(values_column_major: [f64; 4]) -> BatchMatrix<f64> {
        BatchMatrix::from_vec_column_major(2, 2, 1, values_column_major.to_vec())
    }

    fn reference_matmul(lhs: &BatchMatrix<f64>, rhs: &BatchMatrix<f64>) -> BatchMatrix<f64> {
        assert_eq!(lhs.batches(), 1);
        assert_eq!(rhs.batches(), 1);
        assert_eq!(lhs.cols(), rhs.rows());

        let mut out = BatchMatrix::<f64>::new(lhs.rows(), rhs.cols(), 1);

        for col in 0..rhs.cols() {
            for row in 0..lhs.rows() {
                let mut sum = 0.0;

                for k in 0..lhs.cols() {
                    sum += *lhs.get(0, row, k) * *rhs.get(0, k, col);
                }

                out.set(0, row, col, sum);
            }
        }

        out
    }

    fn assert_matrix_close(actual: &BatchMatrix<f64>, expected: &BatchMatrix<f64>, tolerance: f64) {
        assert_eq!(actual.rows(), expected.rows());
        assert_eq!(actual.cols(), expected.cols());

        for col in 0..actual.cols() {
            for row in 0..actual.rows() {
                let actual = *actual.get(0, row, col);
                let expected = *expected.get(0, row, col);
                let error = (actual - expected).abs();

                assert!(
                    error <= tolerance,
                    "({row},{col}): actual={actual}, expected={expected}, \
                     error={error}, tolerance={tolerance}"
                );
            }
        }
    }

    #[test]
    fn quadratic_matrix_product_has_expected_structure() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(1);
        let secret = SecretKey::generate_with_rng(params.high_rlwe(), &mut key_rng);

        let lhs = matrix([1.0, 3.0, 2.0, 4.0]);
        let rhs = matrix([5.0, 7.0, 6.0, 8.0]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(2);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(3);

        let lhs_ct = encrypt_ckks_matrix_with_rng(params, &secret, &lhs, &mut lhs_rng);

        let rhs_ct = encrypt_ckks_matrix_with_rng(params, &secret, &rhs, &mut rhs_rng);

        let quadratic = ccmm_quadratic(&lhs_ct, &rhs_ct);

        assert_eq!(quadratic.c0(), &lhs_ct.b().matmul(rhs_ct.b()));

        assert_eq!(
            quadratic.c1(),
            &lhs_ct
                .b()
                .matmul(rhs_ct.a())
                .add(&lhs_ct.a().matmul(rhs_ct.b()))
        );

        assert_eq!(quadratic.c2(), &lhs_ct.a().matmul(rhs_ct.a()));
    }

    #[test]
    fn deterministic_native_ccmm_matches_2x2_reference() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(10);
        let high_secret = SecretKey::generate_with_rng(params.high_rlwe(), &mut key_rng);

        let low_secret = project_secret_key_to_low(params, &high_secret);

        let lhs = matrix([1.0, 3.0, 2.0, 4.0]);
        let rhs = matrix([5.0, 7.0, 6.0, 8.0]);

        let expected = matrix([19.0, 43.0, 22.0, 50.0]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(11);
        let mut rhs_rng = ChaCha20Rng::seed_from_u64(12);

        let lhs_ct = encrypt_ckks_matrix_with_rng(params, &high_secret, &lhs, &mut lhs_rng);

        let rhs_ct = encrypt_ckks_matrix_with_rng(params, &high_secret, &rhs, &mut rhs_rng);

        let mut eval_rng = ChaCha20Rng::seed_from_u64(13);

        let multiplication_key = MultiplicationKey::generate_with_rng(
            params.high_rlwe(),
            &high_secret,
            65_536,
            &mut eval_rng,
        );

        let result = ccmm(params, &lhs_ct, &rhs_ct, &multiplication_key);

        assert_eq!(result.level(), 0);
        assert_eq!(result.b().modulus(), params.low_modulus());

        let actual = decrypt_ckks_matrix(params, &low_secret, &result);

        assert_matrix_close(&actual, &expected, 1.0e-2);
    }

    #[test]
    fn native_ccmm_matches_cleartext_matrix_multiplication() {
        let params = params();

        let lhs = matrix([0.5, -0.25, 1.25, 0.75]);

        let rhs = matrix([1.5, 0.25, -0.5, 2.0]);

        let expected = reference_matmul(&lhs, &rhs);

        for seed in 0_u64..16 {
            let mut key_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1000);

            let high_secret = SecretKey::generate_with_rng(params.high_rlwe(), &mut key_rng);

            let low_secret = project_secret_key_to_low(params, &high_secret);

            let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x2000);

            let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x3000);

            let lhs_ct = encrypt_ckks_matrix_with_rng(params, &high_secret, &lhs, &mut lhs_rng);

            let rhs_ct = encrypt_ckks_matrix_with_rng(params, &high_secret, &rhs, &mut rhs_rng);

            let mut eval_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x4000);

            let multiplication_key = MultiplicationKey::generate_with_rng(
                params.high_rlwe(),
                &high_secret,
                65_536,
                &mut eval_rng,
            );

            let result = ccmm(params, &lhs_ct, &rhs_ct, &multiplication_key);

            let actual = decrypt_ckks_matrix(params, &low_secret, &result);

            assert_matrix_close(&actual, &expected, 1.5e-2);
        }
    }
}
