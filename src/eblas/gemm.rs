//! Executable eBLAS GEMM front-end.
//!
//! The functions in this module deliberately reuse already validated kernels.
//! R3.5b adds dispatch and semantic checks; it does not introduce new
//! cryptographic arithmetic.

use crate::eblas::{GemmBackend, GemmSpec, MatrixLayout, PrivacyMode};
use crate::grafting::BoundedRnsMultiplicationKey;
use crate::matrix::{BatchMatrix, RnsCkksCiphertextMatrix, RnsCkksPlaintextMatrix};
use crate::ring::{ModulusChain, RnsNttPlan};

fn assert_reference_shapes(
    spec: GemmSpec,
    lhs_rows: usize,
    lhs_cols: usize,
    rhs_rows: usize,
    rhs_cols: usize,
) {
    assert_eq!(spec.layout(), MatrixLayout::ColumnMajor);
    assert_eq!(lhs_rows, spec.shape().lhs().rows());
    assert_eq!(lhs_cols, spec.shape().lhs().cols());
    assert_eq!(rhs_rows, spec.shape().rhs().rows());
    assert_eq!(rhs_cols, spec.shape().rhs().cols());
}

/// Executes plaintext/plaintext GEMM using the dependency-free reference path.
///
/// R3.5b intentionally supports one logical matrix per call. Batched GEMM is a
/// separate eBLAS operation introduced later.
pub fn gemm_pp(spec: GemmSpec, lhs: &BatchMatrix<f64>, rhs: &BatchMatrix<f64>) -> BatchMatrix<f64> {
    assert_eq!(spec.privacy(), PrivacyMode::Pp);
    assert!(spec.supports_backend(GemmBackend::Reference));
    assert_eq!(
        lhs.batches(),
        1,
        "eBLAS GEMM PP currently requires one batch"
    );
    assert_eq!(
        rhs.batches(),
        1,
        "eBLAS GEMM PP currently requires one batch"
    );

    assert_reference_shapes(spec, lhs.rows(), lhs.cols(), rhs.rows(), rhs.cols());

    let mut out = BatchMatrix::new(
        spec.shape().output().rows(),
        spec.shape().output().cols(),
        1,
    );

    for col in 0..rhs.cols() {
        for row in 0..lhs.rows() {
            let mut sum = 0.0_f64;
            for inner in 0..lhs.cols() {
                sum += *lhs.get(0, row, inner) * *rhs.get(0, inner, col);
            }
            out.set(0, row, col, sum);
        }
    }

    out
}

/// Executes ciphertext/plaintext GEMM using the validated realistic RNS/NTT
/// CP path.
///
/// Each output dot product is accumulated before one CKKS rescale.
pub fn gemm_cp(
    spec: GemmSpec,
    lhs: &RnsCkksCiphertextMatrix,
    rhs: &RnsCkksPlaintextMatrix,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    assert_eq!(spec.privacy(), PrivacyMode::Cp);
    assert!(spec.supports_backend(GemmBackend::CpDirect));

    assert_reference_shapes(spec, lhs.rows(), lhs.cols(), rhs.rows(), rhs.cols());

    lhs.matmul_plain_with_ntt(rhs, chain, plan)
}

/// Executes plaintext/ciphertext GEMM by reducing it to the validated CP path.
///
/// Uses `A B = (B^T A^T)^T`; transposes are representation permutations.
pub fn gemm_pc(
    spec: GemmSpec,
    lhs: &RnsCkksPlaintextMatrix,
    rhs: &RnsCkksCiphertextMatrix,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    assert_eq!(spec.privacy(), PrivacyMode::Pc);
    assert!(spec.supports_backend(GemmBackend::CpDirect));
    assert_reference_shapes(spec, lhs.rows(), lhs.cols(), rhs.rows(), rhs.cols());
    rhs.transpose()
        .matmul_plain_with_ntt(&lhs.transpose(), chain, plan)
        .transpose()
}

/// Executes ciphertext/ciphertext GEMM through an explicitly selected backend.
///
/// R3.5b supports:
///
/// - [`GemmBackend::CcScalar`]: scalar multiply/relinearize/rescale for each
///   logical product;
/// - [`GemmBackend::CcStructured`]: accumulate each degree-two RLWE dot product
///   before one relinearization and rescale per output.
///
/// No automatic dispatch is performed.
pub fn gemm_cc(
    spec: GemmSpec,
    backend: GemmBackend,
    lhs: &RnsCkksCiphertextMatrix,
    rhs: &RnsCkksCiphertextMatrix,
    multiplication_key: &BoundedRnsMultiplicationKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    assert_eq!(spec.privacy(), PrivacyMode::Cc);
    assert!(spec.supports_backend(backend));

    assert_reference_shapes(spec, lhs.rows(), lhs.cols(), rhs.rows(), rhs.cols());

    match backend {
        GemmBackend::CcScalar => lhs.matmul_bounded_with_ntt(rhs, multiplication_key, chain, plan),
        GemmBackend::CcStructured => {
            lhs.matmul_structured_bounded_with_ntt(rhs, multiplication_key, chain, plan)
        }
        GemmBackend::Reference | GemmBackend::CpDirect => {
            unreachable!("privacy/backend compatibility was validated above")
        }
    }
}

/// Executes ciphertext/ciphertext GEMM using the frozen R3.5 backend policy.
///
/// Explicit `gemm_cc` selection remains available for reproducible experiments.
pub fn gemm_cc_auto(
    spec: GemmSpec,
    lhs: &RnsCkksCiphertextMatrix,
    rhs: &RnsCkksCiphertextMatrix,
    multiplication_key: &BoundedRnsMultiplicationKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    let backend = crate::eblas::select_cc_backend(spec);
    gemm_cc(spec, backend, lhs, rhs, multiplication_key, chain, plan)
}

#[cfg(test)]
mod tests {
    use super::gemm_pp;
    use crate::eblas::{GemmShape, GemmSpec, MatrixShape, PrivacyMode};
    use crate::matrix::BatchMatrix;

    fn matrix(rows: usize, cols: usize, data: Vec<f64>) -> BatchMatrix<f64> {
        BatchMatrix::from_vec_column_major(rows, cols, 1, data)
    }

    #[test]
    fn pp_gemm_matches_rectangular_reference() {
        let lhs = matrix(2, 3, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let rhs = matrix(3, 2, vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0]);

        let spec = GemmSpec::new(
            GemmShape::new(MatrixShape::new(2, 3), MatrixShape::new(3, 2)),
            PrivacyMode::Pp,
        );

        let actual = gemm_pp(spec, &lhs, &rhs);

        assert_eq!(actual.rows(), 2);
        assert_eq!(actual.cols(), 2);
        assert_eq!(*actual.get(0, 0, 0), 58.0);
        assert_eq!(*actual.get(0, 1, 0), 139.0);
        assert_eq!(*actual.get(0, 0, 1), 64.0);
        assert_eq!(*actual.get(0, 1, 1), 154.0);
    }

    #[test]
    #[should_panic]
    fn pp_gemm_rejects_cp_contract() {
        let lhs = matrix(1, 1, vec![1.0]);
        let rhs = matrix(1, 1, vec![1.0]);
        let spec = GemmSpec::new(
            GemmShape::new(MatrixShape::new(1, 1), MatrixShape::new(1, 1)),
            PrivacyMode::Cp,
        );

        let _ = gemm_pp(spec, &lhs, &rhs);
    }
}
