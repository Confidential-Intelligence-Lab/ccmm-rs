//! eBLAS GEMV and DOT front-ends.
//!
//! R3.5c intentionally reuses the already validated GEMM kernels so that
//! GEMV/DOT semantics are established without duplicating cryptographic
//! arithmetic. Specialized kernels may replace these reductions later.

use crate::eblas::{
    gemm_cc, gemm_cc_auto, gemm_cp, gemm_pp, GemmBackend, GemmShape, GemmSpec, MatrixShape,
    PrivacyMode,
};
use crate::grafting::BoundedRnsMultiplicationKey;
use crate::matrix::{BatchMatrix, RnsCkksCiphertextMatrix, RnsCkksPlaintextMatrix};
use crate::ring::{ModulusChain, RnsNttPlan};

/// Shape contract for `y = A x`, where `A` is `M x K` and `x` is `K x 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemvShape {
    matrix: MatrixShape,
    vector: MatrixShape,
    output: MatrixShape,
}

impl GemvShape {
    /// Creates a valid GEMV shape.
    pub fn new(matrix: MatrixShape, vector: MatrixShape) -> Self {
        assert_eq!(
            vector.cols(),
            1,
            "eBLAS GEMV right operand must be a column vector"
        );
        assert_eq!(
            matrix.cols(),
            vector.rows(),
            "eBLAS GEMV inner dimensions must match"
        );

        Self {
            matrix,
            vector,
            output: MatrixShape::new(matrix.rows(), 1),
        }
    }

    pub const fn matrix(self) -> MatrixShape {
        self.matrix
    }

    pub const fn vector(self) -> MatrixShape {
        self.vector
    }

    pub const fn output(self) -> MatrixShape {
        self.output
    }

    pub const fn inner_dimension(self) -> usize {
        self.matrix.cols()
    }

    fn as_gemm(self, privacy: PrivacyMode) -> GemmSpec {
        GemmSpec::new(GemmShape::new(self.matrix, self.vector), privacy)
    }
}

/// Shape contract for `x^T y`, represented as `(1 x K) * (K x 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DotShape {
    length: usize,
}

impl DotShape {
    pub fn new(length: usize) -> Self {
        assert!(length > 0, "eBLAS DOT vector length must be positive");
        Self { length }
    }

    pub const fn length(self) -> usize {
        self.length
    }

    fn lhs(self) -> MatrixShape {
        MatrixShape::new(1, self.length)
    }

    fn rhs(self) -> MatrixShape {
        MatrixShape::new(self.length, 1)
    }

    fn as_gemm(self, privacy: PrivacyMode) -> GemmSpec {
        GemmSpec::new(GemmShape::new(self.lhs(), self.rhs()), privacy)
    }
}

pub fn gemv_pp(
    shape: GemvShape,
    matrix: &BatchMatrix<f64>,
    vector: &BatchMatrix<f64>,
) -> BatchMatrix<f64> {
    gemm_pp(shape.as_gemm(PrivacyMode::Pp), matrix, vector)
}

pub fn gemv_cp(
    shape: GemvShape,
    matrix: &RnsCkksCiphertextMatrix,
    vector: &RnsCkksPlaintextMatrix,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    gemm_cp(shape.as_gemm(PrivacyMode::Cp), matrix, vector, chain, plan)
}

pub fn gemv_cc(
    shape: GemvShape,
    backend: GemmBackend,
    matrix: &RnsCkksCiphertextMatrix,
    vector: &RnsCkksCiphertextMatrix,
    multiplication_key: &BoundedRnsMultiplicationKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    gemm_cc(
        shape.as_gemm(PrivacyMode::Cc),
        backend,
        matrix,
        vector,
        multiplication_key,
        chain,
        plan,
    )
}

/// Executes ciphertext/ciphertext GEMV using the frozen R3.5 backend policy.
pub fn gemv_cc_auto(
    shape: GemvShape,
    matrix: &RnsCkksCiphertextMatrix,
    vector: &RnsCkksCiphertextMatrix,
    multiplication_key: &BoundedRnsMultiplicationKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    gemm_cc_auto(
        shape.as_gemm(PrivacyMode::Cc),
        matrix,
        vector,
        multiplication_key,
        chain,
        plan,
    )
}

pub fn dot_pp(shape: DotShape, lhs: &BatchMatrix<f64>, rhs: &BatchMatrix<f64>) -> BatchMatrix<f64> {
    gemm_pp(shape.as_gemm(PrivacyMode::Pp), lhs, rhs)
}

pub fn dot_cp(
    shape: DotShape,
    lhs: &RnsCkksCiphertextMatrix,
    rhs: &RnsCkksPlaintextMatrix,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    gemm_cp(shape.as_gemm(PrivacyMode::Cp), lhs, rhs, chain, plan)
}

pub fn dot_cc(
    shape: DotShape,
    backend: GemmBackend,
    lhs: &RnsCkksCiphertextMatrix,
    rhs: &RnsCkksCiphertextMatrix,
    multiplication_key: &BoundedRnsMultiplicationKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    gemm_cc(
        shape.as_gemm(PrivacyMode::Cc),
        backend,
        lhs,
        rhs,
        multiplication_key,
        chain,
        plan,
    )
}

/// Executes ciphertext/ciphertext DOT using the frozen R3.5 backend policy.
pub fn dot_cc_auto(
    shape: DotShape,
    lhs: &RnsCkksCiphertextMatrix,
    rhs: &RnsCkksCiphertextMatrix,
    multiplication_key: &BoundedRnsMultiplicationKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertextMatrix {
    gemm_cc_auto(
        shape.as_gemm(PrivacyMode::Cc),
        lhs,
        rhs,
        multiplication_key,
        chain,
        plan,
    )
}

#[cfg(test)]
mod tests {
    use super::{dot_pp, gemv_pp, DotShape, GemvShape};
    use crate::eblas::MatrixShape;
    use crate::matrix::BatchMatrix;

    #[test]
    fn pp_gemv_matches_reference() {
        let matrix =
            BatchMatrix::from_vec_column_major(2, 3, 1, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let vector = BatchMatrix::from_vec_column_major(3, 1, 1, vec![7.0, 8.0, 9.0]);

        let result = gemv_pp(
            GemvShape::new(MatrixShape::new(2, 3), MatrixShape::new(3, 1)),
            &matrix,
            &vector,
        );

        assert_eq!(*result.get(0, 0, 0), 50.0);
        assert_eq!(*result.get(0, 1, 0), 122.0);
    }

    #[test]
    fn pp_dot_matches_reference() {
        let lhs = BatchMatrix::from_vec_column_major(1, 3, 1, vec![1.0, 2.0, 3.0]);
        let rhs = BatchMatrix::from_vec_column_major(3, 1, 1, vec![4.0, 5.0, 6.0]);

        let result = dot_pp(DotShape::new(3), &lhs, &rhs);
        assert_eq!(*result.get(0, 0, 0), 32.0);
    }

    #[test]
    #[should_panic(expected = "column vector")]
    fn gemv_rejects_non_vector_rhs() {
        let _ = GemvShape::new(MatrixShape::new(2, 3), MatrixShape::new(3, 2));
    }
}
