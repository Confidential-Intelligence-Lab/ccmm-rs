//! Batched eBLAS GEMM.
//!
//! R3.5f defines batching as a sequence of independent GEMMs with identical
//! matrix dimensions, privacy mode, layout, and backend. Batching here is an
//! application/eBLAS abstraction; it is not CKKS SIMD slot packing.

use crate::eblas::{
    gemm_cc, gemm_cp, gemm_pc, gemm_pp, GemmBackend, GemmOperationCount, GemmShape, GemmSpec,
    MatrixLayout, PrivacyMode,
};
use crate::grafting::BoundedRnsMultiplicationKey;
use crate::matrix::{BatchMatrix, RnsCkksCiphertextMatrix, RnsCkksPlaintextMatrix};
use crate::ring::{ModulusChain, RnsNttPlan};

/// Shape of a batched GEMM consisting of independent, identically shaped GEMMs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedGemmShape {
    batches: usize,
    gemm: GemmShape,
}

impl BatchedGemmShape {
    /// Creates a batched GEMM shape.
    pub fn new(batches: usize, gemm: GemmShape) -> Self {
        assert!(
            batches > 0,
            "eBLAS batched GEMM batch count must be positive"
        );
        Self { batches, gemm }
    }

    /// Number of independent GEMMs.
    pub const fn batches(self) -> usize {
        self.batches
    }

    /// Shape of every GEMM in the batch.
    pub const fn gemm(self) -> GemmShape {
        self.gemm
    }
}

/// Semantic contract for batched GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedGemmSpec {
    shape: BatchedGemmShape,
    privacy: PrivacyMode,
    layout: MatrixLayout,
}

impl BatchedGemmSpec {
    /// Creates a column-major batched GEMM contract.
    pub fn new(shape: BatchedGemmShape, privacy: PrivacyMode) -> Self {
        Self {
            shape,
            privacy,
            layout: MatrixLayout::ColumnMajor,
        }
    }

    /// Batched GEMM dimensions.
    pub const fn shape(self) -> BatchedGemmShape {
        self.shape
    }

    /// Operand privacy mode.
    pub const fn privacy(self) -> PrivacyMode {
        self.privacy
    }

    /// Matrix storage layout.
    pub const fn layout(self) -> MatrixLayout {
        self.layout
    }

    /// Returns whether the backend is supported for this privacy mode.
    pub fn supports_backend(self, backend: GemmBackend) -> bool {
        GemmSpec::new(self.shape.gemm(), self.privacy).supports_backend(backend)
    }

    fn scalar_spec(self) -> GemmSpec {
        GemmSpec::new(self.shape.gemm(), self.privacy)
    }
}

/// Batched operation accounting. Counts are exactly batch-count multiples of
/// the corresponding scalar GEMM schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedGemmOperationCount {
    /// Logical scalar products.
    pub scalar_products: usize,
    /// Dot-product additions.
    pub additions: usize,
    /// Relinearizations.
    pub relinearizations: usize,
    /// Rescales.
    pub rescales: usize,
}

impl BatchedGemmOperationCount {
    /// Computes static accounting for a supported batched GEMM backend.
    pub fn for_backend(spec: BatchedGemmSpec, backend: GemmBackend) -> Self {
        assert!(
            spec.supports_backend(backend),
            "eBLAS batched GEMM backend is not supported for this privacy mode"
        );
        let one = GemmOperationCount::for_backend(spec.scalar_spec(), backend);
        let batches = spec.shape().batches();

        Self {
            scalar_products: one
                .scalar_products
                .checked_mul(batches)
                .expect("eBLAS batched GEMM scalar-product count overflow"),
            additions: one
                .additions
                .checked_mul(batches)
                .expect("eBLAS batched GEMM addition count overflow"),
            relinearizations: one
                .relinearizations
                .checked_mul(batches)
                .expect("eBLAS batched GEMM relinearization count overflow"),
            rescales: one
                .rescales
                .checked_mul(batches)
                .expect("eBLAS batched GEMM rescale count overflow"),
        }
    }
}

fn extract_pp_batch(matrix: &BatchMatrix<f64>, batch: usize) -> BatchMatrix<f64> {
    let mut data = Vec::with_capacity(matrix.rows() * matrix.cols());
    for col in 0..matrix.cols() {
        for row in 0..matrix.rows() {
            data.push(*matrix.get(batch, row, col));
        }
    }
    BatchMatrix::from_vec_column_major(matrix.rows(), matrix.cols(), 1, data)
}

/// Executes PP batched GEMM pairwise across corresponding batches.
pub fn batched_gemm_pp(
    spec: BatchedGemmSpec,
    lhs: &BatchMatrix<f64>,
    rhs: &BatchMatrix<f64>,
) -> BatchMatrix<f64> {
    assert_eq!(spec.privacy(), PrivacyMode::Pp);
    assert!(spec.supports_backend(GemmBackend::Reference));
    assert_eq!(lhs.batches(), spec.shape().batches());
    assert_eq!(rhs.batches(), spec.shape().batches());

    let one_spec = spec.scalar_spec();
    let output = one_spec.shape().output();
    let mut data = Vec::with_capacity(
        spec.shape()
            .batches()
            .checked_mul(output.elements())
            .expect("eBLAS batched GEMM output size overflow"),
    );

    for batch in 0..spec.shape().batches() {
        let lhs_batch = extract_pp_batch(lhs, batch);
        let rhs_batch = extract_pp_batch(rhs, batch);
        let result = gemm_pp(one_spec, &lhs_batch, &rhs_batch);
        data.extend_from_slice(result.raw());
    }

    BatchMatrix::from_vec_column_major(output.rows(), output.cols(), spec.shape().batches(), data)
}

/// Executes CP batched GEMM independently for each batch.
pub fn batched_gemm_cp(
    spec: BatchedGemmSpec,
    lhs: &[RnsCkksCiphertextMatrix],
    rhs: &[RnsCkksPlaintextMatrix],
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> Vec<RnsCkksCiphertextMatrix> {
    assert_eq!(spec.privacy(), PrivacyMode::Cp);
    assert!(spec.supports_backend(GemmBackend::CpDirect));
    assert_eq!(lhs.len(), spec.shape().batches());
    assert_eq!(rhs.len(), spec.shape().batches());

    let one_spec = spec.scalar_spec();
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| gemm_cp(one_spec, a, b, chain, plan))
        .collect()
}

/// Executes PC batched GEMM independently for each batch.
pub fn batched_gemm_pc(
    spec: BatchedGemmSpec,
    lhs: &[RnsCkksPlaintextMatrix],
    rhs: &[RnsCkksCiphertextMatrix],
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> Vec<RnsCkksCiphertextMatrix> {
    assert_eq!(spec.privacy(), PrivacyMode::Pc);
    assert!(spec.supports_backend(GemmBackend::CpDirect));
    assert_eq!(lhs.len(), spec.shape().batches());
    assert_eq!(rhs.len(), spec.shape().batches());

    let one_spec = spec.scalar_spec();
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| gemm_pc(one_spec, a, b, chain, plan))
        .collect()
}

/// Executes CC batched GEMM independently for each batch.
pub fn batched_gemm_cc(
    spec: BatchedGemmSpec,
    backend: GemmBackend,
    lhs: &[RnsCkksCiphertextMatrix],
    rhs: &[RnsCkksCiphertextMatrix],
    multiplication_key: &BoundedRnsMultiplicationKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> Vec<RnsCkksCiphertextMatrix> {
    assert_eq!(spec.privacy(), PrivacyMode::Cc);
    assert!(spec.supports_backend(backend));
    assert_eq!(lhs.len(), spec.shape().batches());
    assert_eq!(rhs.len(), spec.shape().batches());

    let one_spec = spec.scalar_spec();
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| gemm_cc(one_spec, backend, a, b, multiplication_key, chain, plan))
        .collect()
}

/// Executes batched CC GEMM using the frozen R3.5 backend policy.
pub fn batched_gemm_cc_auto(
    spec: BatchedGemmSpec,
    lhs: &[RnsCkksCiphertextMatrix],
    rhs: &[RnsCkksCiphertextMatrix],
    multiplication_key: &BoundedRnsMultiplicationKey,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> Vec<RnsCkksCiphertextMatrix> {
    let gemm_spec = GemmSpec::new(spec.shape().gemm(), PrivacyMode::Cc);
    let backend = crate::eblas::select_cc_backend(gemm_spec);
    batched_gemm_cc(spec, backend, lhs, rhs, multiplication_key, chain, plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eblas::{MatrixShape, PrivacyMode};

    fn shape() -> GemmShape {
        GemmShape::new(MatrixShape::new(2, 3), MatrixShape::new(3, 4))
    }

    #[test]
    fn batched_shape_preserves_count_and_gemm_shape() {
        let shape = BatchedGemmShape::new(3, shape());
        assert_eq!(shape.batches(), 3);
        assert_eq!(shape.gemm().output(), MatrixShape::new(2, 4));
    }

    #[test]
    #[should_panic(expected = "batch count must be positive")]
    fn batched_shape_rejects_zero_batches() {
        let _ = BatchedGemmShape::new(0, shape());
    }

    #[test]
    fn operation_accounting_is_exact_batch_multiple() {
        let cp = BatchedGemmSpec::new(BatchedGemmShape::new(3, shape()), PrivacyMode::Cp);
        let count = BatchedGemmOperationCount::for_backend(cp, GemmBackend::CpDirect);

        // One 2x3 * 3x4 GEMM: products=24, additions=16, outputs/rescales=8.
        assert_eq!(count.scalar_products, 72);
        assert_eq!(count.additions, 48);
        assert_eq!(count.relinearizations, 0);
        assert_eq!(count.rescales, 24);

        let cc = BatchedGemmSpec::new(BatchedGemmShape::new(3, shape()), PrivacyMode::Cc);
        let structured = BatchedGemmOperationCount::for_backend(cc, GemmBackend::CcStructured);
        assert_eq!(structured.scalar_products, 72);
        assert_eq!(structured.additions, 48);
        assert_eq!(structured.relinearizations, 24);
        assert_eq!(structured.rescales, 24);
    }

    #[test]
    fn pp_batched_gemm_is_pairwise_and_batch_isolated() {
        let spec = BatchedGemmSpec::new(BatchedGemmShape::new(2, shape()), PrivacyMode::Pp);

        // Two distinct 2x3 batches, column-major within each batch.
        let lhs = BatchMatrix::from_vec_column_major(
            2,
            3,
            2,
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 2.0, 1.0, 0.0, 3.0, 4.0, -1.0],
        );
        // Two distinct 3x4 batches.
        let rhs = BatchMatrix::from_vec_column_major(
            3,
            4,
            2,
            vec![
                // Batch 0: 3 x 4, column-major.
                1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, -1.0, 2.0, 1.0,
                // Batch 1: distinct 3 x 4 matrix.
                1.0, 2.0, 0.0, 2.0, 1.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0,
            ],
        );

        let out = batched_gemm_pp(spec, &lhs, &rhs);
        assert_eq!(out.batches(), 2);
        assert_eq!(out.rows(), 2);
        assert_eq!(out.cols(), 4);

        // Verify each batch independently against scalar gemm_pp.
        for batch in 0..2 {
            let a = extract_pp_batch(&lhs, batch);
            let b = extract_pp_batch(&rhs, batch);
            let expected = gemm_pp(GemmSpec::new(shape(), PrivacyMode::Pp), &a, &b);
            for col in 0..4 {
                for row in 0..2 {
                    assert_eq!(*out.get(batch, row, col), *expected.get(0, row, col));
                }
            }
        }

        assert_ne!(out.get(0, 0, 0), out.get(1, 0, 0));
    }
}
