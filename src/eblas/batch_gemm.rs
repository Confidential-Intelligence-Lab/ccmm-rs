//! Packed Batch GEMM mechanisms.
//!
//! This module connects the eBLAS GEMM contract to the SinC-packed Batch
//! matrix-multiplication mechanisms. It is distinct from `batched_gemm`:
//!
//! - `batched_gemm` executes a sequence of independent logical GEMMs;
//! - `batch_gemm` uses one packed cryptographic representation to realize
//!   multiple real GEMMs simultaneously.
//!
//! The mathematical operation remains GEMM. The mechanism determines the
//! encrypted representation and execution strategy.

use crate::ccmm::batch::BatchCcmmExecutionContext;
use crate::ckks::RnsCkksCiphertext;
use crate::eblas::{BatchGemmGeometry, BatchGemmMechanism, PrivacyMode};
use crate::grafting::RnsRlweCiphertext;
use crate::ring::{ModulusChain, RnsNttPlan, RnsPolynomial};

/// Executes packed ciphertext/plaintext GEMM through the Batch CPMM mechanism.
///
/// Encoding, encryption, key generation, decryption, and SinC decoding are
/// outside this eBLAS execution boundary.
pub fn batch_gemm_cpmm(
    geometry: BatchGemmGeometry,
    lhs: &[RnsCkksCiphertext],
    rhs: &[Vec<RnsPolynomial>],
    scalar_plan: &RnsNttPlan,
    chain: &ModulusChain,
    scale: f64,
) -> Vec<RnsCkksCiphertext> {
    assert_eq!(
        geometry.mechanism(),
        BatchGemmMechanism::Cpmm,
        "eBLAS Batch CPMM requires the CPMM mechanism"
    );
    assert_eq!(
        geometry.privacy(),
        PrivacyMode::Cp,
        "eBLAS Batch CPMM requires ciphertext/plaintext privacy"
    );
    assert_eq!(
        scalar_plan.degree(),
        geometry.scalar_degree(),
        "eBLAS Batch CPMM scalar plan degree must match its geometry"
    );

    for ciphertext in lhs {
        assert_eq!(
            ciphertext.rlwe().degree(),
            geometry.large_degree(),
            "eBLAS Batch CPMM ciphertext degree must match its geometry"
        );
    }

    crate::ccmm::batch::batch_cpmm_execute(
        lhs,
        rhs,
        geometry.dimension(),
        scalar_plan,
        chain,
        scale,
    )
}

/// Executes packed ciphertext/plaintext GEMV through the Batch CPMM mechanism.
///
/// `lhs` is the encrypted packed representation of a batch of square matrices.
/// `rhs` is an already encoded `d x d` plaintext embedding whose first column
/// contains the logical public vectors and whose remaining columns are zero.
///
/// The CPMM execution therefore computes `[A * x, 0, ..., 0]`. This adapter
/// returns only the first packed ciphertext column, which is the logical GEMV
/// result. Packing, plaintext encoding, encryption, and decoding remain outside
/// the eBLAS execution boundary.
///
/// This is a semantic GEMV adapter over the existing square CPMM mechanism,
/// not a specialized lower-complexity GEMV kernel.
pub fn batch_gemv_cpmm(
    geometry: BatchGemmGeometry,
    lhs: &[RnsCkksCiphertext],
    rhs: &[Vec<RnsPolynomial>],
    scalar_plan: &RnsNttPlan,
    chain: &ModulusChain,
    scale: f64,
) -> RnsCkksCiphertext {
    let dimension = geometry.dimension();

    assert_eq!(
        rhs.len(),
        dimension,
        "eBLAS Batch CPMM GEMV plaintext embedding must have one row per matrix row"
    );
    assert!(
        rhs.iter().all(|row| row.len() == dimension),
        "eBLAS Batch CPMM GEMV plaintext embedding must be square"
    );

    let mut output = batch_gemm_cpmm(geometry, lhs, rhs, scalar_plan, chain, scale);

    assert_eq!(
        output.len(),
        dimension,
        "eBLAS Batch CPMM GEMV square embedding returned unexpected output width"
    );

    output.remove(0)
}

/// Executes packed ciphertext/plaintext DOT through the Batch CPMM mechanism.
///
/// Each logical encrypted vector must be represented as the first row of its
/// square left-operand matrix, with every remaining row zero. Each logical
/// public vector uses the GEMV plaintext embedding: the vector occupies the
/// first column of the square right operand and every remaining column is zero.
///
/// The existing packed GEMV execution therefore produces a logical vector
/// `[x^T * y, 0, ..., 0]^T` for every packed batch. This function returns that
/// packed ciphertext representation; scalar extraction remains part of SinC
/// decoding outside the eBLAS execution boundary.
///
/// This is a semantic DOT adapter over the validated Batch CPMM GEMV path, not
/// a specialized lower-complexity DOT kernel.
pub fn batch_dot_cpmm(
    geometry: BatchGemmGeometry,
    lhs: &[RnsCkksCiphertext],
    rhs: &[Vec<RnsPolynomial>],
    scalar_plan: &RnsNttPlan,
    chain: &ModulusChain,
    scale: f64,
) -> RnsCkksCiphertext {
    batch_gemv_cpmm(geometry, lhs, rhs, scalar_plan, chain, scale)
}

/// Executes packed ciphertext/ciphertext GEMM through the Batch CCMM mechanism.
///
/// Encoding, encryption, evaluation-key generation/preparation, decryption,
/// and SinC decoding are outside this eBLAS execution boundary.
pub fn batch_gemm_ccmm(
    geometry: BatchGemmGeometry,
    lhs: &[RnsRlweCiphertext],
    rhs: &[RnsRlweCiphertext],
    context: &BatchCcmmExecutionContext<'_>,
) -> Vec<RnsCkksCiphertext> {
    assert_eq!(
        geometry.mechanism(),
        BatchGemmMechanism::Ccmm,
        "eBLAS Batch CCMM requires the CCMM mechanism"
    );
    assert_eq!(
        geometry.privacy(),
        PrivacyMode::Cc,
        "eBLAS Batch CCMM requires ciphertext/ciphertext privacy"
    );
    assert_eq!(
        context.scalar_plan.degree(),
        geometry.scalar_degree(),
        "eBLAS Batch CCMM scalar plan degree must match its geometry"
    );
    assert_eq!(
        context.prepared_large_plan.degree(),
        geometry.large_degree(),
        "eBLAS Batch CCMM large plan degree must match its geometry"
    );

    for ciphertext in lhs.iter().chain(rhs.iter()) {
        assert_eq!(
            ciphertext.degree(),
            geometry.large_degree(),
            "eBLAS Batch CCMM ciphertext degree must match its geometry"
        );
    }

    crate::ccmm::batch::batch_ccmm_execute(lhs, rhs, geometry.dimension(), context)
}

/// Executes packed ciphertext/ciphertext GEMV through the Batch CCMM mechanism.
///
/// `lhs` is the encrypted packed representation of a batch of square matrices.
/// `rhs` is the encrypted packed representation of a `d x d` embedding whose
/// first column contains the logical encrypted vectors and whose remaining
/// columns are zero.
///
/// CCMM therefore computes `[A * x, 0, ..., 0]`. This adapter returns only the
/// first packed ciphertext column, which is the logical GEMV result.
///
/// Packing, encryption, evaluation-key preparation, decryption, and SinC
/// decoding remain outside the eBLAS execution boundary.
///
/// This is a semantic GEMV adapter over the existing square CCMM mechanism,
/// not a specialized lower-complexity GEMV kernel.
pub fn batch_gemv_ccmm(
    geometry: BatchGemmGeometry,
    lhs: &[RnsRlweCiphertext],
    rhs: &[RnsRlweCiphertext],
    context: &BatchCcmmExecutionContext<'_>,
) -> RnsCkksCiphertext {
    let dimension = geometry.dimension();

    assert_eq!(
        lhs.len(),
        dimension,
        "eBLAS Batch CCMM GEMV left operand must contain d ciphertext columns"
    );
    assert_eq!(
        rhs.len(),
        dimension,
        "eBLAS Batch CCMM GEMV encrypted embedding must contain d ciphertext columns"
    );

    let mut output = batch_gemm_ccmm(geometry, lhs, rhs, context);

    assert_eq!(
        output.len(),
        dimension,
        "eBLAS Batch CCMM GEMV square embedding returned unexpected output width"
    );

    output.remove(0)
}

/// Executes packed ciphertext/ciphertext DOT through the Batch CCMM mechanism.
///
/// Each logical encrypted left vector must be represented as the first row of
/// its square left-operand matrix, with every remaining row zero. Each logical
/// encrypted right vector must be represented as the first column of its square
/// right-operand matrix, with every remaining column zero.
///
/// The existing packed CCMM GEMV execution therefore produces
/// `[x^T * y, 0, ..., 0]^T` for every packed batch. This function returns that
/// packed ciphertext representation; scalar extraction remains part of SinC
/// decoding outside the eBLAS execution boundary.
///
/// This is a semantic DOT adapter over the validated Batch CCMM GEMV path, not
/// a specialized lower-complexity DOT kernel.
pub fn batch_dot_ccmm(
    geometry: BatchGemmGeometry,
    lhs: &[RnsRlweCiphertext],
    rhs: &[RnsRlweCiphertext],
    context: &BatchCcmmExecutionContext<'_>,
) -> RnsCkksCiphertext {
    batch_gemv_ccmm(geometry, lhs, rhs, context)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_cpmm_geometry_exposes_cp_contract() {
        let geometry = BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 64, 256, 8192);

        assert_eq!(geometry.mechanism(), BatchGemmMechanism::Cpmm);
        assert_eq!(geometry.privacy(), PrivacyMode::Cp);
        assert_eq!(geometry.dimension(), 64);
        assert_eq!(geometry.scalar_degree(), 256);
        assert_eq!(geometry.large_degree(), 8192);
        assert_eq!(geometry.batch_count(), 128);
    }

    #[test]
    fn batch_ccmm_geometry_exposes_cc_contract() {
        let geometry = BatchGemmGeometry::new(BatchGemmMechanism::Ccmm, 64, 128, 8192);

        assert_eq!(geometry.mechanism(), BatchGemmMechanism::Ccmm);
        assert_eq!(geometry.privacy(), PrivacyMode::Cc);
        assert_eq!(geometry.dimension(), 64);
        assert_eq!(geometry.scalar_degree(), 128);
        assert_eq!(geometry.large_degree(), 8192);
        assert_eq!(geometry.batch_count(), 64);
    }
}
