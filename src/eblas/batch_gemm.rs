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
