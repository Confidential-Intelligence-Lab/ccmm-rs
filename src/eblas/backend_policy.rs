//! Evidence-backed eBLAS backend-selection policy.
//!
//! R3.5h measured scalar and structured ciphertext/ciphertext GEMM under the
//! research-4096 profile. At reduction width K=1 the schedules are effectively
//! equivalent (1.003x measured ratio); by K=2 structured execution is already
//! 1.648x faster, with the advantage increasing through the tested K=16 point.
//!
//! The policy therefore keeps the scalar backend for K=1 and selects the
//! structured backend for K>=2. Explicit backend selection remains available
//! everywhere for reproducibility and future characterization.

use crate::eblas::{GemmBackend, GemmSpec, PrivacyMode};

/// Selects the concrete CC GEMM backend from the frozen R3.5h policy.
pub fn select_cc_backend(spec: GemmSpec) -> GemmBackend {
    assert_eq!(
        spec.privacy(),
        PrivacyMode::Cc,
        "automatic CC backend selection requires CC privacy"
    );

    if spec.shape().lhs().cols() == 1 {
        GemmBackend::CcScalar
    } else {
        GemmBackend::CcStructured
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eblas::{GemmShape, MatrixShape};

    fn cc_spec(m: usize, k: usize, n: usize) -> GemmSpec {
        GemmSpec::new(
            GemmShape::new(MatrixShape::new(m, k), MatrixShape::new(k, n)),
            PrivacyMode::Cc,
        )
    }

    #[test]
    fn auto_policy_keeps_scalar_at_k1() {
        assert_eq!(select_cc_backend(cc_spec(1, 1, 1)), GemmBackend::CcScalar);
        assert_eq!(select_cc_backend(cc_spec(8, 1, 8)), GemmBackend::CcScalar);
    }

    #[test]
    fn auto_policy_selects_structured_from_k2() {
        for k in [2, 3, 4, 8, 16] {
            assert_eq!(
                select_cc_backend(cc_spec(2, k, 4)),
                GemmBackend::CcStructured
            );
        }
    }

    #[test]
    #[should_panic(expected = "requires CC privacy")]
    fn auto_policy_rejects_non_cc_specs() {
        let spec = GemmSpec::new(
            GemmShape::new(MatrixShape::new(2, 3), MatrixShape::new(3, 4)),
            PrivacyMode::Cp,
        );
        let _ = select_cc_backend(spec);
    }
}
