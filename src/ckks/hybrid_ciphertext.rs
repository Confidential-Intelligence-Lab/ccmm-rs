use crate::grafting::{HybridRlweCiphertext, Pow2GraftedBasis};

use super::CkksChainState;

/// Leveled CKKS metadata attached to a native Grafting/hybrid RLWE
/// ciphertext.
///
/// The CKKS state tracks the ordinary odd-RNS chain. The separately
/// managed power-of-two sprout is tracked by the hybrid ciphertext.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridCkksCiphertext {
    inner: HybridRlweCiphertext,
    state: CkksChainState,
}

impl HybridCkksCiphertext {
    pub fn new(inner: HybridRlweCiphertext, state: CkksChainState) -> Self {
        assert_eq!(
            inner.ordinary().basis(),
            state.basis(),
            "hybrid ciphertext ordinary basis must match CKKS state basis"
        );

        Self { inner, state }
    }

    pub fn inner(&self) -> &HybridRlweCiphertext {
        &self.inner
    }

    pub fn state(&self) -> &CkksChainState {
        &self.state
    }

    pub fn level(&self) -> usize {
        self.state.level()
    }

    pub fn scale(&self) -> f64 {
        self.state.scale()
    }

    pub fn ordinary_basis(&self) -> &crate::ring::ModulusBasis {
        self.inner.ordinary().basis()
    }

    pub fn sprout_bits(&self) -> u32 {
        self.inner.sprout().bits()
    }

    pub fn grafted_basis(&self) -> Pow2GraftedBasis {
        Pow2GraftedBasis::new(self.ordinary_basis().clone(), self.sprout_bits())
    }

    pub fn into_parts(self) -> (HybridRlweCiphertext, CkksChainState) {
        (self.inner, self.state)
    }
}
