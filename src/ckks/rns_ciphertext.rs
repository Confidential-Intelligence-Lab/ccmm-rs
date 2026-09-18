use crate::grafting::RnsRlweCiphertext;
use crate::ring::{ModulusBasis, ModulusChain};

use super::CkksChainState;

/// CKKS ciphertext represented explicitly over the active RNS
/// modulus-chain basis.
///
/// The chain state is not advisory metadata: construction verifies
/// that the ciphertext's RNS basis exactly equals the basis associated
/// with the CKKS level.
#[derive(Debug, Clone, PartialEq)]
pub struct RnsCkksCiphertext {
    inner: RnsRlweCiphertext,
    state: CkksChainState,
}

impl RnsCkksCiphertext {
    pub fn new(inner: RnsRlweCiphertext, state: CkksChainState, chain: &ModulusChain) -> Self {
        state.assert_matches_chain(chain);

        assert_eq!(
            inner.basis(),
            state.basis(),
            "RNS CKKS ciphertext basis must match CKKS chain state"
        );

        Self { inner, state }
    }

    pub fn rlwe(&self) -> &RnsRlweCiphertext {
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

    pub fn basis(&self) -> &ModulusBasis {
        self.inner.basis()
    }

    pub fn into_parts(self) -> (RnsRlweCiphertext, CkksChainState) {
        (self.inner, self.state)
    }

    pub fn assert_matches_chain(&self, chain: &ModulusChain) {
        self.state.assert_matches_chain(chain);

        assert_eq!(
            self.inner.basis(),
            chain.level(self.level()),
            "RNS CKKS ciphertext basis must equal active chain level"
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::grafting::RnsRlweCiphertext;
    use crate::ring::{Modulus, ModulusBasis, Polynomial};
    use crate::rlwe::RlweCiphertext;

    use super::*;

    fn chain() -> ModulusChain {
        ModulusChain::from_top_basis(ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]))
    }

    fn zero_rns_ciphertext(basis: &ModulusBasis, degree: usize) -> RnsRlweCiphertext {
        let limbs = basis
            .moduli()
            .iter()
            .copied()
            .map(|modulus| {
                RlweCiphertext::new(
                    Polynomial::zero(modulus, degree),
                    Polynomial::zero(modulus, degree),
                )
            })
            .collect();

        RnsRlweCiphertext::from_limbs(limbs)
    }

    #[test]
    fn top_level_ciphertext_matches_top_chain_basis() {
        let chain = chain();

        let state = CkksChainState::top(&chain, 65_537.0);

        let ciphertext = RnsCkksCiphertext::new(zero_rns_ciphertext(chain.top(), 8), state, &chain);

        assert_eq!(ciphertext.level(), 0);

        assert_eq!(ciphertext.basis(), chain.level(0));

        assert_eq!(ciphertext.scale(), 65_537.0);
    }

    #[test]
    fn lower_level_ciphertext_matches_lower_basis() {
        let chain = chain();

        let state = CkksChainState::new(&chain, 1, 40_961.0);

        let ciphertext =
            RnsCkksCiphertext::new(zero_rns_ciphertext(chain.level(1), 8), state, &chain);

        assert_eq!(ciphertext.level(), 1);

        assert_eq!(ciphertext.basis(), chain.level(1));
    }

    #[test]
    #[should_panic(expected = "basis must match CKKS chain state")]
    fn rejects_ciphertext_from_wrong_chain_level() {
        let chain = chain();

        let state = CkksChainState::new(&chain, 1, 1.0);

        let _ = RnsCkksCiphertext::new(zero_rns_ciphertext(chain.level(0), 8), state, &chain);
    }

    #[test]
    fn state_and_ciphertext_survive_into_parts() {
        let chain = chain();

        let state = CkksChainState::top(&chain, 256.0);

        let ciphertext =
            RnsCkksCiphertext::new(zero_rns_ciphertext(chain.top(), 8), state.clone(), &chain);

        let (inner, recovered_state) = ciphertext.into_parts();

        assert_eq!(inner.basis(), chain.top());

        assert_eq!(recovered_state, state);
    }
}
