use crate::ring::{Modulus, ModulusBasis, ModulusChain};

/// Cryptographically meaningful CKKS state at one modulus-chain level.
///
/// Level numbering follows `ModulusChain`:
///
/// - level 0 is the highest modulus basis;
/// - increasing the level drops one trailing modulus at a time.
#[derive(Debug, Clone, PartialEq)]
pub struct CkksChainState {
    level: usize,
    basis: ModulusBasis,
    scale: f64,
}

impl CkksChainState {
    pub fn new(chain: &ModulusChain, level: usize, scale: f64) -> Self {
        assert!(
            level <= chain.max_level(),
            "CKKS level must exist in the modulus chain"
        );
        assert!(
            scale.is_finite() && scale > 0.0,
            "CKKS scale must be positive and finite"
        );

        Self {
            level,
            basis: chain.level(level).clone(),
            scale,
        }
    }

    pub fn top(chain: &ModulusChain, scale: f64) -> Self {
        Self::new(chain, 0, scale)
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn basis(&self) -> &ModulusBasis {
        &self.basis
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn composite_modulus(&self) -> u128 {
        self.basis.composite_modulus()
    }

    pub fn can_rescale(&self, chain: &ModulusChain) -> bool {
        chain.has_next_level(self.level)
    }

    pub fn rescale_divisor(&self, chain: &ModulusChain) -> Option<Modulus> {
        self.assert_matches_chain(chain);
        chain.dropped_modulus(self.level)
    }

    pub fn after_rescale(&self, chain: &ModulusChain) -> Self {
        self.assert_matches_chain(chain);

        let divisor = self
            .rescale_divisor(chain)
            .expect("cannot rescale the final CKKS chain level");

        Self::new(chain, self.level + 1, self.scale / divisor.value() as f64)
    }

    /// Advances one CKKS level while accounting for a Grafting sprout-width
    /// transition.
    ///
    /// Besides division by the trailing odd CKKS modulus, changing the
    /// sprout width rescales the represented value by
    /// `2^(target_sprout_bits - source_sprout_bits)`.
    pub fn after_grafted_rescale(
        &self,
        chain: &ModulusChain,
        source_sprout_bits: u32,
        target_sprout_bits: u32,
    ) -> Self {
        let ordinary = self.after_rescale(chain);

        let exponent = i64::from(target_sprout_bits) - i64::from(source_sprout_bits);
        let sprout_factor = 2.0_f64.powf(exponent as f64);

        Self::new(chain, ordinary.level(), ordinary.scale() * sprout_factor)
    }

    pub fn after_multiply(&self, rhs: &Self, chain: &ModulusChain) -> Self {
        self.assert_matches_chain(chain);
        rhs.assert_matches_chain(chain);

        assert_eq!(
            self.level, rhs.level,
            "CKKS multiplication requires matching levels"
        );
        assert_eq!(
            self.basis, rhs.basis,
            "CKKS multiplication requires matching modulus bases"
        );

        Self::new(chain, self.level, self.scale * rhs.scale)
    }

    pub fn assert_matches_chain(&self, chain: &ModulusChain) {
        assert!(
            self.level <= chain.max_level(),
            "CKKS level must exist in the modulus chain"
        );
        assert_eq!(
            &self.basis,
            chain.level(self.level),
            "CKKS basis must match its modulus-chain level"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> ModulusChain {
        ModulusChain::from_top_basis(ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]))
    }

    #[test]
    fn top_state_matches_top_basis() {
        let chain = chain();
        let state = CkksChainState::top(&chain, 65_537.0);

        assert_eq!(state.level(), 0);
        assert_eq!(state.basis(), chain.level(0));
        assert_eq!(state.scale(), 65_537.0);
    }

    #[test]
    fn multiplication_squares_scale_without_changing_level() {
        let chain = chain();
        let lhs = CkksChainState::top(&chain, 256.0);
        let rhs = CkksChainState::top(&chain, 256.0);

        let product = lhs.after_multiply(&rhs, &chain);

        assert_eq!(product.level(), 0);
        assert_eq!(product.basis(), chain.level(0));
        assert_eq!(product.scale(), 65_536.0);
    }

    #[test]
    fn rescale_drops_exactly_one_chain_modulus() {
        let chain = chain();
        let state = CkksChainState::top(&chain, 65_537.0 * 65_537.0);

        assert_eq!(state.rescale_divisor(&chain), Some(Modulus::new(65_537)));

        let next = state.after_rescale(&chain);

        assert_eq!(next.level(), 1);
        assert_eq!(next.basis(), chain.level(1));
        assert_eq!(next.scale(), 65_537.0);
    }

    #[test]
    fn two_successive_rescales_follow_chain() {
        let chain = chain();

        let initial_scale = 65_537.0 * 40_961.0;
        let level0 = CkksChainState::top(&chain, initial_scale);

        let level1 = level0.after_rescale(&chain);
        assert_eq!(level1.level(), 1);
        assert_eq!(level1.basis(), chain.level(1));
        assert_eq!(level1.scale(), 40_961.0);

        let level2 = level1.after_rescale(&chain);
        assert_eq!(level2.level(), 2);
        assert_eq!(level2.basis(), chain.level(2));
        assert_eq!(level2.scale(), 1.0);
        assert!(!level2.can_rescale(&chain));
    }

    #[test]
    #[should_panic(expected = "cannot rescale the final CKKS chain level")]
    fn final_level_cannot_rescale() {
        let chain = chain();
        let state = CkksChainState::new(&chain, chain.max_level(), 1.0);

        let _ = state.after_rescale(&chain);
    }
}
