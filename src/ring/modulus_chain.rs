use super::{Modulus, ModulusBasis};

/// Ordered sequence of modulus bases representing ciphertext levels.
///
/// Level 0 is the highest-modulus level. Increasing the level index
/// removes one trailing modulus at a time:
///
/// level 0: [q0, q1, q2, q3]
/// level 1: [q0, q1, q2]
/// level 2: [q0, q1]
/// level 3: [q0]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulusChain {
    levels: Vec<ModulusBasis>,
}

impl ModulusChain {
    /// Builds the canonical prefix-dropping chain from a top-level basis.
    pub fn from_top_basis(top: ModulusBasis) -> Self {
        let mut levels = Vec::with_capacity(top.len());

        for length in (1..=top.len()).rev() {
            levels.push(top.prefix(length));
        }

        Self { levels }
    }

    /// Constructs a chain from explicit levels.
    ///
    /// Every level after the first must equal the previous level with
    /// exactly one trailing modulus removed.
    pub fn from_levels(levels: Vec<ModulusBasis>) -> Self {
        assert!(
            !levels.is_empty(),
            "modulus chain must contain at least one level"
        );

        for pair in levels.windows(2) {
            let upper = &pair[0];
            let lower = &pair[1];

            assert_eq!(
                upper.len(),
                lower.len() + 1,
                "adjacent modulus-chain levels must differ by one limb"
            );

            assert_eq!(
                upper.prefix(lower.len()),
                *lower,
                "lower modulus-chain level must be an exact prefix"
            );
        }

        Self { levels }
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    pub fn level(&self, index: usize) -> &ModulusBasis {
        &self.levels[index]
    }

    pub fn levels(&self) -> &[ModulusBasis] {
        &self.levels
    }

    pub fn top(&self) -> &ModulusBasis {
        &self.levels[0]
    }

    pub fn bottom(&self) -> &ModulusBasis {
        self.levels
            .last()
            .expect("validated modulus chain is nonempty")
    }

    pub fn max_level(&self) -> usize {
        self.levels.len() - 1
    }

    pub fn has_next_level(&self, level: usize) -> bool {
        level + 1 < self.levels.len()
    }

    pub fn next_level(&self, level: usize) -> Option<&ModulusBasis> {
        self.levels.get(level + 1)
    }

    /// Modulus removed by the transition `level -> level + 1`.
    pub fn dropped_modulus(&self, level: usize) -> Option<Modulus> {
        let upper = self.levels.get(level)?;
        let lower = self.levels.get(level + 1)?;

        assert_eq!(
            upper.len(),
            lower.len() + 1,
            "validated adjacent levels differ by one modulus"
        );

        Some(upper.modulus(upper.len() - 1))
    }

    pub fn composite_modulus(&self, level: usize) -> u128 {
        self.level(level).composite_modulus()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn top_basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ])
    }

    #[test]
    fn canonical_chain_has_one_level_per_limb() {
        let chain = ModulusChain::from_top_basis(top_basis());

        assert_eq!(chain.len(), 3);
        assert_eq!(chain.max_level(), 2);
    }

    #[test]
    fn canonical_chain_drops_trailing_moduli() {
        let chain = ModulusChain::from_top_basis(top_basis());

        assert_eq!(
            chain.level(0).moduli(),
            &[
                Modulus::new(12_289),
                Modulus::new(40_961),
                Modulus::new(65_537),
            ]
        );

        assert_eq!(
            chain.level(1).moduli(),
            &[Modulus::new(12_289), Modulus::new(40_961),]
        );

        assert_eq!(chain.level(2).moduli(), &[Modulus::new(12_289)]);
    }

    #[test]
    fn top_and_bottom_are_correct() {
        let chain = ModulusChain::from_top_basis(top_basis());

        assert_eq!(chain.top(), chain.level(0));
        assert_eq!(chain.bottom(), chain.level(2));
    }

    #[test]
    fn next_level_is_explicit() {
        let chain = ModulusChain::from_top_basis(top_basis());

        assert_eq!(chain.next_level(0), Some(chain.level(1)));

        assert_eq!(chain.next_level(1), Some(chain.level(2)));

        assert_eq!(chain.next_level(2), None);
    }

    #[test]
    fn dropped_modulus_matches_transition() {
        let chain = ModulusChain::from_top_basis(top_basis());

        assert_eq!(chain.dropped_modulus(0), Some(Modulus::new(65_537)));

        assert_eq!(chain.dropped_modulus(1), Some(Modulus::new(40_961)));

        assert_eq!(chain.dropped_modulus(2), None);
    }

    #[test]
    fn composite_modulus_decreases_by_level() {
        let chain = ModulusChain::from_top_basis(top_basis());

        assert_eq!(
            chain.composite_modulus(0),
            12_289_u128 * 40_961_u128 * 65_537_u128
        );

        assert_eq!(chain.composite_modulus(1), 12_289_u128 * 40_961_u128);

        assert_eq!(chain.composite_modulus(2), 12_289_u128);
    }

    #[test]
    fn explicit_valid_chain_is_accepted() {
        let top = top_basis();

        let chain = ModulusChain::from_levels(vec![top.clone(), top.prefix(2), top.prefix(1)]);

        assert_eq!(chain.len(), 3);
    }

    #[test]
    #[should_panic(expected = "at least one level")]
    fn rejects_empty_chain() {
        let _ = ModulusChain::from_levels(vec![]);
    }

    #[test]
    #[should_panic(expected = "differ by one limb")]
    fn rejects_skipped_level() {
        let top = top_basis();

        let _ = ModulusChain::from_levels(vec![top.clone(), top.prefix(1)]);
    }

    #[test]
    #[should_panic(expected = "exact prefix")]
    fn rejects_non_prefix_level() {
        let top = top_basis();

        let wrong = ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(65_537)]);

        let _ = ModulusChain::from_levels(vec![top, wrong]);
    }
}
