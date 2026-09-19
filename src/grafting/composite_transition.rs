use num_bigint::BigUint;
use num_traits::Zero;

use crate::ring::{composite_modulus_big, CompositeModulusChain, LogicalModulusLevel};

use super::{GraftedBasis, Sprout};

/// Grafting transition driven by one logical CKKS level consumption.
///
/// The ordinary CKKS chain and the Grafting sprout remain separate:
///
/// - `CompositeModulusChain` decides which ordinary physical moduli form one
///   logical CKKS level and therefore which group is consumed;
/// - `Sprout` remains independently replaceable Grafting modulus material;
/// - the transition records the exact source/target modulus ratio using
///   arbitrary precision.
///
/// This is deliberately parallel to the legacy `SproutTransition`, whose
/// semantics consume one trailing ordinary physical modulus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeSproutTransition {
    logical_level: usize,
    source: GraftedBasis,
    target: GraftedBasis,
    dropped_ordinary_level: LogicalModulusLevel,
    ratio_numerator: BigUint,
    ratio_denominator: BigUint,
}

impl CompositeSproutTransition {
    pub fn new(
        chain: &CompositeModulusChain,
        logical_level: usize,
        source_sprout: Sprout,
        target_sprout: Sprout,
    ) -> Self {
        assert!(
            chain.has_next_level(logical_level),
            "cannot transition after the final logical CKKS level"
        );

        let source_ordinary = chain.active_basis(logical_level);
        let target_ordinary = chain.active_basis(logical_level + 1);

        let dropped_ordinary_level = chain
            .dropped_logical_level(logical_level)
            .expect("validated nonterminal logical level")
            .clone();

        let source = GraftedBasis::new(source_ordinary, source_sprout);
        let target = GraftedBasis::new(target_ordinary, target_sprout);

        let source_modulus = composite_modulus_big(source.full_basis());
        let target_modulus = composite_modulus_big(target.full_basis());

        let gcd = gcd_biguint(source_modulus.clone(), target_modulus.clone());

        let ratio_numerator = &source_modulus / &gcd;
        let ratio_denominator = &target_modulus / &gcd;

        let transition = Self {
            logical_level,
            source,
            target,
            dropped_ordinary_level,
            ratio_numerator,
            ratio_denominator,
        };

        transition.assert_exact_modulus_identity();
        transition
    }

    pub fn logical_level(&self) -> usize {
        self.logical_level
    }

    pub fn source(&self) -> &GraftedBasis {
        &self.source
    }

    pub fn target(&self) -> &GraftedBasis {
        &self.target
    }

    pub fn dropped_ordinary_level(&self) -> &LogicalModulusLevel {
        &self.dropped_ordinary_level
    }

    pub fn ratio_numerator(&self) -> &BigUint {
        &self.ratio_numerator
    }

    pub fn ratio_denominator(&self) -> &BigUint {
        &self.ratio_denominator
    }

    pub fn source_composite_modulus_big(&self) -> BigUint {
        composite_modulus_big(self.source.full_basis())
    }

    pub fn target_composite_modulus_big(&self) -> BigUint {
        composite_modulus_big(self.target.full_basis())
    }

    pub fn ordinary_rescale_divisor_big(&self) -> BigUint {
        self.dropped_ordinary_level.composite_modulus_big()
    }

    pub fn assert_exact_modulus_identity(&self) {
        let lhs = self.source_composite_modulus_big() * self.ratio_denominator();

        let rhs = self.target_composite_modulus_big() * self.ratio_numerator();

        assert_eq!(
            lhs, rhs,
            "composite Grafting modulus transition ratio is inconsistent"
        );
    }
}

fn gcd_biguint(mut lhs: BigUint, mut rhs: BigUint) -> BigUint {
    while !rhs.is_zero() {
        let remainder = &lhs % &rhs;
        lhs = rhs;
        rhs = remainder;
    }

    lhs
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;

    use crate::ring::{CompositeModulusChain, Modulus, ModulusBasis};

    use super::{CompositeSproutTransition, Sprout};

    fn ordinary_basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(97),
            Modulus::new(193),
            Modulus::new(257),
            Modulus::new(449),
            Modulus::new(577),
            Modulus::new(641),
        ])
    }

    fn chain() -> CompositeModulusChain {
        CompositeModulusChain::new(ordinary_basis(), vec![2, 2, 2])
    }

    fn source_sprout() -> Sprout {
        Sprout::new(vec![Modulus::new(769)])
    }

    fn target_sprout() -> Sprout {
        Sprout::new(vec![Modulus::new(1153)])
    }

    #[test]
    fn transition_consumes_one_complete_logical_ordinary_group() {
        let chain = chain();

        let transition =
            CompositeSproutTransition::new(&chain, 0, source_sprout(), target_sprout());

        assert_eq!(transition.source().ordinary_basis(), &chain.active_basis(0));
        assert_eq!(transition.target().ordinary_basis(), &chain.active_basis(1));

        assert_eq!(transition.dropped_ordinary_level().physical_range(), 4..6);
        assert_eq!(transition.dropped_ordinary_level().physical_limb_count(), 2);
    }

    #[test]
    fn sprout_replacement_is_independent_of_logical_grouping() {
        let chain = chain();

        let transition =
            CompositeSproutTransition::new(&chain, 0, source_sprout(), target_sprout());

        assert_eq!(transition.source().sprout().factors(), &[Modulus::new(769)]);
        assert_eq!(
            transition.target().sprout().factors(),
            &[Modulus::new(1153)]
        );

        assert_eq!(
            transition.source().ordinary_basis().len() - transition.target().ordinary_basis().len(),
            2
        );
    }

    #[test]
    fn ordinary_divisor_is_product_of_complete_logical_group() {
        let chain = chain();

        let transition =
            CompositeSproutTransition::new(&chain, 0, source_sprout(), target_sprout());

        assert_eq!(
            transition.ordinary_rescale_divisor_big(),
            BigUint::from(577_u64) * BigUint::from(641_u64)
        );
    }

    #[test]
    fn exact_transition_ratio_accounts_for_ordinary_group_and_sprout() {
        let chain = chain();

        let transition =
            CompositeSproutTransition::new(&chain, 0, source_sprout(), target_sprout());

        transition.assert_exact_modulus_identity();

        let source = transition.source_composite_modulus_big();
        let target = transition.target_composite_modulus_big();

        assert_eq!(
            source * transition.ratio_denominator(),
            target * transition.ratio_numerator()
        );
    }

    #[test]
    fn successive_transitions_follow_logical_levels_without_partial_groups() {
        let chain = chain();

        let first = CompositeSproutTransition::new(&chain, 0, source_sprout(), target_sprout());

        let second = CompositeSproutTransition::new(&chain, 1, target_sprout(), source_sprout());

        assert_eq!(
            first.target().ordinary_basis(),
            second.source().ordinary_basis()
        );

        assert_eq!(first.dropped_ordinary_level().physical_range(), 4..6);
        assert_eq!(second.dropped_ordinary_level().physical_range(), 2..4);

        assert_eq!(second.target().ordinary_basis(), &chain.active_basis(2));
        assert_eq!(second.target().ordinary_basis().len(), 2);
    }

    #[test]
    fn mixed_logical_group_sizes_are_preserved() {
        let chain = CompositeModulusChain::new(ordinary_basis(), vec![2, 1, 3]);

        let transition =
            CompositeSproutTransition::new(&chain, 0, source_sprout(), target_sprout());

        assert_eq!(transition.dropped_ordinary_level().physical_limb_count(), 3);
        assert_eq!(transition.target().ordinary_basis().len(), 3);
    }

    #[test]
    fn single_limb_groups_recover_legacy_transition_granularity() {
        let chain = CompositeModulusChain::single_limb_levels(ordinary_basis());

        let transition =
            CompositeSproutTransition::new(&chain, 0, source_sprout(), target_sprout());

        assert_eq!(transition.dropped_ordinary_level().physical_limb_count(), 1);
        assert_eq!(
            transition.source().ordinary_basis().len() - transition.target().ordinary_basis().len(),
            1
        );
    }

    #[test]
    #[should_panic(expected = "cannot transition after the final logical CKKS level")]
    fn rejects_transition_after_logical_chain_exhaustion() {
        let chain = chain();

        let _ = CompositeSproutTransition::new(
            &chain,
            chain.max_level(),
            source_sprout(),
            target_sprout(),
        );
    }
}
