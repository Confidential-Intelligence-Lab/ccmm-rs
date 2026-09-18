use crate::ring::{Modulus, ModulusBasis};

use super::{GraftedBasis, Sprout};

/// Factors that are permitted to reappear as sprouts.
///
/// In a full Grafting implementation this set is derived from the
/// switching-key modulus configuration. Keeping it explicit here makes
/// modulus resurrection auditable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResurrectionPool {
    basis: ModulusBasis,
}

impl ResurrectionPool {
    pub fn new(factors: Vec<Modulus>) -> Self {
        Self {
            basis: ModulusBasis::new(factors),
        }
    }

    pub fn basis(&self) -> &ModulusBasis {
        &self.basis
    }

    pub fn contains(&self, modulus: Modulus) -> bool {
        self.basis.contains(modulus)
    }

    pub fn permits(&self, sprout: &Sprout) -> bool {
        sprout.factors().iter().all(|&factor| self.contains(factor))
    }
}

/// Exact structural description of one Grafting modulus transition.
///
/// A transition:
///
/// 1. consumes the trailing ordinary RNS modulus as a nutrient;
/// 2. replaces the current sprout by a new sprout;
/// 3. records the exact rational modulus ratio.
///
/// No ciphertext coefficient transformation is performed here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SproutTransition {
    source: GraftedBasis,
    target: GraftedBasis,
    nutrient: Modulus,
    ratio_numerator: u128,
    ratio_denominator: u128,
}

impl SproutTransition {
    pub fn new(
        source: GraftedBasis,
        next_sprout: Sprout,
        resurrection_pool: &ResurrectionPool,
    ) -> Self {
        assert!(
            source.ordinary_basis().len() > 1,
            "sprout transition requires at least two ordinary RNS limbs"
        );

        assert!(
            resurrection_pool.permits(&next_sprout),
            "next sprout contains factor outside resurrection pool"
        );

        let nutrient = source
            .ordinary_basis()
            .modulus(source.ordinary_basis().len() - 1);

        let target_ordinary = source.ordinary_basis().without_last();

        let target = GraftedBasis::new(target_ordinary, next_sprout);

        // Q_source / Q_target
        //
        // = (Qordinary * Sold)
        //   /
        //   ((Qordinary / nutrient) * Snew)
        //
        // = nutrient * Sold / Snew.
        let numerator = u128::from(nutrient.value())
            .checked_mul(source.sprout_modulus())
            .expect("transition ratio numerator exceeds u128");

        let denominator = target.sprout_modulus();

        let divisor = gcd_u128(numerator, denominator);

        let ratio_numerator = numerator / divisor;

        let ratio_denominator = denominator / divisor;

        let transition = Self {
            source,
            target,
            nutrient,
            ratio_numerator,
            ratio_denominator,
        };

        transition.assert_exact_modulus_identity();

        transition
    }

    pub fn source(&self) -> &GraftedBasis {
        &self.source
    }

    pub fn target(&self) -> &GraftedBasis {
        &self.target
    }

    pub fn nutrient(&self) -> Modulus {
        self.nutrient
    }

    /// Reduced numerator of Q_source / Q_target.
    pub fn ratio_numerator(&self) -> u128 {
        self.ratio_numerator
    }

    /// Reduced denominator of Q_source / Q_target.
    pub fn ratio_denominator(&self) -> u128 {
        self.ratio_denominator
    }

    pub fn assert_exact_modulus_identity(&self) {
        let lhs = self
            .source
            .composite_modulus()
            .checked_mul(self.ratio_denominator)
            .expect("transition identity lhs exceeds u128");

        let rhs = self
            .target
            .composite_modulus()
            .checked_mul(self.ratio_numerator)
            .expect("transition identity rhs exceeds u128");

        assert_eq!(
            lhs, rhs,
            "Grafting modulus transition ratio is inconsistent"
        );
    }
}

fn gcd_u128(mut lhs: u128, mut rhs: u128) -> u128 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }

    lhs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::ModulusBasis;

    fn ordinary_basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ])
    }

    fn source_sprout() -> Sprout {
        Sprout::new(vec![Modulus::new(114_689)])
    }

    fn alternate_sprout() -> Sprout {
        Sprout::new(vec![Modulus::new(147_457)])
    }

    fn pool() -> ResurrectionPool {
        ResurrectionPool::new(vec![Modulus::new(114_689), Modulus::new(147_457)])
    }

    #[test]
    fn transition_consumes_trailing_ordinary_modulus() {
        let source = GraftedBasis::new(ordinary_basis(), source_sprout());

        let transition = SproutTransition::new(source, alternate_sprout(), &pool());

        assert_eq!(transition.nutrient(), Modulus::new(65_537));

        assert_eq!(
            transition.target().ordinary_basis().moduli(),
            &[Modulus::new(12_289), Modulus::new(40_961),]
        );
    }

    #[test]
    fn transition_replaces_sprout() {
        let source = GraftedBasis::new(ordinary_basis(), source_sprout());

        let transition = SproutTransition::new(source, alternate_sprout(), &pool());

        assert_eq!(transition.target().sprout(), &alternate_sprout());
    }

    #[test]
    fn transition_modulus_ratio_is_exact() {
        let source = GraftedBasis::new(ordinary_basis(), source_sprout());

        let transition = SproutTransition::new(source, alternate_sprout(), &pool());

        transition.assert_exact_modulus_identity();

        assert_eq!(
            transition.source().composite_modulus() * transition.ratio_denominator(),
            transition.target().composite_modulus() * transition.ratio_numerator()
        );
    }

    #[test]
    fn transition_ratio_matches_factor_accounting() {
        let source = GraftedBasis::new(ordinary_basis(), source_sprout());

        let transition = SproutTransition::new(source, alternate_sprout(), &pool());

        let raw_numerator = 65_537_u128 * 114_689_u128;

        let raw_denominator = 147_457_u128;

        let divisor = gcd_u128(raw_numerator, raw_denominator);

        assert_eq!(transition.ratio_numerator(), raw_numerator / divisor);

        assert_eq!(transition.ratio_denominator(), raw_denominator / divisor);
    }

    #[test]
    fn previously_used_sprout_can_be_resurrected() {
        let initial = GraftedBasis::new(ordinary_basis(), source_sprout());

        let first = SproutTransition::new(initial, alternate_sprout(), &pool());

        let second = SproutTransition::new(first.target().clone(), source_sprout(), &pool());

        assert_eq!(second.target().sprout(), &source_sprout());

        assert_eq!(
            second.target().ordinary_basis().moduli(),
            &[Modulus::new(12_289)]
        );

        second.assert_exact_modulus_identity();
    }

    #[test]
    fn resurrection_does_not_accumulate_sprout_factors() {
        let initial = GraftedBasis::new(ordinary_basis(), source_sprout());

        let first = SproutTransition::new(initial, alternate_sprout(), &pool());

        let second = SproutTransition::new(first.target().clone(), source_sprout(), &pool());

        assert_eq!(first.target().sprout().len(), 1);

        assert_eq!(second.target().sprout().len(), 1);
    }

    #[test]
    #[should_panic(expected = "outside resurrection pool")]
    fn rejects_unregistered_sprout_factor() {
        let source = GraftedBasis::new(ordinary_basis(), source_sprout());

        let unknown = Sprout::new(vec![Modulus::new(163_841)]);

        let _ = SproutTransition::new(source, unknown, &pool());
    }

    #[test]
    #[should_panic(expected = "at least two ordinary")]
    fn rejects_transition_after_ordinary_basis_is_exhausted() {
        let source = GraftedBasis::new(
            ModulusBasis::new(vec![Modulus::new(12_289)]),
            source_sprout(),
        );

        let _ = SproutTransition::new(source, alternate_sprout(), &pool());
    }
}
