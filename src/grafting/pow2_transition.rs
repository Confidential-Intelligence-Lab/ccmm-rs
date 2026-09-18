use crate::ring::{Modulus, ModulusBasis};

use super::Pow2RnsPolynomial;

/// Grafting modulus basis with an ordinary odd-prime RNS basis and one
/// separately-managed power-of-two sprout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pow2GraftedBasis {
    ordinary: ModulusBasis,
    sprout_bits: u32,
}

impl Pow2GraftedBasis {
    pub fn new(ordinary: ModulusBasis, sprout_bits: u32) -> Self {
        assert!(
            sprout_bits > 0 && sprout_bits <= 63,
            "power-of-two sprout bits must be in 1..=63"
        );

        Self {
            ordinary,
            sprout_bits,
        }
    }

    pub fn ordinary_basis(&self) -> &ModulusBasis {
        &self.ordinary
    }

    pub fn sprout_bits(&self) -> u32 {
        self.sprout_bits
    }

    pub fn sprout_modulus(&self) -> u128 {
        1_u128 << self.sprout_bits
    }

    pub fn ordinary_modulus(&self) -> u128 {
        self.ordinary.composite_modulus()
    }

    pub fn composite_modulus(&self) -> u128 {
        self.ordinary_modulus()
            .checked_mul(self.sprout_modulus())
            .expect("power-of-two grafted modulus exceeds u128")
    }

    pub fn from_coefficients(&self, coefficients: &[u128]) -> Pow2RnsPolynomial {
        Pow2RnsPolynomial::from_coefficients(&self.ordinary, self.sprout_bits, coefficients)
    }

    pub fn assert_matches(&self, polynomial: &Pow2RnsPolynomial) {
        assert_eq!(
            polynomial.ordinary_basis(),
            &self.ordinary,
            "ordinary basis must match power-of-two grafted basis"
        );

        assert_eq!(
            polynomial.sprout_bits(),
            self.sprout_bits,
            "sprout size must match power-of-two grafted basis"
        );
    }
}

/// Structural transition between two power-of-two Grafting levels.
///
/// One trailing ordinary RNS modulus is consumed as the nutrient while
/// the power-of-two sprout may change size.
///
/// No ciphertext arithmetic is performed by this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pow2SproutTransition {
    source: Pow2GraftedBasis,
    target: Pow2GraftedBasis,
    nutrient: Modulus,
    ratio_numerator: u128,
    ratio_denominator: u128,
}

impl Pow2SproutTransition {
    pub fn new(source: Pow2GraftedBasis, target_sprout_bits: u32) -> Self {
        assert!(
            source.ordinary_basis().len() > 1,
            "power-of-two sprout transition requires at least two ordinary limbs"
        );

        assert!(
            target_sprout_bits > 0 && target_sprout_bits <= 63,
            "target power-of-two sprout bits must be in 1..=63"
        );

        let nutrient = source
            .ordinary_basis()
            .modulus(source.ordinary_basis().len() - 1);

        let target =
            Pow2GraftedBasis::new(source.ordinary_basis().without_last(), target_sprout_bits);

        let raw_numerator = u128::from(nutrient.value())
            .checked_mul(source.sprout_modulus())
            .expect("power-of-two transition numerator exceeds u128");

        let raw_denominator = target.sprout_modulus();

        let divisor = gcd_u128(raw_numerator, raw_denominator);

        let ratio_numerator = raw_numerator / divisor;

        let ratio_denominator = raw_denominator / divisor;

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

    pub fn source(&self) -> &Pow2GraftedBasis {
        &self.source
    }

    pub fn target(&self) -> &Pow2GraftedBasis {
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
            .expect("power-of-two transition lhs exceeds u128");

        let rhs = self
            .target
            .composite_modulus()
            .checked_mul(self.ratio_numerator)
            .expect("power-of-two transition rhs exceeds u128");

        assert_eq!(
            lhs, rhs,
            "power-of-two Grafting modulus ratio is inconsistent"
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

    fn source_basis(sprout_bits: u32) -> Pow2GraftedBasis {
        Pow2GraftedBasis::new(
            ModulusBasis::new(vec![
                Modulus::new(12_289),
                Modulus::new(40_961),
                Modulus::new(65_537),
            ]),
            sprout_bits,
        )
    }

    #[test]
    fn basis_composite_modulus_factors_correctly() {
        let basis = source_basis(12);

        assert_eq!(
            basis.composite_modulus(),
            12_289_u128 * 40_961_u128 * 65_537_u128 * (1_u128 << 12)
        );
    }

    #[test]
    fn basis_roundtrips_coefficients() {
        let basis = source_basis(12);

        let q = basis.composite_modulus();

        let coefficients = vec![0, 1, 17, 42, 4_095, 65_536, q / 2, q - 1];

        let polynomial = basis.from_coefficients(&coefficients);

        basis.assert_matches(&polynomial);

        assert_eq!(polynomial.reconstruct_coefficients(), coefficients);
    }

    #[test]
    fn transition_consumes_trailing_nutrient() {
        let transition = Pow2SproutTransition::new(source_basis(12), 17);

        assert_eq!(transition.nutrient(), Modulus::new(65_537));

        assert_eq!(
            transition.target().ordinary_basis().moduli(),
            &[Modulus::new(12_289), Modulus::new(40_961),]
        );
    }

    #[test]
    fn transition_changes_sprout_size_exactly() {
        let transition = Pow2SproutTransition::new(source_basis(12), 17);

        assert_eq!(transition.source().sprout_bits(), 12);

        assert_eq!(transition.target().sprout_bits(), 17);
    }

    #[test]
    fn transition_ratio_matches_factor_accounting() {
        let transition = Pow2SproutTransition::new(source_basis(12), 17);

        let raw_numerator = 65_537_u128 * (1_u128 << 12);

        let raw_denominator = 1_u128 << 17;

        let divisor = gcd_u128(raw_numerator, raw_denominator);

        assert_eq!(transition.ratio_numerator(), raw_numerator / divisor);

        assert_eq!(transition.ratio_denominator(), raw_denominator / divisor);
    }

    #[test]
    fn transition_modulus_identity_is_exact() {
        for source_bits in [4_u32, 8, 12, 16, 20] {
            for target_bits in [4_u32, 8, 12, 16, 20] {
                let transition = Pow2SproutTransition::new(source_basis(source_bits), target_bits);

                transition.assert_exact_modulus_identity();

                assert_eq!(
                    transition.source().composite_modulus() * transition.ratio_denominator(),
                    transition.target().composite_modulus() * transition.ratio_numerator()
                );
            }
        }
    }

    #[test]
    fn repeated_transitions_do_not_accumulate_ordinary_limbs() {
        let first = Pow2SproutTransition::new(source_basis(8), 12);

        let second = Pow2SproutTransition::new(first.target().clone(), 8);

        assert_eq!(first.target().ordinary_basis().len(), 2);

        assert_eq!(second.target().ordinary_basis().len(), 1);

        assert_eq!(second.target().sprout_bits(), 8);
    }

    #[test]
    #[should_panic(expected = "at least two ordinary")]
    fn rejects_transition_after_ordinary_basis_exhaustion() {
        let source = Pow2GraftedBasis::new(ModulusBasis::new(vec![Modulus::new(12_289)]), 8);

        let _ = Pow2SproutTransition::new(source, 12);
    }

    #[test]
    #[should_panic(expected = "1..=63")]
    fn rejects_invalid_source_sprout_size() {
        let _ = Pow2GraftedBasis::new(
            ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(40_961)]),
            64,
        );
    }
}
