pub use hybrid_key_switch::{
    decrypt_hybrid_quadratic_raw, decrypt_hybrid_raw, hybrid_multiply_relinearize_with_backend,
    hybrid_relinearize, hybrid_relinearize_helper_prime, hybrid_relinearize_prepared,
    hybrid_relinearize_with_backend, hybrid_tensor, HybridEvaluationKeyEntry,
    HybridMultiplicationKey, HybridQuadraticCiphertext, HybridRelinearizationBackend,
    PreparedHybridEvaluationKeyEntry, PreparedHybridMultiplicationKey,
};
pub mod hybrid_key_switch;
pub use mixed_gadget::{MixedGadgetDecomposition, MixedGadgetLayout};
pub mod mixed_gadget;
pub use hybrid_rlwe::{
    decrypt_pow2_raw, encrypt_pow2_raw_with_noise_rng, encrypt_pow2_raw_with_rng,
    project_ternary_secret_pow2, reconstruct_hybrid_plaintext, HybridRlweCiphertext,
    Pow2RlweCiphertext,
};
pub mod hybrid_rlwe;
pub use pow2_transition_apply::{
    apply_pow2_sprout_transition, pow2_transition_reference_coefficients,
};
pub mod pow2_transition_apply;
pub use odd_rescale::{odd_rescale_nutrient, odd_rescale_target_basis, rescale_odd_nutrient};
pub mod odd_rescale;
pub use pow2_transition::{Pow2GraftedBasis, Pow2SproutTransition};
pub mod pow2_transition;
pub use prime_resurrection::PrimeSproutKeySwitchPlan;
pub mod prime_resurrection;
pub use rns_key_switch::{
    decrypt_rns_quadratic_raw, decrypt_rns_raw, rns_relinearize, ternary_secret_coefficients,
    RnsKeySwitchKey, RnsMultiplicationKey, RnsQuadraticCiphertext, RnsRlweCiphertext,
};
pub mod rns_key_switch;
pub use rns_gadget::{RnsGadgetBlock, RnsGadgetDecomposition, RnsGadgetLayout};
pub mod rns_gadget;
pub use key_switch::{relinearize_scheduled, GadgetDigitSchedule};
pub mod key_switch;
pub use gadget::{GadgetBlock, GadgetLayout};
pub mod gadget;
pub use pow2_rs::{inv_rs_power_of_two, rs_power_of_two, Pow2RnsPolynomial};
pub mod pow2_rs;
pub use helper_prime::{HelperPrimeNttPlan, HelperPrimeNttPolynomial};
pub mod helper_prime;
pub use pow2::Pow2Polynomial;
pub mod pow2;
pub use universal_sprout::{UniversalSprout, UniversalSproutPolicy};
pub mod universal_sprout;
pub use rescale::{grafted_q_rescale, q_rescale_reference};
pub mod rescale;
pub use transition::{ResurrectionPool, SproutTransition};
pub mod transition;

use crate::ring::{Modulus, ModulusBasis, RnsPolynomial};

/// Separately managed factor of a Grafting ciphertext modulus.
///
/// A sprout may consist of one or more pairwise-coprime factors.
/// Later Grafting stages will define how sprouts are consumed,
/// resurrected, and used for rescaling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sprout {
    basis: ModulusBasis,
}

impl Sprout {
    pub fn new(factors: Vec<Modulus>) -> Self {
        Self {
            basis: ModulusBasis::new(factors),
        }
    }

    pub fn basis(&self) -> &ModulusBasis {
        &self.basis
    }

    pub fn factors(&self) -> &[Modulus] {
        self.basis.moduli()
    }

    pub fn len(&self) -> usize {
        self.basis.len()
    }

    pub fn is_empty(&self) -> bool {
        self.basis.is_empty()
    }

    /// Composite sprout modulus.
    pub fn modulus(&self) -> u128 {
        self.basis.composite_modulus()
    }

    /// Bit length of the composite sprout modulus.
    pub fn bit_length(&self) -> u32 {
        let value = self.modulus();

        u128::BITS - value.leading_zeros()
    }
}

/// Modulus representation used by Grafting.
///
/// The ordinary basis contains the regular RNS limbs. The sprout is
/// represented separately so future level transitions can manipulate it
/// without changing the logical ordinary basis abstraction.
///
/// The full ciphertext basis is the ordered concatenation
///
/// `[ordinary..., sprout...]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraftedBasis {
    ordinary: ModulusBasis,
    sprout: Sprout,
    full: ModulusBasis,
}

impl GraftedBasis {
    pub fn new(ordinary: ModulusBasis, sprout: Sprout) -> Self {
        let mut full_moduli = ordinary.moduli().to_vec();

        full_moduli.extend_from_slice(sprout.factors());

        // Constructing the combined basis validates that ordinary and
        // sprout factors are pairwise coprime across the entire modulus.
        let full = ModulusBasis::new(full_moduli);

        Self {
            ordinary,
            sprout,
            full,
        }
    }

    pub fn ordinary_basis(&self) -> &ModulusBasis {
        &self.ordinary
    }

    pub fn sprout(&self) -> &Sprout {
        &self.sprout
    }

    pub fn full_basis(&self) -> &ModulusBasis {
        &self.full
    }

    pub fn ordinary_modulus(&self) -> u128 {
        self.ordinary.composite_modulus()
    }

    pub fn sprout_modulus(&self) -> u128 {
        self.sprout.modulus()
    }

    pub fn composite_modulus(&self) -> u128 {
        self.full.composite_modulus()
    }

    /// Encodes canonical integer coefficients in the complete grafted
    /// modulus basis.
    pub fn from_coefficients(&self, coefficients: &[u128]) -> RnsPolynomial {
        RnsPolynomial::from_coefficients(self.full.moduli().to_vec(), coefficients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ordinary_basis() -> ModulusBasis {
        ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(40_961)])
    }

    fn sprout() -> Sprout {
        Sprout::new(vec![Modulus::new(65_537), Modulus::new(114_689)])
    }

    #[test]
    fn sprout_preserves_ordered_factors() {
        let sprout = sprout();

        assert_eq!(
            sprout.factors(),
            &[Modulus::new(65_537), Modulus::new(114_689),]
        );
    }

    #[test]
    fn sprout_modulus_is_product_of_factors() {
        let sprout = sprout();

        assert_eq!(sprout.modulus(), 65_537_u128 * 114_689_u128);
    }

    #[test]
    fn single_factor_sprout_is_supported() {
        let sprout = Sprout::new(vec![Modulus::new(65_537)]);

        assert_eq!(sprout.len(), 1);
        assert_eq!(sprout.modulus(), 65_537);
    }

    #[test]
    fn grafted_basis_concatenates_ordinary_and_sprout_factors() {
        let grafted = GraftedBasis::new(ordinary_basis(), sprout());

        assert_eq!(
            grafted.full_basis().moduli(),
            &[
                Modulus::new(12_289),
                Modulus::new(40_961),
                Modulus::new(65_537),
                Modulus::new(114_689),
            ]
        );
    }

    #[test]
    fn grafted_composite_modulus_factors_correctly() {
        let grafted = GraftedBasis::new(ordinary_basis(), sprout());

        assert_eq!(
            grafted.composite_modulus(),
            grafted.ordinary_modulus() * grafted.sprout_modulus()
        );
    }

    #[test]
    fn grafted_representation_roundtrips_through_crt() {
        let grafted = GraftedBasis::new(ordinary_basis(), sprout());

        let q = grafted.composite_modulus();

        let coefficients = vec![0, 1, 17, 42, 65_536, 1_000_000, q / 2, q - 1];

        let polynomial = grafted.from_coefficients(&coefficients);

        assert_eq!(polynomial.basis(), grafted.full_basis());

        assert_eq!(polynomial.reconstruct_coefficients(), coefficients);
    }

    #[test]
    fn sprout_bit_length_matches_composite_value() {
        let sprout = sprout();

        let expected = u128::BITS - sprout.modulus().leading_zeros();

        assert_eq!(sprout.bit_length(), expected);
    }

    #[test]
    #[should_panic(expected = "pairwise coprime")]
    fn rejects_overlap_between_ordinary_basis_and_sprout() {
        let ordinary = ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(40_961)]);

        let overlapping = Sprout::new(vec![Modulus::new(40_961), Modulus::new(65_537)]);

        let _ = GraftedBasis::new(ordinary, overlapping);
    }

    #[test]
    #[should_panic(expected = "pairwise coprime")]
    fn rejects_non_coprime_sprout_factors() {
        let _ = Sprout::new(vec![Modulus::new(15), Modulus::new(21)]);
    }
}

#[cfg(test)]
mod differential;
