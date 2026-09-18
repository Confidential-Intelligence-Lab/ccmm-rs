use rand::{CryptoRng, RngCore};

use super::{
    grafted_q_rescale, rns_relinearize, RnsGadgetLayout, RnsMultiplicationKey,
    RnsQuadraticCiphertext, RnsRlweCiphertext, SproutTransition,
};

/// Prime-sprout Grafting key-switch plan.
///
/// The source and target gadget layouts are explicit because a Grafting
/// transition changes the RNS basis: one ordinary nutrient is consumed
/// and a new sprout is installed.
///
/// This stage does not claim evaluation-key reuse across incompatible
/// bases. Instead it proves that key switching remains correct after the
/// Grafting modulus transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimeSproutKeySwitchPlan {
    transition: SproutTransition,
    source_layout: RnsGadgetLayout,
    target_layout: RnsGadgetLayout,
}

impl PrimeSproutKeySwitchPlan {
    pub fn new(
        transition: SproutTransition,
        source_block_sizes: Vec<usize>,
        target_block_sizes: Vec<usize>,
    ) -> Self {
        let source_layout =
            RnsGadgetLayout::new(transition.source().full_basis().clone(), source_block_sizes);

        let target_layout =
            RnsGadgetLayout::new(transition.target().full_basis().clone(), target_block_sizes);

        assert!(
            source_layout
                .blocks()
                .last()
                .expect("source gadget layout is nonempty")
                .basis()
                .moduli()
                .ends_with(transition.source().sprout().factors()),
            "source sprout must reside in final gadget block"
        );

        assert!(
            target_layout
                .blocks()
                .last()
                .expect("target gadget layout is nonempty")
                .basis()
                .moduli()
                .ends_with(transition.target().sprout().factors()),
            "target sprout must reside in final gadget block"
        );

        Self {
            transition,
            source_layout,
            target_layout,
        }
    }

    pub fn transition(&self) -> &SproutTransition {
        &self.transition
    }

    pub fn source_layout(&self) -> &RnsGadgetLayout {
        &self.source_layout
    }

    pub fn target_layout(&self) -> &RnsGadgetLayout {
        &self.target_layout
    }

    /// Applies the polynomial-level Grafting Q-Rescale independently to
    /// every component of a degree-2 ciphertext.
    pub fn transition_quadratic(&self, source: &RnsQuadraticCiphertext) -> RnsQuadraticCiphertext {
        assert_eq!(
            source.c0().basis(),
            self.transition.source().full_basis(),
            "source quadratic ciphertext must use transition source basis"
        );

        RnsQuadraticCiphertext::from_components(
            grafted_q_rescale(source.c0(), &self.transition),
            grafted_q_rescale(source.c1(), &self.transition),
            grafted_q_rescale(source.c2(), &self.transition),
        )
    }

    /// Generates an RNS multiplication key for the post-transition basis.
    pub fn generate_target_key_with_rng<R>(
        &self,
        degree: usize,
        plaintext_modulus: u64,
        noise_bound: i64,
        secret_coefficients: &[i8],
        rng: &mut R,
    ) -> RnsMultiplicationKey
    where
        R: RngCore + CryptoRng,
    {
        RnsMultiplicationKey::generate_with_rng(
            degree,
            plaintext_modulus,
            noise_bound,
            secret_coefficients,
            self.target_layout.clone(),
            rng,
        )
    }

    /// Grafting transition followed by target-basis RNS relinearization.
    pub fn transition_and_relinearize(
        &self,
        source: &RnsQuadraticCiphertext,
        target_key: &RnsMultiplicationKey,
    ) -> RnsRlweCiphertext {
        assert_eq!(
            target_key.layout(),
            &self.target_layout,
            "target multiplication-key layout must match resurrection plan"
        );

        let transitioned = self.transition_quadratic(source);

        rns_relinearize(&transitioned, target_key)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::grafting::{
        decrypt_rns_quadratic_raw, decrypt_rns_raw, ternary_secret_coefficients, GraftedBasis,
        ResurrectionPool, Sprout,
    };
    use crate::ring::{Modulus, ModulusBasis};
    use crate::rlwe::{encrypt_with_rng, tensor, RlweParameters, RlwePlaintext, SecretKey};

    use super::*;

    fn transition() -> SproutTransition {
        let ordinary = ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]);

        let source_sprout = Sprout::new(vec![Modulus::new(114_689)]);

        let target_sprout = Sprout::new(vec![Modulus::new(147_457)]);

        let pool = ResurrectionPool::new(vec![Modulus::new(114_689), Modulus::new(147_457)]);

        SproutTransition::new(
            GraftedBasis::new(ordinary, source_sprout),
            target_sprout,
            &pool,
        )
    }

    fn plan() -> PrimeSproutKeySwitchPlan {
        PrimeSproutKeySwitchPlan::new(transition(), vec![2, 2], vec![2, 1])
    }

    fn params() -> RlweParameters {
        RlweParameters::new(8, Modulus::new(12_289), 16, 0)
    }

    #[test]
    fn layouts_track_source_and_target_bases_exactly() {
        let plan = plan();

        assert_eq!(
            plan.source_layout().full_basis(),
            plan.transition().source().full_basis()
        );

        assert_eq!(
            plan.target_layout().full_basis(),
            plan.transition().target().full_basis()
        );
    }

    #[test]
    fn transition_consumes_nutrient_and_replaces_sprout() {
        let plan = plan();

        assert_eq!(plan.transition().nutrient(), Modulus::new(65_537));

        assert_eq!(
            plan.target_layout().full_basis().moduli(),
            &[
                Modulus::new(12_289),
                Modulus::new(40_961),
                Modulus::new(147_457),
            ]
        );
    }

    #[test]
    fn transitioned_quadratic_uses_target_basis() {
        let params = params();

        let mut key_rng = ChaCha20Rng::seed_from_u64(1);

        let secret = SecretKey::generate_with_rng(params, &mut key_rng);

        let plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(2);

        let mut rhs_rng = ChaCha20Rng::seed_from_u64(3);

        let product = tensor(
            &encrypt_with_rng(params, &secret, &plaintext, &mut lhs_rng),
            &encrypt_with_rng(params, &secret, &plaintext, &mut rhs_rng),
        );

        let plan = plan();

        let source = RnsQuadraticCiphertext::from_coefficient_ciphertext(
            &product,
            plan.source_layout().full_basis(),
        );

        let target = plan.transition_quadratic(&source);

        assert_eq!(target.c0().basis(), plan.target_layout().full_basis());

        assert_eq!(target.c1().basis(), plan.target_layout().full_basis());

        assert_eq!(target.c2().basis(), plan.target_layout().full_basis());
    }

    #[test]
    fn post_transition_relinearization_preserves_target_quadratic_semantics() {
        let params = params();

        let mut secret_rng = ChaCha20Rng::seed_from_u64(10);

        let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

        let ternary = ternary_secret_coefficients(&secret);

        let lhs_plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let rhs_plaintext = RlwePlaintext::new(params, vec![8, 7, 6, 5, 4, 3, 2, 1]);

        let mut lhs_rng = ChaCha20Rng::seed_from_u64(11);

        let mut rhs_rng = ChaCha20Rng::seed_from_u64(12);

        let product = tensor(
            &encrypt_with_rng(params, &secret, &lhs_plaintext, &mut lhs_rng),
            &encrypt_with_rng(params, &secret, &rhs_plaintext, &mut rhs_rng),
        );

        let plan = plan();

        let source = RnsQuadraticCiphertext::from_coefficient_ciphertext(
            &product,
            plan.source_layout().full_basis(),
        );

        let target_quadratic = plan.transition_quadratic(&source);

        let mut eval_rng = ChaCha20Rng::seed_from_u64(13);

        let target_key = plan.generate_target_key_with_rng(
            params.degree(),
            params.plaintext_modulus(),
            0,
            &ternary,
            &mut eval_rng,
        );

        let target_linear = plan.transition_and_relinearize(&source, &target_key);

        assert_eq!(
            decrypt_rns_raw(&target_linear, &ternary,),
            decrypt_rns_quadratic_raw(&target_quadratic, &ternary,)
        );
    }

    #[test]
    fn transition_key_switch_campaign_is_exact_without_noise() {
        let params = params();

        for seed in 0_u64..32 {
            let mut secret_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1111);

            let secret = SecretKey::generate_with_rng(params, &mut secret_rng);

            let ternary = ternary_secret_coefficients(&secret);

            let lhs_plaintext = RlwePlaintext::new(params, vec![1, 2, 3, 4, 5, 6, 7, 8]);

            let rhs_plaintext = RlwePlaintext::new(params, vec![8, 7, 6, 5, 4, 3, 2, 1]);

            let mut lhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x2222);

            let mut rhs_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x3333);

            let product = tensor(
                &encrypt_with_rng(params, &secret, &lhs_plaintext, &mut lhs_rng),
                &encrypt_with_rng(params, &secret, &rhs_plaintext, &mut rhs_rng),
            );

            for (source_blocks, target_blocks) in [
                (vec![3, 1], vec![2, 1]),
                (vec![2, 2], vec![1, 2]),
                (vec![1, 2, 1], vec![1, 1, 1]),
                (vec![1, 1, 1, 1], vec![1, 1, 1]),
            ] {
                let plan =
                    PrimeSproutKeySwitchPlan::new(transition(), source_blocks, target_blocks);

                let source = RnsQuadraticCiphertext::from_coefficient_ciphertext(
                    &product,
                    plan.source_layout().full_basis(),
                );

                let target_quadratic = plan.transition_quadratic(&source);

                let mut eval_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x4444);

                let target_key = plan.generate_target_key_with_rng(
                    params.degree(),
                    params.plaintext_modulus(),
                    0,
                    &ternary,
                    &mut eval_rng,
                );

                let target_linear = plan.transition_and_relinearize(&source, &target_key);

                assert_eq!(
                    decrypt_rns_raw(&target_linear, &ternary,),
                    decrypt_rns_quadratic_raw(&target_quadratic, &ternary,),
                    "post-transition key-switch mismatch for seed {seed}"
                );
            }
        }
    }
}
