use crate::grafting::{
    apply_pow2_sprout_transition, hybrid_relinearize_with_backend, hybrid_tensor,
    HelperPrimeNttPlan, HybridMultiplicationKey, HybridQuadraticCiphertext,
    HybridRelinearizationBackend, Pow2SproutTransition, PreparedHybridMultiplicationKey,
};
use crate::ring::ModulusChain;

use super::HybridCkksCiphertext;

/// Backend-specific execution material for one hybrid CKKS multiplication.
#[derive(Clone, Copy)]
pub struct HybridCkksMultiplyConfig<'a> {
    pub target_sprout_bits: u32,
    pub key: &'a HybridMultiplicationKey,
    pub backend: HybridRelinearizationBackend,
    pub helper_plan: Option<&'a HelperPrimeNttPlan>,
    pub prepared_key: Option<&'a PreparedHybridMultiplicationKey>,
}

/// Multiplies two hybrid CKKS ciphertexts, performs one complete
/// Grafting modulus transition, and relinearizes under target-level
/// evaluation material.
///
/// Ordering:
///
/// 1. tensor product at the source Grafting basis;
/// 2. transition every degree-two component to the target basis;
/// 3. relinearize using target-level evaluation material;
/// 4. advance the CKKS chain state by one level.
pub fn multiply_transition_relinearize_hybrid_ckks(
    lhs: &HybridCkksCiphertext,
    rhs: &HybridCkksCiphertext,
    config: HybridCkksMultiplyConfig<'_>,
    chain: &ModulusChain,
) -> HybridCkksCiphertext {
    lhs.state().assert_matches_chain(chain);
    rhs.state().assert_matches_chain(chain);

    assert_eq!(
        lhs.level(),
        rhs.level(),
        "hybrid CKKS multiplication requires matching levels"
    );

    assert_eq!(
        lhs.ordinary_basis(),
        rhs.ordinary_basis(),
        "hybrid CKKS multiplication requires matching ordinary bases"
    );

    assert_eq!(
        lhs.sprout_bits(),
        rhs.sprout_bits(),
        "hybrid CKKS multiplication requires matching sprout sizes"
    );

    assert!(
        lhs.state().can_rescale(chain),
        "hybrid CKKS multiplication requires a remaining CKKS level"
    );

    let source_basis = lhs.grafted_basis();

    let transition = Pow2SproutTransition::new(source_basis, config.target_sprout_bits);

    assert_eq!(
        transition.target().ordinary_basis(),
        chain.level(lhs.level() + 1),
        "Grafting transition target must match next CKKS chain level"
    );

    assert_eq!(
        config.key.layout().ordinary_basis(),
        transition.target().ordinary_basis(),
        "hybrid multiplication key must use target CKKS ordinary basis"
    );

    assert_eq!(
        config.key.layout().sprout_bits(),
        transition.target().sprout_bits(),
        "hybrid multiplication key must use target Grafting sprout size"
    );

    let product = hybrid_tensor(lhs.inner(), rhs.inner());

    let transitioned = transition_quadratic(&product, &transition);

    let inner = hybrid_relinearize_with_backend(
        &transitioned,
        config.key,
        config.backend,
        config.helper_plan,
        config.prepared_key,
    );

    let product_state = lhs.state().after_multiply(rhs.state(), chain);

    let next_state =
        product_state.after_grafted_rescale(chain, lhs.sprout_bits(), config.target_sprout_bits);

    HybridCkksCiphertext::new(inner, next_state)
}

fn transition_quadratic(
    product: &HybridQuadraticCiphertext,
    transition: &Pow2SproutTransition,
) -> HybridQuadraticCiphertext {
    HybridQuadraticCiphertext::new(
        apply_pow2_sprout_transition(product.c0(), transition),
        apply_pow2_sprout_transition(product.c1(), transition),
        apply_pow2_sprout_transition(product.c2(), transition),
    )
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::ckks::{CkksChainState, HybridCkksCiphertext, HybridCkksMultiplyConfig};
    use crate::grafting::{
        decrypt_hybrid_quadratic_raw, decrypt_hybrid_raw, encrypt_pow2_raw_with_rng,
        project_ternary_secret_pow2, HelperPrimeNttPlan, HybridMultiplicationKey,
        HybridRelinearizationBackend, HybridRlweCiphertext, MixedGadgetLayout, Pow2Polynomial,
        PreparedHybridMultiplicationKey, RnsRlweCiphertext,
    };
    use crate::ring::{Modulus, ModulusBasis, ModulusChain, Polynomial};
    use crate::rlwe::{encrypt_raw_with_rng, RlweParameters, SecretKey};

    use super::*;

    const HELPER_PRIME: u64 = 2_013_265_921;

    fn chain() -> ModulusChain {
        ModulusChain::from_top_basis(ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]))
    }

    fn secret() -> Vec<i8> {
        vec![-1, 0, 1, 1, 0, -1, 1, 0]
    }

    fn project_secret(modulus: Modulus, coefficients: &[i8]) -> SecretKey {
        SecretKey::from_polynomial(Polynomial::new(
            modulus,
            coefficients
                .iter()
                .map(|&value| match value {
                    -1 => modulus.value() - 1,
                    0 => 0,
                    1 => 1,
                    _ => panic!("secret must be ternary"),
                })
                .collect(),
        ))
    }

    fn hybrid_encrypt_zero(
        chain: &ModulusChain,
        level: usize,
        sprout_bits: u32,
        scale: f64,
        secret: &[i8],
        seed: u64,
    ) -> HybridCkksCiphertext {
        let ordinary_basis = chain.level(level);

        let degree = secret.len();

        let ordinary_limbs = ordinary_basis
            .moduli()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, modulus)| {
                let params = RlweParameters::new(degree, modulus, 2, 0);

                let key = project_secret(modulus, secret);

                let message = Polynomial::zero(modulus, degree);

                let mut rng = ChaCha20Rng::seed_from_u64(seed ^ index as u64);

                encrypt_raw_with_rng(params, &key, &message, &mut rng)
            })
            .collect();

        let ordinary = RnsRlweCiphertext::from_limbs(ordinary_limbs);

        let sprout_secret = project_ternary_secret_pow2(sprout_bits, secret);

        let sprout_message = Pow2Polynomial::new(sprout_bits, vec![0; degree]);

        let mut sprout_rng = ChaCha20Rng::seed_from_u64(seed ^ 0xFFFF);

        let sprout = encrypt_pow2_raw_with_rng(&sprout_secret, &sprout_message, &mut sprout_rng);

        HybridCkksCiphertext::new(
            HybridRlweCiphertext::new(ordinary, sprout),
            CkksChainState::new(chain, level, scale),
        )
    }

    fn target_key(
        chain: &ModulusChain,
        target_level: usize,
        sprout_bits: u32,
        secret: &[i8],
        seed: u64,
    ) -> HybridMultiplicationKey {
        let basis = chain.level(target_level).clone();

        let blocks = match basis.len() {
            2 => vec![1, 1],
            1 => vec![1],
            _ => panic!("unexpected target basis size"),
        };

        let layout = MixedGadgetLayout::new(basis, blocks, sprout_bits);

        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        HybridMultiplicationKey::generate_with_rng(secret.len(), 2, 0, secret, layout, &mut rng)
    }

    #[test]
    fn hybrid_ckks_backends_match_after_transition() {
        let chain = chain();

        let secret = secret();

        let lhs = hybrid_encrypt_zero(&chain, 0, 8, 65_537.0, &secret, 0xCE00);

        let rhs = hybrid_encrypt_zero(&chain, 0, 8, 65_537.0, &secret, 0xCE01);

        let target_bits = 8;

        let key = target_key(&chain, 1, target_bits, &secret, 0xCE02);

        let helper_plan =
            HelperPrimeNttPlan::new(target_bits, secret.len(), Modulus::new(HELPER_PRIME));

        let prepared = PreparedHybridMultiplicationKey::prepare(&key, &helper_plan);

        let direct = multiply_transition_relinearize_hybrid_ckks(
            &lhs,
            &rhs,
            HybridCkksMultiplyConfig {
                target_sprout_bits: target_bits,
                key: &key,
                backend: HybridRelinearizationBackend::Direct,
                helper_plan: None,
                prepared_key: None,
            },
            &chain,
        );

        let helper = multiply_transition_relinearize_hybrid_ckks(
            &lhs,
            &rhs,
            HybridCkksMultiplyConfig {
                target_sprout_bits: target_bits,
                key: &key,
                backend: HybridRelinearizationBackend::HelperPrime,
                helper_plan: Some(&helper_plan),
                prepared_key: None,
            },
            &chain,
        );

        let cached = multiply_transition_relinearize_hybrid_ckks(
            &lhs,
            &rhs,
            HybridCkksMultiplyConfig {
                target_sprout_bits: target_bits,
                key: &key,
                backend: HybridRelinearizationBackend::Prepared,
                helper_plan: Some(&helper_plan),
                prepared_key: Some(&prepared),
            },
            &chain,
        );

        assert_eq!(direct.inner(), helper.inner());

        assert_eq!(direct.inner(), cached.inner());

        assert_eq!(direct.level(), 1);

        assert_eq!(direct.ordinary_basis(), chain.level(1));

        assert_eq!(direct.sprout_bits(), target_bits);
    }

    #[test]
    fn hybrid_ckks_scale_advances_like_ckks_multiply_rescale() {
        let chain = chain();

        let secret = secret();

        let lhs_scale = 256.0;

        let rhs_scale = 512.0;

        let lhs = hybrid_encrypt_zero(&chain, 0, 8, lhs_scale, &secret, 0xCE10);

        let rhs = hybrid_encrypt_zero(&chain, 0, 8, rhs_scale, &secret, 0xCE11);

        let key = target_key(&chain, 1, 8, &secret, 0xCE12);

        let result = multiply_transition_relinearize_hybrid_ckks(
            &lhs,
            &rhs,
            HybridCkksMultiplyConfig {
                target_sprout_bits: 8,
                key: &key,
                backend: HybridRelinearizationBackend::Direct,
                helper_plan: None,
                prepared_key: None,
            },
            &chain,
        );

        let nutrient = chain.level(0).modulus(chain.level(0).len() - 1).value() as f64;

        let expected = lhs_scale * rhs_scale / nutrient;

        assert!(
            (result.scale() - expected).abs() < 1.0e-12,
            "actual={}, expected={expected}",
            result.scale(),
        );
    }

    fn assert_grafted_scale_transition(
        source_bits: u32,
        target_bits: u32,
        expected_pow2_factor: f64,
        seed: u64,
    ) {
        let chain = chain();
        let secret = secret();

        let lhs_scale = 256.0;
        let rhs_scale = 512.0;

        let lhs = hybrid_encrypt_zero(&chain, 0, source_bits, lhs_scale, &secret, seed);
        let rhs = hybrid_encrypt_zero(&chain, 0, source_bits, rhs_scale, &secret, seed ^ 1);

        let key = target_key(&chain, 1, target_bits, &secret, seed ^ 2);

        let result = multiply_transition_relinearize_hybrid_ckks(
            &lhs,
            &rhs,
            HybridCkksMultiplyConfig {
                target_sprout_bits: target_bits,
                key: &key,
                backend: HybridRelinearizationBackend::Direct,
                helper_plan: None,
                prepared_key: None,
            },
            &chain,
        );

        let nutrient = chain.level(0).modulus(chain.level(0).len() - 1).value() as f64;
        let expected = lhs_scale * rhs_scale / nutrient * expected_pow2_factor;

        assert!(
            (result.scale() - expected).abs() < 1.0e-12,
            "source_bits={source_bits}, target_bits={target_bits}, actual={}, expected={expected}",
            result.scale(),
        );

        assert_eq!(result.level(), 1);
        assert_eq!(result.sprout_bits(), target_bits);
        assert_eq!(result.ordinary_basis(), chain.level(1));
    }

    #[test]
    fn hybrid_ckks_scale_tracks_shrinking_sprout_8_to_6() {
        assert_grafted_scale_transition(8, 6, 0.25, 0xCE30);
    }

    #[test]
    fn hybrid_ckks_scale_tracks_growing_sprout_6_to_8() {
        assert_grafted_scale_transition(6, 8, 4.0, 0xCE40);
    }

    #[test]
    fn hybrid_ckks_scale_tracks_shrinking_sprout_8_to_4() {
        assert_grafted_scale_transition(8, 4, 1.0 / 16.0, 0xCE50);
    }

    #[test]
    fn hybrid_ckks_direct_path_preserves_transitioned_quadratic_semantics() {
        let chain = chain();

        let secret = secret();

        let lhs = hybrid_encrypt_zero(&chain, 0, 8, 65_537.0, &secret, 0xCE20);

        let rhs = hybrid_encrypt_zero(&chain, 0, 8, 65_537.0, &secret, 0xCE21);

        let transition = crate::grafting::Pow2SproutTransition::new(lhs.grafted_basis(), 8);

        let product = crate::grafting::hybrid_tensor(lhs.inner(), rhs.inner());

        let transitioned = HybridQuadraticCiphertext::new(
            crate::grafting::apply_pow2_sprout_transition(product.c0(), &transition),
            crate::grafting::apply_pow2_sprout_transition(product.c1(), &transition),
            crate::grafting::apply_pow2_sprout_transition(product.c2(), &transition),
        );

        let key = target_key(&chain, 1, 8, &secret, 0xCE22);

        let result = multiply_transition_relinearize_hybrid_ckks(
            &lhs,
            &rhs,
            HybridCkksMultiplyConfig {
                target_sprout_bits: 8,
                key: &key,
                backend: HybridRelinearizationBackend::Direct,
                helper_plan: None,
                prepared_key: None,
            },
            &chain,
        );

        let observed = decrypt_hybrid_raw(result.inner(), &secret);

        let expected = decrypt_hybrid_quadratic_raw(&transitioned, &secret);

        assert_eq!(observed, expected);
    }

    #[test]
    #[should_panic(expected = "hybrid CKKS multiplication requires matching sprout sizes")]
    fn hybrid_ckks_rejects_mismatched_sprout_widths() {
        let chain = chain();
        let secret = secret();

        let lhs = hybrid_encrypt_zero(&chain, 0, 8, 65_537.0, &secret, 0xCF00);

        let rhs = hybrid_encrypt_zero(&chain, 0, 6, 65_537.0, &secret, 0xCF01);

        let key = target_key(&chain, 1, 8, &secret, 0xCF02);

        let _ = multiply_transition_relinearize_hybrid_ckks(
            &lhs,
            &rhs,
            HybridCkksMultiplyConfig {
                target_sprout_bits: 8,
                key: &key,
                backend: HybridRelinearizationBackend::Direct,
                helper_plan: None,
                prepared_key: None,
            },
            &chain,
        );
    }

    #[test]
    #[should_panic(expected = "hybrid multiplication key must use target Grafting sprout size")]
    fn hybrid_ckks_rejects_wrong_target_sprout_key() {
        let chain = chain();
        let secret = secret();

        let lhs = hybrid_encrypt_zero(&chain, 0, 8, 65_537.0, &secret, 0xCF10);

        let rhs = hybrid_encrypt_zero(&chain, 0, 8, 65_537.0, &secret, 0xCF11);

        /*
         * Transition requests an 8-bit target sprout but the key
         * was generated for a 6-bit target sprout.
         */
        let key = target_key(&chain, 1, 6, &secret, 0xCF12);

        let _ = multiply_transition_relinearize_hybrid_ckks(
            &lhs,
            &rhs,
            HybridCkksMultiplyConfig {
                target_sprout_bits: 8,
                key: &key,
                backend: HybridRelinearizationBackend::Direct,
                helper_plan: None,
                prepared_key: None,
            },
            &chain,
        );
    }

    #[test]
    #[should_panic(expected = "helper-prime hybrid backend requires a helper-prime plan")]
    fn hybrid_ckks_helper_backend_requires_helper_plan() {
        let chain = chain();
        let secret = secret();

        let lhs = hybrid_encrypt_zero(&chain, 0, 8, 65_537.0, &secret, 0xCF20);

        let rhs = hybrid_encrypt_zero(&chain, 0, 8, 65_537.0, &secret, 0xCF21);

        let key = target_key(&chain, 1, 8, &secret, 0xCF22);

        let _ = multiply_transition_relinearize_hybrid_ckks(
            &lhs,
            &rhs,
            HybridCkksMultiplyConfig {
                target_sprout_bits: 8,
                key: &key,
                backend: HybridRelinearizationBackend::HelperPrime,
                helper_plan: None,
                prepared_key: None,
            },
            &chain,
        );
    }

    #[test]
    #[should_panic(expected = "prepared hybrid backend requires a prepared multiplication key")]
    fn hybrid_ckks_prepared_backend_requires_prepared_key() {
        let chain = chain();
        let secret = secret();

        let lhs = hybrid_encrypt_zero(&chain, 0, 8, 65_537.0, &secret, 0xCF30);

        let rhs = hybrid_encrypt_zero(&chain, 0, 8, 65_537.0, &secret, 0xCF31);

        let key = target_key(&chain, 1, 8, &secret, 0xCF32);

        let helper = HelperPrimeNttPlan::new(8, secret.len(), Modulus::new(HELPER_PRIME));

        let _ = multiply_transition_relinearize_hybrid_ckks(
            &lhs,
            &rhs,
            HybridCkksMultiplyConfig {
                target_sprout_bits: 8,
                key: &key,
                backend: HybridRelinearizationBackend::Prepared,
                helper_plan: Some(&helper),
                prepared_key: None,
            },
            &chain,
        );
    }

    #[test]
    #[should_panic(expected = "hybrid CKKS multiplication requires matching levels")]
    fn hybrid_ckks_rejects_mismatched_levels() {
        let chain = chain();
        let secret = secret();

        let lhs = hybrid_encrypt_zero(&chain, 0, 8, 65_537.0, &secret, 0xCF40);

        let rhs = hybrid_encrypt_zero(&chain, 1, 8, 65_537.0, &secret, 0xCF41);

        let key = target_key(&chain, 1, 8, &secret, 0xCF42);

        let _ = multiply_transition_relinearize_hybrid_ckks(
            &lhs,
            &rhs,
            HybridCkksMultiplyConfig {
                target_sprout_bits: 8,
                key: &key,
                backend: HybridRelinearizationBackend::Direct,
                helper_plan: None,
                prepared_key: None,
            },
            &chain,
        );
    }

    #[test]
    #[should_panic(expected = "hybrid CKKS multiplication requires a remaining CKKS level")]
    fn hybrid_ckks_rejects_multiplication_at_terminal_level() {
        let chain = chain();
        let secret = secret();

        let terminal = chain.max_level();

        let lhs = hybrid_encrypt_zero(&chain, terminal, 8, 65_537.0, &secret, 0xCF50);

        let rhs = hybrid_encrypt_zero(&chain, terminal, 8, 65_537.0, &secret, 0xCF51);

        /*
         * This key should never actually be consumed because
         * terminal-level rejection occurs first.
         */
        let key = target_key(&chain, terminal, 8, &secret, 0xCF52);

        let _ = multiply_transition_relinearize_hybrid_ckks(
            &lhs,
            &rhs,
            HybridCkksMultiplyConfig {
                target_sprout_bits: 8,
                key: &key,
                backend: HybridRelinearizationBackend::Direct,
                helper_plan: None,
                prepared_key: None,
            },
            &chain,
        );
    }
}
