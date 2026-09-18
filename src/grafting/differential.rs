use super::{
    apply_pow2_sprout_transition, decrypt_hybrid_quadratic_raw, decrypt_hybrid_raw,
    hybrid_relinearize, HybridMultiplicationKey, HybridQuadraticCiphertext, MixedGadgetLayout,
    Pow2GraftedBasis, Pow2SproutTransition,
};
use crate::ring::{Modulus, ModulusBasis};

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use super::*;

    fn source_basis(bits: u32) -> Pow2GraftedBasis {
        Pow2GraftedBasis::new(
            ModulusBasis::new(vec![
                Modulus::new(12_289),
                Modulus::new(40_961),
                Modulus::new(65_537),
            ]),
            bits,
        )
    }

    fn ternary_secret() -> Vec<i8> {
        vec![-1, 0, 1, 1, -1, 0, 1, 0]
    }

    fn make_component(basis: &Pow2GraftedBasis, seed: u128) -> super::super::Pow2RnsPolynomial {
        let q = basis.composite_modulus();

        let coefficients: Vec<u128> = (0..8_u128)
            .map(|index| (97 * seed + 31 * index + 7 * index * index + 11) % q)
            .collect();

        basis.from_coefficients(&coefficients)
    }

    fn source_quadratic(basis: &Pow2GraftedBasis, seed: u128) -> HybridQuadraticCiphertext {
        HybridQuadraticCiphertext::new(
            make_component(basis, seed + 1),
            make_component(basis, seed + 101),
            make_component(basis, seed + 1001),
        )
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

    #[test]
    fn transitioned_quadratic_uses_exact_target_basis() {
        for (source_bits, target_bits) in [(8_u32, 8_u32), (8, 12), (12, 8), (12, 16), (16, 12)] {
            let transition = Pow2SproutTransition::new(source_basis(source_bits), target_bits);

            let source = source_quadratic(transition.source(), 7);

            let target = transition_quadratic(&source, &transition);

            for component in [target.c0(), target.c1(), target.c2()] {
                transition.target().assert_matches(component);
            }
        }
    }

    #[test]
    fn transition_then_relinearize_preserves_target_quadratic_semantics() {
        let transition = Pow2SproutTransition::new(source_basis(8), 12);

        let source = source_quadratic(transition.source(), 17);

        let target_quadratic = transition_quadratic(&source, &transition);

        let layout = MixedGadgetLayout::new(
            transition.target().ordinary_basis().clone(),
            vec![1, 1],
            transition.target().sprout_bits(),
        );

        let mut rng = ChaCha20Rng::seed_from_u64(1);

        let key = HybridMultiplicationKey::generate_with_rng(
            8,
            16,
            0,
            &ternary_secret(),
            layout,
            &mut rng,
        );

        let linear = hybrid_relinearize(&target_quadratic, &key);

        assert_eq!(
            decrypt_hybrid_raw(&linear, &ternary_secret(),),
            decrypt_hybrid_quadratic_raw(&target_quadratic, &ternary_secret(),)
        );
    }

    #[test]
    fn full_grafting_differential_campaign() {
        for (source_bits, target_bits) in [
            (4_u32, 4_u32),
            (4, 8),
            (8, 4),
            (8, 8),
            (8, 12),
            (12, 8),
            (12, 12),
            (12, 16),
            (16, 12),
            (16, 16),
        ] {
            for seed in 0_u64..32 {
                let transition = Pow2SproutTransition::new(source_basis(source_bits), target_bits);

                let source = source_quadratic(transition.source(), u128::from(seed));

                let target_quadratic = transition_quadratic(&source, &transition);

                for block_sizes in [vec![2], vec![1, 1]] {
                    let layout = MixedGadgetLayout::new(
                        transition.target().ordinary_basis().clone(),
                        block_sizes,
                        target_bits,
                    );

                    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xA5A5_5A5A);

                    let key = HybridMultiplicationKey::generate_with_rng(
                        8,
                        16,
                        0,
                        &ternary_secret(),
                        layout,
                        &mut rng,
                    );

                    let linear = hybrid_relinearize(&target_quadratic, &key);

                    let actual = decrypt_hybrid_raw(&linear, &ternary_secret());

                    let expected =
                        decrypt_hybrid_quadratic_raw(&target_quadratic, &ternary_secret());

                    assert_eq!(
                        actual, expected,
                        "full Grafting mismatch: \
                         source_bits={source_bits}, \
                         target_bits={target_bits}, \
                         seed={seed}"
                    );
                }
            }
        }
    }

    #[test]
    fn two_transition_campaign_preserves_semantics() {
        for seed in 0_u64..16 {
            let first = Pow2SproutTransition::new(source_basis(8), 12);

            let second = Pow2SproutTransition::new(first.target().clone(), 8);

            let source = source_quadratic(first.source(), u128::from(seed));

            let middle = transition_quadratic(&source, &first);

            let target = transition_quadratic(&middle, &second);

            let layout = MixedGadgetLayout::new(
                second.target().ordinary_basis().clone(),
                vec![1],
                second.target().sprout_bits(),
            );

            let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xCAFE_BABE);

            let key = HybridMultiplicationKey::generate_with_rng(
                8,
                16,
                0,
                &ternary_secret(),
                layout,
                &mut rng,
            );

            let linear = hybrid_relinearize(&target, &key);

            assert_eq!(
                decrypt_hybrid_raw(&linear, &ternary_secret(),),
                decrypt_hybrid_quadratic_raw(&target, &ternary_secret(),),
                "two-transition Grafting mismatch for seed {seed}"
            );
        }
    }

    #[test]
    fn target_relinearization_is_independent_of_gadget_partition() {
        let transition = Pow2SproutTransition::new(source_basis(12), 8);

        let source = source_quadratic(transition.source(), 123);

        let target = transition_quadratic(&source, &transition);

        let expected = decrypt_hybrid_quadratic_raw(&target, &ternary_secret());

        for block_sizes in [vec![2], vec![1, 1]] {
            let layout = MixedGadgetLayout::new(
                transition.target().ordinary_basis().clone(),
                block_sizes,
                8,
            );

            let mut rng = ChaCha20Rng::seed_from_u64(99);

            let key = HybridMultiplicationKey::generate_with_rng(
                8,
                16,
                0,
                &ternary_secret(),
                layout,
                &mut rng,
            );

            let actual = decrypt_hybrid_raw(&hybrid_relinearize(&target, &key), &ternary_secret());

            assert_eq!(actual, expected);
        }
    }
}
