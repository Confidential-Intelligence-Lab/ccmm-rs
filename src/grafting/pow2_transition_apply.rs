use super::{
    inv_rs_power_of_two, rescale_odd_nutrient, rs_power_of_two, Pow2RnsPolynomial,
    Pow2SproutTransition,
};

/// Applies one complete power-of-two Grafting modulus transition.
///
/// The transition is factored into:
///
/// 1. exact division by the trailing odd nutrient;
/// 2. adjustment of the sprout size with RS or Inv-RS.
///
/// No floating-point arithmetic is used.
pub fn apply_pow2_sprout_transition(
    polynomial: &Pow2RnsPolynomial,
    transition: &Pow2SproutTransition,
) -> Pow2RnsPolynomial {
    transition.source().assert_matches(polynomial);

    let after_odd = rescale_odd_nutrient(polynomial);

    let source_bits = transition.source().sprout_bits();

    let target_bits = transition.target().sprout_bits();

    let result = if target_bits > source_bits {
        inv_rs_power_of_two(&after_odd, target_bits - source_bits)
    } else if target_bits < source_bits {
        rs_power_of_two(&after_odd, source_bits - target_bits)
    } else {
        after_odd
    };

    transition.target().assert_matches(&result);

    result
}

/// Independent exact integer reference for the composed transition.
///
/// Semantics:
///
/// 1. remove the residue modulo the odd nutrient and divide by it;
/// 2. if the target sprout is larger, multiply by the corresponding
///    power of two;
/// 3. if the target sprout is smaller, apply the exact RS definition.
pub fn pow2_transition_reference_coefficients(
    coefficients: &[u128],
    transition: &Pow2SproutTransition,
) -> Vec<u128> {
    let nutrient = u128::from(transition.nutrient().value());

    let source_bits = transition.source().sprout_bits();

    let target_bits = transition.target().sprout_bits();

    coefficients
        .iter()
        .map(|&coefficient| {
            let after_odd = (coefficient - coefficient % nutrient) / nutrient;

            if target_bits > source_bits {
                let shift = target_bits - source_bits;

                let target_modulus = transition.target().composite_modulus();

                after_odd
                    .checked_mul(1_u128 << shift)
                    .expect("reference Inv-RS multiplication exceeds u128")
                    % target_modulus
            } else if target_bits < source_bits {
                let shift = source_bits - target_bits;

                let divisor = 1_u128 << shift;

                let low = after_odd % divisor;

                (after_odd - low) / divisor
            } else {
                after_odd
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grafting::{Pow2GraftedBasis, Pow2SproutTransition};
    use crate::ring::{Modulus, ModulusBasis};

    fn basis(bits: u32) -> Pow2GraftedBasis {
        Pow2GraftedBasis::new(
            ModulusBasis::new(vec![
                Modulus::new(12_289),
                Modulus::new(40_961),
                Modulus::new(65_537),
            ]),
            bits,
        )
    }

    #[test]
    fn equal_sprout_size_is_odd_rescale_only() {
        let transition = Pow2SproutTransition::new(basis(12), 12);

        let modulus = transition.source().composite_modulus();

        let coefficients = vec![0, 1, 17, 42, 65_536, 65_537, 1_000_000, modulus / 3];

        let source = transition.source().from_coefficients(&coefficients);

        let actual = apply_pow2_sprout_transition(&source, &transition);

        let expected = pow2_transition_reference_coefficients(&coefficients, &transition);

        assert_eq!(actual.reconstruct_coefficients(), expected);
    }

    #[test]
    fn growing_sprout_matches_reference() {
        let transition = Pow2SproutTransition::new(basis(8), 13);

        let modulus = transition.source().composite_modulus();

        let coefficients = vec![0, 1, 17, 42, 65_537, 1_000_003, modulus / 5, modulus - 1];

        let source = transition.source().from_coefficients(&coefficients);

        assert_eq!(
            apply_pow2_sprout_transition(&source, &transition,).reconstruct_coefficients(),
            pow2_transition_reference_coefficients(&coefficients, &transition,)
        );
    }

    #[test]
    fn shrinking_sprout_matches_reference() {
        let transition = Pow2SproutTransition::new(basis(16), 11);

        let modulus = transition.source().composite_modulus();

        let coefficients = vec![0, 1, 17, 42, 65_537, 1_000_003, modulus / 7, modulus - 1];

        let source = transition.source().from_coefficients(&coefficients);

        assert_eq!(
            apply_pow2_sprout_transition(&source, &transition,).reconstruct_coefficients(),
            pow2_transition_reference_coefficients(&coefficients, &transition,)
        );
    }

    #[test]
    fn result_uses_exact_target_basis() {
        for (source_bits, target_bits) in [(8_u32, 8_u32), (8, 13), (16, 11)] {
            let transition = Pow2SproutTransition::new(basis(source_bits), target_bits);

            let source = transition.source().from_coefficients(&[1, 2, 3, 4]);

            let result = apply_pow2_sprout_transition(&source, &transition);

            transition.target().assert_matches(&result);
        }
    }

    #[test]
    fn differential_campaign_matches_reference() {
        for (source_bits, target_bits) in [
            (4_u32, 4_u32),
            (4, 8),
            (8, 4),
            (8, 12),
            (12, 8),
            (12, 16),
            (16, 12),
            (16, 16),
        ] {
            let transition = Pow2SproutTransition::new(basis(source_bits), target_bits);

            let modulus = transition.source().composite_modulus();

            for seed in 0_u128..32 {
                let coefficients: Vec<u128> = (0..32_u128)
                    .map(|index| (97 * seed + 31 * index + 7 * index * index + 11) % modulus)
                    .collect();

                let source = transition.source().from_coefficients(&coefficients);

                let actual =
                    apply_pow2_sprout_transition(&source, &transition).reconstruct_coefficients();

                let expected = pow2_transition_reference_coefficients(&coefficients, &transition);

                assert_eq!(
                    actual, expected,
                    "power-of-two transition mismatch: \
                     source_bits={source_bits}, \
                     target_bits={target_bits}, \
                     seed={seed}"
                );
            }
        }
    }

    #[test]
    fn two_successive_transitions_match_two_reference_steps() {
        let first = Pow2SproutTransition::new(basis(8), 12);

        let second = Pow2SproutTransition::new(first.target().clone(), 8);

        let modulus = first.source().composite_modulus();

        let coefficients: Vec<u128> = (0..32_u128)
            .map(|index| (101 + 43 * index + 5 * index * index) % modulus)
            .collect();

        let source = first.source().from_coefficients(&coefficients);

        let actual_first = apply_pow2_sprout_transition(&source, &first);

        let actual_second = apply_pow2_sprout_transition(&actual_first, &second);

        let expected_first = pow2_transition_reference_coefficients(&coefficients, &first);

        let expected_second = pow2_transition_reference_coefficients(&expected_first, &second);

        assert_eq!(actual_second.reconstruct_coefficients(), expected_second);
    }
}
