use crate::ring::RnsPolynomial;

use super::SproutTransition;

/// Applies a correctness-oriented Grafting Q-Rescale to an RNS polynomial.
///
/// For source modulus `Q` and target modulus `Q'`, each centered
/// coefficient `c` is mapped to
///
/// ```text
/// round(c * Q' / Q) mod Q'
/// ```
///
/// `SproutTransition` stores the exact reduced ratio
///
/// ```text
/// Q / Q' = numerator / denominator
/// ```
///
/// so the implementation equivalently evaluates
///
/// ```text
/// round(c * denominator / numerator)
/// ```
///
/// No floating-point arithmetic is used.
///
/// This is the polynomial-level reference semantics. Ciphertext noise,
/// scale metadata, and key-switching integration are handled in later
/// Grafting stages.
pub fn grafted_q_rescale(
    polynomial: &RnsPolynomial,
    transition: &SproutTransition,
) -> RnsPolynomial {
    assert_eq!(
        polynomial.basis(),
        transition.source().full_basis(),
        "polynomial basis must match Grafting transition source"
    );

    let source_modulus = transition.source().composite_modulus();

    let target_modulus = transition.target().composite_modulus();

    let numerator = transition.ratio_numerator();

    let denominator = transition.ratio_denominator();

    let coefficients = polynomial.reconstruct_coefficients();

    let rescaled: Vec<u128> = coefficients
        .into_iter()
        .map(|coefficient| {
            let centered = center(coefficient, source_modulus);

            let rounded_magnitude = mul_div_round(centered.magnitude, denominator, numerator);

            canonicalize_signed(centered.negative, rounded_magnitude, target_modulus)
        })
        .collect();

    RnsPolynomial::from_coefficients(
        transition.target().full_basis().moduli().to_vec(),
        &rescaled,
    )
}

/// Independent direct reference using Q'/Q rather than the reduced
/// transition ratio.
pub fn q_rescale_reference(
    polynomial: &RnsPolynomial,
    transition: &SproutTransition,
) -> RnsPolynomial {
    assert_eq!(
        polynomial.basis(),
        transition.source().full_basis(),
        "polynomial basis must match Grafting transition source"
    );

    let source_modulus = transition.source().composite_modulus();

    let target_modulus = transition.target().composite_modulus();

    let coefficients = polynomial.reconstruct_coefficients();

    let rescaled: Vec<u128> = coefficients
        .into_iter()
        .map(|coefficient| {
            let centered = center(coefficient, source_modulus);

            let rounded_magnitude =
                mul_div_round(centered.magnitude, target_modulus, source_modulus);

            canonicalize_signed(centered.negative, rounded_magnitude, target_modulus)
        })
        .collect();

    RnsPolynomial::from_coefficients(
        transition.target().full_basis().moduli().to_vec(),
        &rescaled,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Centered {
    negative: bool,
    magnitude: u128,
}

/// Converts the canonical representative in `[0,Q)` into centered form.
///
/// All current Grafting moduli are odd, so there is no ambiguous `Q/2`
/// representative.
fn center(value: u128, modulus: u128) -> Centered {
    assert!(
        value < modulus,
        "coefficient must be canonical modulo source modulus"
    );

    if value <= modulus / 2 {
        Centered {
            negative: false,
            magnitude: value,
        }
    } else {
        Centered {
            negative: true,
            magnitude: modulus - value,
        }
    }
}

/// Computes `round(value * multiplier / divisor)` using exact integer
/// arithmetic.
///
/// Half-way cases are rounded away from zero; sign handling is external.
fn mul_div_round(value: u128, multiplier: u128, divisor: u128) -> u128 {
    assert!(divisor > 0, "rescale divisor must be positive");

    let product = value
        .checked_mul(multiplier)
        .expect("Q-Rescale intermediate exceeds u128");

    let quotient = product / divisor;
    let remainder = product % divisor;

    // `remainder >= ceil(divisor / 2)` avoids evaluating
    // `2 * remainder`, which could overflow.
    let round_threshold = divisor / 2 + divisor % 2;

    if remainder >= round_threshold {
        quotient
            .checked_add(1)
            .expect("Q-Rescale rounding exceeds u128")
    } else {
        quotient
    }
}

fn canonicalize_signed(negative: bool, magnitude: u128, modulus: u128) -> u128 {
    let reduced = magnitude % modulus;

    if negative && reduced != 0 {
        modulus - reduced
    } else {
        reduced
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grafting::{GraftedBasis, ResurrectionPool, Sprout};
    use crate::ring::{Modulus, ModulusBasis};

    fn transition() -> SproutTransition {
        let ordinary = ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]);

        let old_sprout = Sprout::new(vec![Modulus::new(114_689)]);

        let new_sprout = Sprout::new(vec![Modulus::new(147_457)]);

        let pool = ResurrectionPool::new(vec![Modulus::new(114_689), Modulus::new(147_457)]);

        SproutTransition::new(GraftedBasis::new(ordinary, old_sprout), new_sprout, &pool)
    }

    fn polynomial(transition: &SproutTransition, coefficients: &[u128]) -> RnsPolynomial {
        RnsPolynomial::from_coefficients(
            transition.source().full_basis().moduli().to_vec(),
            coefficients,
        )
    }

    #[test]
    fn zero_rescales_to_zero() {
        let transition = transition();

        let input = polynomial(&transition, &[0; 8]);

        let output = grafted_q_rescale(&input, &transition);

        assert_eq!(output.reconstruct_coefficients(), vec![0; 8]);
    }

    #[test]
    fn grafted_rescale_matches_direct_q_ratio_reference() {
        let transition = transition();

        let source_q = transition.source().composite_modulus();

        let coefficients = vec![
            0,
            1,
            17,
            42,
            source_q / 8,
            source_q / 3,
            source_q - 1,
            source_q - 42,
        ];

        let input = polynomial(&transition, &coefficients);

        assert_eq!(
            grafted_q_rescale(&input, &transition,),
            q_rescale_reference(&input, &transition,)
        );
    }

    #[test]
    fn positive_and_negative_centered_coefficients_are_symmetric() {
        let transition = transition();

        let source_q = transition.source().composite_modulus();

        let target_q = transition.target().composite_modulus();

        let magnitude = 12_345_u128;

        let input = polynomial(&transition, &[magnitude, source_q - magnitude]);

        let output = grafted_q_rescale(&input, &transition).reconstruct_coefficients();

        assert_eq!(
            output[1],
            if output[0] == 0 {
                0
            } else {
                target_q - output[0]
            }
        );
    }

    #[test]
    fn rescale_rounding_error_is_at_most_half_source_unit() {
        let transition = transition();

        let source_q = transition.source().composite_modulus();

        let target_q = transition.target().composite_modulus();

        for value in [0_u128, 1, 7, 42, 1_001, 100_003, source_q / 7, source_q / 3] {
            let input = polynomial(&transition, &[value]);

            let output = grafted_q_rescale(&input, &transition).reconstruct_coefficients()[0];

            // These test values are positive centered representatives.
            //
            // If y = round(x Q'/Q), then
            //
            // |y Q - x Q'| <= Q/2.
            let lhs = output
                .checked_mul(source_q)
                .expect("test product exceeds u128");

            let rhs = value
                .checked_mul(target_q)
                .expect("test product exceeds u128");

            let error = lhs.abs_diff(rhs);

            assert!(
                error <= source_q / 2 + 1,
                "Q-Rescale rounding bound exceeded"
            );
        }
    }

    #[test]
    fn rescaled_polynomial_uses_target_grafted_basis() {
        let transition = transition();

        let input = polynomial(&transition, &[1, 2, 3, 4]);

        let output = grafted_q_rescale(&input, &transition);

        assert_eq!(output.basis(), transition.target().full_basis());
    }

    #[test]
    fn deterministic_campaign_matches_independent_reference() {
        let transition = transition();

        let source_q = transition.source().composite_modulus();

        for seed in 0_u128..64 {
            let coefficients: Vec<u128> = (0..32_u128)
                .map(|index| (97 * seed + 31 * index + 7 * index * index + 11) % source_q)
                .collect();

            let input = polynomial(&transition, &coefficients);

            assert_eq!(
                grafted_q_rescale(&input, &transition,),
                q_rescale_reference(&input, &transition,),
                "Q-Rescale mismatch for seed {seed}"
            );
        }
    }

    #[test]
    fn two_successive_grafted_rescales_match_two_reference_rescales() {
        let first = transition();

        let second_sprout = Sprout::new(vec![Modulus::new(114_689)]);

        let pool = ResurrectionPool::new(vec![Modulus::new(114_689), Modulus::new(147_457)]);

        let second = SproutTransition::new(first.target().clone(), second_sprout, &pool);

        let source_q = first.source().composite_modulus();

        let coefficients = vec![1_u128, 17, 42, 1_001, 100_003, source_q / 5];

        let input = polynomial(&first, &coefficients);

        let actual_first = grafted_q_rescale(&input, &first);

        let actual_second = grafted_q_rescale(&actual_first, &second);

        let reference_first = q_rescale_reference(&input, &first);

        let reference_second = q_rescale_reference(&reference_first, &second);

        assert_eq!(actual_second, reference_second);
    }
}
