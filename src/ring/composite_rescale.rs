use num_bigint::BigUint;

use super::{
    reconstruct_coefficients_big, CompositeModulusChain, Modulus, Polynomial, RnsPolynomial,
};

/// Drops one physical trailing modulus using native RNS arithmetic.
///
/// For a logical coefficient `a` and dropped modulus `q_d`,
///
/// `(a - (a mod q_d)) / q_d`
///
/// is represented directly in each surviving residue limb.
fn drop_one_trailing_modulus(polynomial: &RnsPolynomial, dropped: Modulus) -> RnsPolynomial {
    let basis = polynomial.basis();

    assert!(
        basis.len() > 1,
        "composite rescale requires at least two physical limbs"
    );

    assert_eq!(
        basis.modulus(basis.len() - 1),
        dropped,
        "composite rescale must drop the trailing physical modulus"
    );

    let next_basis = basis.prefix(basis.len() - 1);
    let dropped_residue = polynomial.residue(basis.len() - 1);
    let q_drop = dropped.value();

    let residues = next_basis
        .moduli()
        .iter()
        .copied()
        .enumerate()
        .map(|(limb_index, modulus)| {
            let qi = modulus.value();
            let inverse =
                inverse_mod_u64(q_drop % qi, qi).expect("RNS moduli must be pairwise coprime");

            let source = polynomial.residue(limb_index);

            let coefficients = source
                .coefficients()
                .iter()
                .zip(dropped_residue.coefficients())
                .map(|(&value, &dropped_value)| {
                    let dropped_mod_qi = dropped_value % qi;
                    let difference = if value >= dropped_mod_qi {
                        value - dropped_mod_qi
                    } else {
                        qi - (dropped_mod_qi - value)
                    };

                    modulus.mul(difference, inverse)
                })
                .collect();

            Polynomial::new(modulus, coefficients)
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

fn inverse_mod_u64(value: u64, modulus: u64) -> Option<u64> {
    if value == 0 || modulus <= 1 {
        return None;
    }

    let mut old_r = i128::from(value);
    let mut r = i128::from(modulus);
    let mut old_s = 1_i128;
    let mut s = 0_i128;

    while r != 0 {
        let quotient = old_r / r;

        let next_r = old_r - quotient * r;
        old_r = r;
        r = next_r;

        let next_s = old_s - quotient * s;
        old_s = s;
        s = next_s;
    }

    if old_r != 1 {
        return None;
    }

    Some(old_s.rem_euclid(i128::from(modulus)) as u64)
}

/// Consumes one logical CKKS level by dropping every physical modulus in that
/// logical group.
///
/// The transition remains RNS-native: physical factors are removed
/// sequentially using exact residue arithmetic. No full composite-`Q`
/// reconstruction is required on the execution path.
///
/// The logical divisor is the product of every physical modulus in the group
/// selected by `CompositeModulusChain::dropped_logical_level`.
pub fn rescale_composite_level_to_next(
    polynomial: &RnsPolynomial,
    chain: &CompositeModulusChain,
    level: usize,
) -> RnsPolynomial {
    assert_eq!(
        polynomial.basis(),
        &chain.active_basis(level),
        "RNS polynomial basis must match the active composite CKKS level"
    );

    let dropped = chain
        .dropped_logical_level(level)
        .expect("cannot rescale the final composite CKKS level");

    let mut current = polynomial.clone();

    // Groups are contiguous and trailing in the active basis. Drop from the
    // highest physical index downward so every step remains a prefix basis.
    for modulus in dropped.basis().moduli().iter().rev().copied() {
        current = drop_one_trailing_modulus(&current, modulus);
    }

    assert_eq!(
        current.basis(),
        &chain.active_basis(level + 1),
        "composite rescale must land on the next logical active basis"
    );

    current
}

/// Exact arbitrary-precision reference for one logical composite rescale.
///
/// For canonical `a in [0, Q)`, this computes
///
/// `floor(a / D)`
///
/// where `D` is the product of all physical moduli in the consumed logical
/// level. This is equivalent to sequential exact trailing-prime removal for
/// canonical representatives.
pub fn composite_rescale_reference_coefficients(
    polynomial: &RnsPolynomial,
    chain: &CompositeModulusChain,
    level: usize,
) -> Vec<BigUint> {
    assert_eq!(
        polynomial.basis(),
        &chain.active_basis(level),
        "reference polynomial basis must match active composite CKKS level"
    );

    let divisor = chain
        .rescale_divisor_big(level)
        .expect("cannot rescale the final composite CKKS level");

    reconstruct_coefficients_big(polynomial)
        .into_iter()
        .map(|value| value / &divisor)
        .collect()
}

/// Re-encodes exact reference coefficients in the next logical active basis.
pub fn composite_rescale_reference_polynomial(
    polynomial: &RnsPolynomial,
    chain: &CompositeModulusChain,
    level: usize,
) -> RnsPolynomial {
    let coefficients = composite_rescale_reference_coefficients(polynomial, chain, level);

    super::rns_from_big_coefficients(
        chain.active_basis(level + 1).moduli().to_vec(),
        &coefficients,
    )
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;

    use crate::ring::{
        reconstruct_coefficients_big, rns_from_big_coefficients, CompositeModulusChain, Modulus,
        ModulusBasis,
    };

    use super::{composite_rescale_reference_polynomial, rescale_composite_level_to_next};

    fn basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(97),
            Modulus::new(193),
            Modulus::new(257),
            Modulus::new(449),
            Modulus::new(577),
            Modulus::new(641),
        ])
    }

    fn coefficients() -> Vec<BigUint> {
        vec![
            BigUint::from(0_u64),
            BigUint::from(1_u64),
            BigUint::from(42_u64),
            BigUint::from(12_345_u64),
            BigUint::from(9_876_543_u64),
            BigUint::from(123_456_789_012_345_u64),
        ]
    }

    #[test]
    fn two_prime_logical_rescale_matches_biguint_reference() {
        let chain = CompositeModulusChain::new(basis(), vec![2, 2, 2]);
        let input =
            rns_from_big_coefficients(chain.active_basis(0).moduli().to_vec(), &coefficients());

        let actual = rescale_composite_level_to_next(&input, &chain, 0);
        let expected = composite_rescale_reference_polynomial(&input, &chain, 0);

        assert_eq!(actual, expected);
        assert_eq!(actual.basis(), &chain.active_basis(1));
    }

    #[test]
    fn two_successive_composite_rescales_match_exact_reference() {
        let chain = CompositeModulusChain::new(basis(), vec![2, 2, 2]);
        let input =
            rns_from_big_coefficients(chain.active_basis(0).moduli().to_vec(), &coefficients());

        let level1 = rescale_composite_level_to_next(&input, &chain, 0);
        let level2 = rescale_composite_level_to_next(&level1, &chain, 1);

        let divisor0 = chain.rescale_divisor_big(0).unwrap();
        let divisor1 = chain.rescale_divisor_big(1).unwrap();

        let expected: Vec<BigUint> = coefficients()
            .into_iter()
            .map(|value| (value / &divisor0) / &divisor1)
            .collect();

        assert_eq!(reconstruct_coefficients_big(&level2), expected);
        assert_eq!(level2.basis(), &chain.active_basis(2));
    }

    #[test]
    fn mixed_group_sizes_are_supported() {
        let chain = CompositeModulusChain::new(basis(), vec![2, 1, 3]);
        let input =
            rns_from_big_coefficients(chain.active_basis(0).moduli().to_vec(), &coefficients());

        let level1 = rescale_composite_level_to_next(&input, &chain, 0);

        assert_eq!(level1.basis(), &chain.active_basis(1));
        assert_eq!(level1.basis().len(), 3);

        let expected = composite_rescale_reference_polynomial(&input, &chain, 0);

        assert_eq!(level1, expected);
    }

    #[test]
    fn single_limb_groups_reduce_to_legacy_semantics() {
        let chain = CompositeModulusChain::single_limb_levels(basis());
        let input =
            rns_from_big_coefficients(chain.active_basis(0).moduli().to_vec(), &coefficients());

        let actual = rescale_composite_level_to_next(&input, &chain, 0);
        let expected = composite_rescale_reference_polynomial(&input, &chain, 0);

        assert_eq!(actual, expected);
        assert_eq!(actual.basis().len(), 5);
    }

    #[test]
    #[should_panic(expected = "cannot rescale the final composite CKKS level")]
    fn final_logical_level_cannot_rescale() {
        let chain = CompositeModulusChain::new(basis(), vec![2, 2, 2]);
        let final_basis = chain.active_basis(chain.max_level());
        let input =
            rns_from_big_coefficients(final_basis.moduli().to_vec(), &[BigUint::from(1_u64)]);

        let _ = rescale_composite_level_to_next(&input, &chain, chain.max_level());
    }
}
