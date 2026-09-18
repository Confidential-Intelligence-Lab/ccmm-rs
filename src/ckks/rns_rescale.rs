use crate::grafting::RnsRlweCiphertext;
use crate::ring::{Modulus, ModulusChain, Polynomial};
use crate::rlwe::RlweCiphertext;

use super::RnsCkksCiphertext;

/// Rescales one RNS CKKS ciphertext to the next modulus-chain level.
///
/// For every logical coefficient `a` and dropped trailing modulus
/// `q_drop`, the transition is
///
/// ```text
/// (a - (a mod q_drop)) / q_drop
/// ```
///
/// represented directly in the remaining RNS limbs.
///
/// The CKKS scale is divided by `q_drop`.
pub fn rescale_rns_ckks_to_next(
    ciphertext: &RnsCkksCiphertext,
    chain: &ModulusChain,
) -> RnsCkksCiphertext {
    ciphertext.assert_matches_chain(chain);

    let level = ciphertext.level();

    let dropped = chain
        .dropped_modulus(level)
        .expect("cannot rescale the final CKKS chain level");

    let next_state = ciphertext.state().after_rescale(chain);

    let inner = rescale_rns_rlwe(ciphertext.rlwe(), dropped);

    RnsCkksCiphertext::new(inner, next_state, chain)
}

/// Rescales both RLWE components limb-by-limb.
fn rescale_rns_rlwe(ciphertext: &RnsRlweCiphertext, dropped: Modulus) -> RnsRlweCiphertext {
    assert!(
        ciphertext.basis().len() > 1,
        "RNS CKKS rescale requires at least two modulus limbs"
    );

    let dropped_index = ciphertext.basis().len() - 1;

    assert_eq!(
        ciphertext.basis().modulus(dropped_index),
        dropped,
        "dropped modulus must be the trailing RNS limb"
    );

    let dropped_limb = ciphertext.limb(dropped_index);

    let output_limbs = ciphertext.basis().moduli()[..dropped_index]
        .iter()
        .copied()
        .enumerate()
        .map(|(index, _)| {
            let source = ciphertext.limb(index);

            let b = rescale_polynomial_limb(source.b(), dropped_limb.b(), dropped);

            let a = rescale_polynomial_limb(source.a(), dropped_limb.a(), dropped);

            RlweCiphertext::new(b, a)
        })
        .collect();

    RnsRlweCiphertext::from_limbs(output_limbs)
}

/// Computes one target residue limb of the exact odd-prime rescale.
///
/// Given source residue `x mod q_i` and dropped residue
/// `r = x mod q_drop`, computes
///
/// ```text
/// ((x - r) / q_drop) mod q_i
/// ```
///
/// using only residues.
fn rescale_polynomial_limb(
    source: &Polynomial,
    dropped_residue: &Polynomial,
    dropped: Modulus,
) -> Polynomial {
    let modulus = source.modulus();

    assert_eq!(
        source.degree(),
        dropped_residue.degree(),
        "RNS rescale limb degrees must match"
    );

    assert_eq!(
        dropped_residue.modulus(),
        dropped,
        "dropped residue polynomial must use dropped modulus"
    );

    let inverse = modulus.inverse_prime(dropped.value() % modulus.value());

    let coefficients = source
        .coefficients()
        .iter()
        .zip(dropped_residue.coefficients())
        .map(|(&value, &remainder)| {
            /*
             * CKKS nearest rescale:
             *
             *   round(x / q_drop)
             *     = (x - centered(x mod q_drop)) / q_drop.
             */
            let dropped_q = dropped.value();

            let centered_remainder = if remainder > dropped_q / 2 {
                i128::from(remainder) - i128::from(dropped_q)
            } else {
                i128::from(remainder)
            };

            let remainder_mod_q = if centered_remainder >= 0 {
                (centered_remainder as u128 % u128::from(modulus.value())) as u64
            } else {
                let magnitude =
                    (centered_remainder.unsigned_abs() % u128::from(modulus.value())) as u64;

                if magnitude == 0 {
                    0
                } else {
                    modulus.value() - magnitude
                }
            };

            let difference = modulus.sub(value, remainder_mod_q);

            modulus.mul(difference, inverse)
        })
        .collect();

    Polynomial::new(modulus, coefficients)
}

#[cfg(test)]
mod tests {
    use crate::ckks::CkksChainState;
    use crate::ring::{ModulusBasis, RnsPolynomial};

    use super::*;

    fn ckks_rescale_reference(
        value: u128,
        source_modulus: u128,
        divisor: u128,
        target_modulus: u128,
    ) -> u128 {
        let source = i128::try_from(source_modulus).expect("source modulus exceeds i128");

        let divisor = i128::try_from(divisor).expect("divisor exceeds i128");

        let target = i128::try_from(target_modulus).expect("target modulus exceeds i128");

        let canonical = i128::try_from(value).expect("coefficient exceeds i128");

        /*
         * CKKS interprets the source coefficient in its centered
         * representative interval.
         */
        let centered = if canonical > source / 2 {
            canonical - source
        } else {
            canonical
        };

        /*
         * Signed nearest-integer division.
         */
        let rounded = if centered >= 0 {
            (centered + divisor / 2) / divisor
        } else {
            -((-centered + divisor / 2) / divisor)
        };

        /*
         * Return the canonical representative in the target modulus.
         */
        ((rounded % target) + target) as u128 % target_modulus
    }

    fn chain() -> ModulusChain {
        ModulusChain::from_top_basis(ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ]))
    }

    fn rns_rlwe_from_coefficients(
        basis: &ModulusBasis,
        coefficients_b: &[u128],
        coefficients_a: &[u128],
    ) -> RnsRlweCiphertext {
        let b = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), coefficients_b);

        let a = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), coefficients_a);

        let limbs = basis
            .moduli()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, _)| {
                RlweCiphertext::new(b.residue(index).clone(), a.residue(index).clone())
            })
            .collect();

        RnsRlweCiphertext::from_limbs(limbs)
    }

    fn reconstruct_component(ciphertext: &RnsRlweCiphertext, use_b: bool) -> Vec<u128> {
        let residues = ciphertext
            .limbs()
            .iter()
            .map(|limb| {
                if use_b {
                    limb.b().clone()
                } else {
                    limb.a().clone()
                }
            })
            .collect();

        RnsPolynomial::from_residues(residues).reconstruct_coefficients()
    }

    fn ciphertext_from_coefficients(
        chain: &ModulusChain,
        coefficients_b: &[u128],
        coefficients_a: &[u128],
        scale: f64,
    ) -> RnsCkksCiphertext {
        let inner = rns_rlwe_from_coefficients(chain.top(), coefficients_b, coefficients_a);

        RnsCkksCiphertext::new(inner, CkksChainState::top(chain, scale), chain)
    }

    #[test]
    fn rescale_moves_to_next_chain_level() {
        let chain = chain();

        let ciphertext = ciphertext_from_coefficients(
            &chain,
            &[1, 17, 42, 65_537],
            &[9, 21, 77, 131],
            65_537.0 * 65_537.0,
        );

        let next = rescale_rns_ckks_to_next(&ciphertext, &chain);

        assert_eq!(next.level(), 1);

        assert_eq!(next.basis(), chain.level(1));

        assert_eq!(next.scale(), 65_537.0);
    }

    #[test]
    fn rescale_matches_integer_reference_exactly() {
        let chain = chain();

        let modulus = chain.composite_modulus(0);

        let coefficients = vec![
            0,
            1,
            17,
            42,
            65_536,
            65_537,
            1_000_000,
            modulus / 3,
            modulus - 1,
        ];

        let ciphertext = ciphertext_from_coefficients(&chain, &coefficients, &coefficients, 1.0);

        let next = rescale_rns_ckks_to_next(&ciphertext, &chain);

        let dropped = u128::from(chain.dropped_modulus(0).unwrap().value());

        let expected: Vec<u128> = coefficients
            .iter()
            .map(|&value| {
                ckks_rescale_reference(
                    value,
                    chain.composite_modulus(0),
                    dropped,
                    chain.composite_modulus(1),
                )
            })
            .collect();

        assert_eq!(reconstruct_component(next.rlwe(), true,), expected);

        assert_eq!(reconstruct_component(next.rlwe(), false,), expected);
    }

    #[test]
    fn two_successive_rescales_follow_chain_exactly() {
        let chain = chain();

        let top_q = chain.composite_modulus(0);

        let coefficients: Vec<u128> = (0_u128..16)
            .map(|index| (101 + 43 * index + 5 * index * index) % top_q)
            .collect();

        let level0 =
            ciphertext_from_coefficients(&chain, &coefficients, &coefficients, 65_537.0 * 40_961.0);

        let level1 = rescale_rns_ckks_to_next(&level0, &chain);

        let level2 = rescale_rns_ckks_to_next(&level1, &chain);

        assert_eq!(level2.level(), 2);

        assert_eq!(level2.basis(), chain.level(2));

        assert_eq!(level2.scale(), 1.0);

        let first_divisor = 65_537_u128;

        let second_divisor = 40_961_u128;

        let expected: Vec<u128> = coefficients
            .iter()
            .map(|&value| {
                let first = (value - value % first_divisor) / first_divisor;

                (first - first % second_divisor) / second_divisor
            })
            .collect();

        assert_eq!(reconstruct_component(level2.rlwe(), true,), expected);
    }

    #[test]
    #[should_panic(expected = "cannot rescale the final CKKS chain level")]
    fn final_level_cannot_rescale() {
        let chain = chain();

        let basis = chain.bottom();

        let inner = rns_rlwe_from_coefficients(basis, &[1, 2, 3, 4], &[5, 6, 7, 8]);

        let ciphertext = RnsCkksCiphertext::new(
            inner,
            CkksChainState::new(&chain, chain.max_level(), 1.0),
            &chain,
        );

        let _ = rescale_rns_ckks_to_next(&ciphertext, &chain);
    }
}
