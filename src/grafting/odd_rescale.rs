use crate::ring::{Modulus, Polynomial, RnsPolynomial};

use super::{Pow2GraftedBasis, Pow2RnsPolynomial};

/// Exact rescale by the trailing odd RNS modulus.
///
/// If the represented integer coefficient is `a` and the consumed
/// nutrient is `q`, this computes
///
/// ```text
/// (a - (a mod q)) / q
/// ```
///
/// and returns the result in the basis obtained by removing `q` while
/// preserving the existing power-of-two sprout.
///
/// This is fundamentally different from a structural RNS limb drop:
/// the represented integer is divided by the consumed modulus.
pub fn rescale_odd_nutrient(polynomial: &Pow2RnsPolynomial) -> Pow2RnsPolynomial {
    assert!(
        polynomial.ordinary_basis().len() > 1,
        "odd-nutrient rescale requires at least two ordinary limbs"
    );

    let source_basis = polynomial.ordinary_basis();

    let nutrient_index = source_basis.len() - 1;

    let nutrient = source_basis.modulus(nutrient_index);

    let nutrient_residue = polynomial.ordinary().residue(nutrient_index);

    let target_basis = source_basis.without_last();

    let ordinary_residues: Vec<Polynomial> = target_basis
        .moduli()
        .iter()
        .enumerate()
        .map(|(index, &modulus)| {
            let inverse = modulus.inverse_prime(nutrient.value() % modulus.value());

            let coefficients = polynomial
                .ordinary()
                .residue(index)
                .coefficients()
                .iter()
                .zip(nutrient_residue.coefficients())
                .map(|(&value, &remainder)| {
                    let difference = modulus.sub(value, remainder % modulus.value());

                    modulus.mul(difference, inverse)
                })
                .collect();

            Polynomial::new(modulus, coefficients)
        })
        .collect();

    let ordinary = RnsPolynomial::from_residues(ordinary_residues);

    let bits = polynomial.sprout_bits();

    let pow2_modulus = 1_u64 << bits;

    let mask = pow2_modulus - 1;

    let inverse = inverse_odd_mod_power_of_two(nutrient.value() & mask, bits);

    let sprout_coefficients = polynomial
        .sprout()
        .coefficients()
        .iter()
        .zip(nutrient_residue.coefficients())
        .map(|(&value, &remainder)| {
            let remainder = remainder & mask;

            let difference = value.wrapping_sub(remainder) & mask;

            difference.wrapping_mul(inverse) & mask
        })
        .collect();

    Pow2RnsPolynomial::from_parts(
        ordinary,
        super::Pow2Polynomial::new(bits, sprout_coefficients),
    )
}

/// Returns the exact target basis for one odd-nutrient rescale.
pub fn odd_rescale_target_basis(source: &Pow2GraftedBasis) -> Pow2GraftedBasis {
    assert!(
        source.ordinary_basis().len() > 1,
        "odd-nutrient rescale requires at least two ordinary limbs"
    );

    Pow2GraftedBasis::new(source.ordinary_basis().without_last(), source.sprout_bits())
}

/// Nutrient consumed by an odd rescale.
pub fn odd_rescale_nutrient(source: &Pow2GraftedBasis) -> Modulus {
    assert!(
        source.ordinary_basis().len() > 1,
        "odd-nutrient rescale requires at least two ordinary limbs"
    );

    source
        .ordinary_basis()
        .modulus(source.ordinary_basis().len() - 1)
}

fn inverse_odd_mod_power_of_two(value: u64, bits: u32) -> u64 {
    assert!(value & 1 == 1, "power-of-two inverse requires odd input");

    assert!(
        bits > 0 && bits <= 63,
        "power-of-two inverse bits must be in 1..=63"
    );

    let mut inverse = 1_u64;

    // Newton iteration doubles the number of correct bits.
    for _ in 0..6 {
        inverse = inverse.wrapping_mul(2_u64.wrapping_sub(value.wrapping_mul(inverse)));
    }

    let mask = (1_u64 << bits) - 1;

    inverse & mask
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn target_basis_drops_exactly_the_nutrient() {
        let source = basis(12);

        let target = odd_rescale_target_basis(&source);

        assert_eq!(odd_rescale_nutrient(&source,), Modulus::new(65_537));

        assert_eq!(
            target.ordinary_basis().moduli(),
            &[Modulus::new(12_289), Modulus::new(40_961),]
        );

        assert_eq!(target.sprout_bits(), 12);
    }

    #[test]
    fn odd_rescale_matches_exact_integer_definition() {
        let source_basis = basis(12);

        let q = source_basis.composite_modulus();

        let coefficients = vec![0, 1, 17, 42, 65_536, 65_537, 1_000_000, q / 7, q / 3, q - 1];

        let source = source_basis.from_coefficients(&coefficients);

        let actual = rescale_odd_nutrient(&source);

        let nutrient = 65_537_u128;

        let expected: Vec<u128> = coefficients
            .iter()
            .map(|&value| (value - value % nutrient) / nutrient)
            .collect();

        assert_eq!(actual.reconstruct_coefficients(), expected);
    }

    #[test]
    fn exact_multiples_divide_without_error() {
        let source_basis = basis(8);

        let nutrient = 65_537_u128;

        let coefficients: Vec<u128> = (0_u128..32).map(|value| value * nutrient).collect();

        let source = source_basis.from_coefficients(&coefficients);

        assert_eq!(
            rescale_odd_nutrient(&source,).reconstruct_coefficients(),
            (0_u128..32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn odd_rescale_preserves_power_of_two_sprout_size() {
        for bits in [4_u32, 8, 12, 16, 24, 32] {
            let source_basis = basis(bits);

            let source = source_basis.from_coefficients(&[1, 17, 42, 1_001]);

            let target = rescale_odd_nutrient(&source);

            assert_eq!(target.sprout_bits(), bits);

            assert_eq!(
                target.ordinary_basis().len(),
                source.ordinary_basis().len() - 1
            );
        }
    }

    #[test]
    fn differential_campaign_matches_integer_reference() {
        for bits in [4_u32, 8, 12, 16] {
            let source_basis = basis(bits);

            let modulus = source_basis.composite_modulus();

            let nutrient = 65_537_u128;

            for seed in 0_u128..32 {
                let coefficients: Vec<u128> = (0..32_u128)
                    .map(|index| (97 * seed + 31 * index + 7 * index * index + 11) % modulus)
                    .collect();

                let source = source_basis.from_coefficients(&coefficients);

                let actual = rescale_odd_nutrient(&source).reconstruct_coefficients();

                let expected: Vec<u128> = coefficients
                    .iter()
                    .map(|&value| (value - value % nutrient) / nutrient)
                    .collect();

                assert_eq!(
                    actual, expected,
                    "odd rescale mismatch: bits={bits}, seed={seed}"
                );
            }
        }
    }

    #[test]
    fn two_successive_odd_rescales_match_integer_reference() {
        let source_basis = Pow2GraftedBasis::new(
            ModulusBasis::new(vec![
                Modulus::new(12_289),
                Modulus::new(40_961),
                Modulus::new(65_537),
                Modulus::new(114_689),
            ]),
            12,
        );

        let modulus = source_basis.composite_modulus();

        let coefficients: Vec<u128> = (0..32_u128)
            .map(|index| (101 + 43 * index + 5 * index * index) % modulus)
            .collect();

        let source = source_basis.from_coefficients(&coefficients);

        let first = rescale_odd_nutrient(&source);

        let second = rescale_odd_nutrient(&first);

        let expected: Vec<u128> = coefficients
            .iter()
            .map(|&value| {
                let first = (value - value % 114_689_u128) / 114_689_u128;

                (first - first % 65_537_u128) / 65_537_u128
            })
            .collect();

        assert_eq!(second.reconstruct_coefficients(), expected);
    }
}
