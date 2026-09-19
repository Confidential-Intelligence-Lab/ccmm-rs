use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, ToPrimitive, Zero};

use super::{Modulus, ModulusBasis, Polynomial, RnsPolynomial};

/// Exact arbitrary-precision product of all RNS moduli.
///
/// The performance-critical RNS/NTT path remains limb-wise over native
/// moduli. This helper exists for reference arithmetic whose logical
/// composite modulus can exceed `u128`.
pub fn composite_modulus_big(basis: &ModulusBasis) -> BigUint {
    basis
        .moduli()
        .iter()
        .fold(BigUint::one(), |product, modulus| {
            product * BigUint::from(modulus.value())
        })
}

/// Constructs an RNS polynomial from arbitrary-precision canonical
/// coefficients.
///
/// Every coefficient is reduced independently modulo each native RNS limb.
pub fn rns_from_big_coefficients(moduli: Vec<Modulus>, coefficients: &[BigUint]) -> RnsPolynomial {
    assert!(
        !coefficients.is_empty(),
        "RNS polynomial degree must be positive"
    );

    let residues = moduli
        .into_iter()
        .map(|modulus| {
            let q = BigUint::from(modulus.value());

            let reduced = coefficients
                .iter()
                .map(|coefficient| {
                    (coefficient % &q)
                        .to_u64()
                        .expect("coefficient reduced modulo u64 must fit u64")
                })
                .collect();

            Polynomial::new(modulus, reduced)
        })
        .collect();

    RnsPolynomial::from_residues(residues)
}

/// Reconstructs the canonical coefficient representatives modulo the exact
/// composite RNS modulus using arbitrary precision.
///
/// This uses an incremental Garner-style CRT update and therefore never
/// requires the full composite modulus to fit in a machine integer.
pub fn reconstruct_coefficients_big(polynomial: &RnsPolynomial) -> Vec<BigUint> {
    (0..polynomial.degree())
        .map(|coefficient_index| {
            let mut value = BigUint::zero();
            let mut accumulated_modulus = BigUint::one();

            for residue in polynomial.residues() {
                let modulus = residue.modulus().value();
                let residue_value = residue.coefficients()[coefficient_index];

                let value_mod = (&value % modulus)
                    .to_u64()
                    .expect("BigUint modulo u64 must fit u64");

                let accumulated_modulus_mod = (&accumulated_modulus % modulus)
                    .to_u64()
                    .expect("BigUint modulo u64 must fit u64");

                let inverse = inverse_mod_u64(accumulated_modulus_mod, modulus);

                let delta = if residue_value >= value_mod {
                    residue_value - value_mod
                } else {
                    modulus - (value_mod - residue_value)
                };

                let correction =
                    ((u128::from(delta) * u128::from(inverse)) % u128::from(modulus)) as u64;

                value += &accumulated_modulus * correction;
                accumulated_modulus *= modulus;
            }

            value
        })
        .collect()
}

/// Maps a canonical representative in `[0, Q)` to the centered interval
/// `(-Q/2, Q/2]` using arbitrary precision.
pub fn centered_representative_big(value: &BigUint, modulus: &BigUint) -> BigInt {
    assert!(
        value < modulus,
        "canonical representative must be smaller than modulus"
    );
    assert!(!modulus.is_zero(), "modulus must be nonzero");

    let half = modulus >> 1_usize;

    if value > &half {
        BigInt::from_biguint(Sign::Plus, value.clone())
            - BigInt::from_biguint(Sign::Plus, modulus.clone())
    } else {
        BigInt::from_biguint(Sign::Plus, value.clone())
    }
}

fn inverse_mod_u64(value: u64, modulus: u64) -> u64 {
    assert!(modulus > 1, "modulus must exceed one");

    let mut old_r = i128::from(value);
    let mut r = i128::from(modulus);
    let mut old_s = 1_i128;
    let mut s = 0_i128;

    while r != 0 {
        let quotient = old_r / r;

        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }

    assert_eq!(old_r, 1, "value must be invertible modulo modulus");

    old_s.rem_euclid(i128::from(modulus)) as u64
}

#[cfg(test)]
mod tests {
    use num_bigint::{BigInt, BigUint};
    use num_traits::{One, ToPrimitive};

    use crate::ckks::{research_profile_4096, research_profile_8192};

    use super::{
        centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
        rns_from_big_coefficients,
    };

    #[test]
    fn small_composite_matches_existing_u128_path() {
        let basis = research_profile_4096().modulus_basis();

        let wide = composite_modulus_big(&basis);
        let narrow = basis.composite_modulus();

        assert_eq!(wide.to_u128(), Some(narrow));
    }

    #[test]
    fn small_reconstruction_matches_existing_u128_path() {
        let basis = research_profile_4096().modulus_basis();

        let coefficients = vec![
            BigUint::from(0_u64),
            BigUint::from(1_u64),
            BigUint::from(17_u64),
            BigUint::from(1_000_003_u64),
            BigUint::from(9_876_543_210_u64),
        ];

        let polynomial = rns_from_big_coefficients(basis.moduli().to_vec(), &coefficients);

        let wide = reconstruct_coefficients_big(&polynomial);
        let narrow: Vec<BigUint> = polynomial
            .reconstruct_coefficients()
            .into_iter()
            .map(BigUint::from)
            .collect();

        assert_eq!(wide, narrow);
        assert_eq!(wide, coefficients);
    }

    #[test]
    fn research_8192_composite_exceeds_u128_without_overflow() {
        let basis = research_profile_8192().modulus_basis();
        let composite = composite_modulus_big(&basis);

        assert!(composite.bits() > 128);
        assert!(composite.bits() <= 200);
    }

    #[test]
    fn research_8192_roundtrip_supports_coefficients_above_u128() {
        let basis = research_profile_8192().modulus_basis();
        let composite = composite_modulus_big(&basis);

        let very_large = (BigUint::one() << 150_usize) + BigUint::from(0x1234_5678_u64);
        assert!(very_large < composite);

        let coefficients = vec![
            BigUint::from(0_u64),
            BigUint::from(1_u64),
            very_large,
            &composite - BigUint::from(7_u64),
        ];

        let polynomial = rns_from_big_coefficients(basis.moduli().to_vec(), &coefficients);

        let reconstructed = reconstruct_coefficients_big(&polynomial);

        assert_eq!(reconstructed, coefficients);
    }

    #[test]
    fn centered_representative_handles_wide_negative_value() {
        let basis = research_profile_8192().modulus_basis();
        let composite = composite_modulus_big(&basis);

        let canonical = &composite - BigUint::from(3_u64);
        let centered = centered_representative_big(&canonical, &composite);

        assert_eq!(centered, BigInt::from(-3_i64));
    }
}
