//! R16.5.18a: exact reference arithmetic for auxiliary-modulus reduction.
//!
//! Correctness oracle only; this is not a production key-switching backend.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{Signed, Zero};

use super::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
    rns_from_big_coefficients, ModulusBasis, RnsPolynomial,
};

/// Round x / divisor to nearest integer, ties away from zero.
fn round_divide_signed(value: &BigInt, divisor: &BigInt) -> BigInt {
    assert!(divisor > &BigInt::zero(), "divisor must be positive");

    let magnitude = value.abs();
    let rounded: BigInt = (magnitude + (divisor / BigInt::from(2_u8))) / divisor;

    if value.sign() == Sign::Minus {
        -rounded
    } else {
        rounded
    }
}

/// Reduce a signed integer modulo a positive modulus to [0, modulus).
fn canonical_signed(value: &BigInt, modulus: &BigInt) -> BigUint {
    assert!(modulus > &BigInt::zero());

    let mut reduced = value % modulus;
    if reduced.sign() == Sign::Minus {
        reduced += modulus;
    }

    reduced
        .to_biguint()
        .expect("canonical residue must be nonnegative")
}

/// Reference centered division by an auxiliary modulus P.
///
/// The input basis must be Q followed by the auxiliary basis P.
/// The output is represented over Q.
///
/// This routine reconstructs each coefficient using arbitrary precision.
/// It is intended for correctness testing, not performance-critical execution.
pub fn auxiliary_mod_down_reference(
    polynomial: &RnsPolynomial,
    q_basis: &ModulusBasis,
    p_basis: &ModulusBasis,
) -> RnsPolynomial {
    assert!(!p_basis.is_empty(), "auxiliary basis must be nonempty");

    let mut expected_moduli = q_basis.moduli().to_vec();
    expected_moduli.extend_from_slice(p_basis.moduli());

    assert_eq!(
        polynomial.moduli(),
        expected_moduli.as_slice(),
        "input basis must be Q followed by P"
    );

    let qp = composite_modulus_big(polynomial.basis());
    let q = composite_modulus_big(q_basis);
    let p = composite_modulus_big(p_basis);

    let q_signed = BigInt::from_biguint(Sign::Plus, q);
    let p_signed = BigInt::from_biguint(Sign::Plus, p);

    let output: Vec<BigUint> = reconstruct_coefficients_big(polynomial)
        .iter()
        .map(|coefficient| {
            let centered = centered_representative_big(coefficient, &qp);
            let divided = round_divide_signed(&centered, &p_signed);
            canonical_signed(&divided, &q_signed)
        })
        .collect();

    rns_from_big_coefficients(q_basis.moduli().to_vec(), &output)
}

/// R16.5.18g.1: exact QP-to-Q modulus-down using only P-side CRT.
///
/// This is an experimental candidate. The reference implementation remains
/// the correctness oracle and the active hybrid-switching backend.
///
/// Requires odd, pairwise-coprime Q and P moduli.
pub fn auxiliary_mod_down_rns_candidate(
    polynomial: &RnsPolynomial,
    q_basis: &ModulusBasis,
    p_basis: &ModulusBasis,
) -> RnsPolynomial {
    use num_traits::ToPrimitive;

    assert!(!p_basis.is_empty(), "auxiliary basis must be nonempty");

    let mut expected = q_basis.moduli().to_vec();
    expected.extend_from_slice(p_basis.moduli());
    assert_eq!(
        polynomial.moduli(),
        expected.as_slice(),
        "input basis must be Q followed by P"
    );

    for modulus in &expected {
        assert_eq!(modulus.value() % 2, 1, "RNS candidate requires odd moduli");
    }

    let p = composite_modulus_big(p_basis);

    let p_polynomial = RnsPolynomial::from_residues(
        (q_basis.len()..expected.len())
            .map(|index| polynomial.residue(index).clone())
            .collect(),
    );

    let p_coefficients = reconstruct_coefficients_big(&p_polynomial);

    let mut output = Vec::with_capacity(q_basis.len());

    for (q_index, &modulus) in q_basis.moduli().iter().enumerate() {
        let qi = modulus.value();
        let p_mod_q = (&p % BigUint::from(qi))
            .to_u64()
            .expect("P residue must fit in u64");

        // Extended Euclidean algorithm over BigInt, performed once per Q limb.
        let mut old_r = BigInt::from(p_mod_q);
        let mut r = BigInt::from(qi);
        let mut old_t = BigInt::from(1);
        let mut t = BigInt::from(0);

        while !r.is_zero() {
            let quotient = &old_r / &r;
            let next_r = &old_r - &quotient * &r;
            let next_t = &old_t - &quotient * &t;
            old_r = r;
            r = next_r;
            old_t = t;
            t = next_t;
        }

        assert_eq!(old_r, BigInt::from(1), "Q and P must be coprime");

        let inverse = canonical_signed(&old_t, &BigInt::from(qi))
            .to_u64()
            .expect("inverse must fit in u64");

        let coefficients = p_coefficients
            .iter()
            .enumerate()
            .map(|(index, coefficient)| {
                let centered = centered_representative_big(coefficient, &p);

                let r_mod_q = canonical_signed(&centered, &BigInt::from(qi))
                    .to_u64()
                    .expect("centered P residue must fit modulo qi");

                let x_mod_q = polynomial.residue(q_index).coefficient(index);
                let difference = modulus.sub(x_mod_q, r_mod_q);

                ((u128::from(difference) * u128::from(inverse)) % u128::from(qi)) as u64
            })
            .collect();

        output.push(super::Polynomial::new(modulus, coefficients));
    }

    RnsPolynomial::from_residues(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::{Modulus, RnsPolynomial};

    fn bases() -> (ModulusBasis, ModulusBasis) {
        (
            ModulusBasis::new(vec![Modulus::new(97), Modulus::new(193)]),
            ModulusBasis::new(vec![Modulus::new(257)]),
        )
    }

    #[test]
    fn r16_5_18a_signed_rounding() {
        let d = BigInt::from(10);

        for (input, expected) in [
            (0, 0),
            (4, 0),
            (5, 1),
            (6, 1),
            (14, 1),
            (15, 2),
            (-4, 0),
            (-5, -1),
            (-6, -1),
            (-14, -1),
            (-15, -2),
        ] {
            assert_eq!(
                round_divide_signed(&BigInt::from(input), &d),
                BigInt::from(expected),
                "rounding mismatch for {input}"
            );
        }
    }

    #[test]
    fn r16_5_18a_mod_down_signed_reference() {
        let (q_basis, p_basis) = bases();
        let mut moduli = q_basis.moduli().to_vec();
        moduli.extend_from_slice(p_basis.moduli());

        let qp = composite_modulus_big(&ModulusBasis::new(moduli.clone()));
        let q = composite_modulus_big(&q_basis);
        let p = composite_modulus_big(&p_basis);

        let signed_values = [
            -1000_i64, -386, -257, -129, -128, -1, 0, 1, 128, 129, 257, 386, 1000,
        ];

        let coefficients: Vec<BigUint> = signed_values
            .iter()
            .map(|&value| canonical_signed(&BigInt::from(value), &BigInt::from(qp.clone())))
            .collect();

        let polynomial = rns_from_big_coefficients(moduli, &coefficients);
        let result = auxiliary_mod_down_reference(&polynomial, &q_basis, &p_basis);

        let actual = reconstruct_coefficients_big(&result);

        for (&value, observed) in signed_values.iter().zip(actual.iter()) {
            let expected = canonical_signed(
                &round_divide_signed(&BigInt::from(value), &BigInt::from(p.clone())),
                &BigInt::from(q.clone()),
            );

            assert_eq!(*observed, expected, "input={value}");
        }
    }

    #[test]
    fn r16_5_18a_rejects_incorrect_basis() {
        let (q_basis, p_basis) = bases();

        let polynomial = RnsPolynomial::zero(q_basis.moduli().to_vec(), 8);

        let result = std::panic::catch_unwind(|| {
            auxiliary_mod_down_reference(&polynomial, &q_basis, &p_basis)
        });

        assert!(result.is_err());
    }
}
