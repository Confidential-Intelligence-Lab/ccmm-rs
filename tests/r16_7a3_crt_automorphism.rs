//! R16.7a.3: canonical CRT reconstruction and negacyclic
//! automorphism commutation.
//!
//! Exact algebraic regression; no cryptographic security claim.

use ccmm_rs::ring::{reconstruct_coefficients_big, Modulus, RnsPolynomial};
use num_bigint::BigUint;
use num_traits::Zero;

fn canonical_automorphism(
    coefficients: &[BigUint],
    exponent: usize,
    modulus: &BigUint,
) -> Vec<BigUint> {
    let degree = coefficients.len();
    let mut result = vec![BigUint::zero(); degree];

    for (index, coefficient) in coefficients.iter().enumerate() {
        let position = (index * exponent) % (2 * degree);
        let destination = position % degree;

        result[destination] = if position >= degree {
            if coefficient.is_zero() {
                BigUint::zero()
            } else {
                modulus - coefficient
            }
        } else {
            coefficient.clone()
        };
    }

    result
}

#[test]
fn r16_7a3_crt_automorphism() {
    let degree = 16usize;
    let moduli = vec![
        Modulus::new(12289),
        Modulus::new(40961),
        Modulus::new(65537),
    ];

    let modulus = moduli
        .iter()
        .fold(BigUint::from(1u8), |acc, q| acc * BigUint::from(q.value()));

    let coefficients: Vec<BigUint> = (0..degree)
        .map(|i| {
            if i % 4 == 0 {
                &modulus - BigUint::from(i as u64 + 1)
            } else {
                BigUint::from((i * 7919 + 17) as u64)
            }
        })
        .collect();

    let polynomial = ccmm_rs::ring::rns_from_big_coefficients(moduli.clone(), &coefficients);

    let reconstructed = reconstruct_coefficients_big(&polynomial);

    assert_eq!(reconstructed, coefficients);

    let mut checks = 0usize;

    for exponent in (1..2 * degree).step_by(2) {
        let expected = canonical_automorphism(&reconstructed, exponent, &modulus);

        let transformed = RnsPolynomial::from_residues(
            polynomial
                .residues()
                .iter()
                .map(|residue| ccmm_rs::ckks::apply_automorphism(residue, exponent))
                .collect(),
        );

        let actual = reconstruct_coefficients_big(&transformed);

        assert_eq!(
            actual, expected,
            "CRT/automorphism mismatch for exponent {exponent}"
        );

        checks += 1;
    }

    println!("R16_7A3_DEGREE={degree}");
    println!("R16_7A3_EXPONENT_CHECKS={checks}");
    println!("R16_7A3_EXACT_EQUALITY=PASS");
}
