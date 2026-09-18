use super::{Modulus, ModulusBasis, Polynomial, RnsPolynomial};

/// Drops trailing RNS limbs when `target_basis` is an exact prefix of
/// the polynomial's current basis.
///
/// No CRT reconstruction is required: the retained residues already
/// represent the correct value modulo every modulus in the target basis.
pub fn drop_basis_prefix(polynomial: &RnsPolynomial, target_basis: &ModulusBasis) -> RnsPolynomial {
    assert!(
        target_basis.len() <= polynomial.basis().len(),
        "target basis cannot be larger than source basis for basis drop"
    );

    assert_eq!(
        &polynomial.basis().moduli()[..target_basis.len()],
        target_basis.moduli(),
        "target basis must be an exact prefix of source basis"
    );

    let residues = polynomial.residues()[..target_basis.len()].to_vec();

    RnsPolynomial::from_residues(residues)
}

/// Extends an RNS polynomial to a larger basis whose prefix is exactly
/// the current basis.
///
/// Existing residue limbs are preserved. New residue limbs are computed
/// from a mixed-radix representation obtained with Garner's algorithm.
///
/// This avoids reconstructing the full coefficient modulo the composite
/// source modulus.
pub fn extend_basis_prefix(
    polynomial: &RnsPolynomial,
    target_basis: &ModulusBasis,
) -> RnsPolynomial {
    let source_basis = polynomial.basis();

    assert!(
        target_basis.len() >= source_basis.len(),
        "target basis cannot be smaller than source basis for basis extension"
    );

    assert_eq!(
        &target_basis.moduli()[..source_basis.len()],
        source_basis.moduli(),
        "source basis must be an exact prefix of target basis"
    );

    if target_basis == source_basis {
        return polynomial.clone();
    }

    let mut residues = polynomial.residues().to_vec();

    for &target_modulus in &target_basis.moduli()[source_basis.len()..] {
        let coefficients = extend_coefficients_to_modulus(polynomial, target_modulus);

        residues.push(Polynomial::new(target_modulus, coefficients));
    }

    RnsPolynomial::from_residues(residues)
}

/// Computes the represented polynomial modulo one new modulus using
/// mixed-radix digits instead of full CRT reconstruction.
fn extend_coefficients_to_modulus(polynomial: &RnsPolynomial, target_modulus: Modulus) -> Vec<u64> {
    assert!(
        !polynomial.basis().contains(target_modulus),
        "target extension modulus must not already exist in source basis"
    );

    (0..polynomial.degree())
        .map(|coefficient_index| {
            let digits = mixed_radix_digits(polynomial, coefficient_index);

            evaluate_mixed_radix_modulus(&digits, polynomial.basis(), target_modulus)
        })
        .collect()
}

/// Converts one coefficient from residue form into mixed-radix digits.
///
/// If the source basis is `[q0, q1, ..., qL]`, the result represents
///
/// `x = c0 + c1*q0 + c2*q0*q1 + ...`.
fn mixed_radix_digits(polynomial: &RnsPolynomial, coefficient_index: usize) -> Vec<u64> {
    let moduli = polynomial.basis().moduli();

    let mut digits: Vec<u64> = polynomial
        .residues()
        .iter()
        .map(|residue| residue.coefficients()[coefficient_index])
        .collect();

    for i in 0..moduli.len() {
        for j in i + 1..moduli.len() {
            let modulus_j = moduli[j];

            let difference = modulus_j.sub(digits[j], digits[i] % modulus_j.value());

            let inverse = modulus_j.inverse_prime(moduli[i].value() % modulus_j.value());

            digits[j] = modulus_j.mul(difference, inverse);
        }
    }

    digits
}

fn evaluate_mixed_radix_modulus(
    digits: &[u64],
    source_basis: &ModulusBasis,
    target_modulus: Modulus,
) -> u64 {
    assert_eq!(
        digits.len(),
        source_basis.len(),
        "mixed-radix digit count must match source basis"
    );

    let mut value = 0_u64;
    let mut product = 1_u64;

    for (index, &digit) in digits.iter().enumerate() {
        value = target_modulus.add(
            value,
            target_modulus.mul(digit % target_modulus.value(), product),
        );

        product = target_modulus.mul(
            product,
            source_basis.modulus(index).value() % target_modulus.value(),
        );
    }

    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::convert_basis;

    fn full_basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
            Modulus::new(114_689),
        ])
    }

    fn source_basis() -> ModulusBasis {
        full_basis().prefix(2)
    }

    #[test]
    fn basis_drop_matches_reference_conversion() {
        let source = full_basis();
        let target = source.prefix(2);

        let polynomial = RnsPolynomial::from_coefficients(
            source.moduli().to_vec(),
            &[
                0,
                1,
                17,
                42,
                1_000,
                65_536,
                1_000_000,
                source.composite_modulus() - 1,
            ],
        );

        assert_eq!(
            drop_basis_prefix(&polynomial, &target,),
            convert_basis(&polynomial, &target,)
        );
    }

    #[test]
    fn basis_drop_preserves_retained_limbs_exactly() {
        let source = full_basis();
        let target = source.prefix(3);

        let polynomial =
            RnsPolynomial::from_coefficients(source.moduli().to_vec(), &[1, 2, 3, 4, 5, 6, 7, 8]);

        let dropped = drop_basis_prefix(&polynomial, &target);

        assert_eq!(dropped.residues(), &polynomial.residues()[..3]);
    }

    #[test]
    fn basis_extension_matches_reference_conversion() {
        let source = source_basis();
        let target = full_basis();

        let coefficients = vec![
            0_u128,
            1,
            42,
            12_345,
            65_536,
            1_000_000,
            source.composite_modulus() / 2,
            source.composite_modulus() - 1,
        ];

        let polynomial = RnsPolynomial::from_coefficients(source.moduli().to_vec(), &coefficients);

        let extended = extend_basis_prefix(&polynomial, &target);

        let reference = convert_basis(&polynomial, &target);

        assert_eq!(extended, reference);

        assert_eq!(extended.reconstruct_coefficients(), coefficients);
    }

    #[test]
    fn basis_extension_preserves_existing_limbs_exactly() {
        let source = source_basis();
        let target = full_basis();

        let polynomial =
            RnsPolynomial::from_coefficients(source.moduli().to_vec(), &[3, 5, 7, 11, 13, 17]);

        let extended = extend_basis_prefix(&polynomial, &target);

        assert_eq!(&extended.residues()[..source.len()], polynomial.residues());
    }

    #[test]
    fn extend_then_drop_roundtrip_is_exact() {
        let source = source_basis();
        let target = full_basis();

        let original = RnsPolynomial::from_coefficients(
            source.moduli().to_vec(),
            &[1, 7, 29, 101, 1_001, 100_003],
        );

        let extended = extend_basis_prefix(&original, &target);

        let recovered = drop_basis_prefix(&extended, &source);

        assert_eq!(recovered, original);
    }

    #[test]
    fn drop_then_reference_extension_agree_on_canonical_representative() {
        let full = full_basis();
        let smaller = full.prefix(2);

        let polynomial = RnsPolynomial::from_coefficients(
            full.moduli().to_vec(),
            &[1, 17, 1_001, 65_537, 1_000_003],
        );

        let dropped = drop_basis_prefix(&polynomial, &smaller);

        let reextended = extend_basis_prefix(&dropped, &full);

        let reference = convert_basis(&dropped, &full);

        assert_eq!(reextended, reference);
    }

    #[test]
    fn extension_campaign_matches_reference() {
        let source = source_basis();
        let target = full_basis();

        let source_q = source.composite_modulus();

        for seed in 0_u128..32 {
            let coefficients: Vec<u128> = (0..16_u128)
                .map(|index| (seed * 17 + index * 31 + index * index * 7 + 11) % source_q)
                .collect();

            let polynomial =
                RnsPolynomial::from_coefficients(source.moduli().to_vec(), &coefficients);

            assert_eq!(
                extend_basis_prefix(&polynomial, &target,),
                convert_basis(&polynomial, &target,),
                "basis extension mismatch for seed {seed}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "exact prefix")]
    fn drop_rejects_non_prefix_target() {
        let source = full_basis();

        let wrong = ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(65_537)]);

        let polynomial = RnsPolynomial::zero(source.moduli().to_vec(), 8);

        let _ = drop_basis_prefix(&polynomial, &wrong);
    }

    #[test]
    #[should_panic(expected = "exact prefix")]
    fn extension_rejects_non_prefix_target() {
        let source = source_basis();

        let wrong = ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(65_537),
            Modulus::new(114_689),
        ]);

        let polynomial = RnsPolynomial::zero(source.moduli().to_vec(), 8);

        let _ = extend_basis_prefix(&polynomial, &wrong);
    }
}
