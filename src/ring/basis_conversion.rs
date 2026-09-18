use super::{ModulusBasis, RnsPolynomial};

/// Converts an RNS polynomial from one modulus basis to another.
///
/// This correctness-oriented reference implementation:
///
/// 1. reconstructs each coefficient as its canonical representative
///    modulo the source composite modulus;
/// 2. reduces that representative into every modulus of the target basis.
///
/// The result therefore represents the same canonical integer
/// coefficients in the target basis.
///
/// This implementation is deliberately not optimized. It serves as the
/// reference oracle for future fast basis-drop, basis-extension, and
/// Grafting implementations.
pub fn convert_basis(polynomial: &RnsPolynomial, target_basis: &ModulusBasis) -> RnsPolynomial {
    let coefficients = polynomial.reconstruct_coefficients();

    RnsPolynomial::from_coefficients(target_basis.moduli().to_vec(), &coefficients)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::{Modulus, ModulusBasis};

    fn source_basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ])
    }

    fn smaller_basis() -> ModulusBasis {
        ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(40_961)])
    }

    fn alternate_basis() -> ModulusBasis {
        ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(147_457)])
    }

    #[test]
    fn identity_conversion_preserves_polynomial() {
        let basis = source_basis();

        let polynomial = RnsPolynomial::from_coefficients(
            basis.moduli().to_vec(),
            &[0, 1, 7, 42, 1_000, 65_536, 1_000_000],
        );

        let converted = convert_basis(&polynomial, &basis);

        assert_eq!(converted, polynomial);
    }

    #[test]
    fn conversion_to_smaller_basis_preserves_canonical_values_mod_target() {
        let source = source_basis();
        let target = smaller_basis();

        let source_q = source.composite_modulus();

        let target_q = target.composite_modulus();

        let coefficients = vec![0, 1, 17, 12_288, 65_536, source_q / 2, source_q - 1];

        let polynomial = RnsPolynomial::from_coefficients(source.moduli().to_vec(), &coefficients);

        let converted = convert_basis(&polynomial, &target);

        let expected: Vec<u128> = coefficients.iter().map(|value| value % target_q).collect();

        assert_eq!(converted.reconstruct_coefficients(), expected);
    }

    #[test]
    fn conversion_to_disjoint_basis_preserves_integer_values() {
        let source = source_basis();
        let target = alternate_basis();

        let coefficients = vec![0_u128, 1, 42, 12_345, 1_000_000, 10_000_000];

        assert!(coefficients.iter().all(|&value| {
            value < source.composite_modulus() && value < target.composite_modulus()
        }));

        let polynomial = RnsPolynomial::from_coefficients(source.moduli().to_vec(), &coefficients);

        let converted = convert_basis(&polynomial, &target);

        assert_eq!(converted.reconstruct_coefficients(), coefficients);
    }

    #[test]
    fn conversion_can_extend_to_larger_basis_via_canonical_lift() {
        let source = ModulusBasis::new(vec![Modulus::new(12_289), Modulus::new(40_961)]);

        let target = source_basis();

        let coefficients = vec![1_u128, 42, 65_536, 1_000_000];

        assert!(coefficients
            .iter()
            .all(|&value| { value < source.composite_modulus() }));

        let polynomial = RnsPolynomial::from_coefficients(source.moduli().to_vec(), &coefficients);

        let converted = convert_basis(&polynomial, &target);

        assert_eq!(converted.reconstruct_coefficients(), coefficients);
    }

    #[test]
    fn roundtrip_through_alternate_basis_preserves_small_coefficients() {
        let source = source_basis();
        let alternate = alternate_basis();

        let coefficients = vec![3_u128, 11, 29, 97, 1_001, 100_003];

        let original = RnsPolynomial::from_coefficients(source.moduli().to_vec(), &coefficients);

        let alternate_representation = convert_basis(&original, &alternate);

        let recovered = convert_basis(&alternate_representation, &source);

        assert_eq!(recovered.reconstruct_coefficients(), coefficients);
    }

    #[test]
    fn arithmetic_then_conversion_matches_conversion_of_reference_result() {
        let source = source_basis();
        let target = alternate_basis();

        let lhs_coefficients = vec![1_u128, 2, 3, 4, 5, 6, 7, 8];

        let rhs_coefficients = vec![8_u128, 7, 6, 5, 4, 3, 2, 1];

        let lhs = RnsPolynomial::from_coefficients(source.moduli().to_vec(), &lhs_coefficients);

        let rhs = RnsPolynomial::from_coefficients(source.moduli().to_vec(), &rhs_coefficients);

        let converted_sum = convert_basis(&lhs.add(&rhs), &target);

        let expected = RnsPolynomial::from_coefficients(target.moduli().to_vec(), &[9_u128; 8]);

        assert_eq!(converted_sum, expected);
    }
}
