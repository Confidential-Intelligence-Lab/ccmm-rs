use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{ToPrimitive, Zero};

use crate::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, ModulusBasis,
    Polynomial, RnsPolynomial,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedGadgetLayout {
    full_basis: ModulusBasis,
    base_log: u32,
    base: u128,
    digit_count: usize,
}

impl BoundedGadgetLayout {
    pub fn new(full_basis: ModulusBasis, base_log: u32) -> Self {
        assert!(
            (1..=63).contains(&base_log),
            "bounded gadget base_log must be in 1..=63"
        );

        let modulus_bits = composite_modulus_big(&full_basis).bits();
        let digit_count = modulus_bits.div_ceil(u64::from(base_log)) as usize;

        Self {
            full_basis,
            base_log,
            base: 1_u128 << base_log,
            digit_count,
        }
    }

    pub fn full_basis(&self) -> &ModulusBasis {
        &self.full_basis
    }

    pub fn base_log(&self) -> u32 {
        self.base_log
    }

    pub fn base(&self) -> u128 {
        self.base
    }

    pub fn digit_count(&self) -> usize {
        self.digit_count
    }

    pub fn maximum_digit_magnitude(&self) -> u128 {
        self.base / 2
    }

    pub fn decompose(&self, polynomial: &RnsPolynomial) -> BoundedGadgetDecomposition {
        assert_eq!(
            polynomial.basis(),
            &self.full_basis,
            "polynomial basis must match bounded gadget layout"
        );

        let modulus = composite_modulus_big(&self.full_basis);
        let base = BigInt::from(self.base);
        let half_base = BigInt::from(self.base / 2);

        let canonical = reconstruct_coefficients_big(polynomial);
        let degree = canonical.len();
        let mut digits = vec![vec![0_i128; degree]; self.digit_count];

        for (coefficient_index, coefficient) in canonical.iter().enumerate() {
            let mut value = centered_representative_big(coefficient, &modulus);

            for digit_row in digits.iter_mut().take(self.digit_count) {
                let mut residue = &value % &base;

                if residue.sign() == Sign::Minus {
                    residue += &base;
                }

                let digit_big = if residue >= half_base {
                    residue - &base
                } else {
                    residue
                };

                let digit = digit_big
                    .to_i128()
                    .expect("bounded gadget digit must fit in i128");

                digit_row[coefficient_index] = digit;
                value = (value - BigInt::from(digit)) / &base;
            }

            assert!(
                value.is_zero(),
                "bounded gadget digit count is insufficient for centered coefficient"
            );
        }

        BoundedGadgetDecomposition {
            layout: self.clone(),
            digits,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedGadgetDecomposition {
    layout: BoundedGadgetLayout,
    digits: Vec<Vec<i128>>,
}

impl BoundedGadgetDecomposition {
    pub fn layout(&self) -> &BoundedGadgetLayout {
        &self.layout
    }

    pub fn digits(&self) -> &[Vec<i128>] {
        &self.digits
    }

    pub fn digit(&self, index: usize) -> &[i128] {
        &self.digits[index]
    }

    pub fn reconstruct_centered_coefficients_big(&self) -> Vec<BigInt> {
        let degree = self.digits[0].len();
        let base = BigInt::from(self.layout.base);
        let mut output = vec![BigInt::zero(); degree];

        for (coefficient_index, output_value) in output.iter_mut().enumerate() {
            let mut power = BigInt::from(1_u8);
            let mut value = BigInt::zero();

            for digit_index in 0..self.digits.len() {
                value += BigInt::from(self.digits[digit_index][coefficient_index]) * &power;

                if digit_index + 1 < self.digits.len() {
                    power *= &base;
                }
            }

            *output_value = value;
        }

        output
    }

    pub fn reconstruct_centered_coefficients(&self) -> Vec<i128> {
        self.reconstruct_centered_coefficients_big()
            .into_iter()
            .map(|value| {
                value
                    .to_i128()
                    .expect("centered coefficient does not fit in legacy i128 API")
            })
            .collect()
    }

    pub fn reconstruct_coefficients_big(&self) -> Vec<BigUint> {
        let modulus = composite_modulus_big(&self.layout.full_basis);

        self.reconstruct_centered_coefficients_big()
            .into_iter()
            .map(|value| match value.sign() {
                Sign::Minus => {
                    let magnitude = value.magnitude();

                    assert!(
                        magnitude <= &modulus,
                        "centered coefficient magnitude exceeds composite modulus"
                    );

                    &modulus - magnitude
                }
                _ => value
                    .to_biguint()
                    .expect("nonnegative centered coefficient must convert to BigUint"),
            })
            .collect()
    }

    pub fn reconstruct_coefficients(&self) -> Vec<u128> {
        self.reconstruct_coefficients_big()
            .into_iter()
            .map(|value| {
                value
                    .to_u128()
                    .expect("coefficient does not fit in legacy u128 API")
            })
            .collect()
    }

    pub fn lift_digit(&self, index: usize) -> RnsPolynomial {
        let residues = self
            .layout
            .full_basis
            .moduli()
            .iter()
            .copied()
            .map(|modulus| {
                let q = i128::from(modulus.value());

                let coefficients = self.digits[index]
                    .iter()
                    .map(|&value| value.rem_euclid(q) as u64)
                    .collect();

                Polynomial::new(modulus, coefficients)
            })
            .collect();

        RnsPolynomial::from_residues(residues)
    }

    pub fn maximum_observed_digit_magnitude(&self) -> u128 {
        self.digits
            .iter()
            .flatten()
            .map(|&value| value.unsigned_abs())
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::Modulus;

    fn basis() -> ModulusBasis {
        ModulusBasis::new(vec![
            Modulus::new(12_289),
            Modulus::new(40_961),
            Modulus::new(65_537),
        ])
    }

    #[test]
    fn layout_reports_expected_power_of_two_base() {
        let layout = BoundedGadgetLayout::new(basis(), 8);
        assert_eq!(layout.base(), 256);
        assert_eq!(layout.maximum_digit_magnitude(), 128);
        assert!(layout.digit_count() > 0);
    }

    #[test]
    fn decomposition_reconstructs_canonical_coefficients_exactly() {
        let basis = basis();
        let q = basis.composite_modulus();
        let coefficients = vec![0, 1, 17, 42, 1_001, q / 3, q / 2, q - 42, q - 1];
        let polynomial = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &coefficients);

        for base_log in [2_u32, 4, 8, 12, 16, 20, 24, 32] {
            let decomposition =
                BoundedGadgetLayout::new(basis.clone(), base_log).decompose(&polynomial);
            assert_eq!(
                decomposition.reconstruct_coefficients(),
                coefficients,
                "bounded gadget reconstruction mismatch for base_log={base_log}"
            );
        }
    }

    #[test]
    fn digits_respect_balanced_bound() {
        let basis = basis();
        let q = basis.composite_modulus();
        let coefficients: Vec<u128> = (0..128_u128)
            .map(|index| (97 * index + 31 * index * index + 11) % q)
            .collect();
        let polynomial = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &coefficients);

        for base_log in [2_u32, 4, 8, 12, 16, 20, 24, 32] {
            let layout = BoundedGadgetLayout::new(basis.clone(), base_log);
            let decomposition = layout.decompose(&polynomial);
            assert!(
                decomposition.maximum_observed_digit_magnitude()
                    <= layout.maximum_digit_magnitude(),
                "digit bound violated for base_log={base_log}"
            );
        }
    }

    #[test]
    fn lifted_digits_use_full_basis() {
        let basis = basis();
        let polynomial =
            RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &[1, 17, 42, 1_001, 65_536]);
        let decomposition = BoundedGadgetLayout::new(basis.clone(), 8).decompose(&polynomial);

        for digit_index in 0..decomposition.layout().digit_count() {
            assert_eq!(decomposition.lift_digit(digit_index).basis(), &basis);
        }
    }

    #[test]
    fn deterministic_campaign_is_exact_across_bases() {
        let basis = basis();
        let q = basis.composite_modulus();

        for seed in 0_u128..64 {
            let coefficients: Vec<u128> = (0..64_u128)
                .map(|index| (97 * seed + 31 * index + 7 * index * index + 11) % q)
                .collect();
            let polynomial =
                RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &coefficients);

            for base_log in [2_u32, 4, 8, 12, 16, 20, 24, 32] {
                let decomposition =
                    BoundedGadgetLayout::new(basis.clone(), base_log).decompose(&polynomial);
                assert_eq!(
                    decomposition.reconstruct_coefficients(),
                    coefficients,
                    "bounded gadget mismatch: seed={seed}, base_log={base_log}"
                );
            }
        }
    }

    #[test]
    fn negative_centered_values_reconstruct_exactly() {
        let basis = basis();
        let q = basis.composite_modulus();
        let coefficients = vec![q - 1, q - 2, q - 17, q - 1_001];
        let polynomial = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &coefficients);
        let decomposition = BoundedGadgetLayout::new(basis, 8).decompose(&polynomial);

        assert_eq!(
            decomposition.reconstruct_centered_coefficients(),
            vec![-1, -2, -17, -1_001]
        );
        assert_eq!(decomposition.reconstruct_coefficients(), coefficients);
    }

    #[test]
    fn realistic_4096_decomposition_reconstructs_exactly() {
        use crate::ckks::research_profile_4096;

        let profile = research_profile_4096();
        let basis = profile.modulus_chain().top().clone();
        let q = basis.composite_modulus();

        let coefficients: Vec<u128> = (0..profile.degree())
            .map(|index| {
                let index = index as u128;
                (index * index * 1_000_003 + index * 97 + q / 3) % q
            })
            .collect();

        let polynomial = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &coefficients);

        for base_log in [4_u32, 8, 12, 16, 20] {
            let layout = BoundedGadgetLayout::new(basis.clone(), base_log);
            let decomposition = layout.decompose(&polynomial);

            assert_eq!(
                decomposition.reconstruct_coefficients(),
                coefficients,
                "N=4096 bounded decomposition mismatch for base_log={base_log}"
            );

            assert!(
                decomposition.maximum_observed_digit_magnitude()
                    <= layout.maximum_digit_magnitude(),
                "N=4096 digit bound violated for base_log={base_log}"
            );

            println!(
                "R3_1B_BASE_LOG={base_log} BASE={} DIGITS={} MAX_DIGIT={} BOUND={}",
                layout.base(),
                layout.digit_count(),
                decomposition.maximum_observed_digit_magnitude(),
                layout.maximum_digit_magnitude(),
            );
        }
    }

    #[test]
    #[should_panic(expected = "base_log must be in 1..=63")]
    fn rejects_zero_base_log() {
        let _ = BoundedGadgetLayout::new(basis(), 0);
    }
}

#[cfg(test)]
mod wide_modulus_tests {
    use num_bigint::BigUint;
    use num_traits::One;

    use crate::ckks::research_profile_8192;
    use crate::ring::{composite_modulus_big, rns_from_big_coefficients};

    use super::BoundedGadgetLayout;

    #[test]
    fn research_8192_bounded_decomposition_roundtrips_above_u128() {
        let profile = research_profile_8192();
        let basis = profile.modulus_basis();
        let modulus = composite_modulus_big(&basis);

        let large = (BigUint::one() << 150_usize) + BigUint::from(0x1234_5678_u64);
        assert!(large < modulus);

        let coefficients = vec![
            BigUint::from(0_u64),
            BigUint::from(1_u64),
            large,
            &modulus - BigUint::from(1_u64),
            &modulus - BigUint::from(17_u64),
        ];

        let polynomial = rns_from_big_coefficients(basis.moduli().to_vec(), &coefficients);

        let layout = BoundedGadgetLayout::new(basis.clone(), 20);
        assert_eq!(layout.digit_count(), 10);

        let decomposition = layout.decompose(&polynomial);

        assert_eq!(decomposition.reconstruct_coefficients_big(), coefficients);

        assert!(
            decomposition.maximum_observed_digit_magnitude() <= layout.maximum_digit_magnitude()
        );

        for digit_index in 0..layout.digit_count() {
            assert_eq!(decomposition.lift_digit(digit_index).basis(), &basis);
        }
    }
}
