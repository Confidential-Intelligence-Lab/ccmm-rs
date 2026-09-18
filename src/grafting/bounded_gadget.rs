use crate::ring::{ModulusBasis, RnsPolynomial};

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

        let q = full_basis.composite_modulus();

        assert!(
            q <= i128::MAX as u128,
            "bounded gadget composite modulus must fit in i128"
        );

        let modulus_bits = u128::BITS - q.leading_zeros();
        let digit_count = modulus_bits.div_ceil(base_log) as usize;

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

        let q = self.full_basis.composite_modulus();
        let base = self.base as i128;
        let half_base = base / 2;
        let canonical = polynomial.reconstruct_coefficients();
        let degree = canonical.len();
        let mut digits = vec![vec![0_i128; degree]; self.digit_count];

        for (coefficient_index, coefficient) in canonical.into_iter().enumerate() {
            let mut value = center(coefficient, q);

            for digit_row in digits.iter_mut().take(self.digit_count) {
                let residue = value.rem_euclid(base);
                let digit = if residue >= half_base {
                    residue - base
                } else {
                    residue
                };

                digit_row[coefficient_index] = digit;
                value = (value - digit) / base;
            }

            assert_eq!(
                value, 0,
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

    pub fn reconstruct_centered_coefficients(&self) -> Vec<i128> {
        let degree = self.digits[0].len();
        let base = self.layout.base as i128;
        let mut output = vec![0_i128; degree];

        for (coefficient_index, output_value) in output.iter_mut().enumerate() {
            let mut power = 1_i128;
            let mut value = 0_i128;

            for digit_index in 0..self.digits.len() {
                let term = self.digits[digit_index][coefficient_index]
                    .checked_mul(power)
                    .expect("bounded gadget reconstruction term exceeds i128");

                value = value
                    .checked_add(term)
                    .expect("bounded gadget reconstruction exceeds i128");

                if digit_index + 1 < self.digits.len() {
                    power = power
                        .checked_mul(base)
                        .expect("bounded gadget radix power exceeds i128");
                }
            }

            *output_value = value;
        }

        output
    }

    pub fn reconstruct_coefficients(&self) -> Vec<u128> {
        let q = self.layout.full_basis.composite_modulus();

        self.reconstruct_centered_coefficients()
            .into_iter()
            .map(|value| canonicalize(value, q))
            .collect()
    }

    pub fn lift_digit(&self, index: usize) -> RnsPolynomial {
        let q = self.layout.full_basis.composite_modulus();

        let coefficients: Vec<u128> = self.digits[index]
            .iter()
            .copied()
            .map(|value| canonicalize(value, q))
            .collect();

        RnsPolynomial::from_coefficients(self.layout.full_basis.moduli().to_vec(), &coefficients)
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

fn center(value: u128, modulus: u128) -> i128 {
    if value > modulus / 2 {
        value as i128 - modulus as i128
    } else {
        value as i128
    }
}

fn canonicalize(value: i128, modulus: u128) -> u128 {
    let modulus = modulus as i128;
    value.rem_euclid(modulus) as u128
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
