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

        // Balanced radix-2^base_log digits have a signed leading
        // digit. Reserve one additional bit of representation
        // capacity to cover the full centered modulus interval.
        let digit_count = (modulus_bits + 1).div_ceil(u64::from(base_log)) as usize;

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

        let canonical = reconstruct_coefficients_big(polynomial);
        self.decompose_canonical(&canonical)
    }

    // R36_4A_CANONICAL_DECOMPOSITION
    /// Decompose canonical coefficients in [0, Q), without repeating CRT.
    ///
    /// This preserves the centered balanced-digit convention used by
    /// the original RNS decomposition.
    pub fn decompose_canonical(&self, canonical: &[BigUint]) -> BoundedGadgetDecomposition {
        let modulus = composite_modulus_big(&self.full_basis);
        let base = BigInt::from(self.base);
        let half_base = BigInt::from(self.base / 2);

        let degree = canonical.len();
        assert!(degree > 0, "canonical decomposition requires coefficients");
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

// R36_4B_PREPARED_CANONICAL_SOURCE
/// Reusable canonical representation of an RNS polynomial.
///
/// The expensive CRT reconstruction is performed once. Subsequent
/// automorphisms operate on canonical coefficients modulo Q.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedCanonicalGaloisSource {
    basis: ModulusBasis,
    canonical: Vec<BigUint>,
    modulus: BigUint,
}

impl PreparedCanonicalGaloisSource {
    pub fn new(polynomial: &RnsPolynomial) -> Self {
        let basis = polynomial.basis().clone();
        let modulus = composite_modulus_big(&basis);
        let canonical = reconstruct_coefficients_big(polynomial);

        Self {
            basis,
            canonical,
            modulus,
        }
    }

    pub fn basis(&self) -> &ModulusBasis {
        &self.basis
    }

    pub fn degree(&self) -> usize {
        self.canonical.len()
    }

    pub fn automorphism_canonical(&self, exponent: usize) -> Vec<BigUint> {
        let degree = self.degree();

        assert!(
            degree >= 2 && degree.is_power_of_two(),
            "canonical Galois source requires power-of-two degree >= 2"
        );

        let two_n = 2 * degree;
        let exponent = exponent % two_n;

        assert!(
            exponent % 2 == 1,
            "canonical Galois exponent must be odd modulo 2N"
        );

        let mut output = vec![BigUint::zero(); degree];

        for (source_index, coefficient) in self.canonical.iter().enumerate() {
            let mapped = (source_index * exponent) % two_n;
            let target = mapped % degree;

            output[target] = if mapped < degree || coefficient.is_zero() {
                coefficient.clone()
            } else {
                &self.modulus - coefficient
            };
        }

        output
    }

    pub fn decompose_automorphism(
        &self,
        layout: &BoundedGadgetLayout,
        exponent: usize,
    ) -> BoundedGadgetDecomposition {
        assert_eq!(
            layout.full_basis(),
            &self.basis,
            "prepared canonical source basis mismatch"
        );

        layout.decompose_canonical(&self.automorphism_canonical(exponent))
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
    fn r36_4b_prepared_source_matches_rotated_rns_digits_exactly() {
        use crate::ckks::apply_automorphism;

        let basis = basis();
        let layout = BoundedGadgetLayout::new(basis.clone(), 8);

        let coefficients = vec![0_u128, 1, 2, 127, 128, 255, 65_535, 1_000_000];

        let polynomial = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &coefficients);

        let prepared = PreparedCanonicalGaloisSource::new(&polynomial);

        assert_eq!(prepared.basis(), &basis);
        assert_eq!(prepared.degree(), polynomial.degree());

        for exponent in (1..16).step_by(2) {
            let rotated_residues = polynomial
                .residues()
                .iter()
                .map(|residue| apply_automorphism(residue, exponent))
                .collect();

            let rotated = RnsPolynomial::from_residues(rotated_residues);

            assert_eq!(
                prepared.decompose_automorphism(&layout, exponent),
                layout.decompose(&rotated),
                "R36.4B exact digit mismatch at exponent {exponent}",
            );
        }

        println!("R36_4B_EXACT_DIGIT_EQUIVALENCE=PASS");
    }

    #[test]
    fn r36_4a_canonical_decomposition_matches_rns_exactly() {
        use crate::ckks::apply_automorphism;
        use crate::ring::reconstruct_coefficients_big;

        let basis = basis();
        let layout = BoundedGadgetLayout::new(basis.clone(), 8);
        let degree = 8;

        let coefficients = vec![0_u128, 1, 2, 127, 128, 255, 65_535, 1_000_000];

        let polynomial = RnsPolynomial::from_coefficients(basis.moduli().to_vec(), &coefficients);

        let canonical = reconstruct_coefficients_big(&polynomial);
        assert_eq!(
            layout.decompose(&polynomial),
            layout.decompose_canonical(&canonical),
        );

        let modulus = composite_modulus_big(&basis);
        let two_n = 2 * degree;

        for exponent in (1..two_n).step_by(2) {
            let mut rotated = vec![BigUint::zero(); degree];

            for (index, coefficient) in canonical.iter().enumerate() {
                let mapped = (index * exponent) % two_n;
                let target = mapped % degree;

                rotated[target] = if mapped < degree || coefficient.is_zero() {
                    coefficient.clone()
                } else {
                    &modulus - coefficient
                };
            }

            let rotated_residues = polynomial
                .residues()
                .iter()
                .map(|residue| apply_automorphism(residue, exponent))
                .collect();

            let rotated_rns = RnsPolynomial::from_residues(rotated_residues);

            assert_eq!(
                layout.decompose(&rotated_rns),
                layout.decompose_canonical(&rotated),
                "R36.4A exact digit mismatch at exponent {exponent}",
            );
        }

        println!("R36_4A_EXACT_DIGIT_EQUIVALENCE=PASS");
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
        assert_eq!(layout.digit_count(), 11);

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

// R36_5C3_PRODUCTION_DECOMPOSITION_BENCHMARK
#[cfg(test)]
mod r36_5c3_benchmark {
    use super::*;
    use crate::ckks::research_profile_65536_for_levels;
    use crate::ring::{ModulusChain, Polynomial};
    use std::time::Instant;

    #[test]
    #[ignore = "production-scale performance characterization"]
    fn r36_5c3_production_canonical_decomposition_benchmark() {
        let degree = 65_536usize;
        let profile = research_profile_65536_for_levels(13, 3);

        assert_eq!(profile.degree(), degree);

        let chain = ModulusChain::from_top_basis(profile.modulus_chain().top().clone());

        assert_eq!(chain.top().moduli().len(), 16);

        let q = chain.level(3).clone();
        assert_eq!(q.moduli().len(), 13);

        let layout = BoundedGadgetLayout::new(q.clone(), 32);

        let polynomial = RnsPolynomial::from_residues(
            q.moduli()
                .iter()
                .map(|&modulus| {
                    Polynomial::new(
                        modulus,
                        (0..degree)
                            .map(|i| (i as u64 * 37 + 11) % modulus.value())
                            .collect(),
                    )
                })
                .collect(),
        );

        let exponents: Vec<usize> = (0..32).map(|i| 2 * i + 3).collect();

        let started = Instant::now();
        let prepared = PreparedCanonicalGaloisSource::new(&polynomial);
        let preparation_seconds = started.elapsed().as_secs_f64();

        let mut legacy_total = 0.0f64;
        let mut canonical_total = 0.0f64;

        for (index, &exponent) in exponents.iter().enumerate() {
            let transformed = RnsPolynomial::from_residues(
                polynomial
                    .residues()
                    .iter()
                    .map(|residue| {
                        let modulus = residue.modulus();
                        let mut output = vec![0_u64; degree];

                        for (i, &value) in residue.coefficients().iter().enumerate() {
                            let mapped = (i * exponent) % (2 * degree);
                            let target = mapped % degree;

                            output[target] = if mapped < degree || value == 0 {
                                value
                            } else {
                                modulus.value() - value
                            };
                        }

                        Polynomial::new(modulus, output)
                    })
                    .collect(),
            );

            let started = Instant::now();
            let legacy = layout.decompose(&transformed);
            legacy_total += started.elapsed().as_secs_f64();

            let started = Instant::now();
            let canonical = prepared.decompose_automorphism(&layout, exponent);
            canonical_total += started.elapsed().as_secs_f64();

            assert_eq!(legacy, canonical, "R36.5C.3 mismatch exponent={exponent}");

            let rotations = index + 1;

            if [1, 2, 4, 8, 16, 32].contains(&rotations) {
                let amortized = preparation_seconds + canonical_total;

                println!(
                    "R36_5C3_RESULT degree={} q_limbs={} base_log=32 rotations={} preparation_seconds={:.6} legacy_seconds={:.6} canonical_seconds={:.6} amortized_canonical_seconds={:.6} speedup={:.6}",
                    degree,
                    q.moduli().len(),
                    rotations,
                    preparation_seconds,
                    legacy_total,
                    canonical_total,
                    amortized,
                    legacy_total / amortized,
                );
            }
        }

        println!("R36_5C3_EXACT_DECOMPOSITION_STATUS=PASS");
    }
}

// R36_5C2_CANONICAL_DECOMPOSITION_BENCHMARK
#[cfg(test)]
mod r36_5c2_benchmark {
    use super::*;
    use crate::ring::{Modulus, Polynomial};
    use std::time::Instant;

    #[test]
    #[ignore = "opt-in performance characterization"]
    fn r36_5c2_canonical_decomposition_benchmark() {
        let degree: usize = std::env::var("FHE_R36_5C2_DEGREE")
            .ok()
            .map(|value| value.parse().expect("invalid degree"))
            .unwrap_or(1024);

        assert!(degree >= 16 && degree.is_power_of_two());

        let q = ModulusBasis::new(
            [12_289_u64, 40_961, 65_537]
                .into_iter()
                .map(Modulus::new)
                .collect(),
        );

        let polynomial = RnsPolynomial::from_residues(
            q.moduli()
                .iter()
                .map(|&modulus| {
                    Polynomial::new(
                        modulus,
                        (0..degree)
                            .map(|i| (i as u64 * 37 + 11) % modulus.value())
                            .collect(),
                    )
                })
                .collect(),
        );

        let layout = BoundedGadgetLayout::new(q, 32);

        let started = Instant::now();
        let prepared = PreparedCanonicalGaloisSource::new(&polynomial);
        let preparation_seconds = started.elapsed().as_secs_f64();

        let exponents = [3_usize, 5, 7, 11, 15, 19, 23, 27];

        for count in [1_usize, 2, 4, 8] {
            let mut legacy_seconds = 0.0;
            let mut canonical_seconds = 0.0;

            for &exponent in exponents.iter().take(count) {
                let transformed = RnsPolynomial::from_residues(
                    polynomial
                        .residues()
                        .iter()
                        .map(|residue| {
                            let modulus = residue.modulus();
                            let coefficients = residue.coefficients();
                            let mut output = vec![0_u64; degree];

                            for (i, &value) in coefficients.iter().enumerate() {
                                let mapped = (i * exponent) % (2 * degree);
                                let target = mapped % degree;

                                output[target] = if mapped < degree || value == 0 {
                                    value
                                } else {
                                    modulus.value() - value
                                };
                            }

                            Polynomial::new(modulus, output)
                        })
                        .collect(),
                );

                let started = Instant::now();
                let legacy = layout.decompose(&transformed);
                legacy_seconds += started.elapsed().as_secs_f64();

                let started = Instant::now();
                let canonical = prepared.decompose_automorphism(&layout, exponent);
                canonical_seconds += started.elapsed().as_secs_f64();

                assert_eq!(
                    legacy, canonical,
                    "R36.5C.2 gadget mismatch exponent={exponent}"
                );
            }

            let amortized_canonical_seconds = preparation_seconds + canonical_seconds;

            println!(
                "R36_5C2_RESULT degree={} q_limbs={} rotations={} preparation_seconds={:.9} legacy_seconds={:.9} canonical_seconds={:.9} amortized_canonical_seconds={:.9} speedup={:.6}",
                degree,
                layout.full_basis().moduli().len(),
                count,
                preparation_seconds,
                legacy_seconds,
                canonical_seconds,
                amortized_canonical_seconds,
                legacy_seconds / amortized_canonical_seconds,
            );
        }

        println!("R36_5C2_EXACT_DECOMPOSITION_STATUS=PASS");
    }
}
