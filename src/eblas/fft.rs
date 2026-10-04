//! Application-facing one-dimensional FFT for eBLAS.
//!
//! This module separates transform semantics from cryptographic execution.
//! The production clear path is an iterative radix-2 Cooley-Tukey FFT.
//! A dense O(N^2) DFT is retained only as a numerical reference oracle.
//!
//! This transform is distinct from the NTT used internally for polynomial
//! arithmetic by the cryptographic substrate.

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::ckks::{mod_switch_rns_ckks_to_next, CkksCanonicalEmbedding, RnsCkksEvaluator};
use crate::matrix::RnsCkksCiphertextMatrix;
use crate::ring::{ModulusChain, RnsNttPlan};

use super::level1::scale_complex_cp;

/// Direction of a complex Fourier transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftDirection {
    /// Forward transform:
    ///
    /// `X[k] = sum_n x[n] exp(-2*pi*i*k*n/N)`.
    Forward,
    /// Inverse transform, normalized by `1/N`.
    Inverse,
}

/// Shape contract for a one-dimensional radix-2 FFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fft1Shape {
    length: usize,
}

impl Fft1Shape {
    /// Creates a radix-2 FFT shape.
    ///
    /// The current implementation requires a non-zero power-of-two length.
    pub fn new(length: usize) -> Self {
        assert!(length > 0, "eBLAS FFT length must be positive");
        assert!(
            length.is_power_of_two(),
            "eBLAS radix-2 FFT length must be a power of two"
        );

        Self { length }
    }

    /// Number of complex values transformed.
    pub const fn length(self) -> usize {
        self.length
    }

    /// Number of radix-2 butterfly stages.
    pub fn stages(self) -> u32 {
        self.length.trailing_zeros()
    }

    /// Total number of radix-2 butterflies.
    pub fn butterflies(self) -> usize {
        self.length
            .checked_mul(self.stages() as usize)
            .expect("eBLAS FFT butterfly count overflow")
            / 2
    }
}

/// One radix-2 butterfly in an FFT execution plan.
///
/// Indices refer to the working vector after the initial bit-reversal
/// permutation. `twiddle_exponent` selects
///
/// `exp(sign * 2*pi*i*twiddle_exponent/span)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fft1Butterfly {
    stage: usize,
    span: usize,
    even_index: usize,
    odd_index: usize,
    twiddle_exponent: usize,
}

impl Fft1Butterfly {
    pub const fn stage(self) -> usize {
        self.stage
    }

    pub const fn span(self) -> usize {
        self.span
    }

    pub const fn even_index(self) -> usize {
        self.even_index
    }

    pub const fn odd_index(self) -> usize {
        self.odd_index
    }

    pub const fn twiddle_exponent(self) -> usize {
        self.twiddle_exponent
    }
}

/// One radix-2 FFT stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fft1Stage {
    index: usize,
    span: usize,
    butterflies: Vec<Fft1Butterfly>,
}

impl Fft1Stage {
    pub const fn index(&self) -> usize {
        self.index
    }

    pub const fn span(&self) -> usize {
        self.span
    }

    pub fn butterflies(&self) -> &[Fft1Butterfly] {
        &self.butterflies
    }
}

/// Explicit application-facing execution plan for FFT1.
///
/// The plan separates transform structure from backend execution:
///
/// 1. bit-reversal permutation;
/// 2. staged radix-2 butterflies;
/// 3. inverse normalization, when requested by the executor.
///
/// Twiddle values themselves are not stored because they are public constants
/// determined by `(direction, span, twiddle_exponent)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fft1Plan {
    shape: Fft1Shape,
    stages: Vec<Fft1Stage>,
}

impl Fft1Plan {
    /// Builds the canonical iterative radix-2 Cooley-Tukey plan.
    pub fn new(shape: Fft1Shape) -> Self {
        let n = shape.length();
        let mut stages = Vec::with_capacity(shape.stages() as usize);

        let mut span = 2usize;
        let mut stage_index = 0usize;

        while span <= n {
            let half = span / 2;
            let mut butterflies = Vec::with_capacity(n / 2);

            for base in (0..n).step_by(span) {
                for offset in 0..half {
                    butterflies.push(Fft1Butterfly {
                        stage: stage_index,
                        span,
                        even_index: base + offset,
                        odd_index: base + offset + half,
                        twiddle_exponent: offset,
                    });
                }
            }

            stages.push(Fft1Stage {
                index: stage_index,
                span,
                butterflies,
            });

            stage_index += 1;

            span = span
                .checked_mul(2)
                .expect("eBLAS FFT plan stage span overflow");
        }

        let plan = Self { shape, stages };

        assert_eq!(
            plan.butterfly_count(),
            shape.butterflies(),
            "eBLAS FFT plan butterfly count must match shape contract"
        );

        plan
    }

    pub const fn shape(&self) -> Fft1Shape {
        self.shape
    }

    pub fn stages(&self) -> &[Fft1Stage] {
        &self.stages
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    pub fn butterfly_count(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.butterflies.len())
            .sum()
    }
}

/// Executes an explicit FFT1 plan.
///
/// This is numerically equivalent to [`fft1_pp`] but consumes the validated
/// structural plan rather than reconstructing butterfly topology internally.
pub fn execute_fft1_plan(
    plan: &Fft1Plan,
    direction: FftDirection,
    input: &[Complex64],
) -> Vec<Complex64> {
    let shape = plan.shape();

    assert_eq!(
        input.len(),
        shape.length(),
        "eBLAS FFT plan input length must match transform shape"
    );

    let mut values = input.to_vec();
    bit_reverse_permute(&mut values);

    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };

    for stage in plan.stages() {
        for butterfly in stage.butterflies() {
            debug_assert_eq!(butterfly.stage(), stage.index());
            debug_assert_eq!(butterfly.span(), stage.span());

            let angle =
                sign * 2.0 * PI * butterfly.twiddle_exponent() as f64 / butterfly.span() as f64;

            let twiddle = Complex64::new(angle.cos(), angle.sin());

            let even = values[butterfly.even_index()];
            let odd = twiddle * values[butterfly.odd_index()];

            values[butterfly.even_index()] = even + odd;
            values[butterfly.odd_index()] = even - odd;
        }
    }

    if direction == FftDirection::Inverse {
        let scale = shape.length() as f64;

        for value in &mut values {
            *value /= scale;
        }
    }

    values
}

/// Executes one radix-2 FFT butterfly over encrypted operands and a public
/// complex twiddle.
///
/// For encrypted matrices `a` and `b`, this computes
///
/// ```text
/// t = twiddle * b
/// u = a + t
/// v = a - t
/// ```
///
/// `scale_complex_cp` performs the public complex multiplication and consumes
/// one CKKS level. `a` is modulus-switched to the same next level without
/// changing its scale, after which ciphertext addition and subtraction preserve
/// the aligned state.
///
/// The transform coefficient is public: this primitive performs no
/// ciphertext-ciphertext multiplication and requires no relinearization.
///
/// This operation consumes exactly one CKKS level.
pub fn fft1_butterfly_cp(
    evaluator: &RnsCkksEvaluator<'_>,
    a: &RnsCkksCiphertextMatrix,
    b: &RnsCkksCiphertextMatrix,
    twiddle: Complex64,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> (RnsCkksCiphertextMatrix, RnsCkksCiphertextMatrix) {
    assert!(
        twiddle.re.is_finite() && twiddle.im.is_finite(),
        "eBLAS FFT1 butterfly twiddle must be finite"
    );

    assert_eq!(
        a.rows(),
        b.rows(),
        "eBLAS FFT1 butterfly operand row counts must match"
    );
    assert_eq!(
        a.cols(),
        b.cols(),
        "eBLAS FFT1 butterfly operand column counts must match"
    );
    assert_eq!(
        a.level(),
        b.level(),
        "eBLAS FFT1 butterfly operands must have matching levels"
    );
    assert_eq!(
        a.scale(),
        b.scale(),
        "eBLAS FFT1 butterfly operands must have matching scales"
    );
    assert_eq!(
        a.get(0, 0).basis(),
        b.get(0, 0).basis(),
        "eBLAS FFT1 butterfly operands must have matching bases"
    );

    let weighted_b = scale_complex_cp(b, twiddle, embedding, chain, plan);

    let mut aligned_a_data = Vec::with_capacity(a.len());
    for col in 0..a.cols() {
        for row in 0..a.rows() {
            aligned_a_data.push(mod_switch_rns_ckks_to_next(a.get(row, col), chain));
        }
    }

    let aligned_a =
        RnsCkksCiphertextMatrix::from_vec_column_major(a.rows(), a.cols(), aligned_a_data);

    assert_eq!(
        aligned_a.level(),
        weighted_b.level(),
        "eBLAS FFT1 butterfly aligned operands must have matching levels"
    );
    assert_eq!(
        aligned_a.scale(),
        weighted_b.scale(),
        "eBLAS FFT1 butterfly aligned operands must have matching scales"
    );
    assert_eq!(
        aligned_a.get(0, 0).basis(),
        weighted_b.get(0, 0).basis(),
        "eBLAS FFT1 butterfly aligned operands must have matching bases"
    );

    let mut upper = Vec::with_capacity(a.len());
    let mut lower = Vec::with_capacity(a.len());

    for col in 0..a.cols() {
        for row in 0..a.rows() {
            upper.push(evaluator.add(aligned_a.get(row, col), weighted_b.get(row, col)));
            lower.push(evaluator.sub(aligned_a.get(row, col), weighted_b.get(row, col)));
        }
    }

    (
        RnsCkksCiphertextMatrix::from_vec_column_major(a.rows(), a.cols(), upper),
        RnsCkksCiphertextMatrix::from_vec_column_major(a.rows(), a.cols(), lower),
    )
}

/// Dense complex DFT reference oracle.
///
/// This intentionally uses O(N^2) work and exists for semantic validation.
/// Production FFT execution must not call this routine.
pub fn dft1_reference(
    shape: Fft1Shape,
    direction: FftDirection,
    input: &[Complex64],
) -> Vec<Complex64> {
    assert_eq!(
        input.len(),
        shape.length(),
        "eBLAS DFT input length must match transform shape"
    );

    let n = shape.length();
    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };

    let mut output = vec![Complex64::new(0.0, 0.0); n];

    for (k, output_value) in output.iter_mut().enumerate() {
        let mut sum = Complex64::new(0.0, 0.0);

        for (sample, &value) in input.iter().enumerate() {
            let angle = sign * 2.0 * PI * (k as f64) * (sample as f64) / (n as f64);

            let twiddle = Complex64::new(angle.cos(), angle.sin());
            sum += value * twiddle;
        }

        *output_value = sum;
    }

    if direction == FftDirection::Inverse {
        let scale = n as f64;
        for value in &mut output {
            *value /= scale;
        }
    }

    output
}

/// Computes an application-facing one-dimensional radix-2 FFT.
///
/// The algorithm is iterative decimation-in-time Cooley-Tukey:
///
/// 1. bit-reverse the input ordering;
/// 2. execute `log2(N)` butterfly stages;
/// 3. normalize by `1/N` for inverse transforms.
///
/// Twiddle factors are public constants.
pub fn fft1_pp(shape: Fft1Shape, direction: FftDirection, input: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(
        input.len(),
        shape.length(),
        "eBLAS FFT input length must match transform shape"
    );

    let n = shape.length();
    let mut values = input.to_vec();

    bit_reverse_permute(&mut values);

    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };

    let mut span = 2usize;

    while span <= n {
        let half = span / 2;
        let angle = sign * 2.0 * PI / span as f64;
        let stage_root = Complex64::new(angle.cos(), angle.sin());

        for base in (0..n).step_by(span) {
            let mut twiddle = Complex64::new(1.0, 0.0);

            for offset in 0..half {
                let even_index = base + offset;
                let odd_index = even_index + half;

                let even = values[even_index];
                let odd = twiddle * values[odd_index];

                values[even_index] = even + odd;
                values[odd_index] = even - odd;

                twiddle *= stage_root;
            }
        }

        span = span.checked_mul(2).expect("eBLAS FFT stage span overflow");
    }

    if direction == FftDirection::Inverse {
        let scale = n as f64;
        for value in &mut values {
            *value /= scale;
        }
    }

    values
}

fn bit_reverse_permute(values: &mut [Complex64]) {
    let n = values.len();

    if n <= 2 {
        return;
    }

    let bits = n.trailing_zeros();

    for index in 0..n {
        let reversed = index.reverse_bits() >> (usize::BITS - bits);

        if reversed > index {
            values.swap(index, reversed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_complex(length: usize, salt: usize) -> Vec<Complex64> {
        (0..length)
            .map(|index| {
                let real_raw = (index * 17 + salt * 13) % 37;
                let imag_raw = (index * 11 + salt * 19) % 31;

                Complex64::new(
                    (real_raw as f64 - 18.0) / 19.0,
                    (imag_raw as f64 - 15.0) / 17.0,
                )
            })
            .collect()
    }

    fn max_abs_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
        assert_eq!(actual.len(), expected.len());

        actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn shape_exposes_radix2_structure() {
        let shape = Fft1Shape::new(8);

        assert_eq!(shape.length(), 8);
        assert_eq!(shape.stages(), 3);
        assert_eq!(shape.butterflies(), 12);
    }

    #[test]
    #[should_panic(expected = "FFT length must be positive")]
    fn shape_rejects_zero_length() {
        let _ = Fft1Shape::new(0);
    }

    #[test]
    #[should_panic(expected = "FFT length must be a power of two")]
    fn shape_rejects_non_power_of_two() {
        let _ = Fft1Shape::new(12);
    }

    #[test]
    fn impulse_has_flat_forward_spectrum() {
        let shape = Fft1Shape::new(8);

        let mut input = vec![Complex64::new(0.0, 0.0); 8];
        input[0] = Complex64::new(1.0, 0.0);

        let output = fft1_pp(shape, FftDirection::Forward, &input);

        for value in output {
            assert!((value - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
        }
    }

    #[test]
    fn constant_signal_maps_to_dc() {
        let shape = Fft1Shape::new(8);
        let input = vec![Complex64::new(2.0, 0.0); 8];

        let output = fft1_pp(shape, FftDirection::Forward, &input);

        assert!((output[0] - Complex64::new(16.0, 0.0)).norm() < 1.0e-12);

        for value in &output[1..] {
            assert!(value.norm() < 1.0e-12);
        }
    }

    #[test]
    fn radix2_fft_matches_dense_dft_reference() {
        for &(length, salt) in &[
            (1usize, 1usize),
            (2, 2),
            (4, 3),
            (8, 4),
            (16, 5),
            (32, 6),
            (64, 7),
            (128, 8),
            (256, 9),
        ] {
            let shape = Fft1Shape::new(length);
            let input = deterministic_complex(length, salt);

            let expected = dft1_reference(shape, FftDirection::Forward, &input);

            let actual = fft1_pp(shape, FftDirection::Forward, &input);

            let error = max_abs_error(&actual, &expected);

            println!(
                "FFT1_PP_DFT_CASE=N{} STAGES{} BUTTERFLIES{} MAX_ABS={:.12e}",
                length,
                shape.stages(),
                shape.butterflies(),
                error,
            );

            assert!(
                error < 1.0e-10,
                "radix-2 FFT maximum absolute error {error:e}"
            );
        }
    }

    #[test]
    fn inverse_fft_matches_dense_inverse_dft_reference() {
        for &(length, salt) in &[
            (1usize, 11usize),
            (2, 12),
            (4, 13),
            (8, 14),
            (16, 15),
            (32, 16),
            (64, 17),
        ] {
            let shape = Fft1Shape::new(length);
            let input = deterministic_complex(length, salt);

            let expected = dft1_reference(shape, FftDirection::Inverse, &input);

            let actual = fft1_pp(shape, FftDirection::Inverse, &input);

            let error = max_abs_error(&actual, &expected);

            println!("IFFT1_PP_DFT_CASE=N{} MAX_ABS={:.12e}", length, error,);

            assert!(
                error < 1.0e-10,
                "radix-2 inverse FFT maximum absolute error {error:e}"
            );
        }
    }

    #[test]
    fn inverse_roundtrip_recovers_input() {
        for &(length, salt) in &[
            (1usize, 21usize),
            (2, 22),
            (4, 23),
            (8, 24),
            (16, 25),
            (32, 26),
            (64, 27),
            (128, 28),
            (256, 29),
        ] {
            let shape = Fft1Shape::new(length);
            let input = deterministic_complex(length, salt);

            let spectrum = fft1_pp(shape, FftDirection::Forward, &input);

            let recovered = fft1_pp(shape, FftDirection::Inverse, &spectrum);

            let error = max_abs_error(&recovered, &input);

            println!("FFT1_PP_ROUNDTRIP_CASE=N{} MAX_ABS={:.12e}", length, error,);

            assert!(
                error < 1.0e-10,
                "FFT roundtrip maximum absolute error {error:e}"
            );
        }
    }

    #[test]
    fn plan_exposes_expected_n8_provenance() {
        let shape = Fft1Shape::new(8);
        let plan = Fft1Plan::new(shape);

        assert_eq!(plan.stage_count(), 3);
        assert_eq!(plan.butterfly_count(), 12);

        let stages = plan.stages();

        assert_eq!(stages[0].index(), 0);
        assert_eq!(stages[0].span(), 2);

        assert_eq!(
            stages[0].butterflies(),
            &[
                Fft1Butterfly {
                    stage: 0,
                    span: 2,
                    even_index: 0,
                    odd_index: 1,
                    twiddle_exponent: 0,
                },
                Fft1Butterfly {
                    stage: 0,
                    span: 2,
                    even_index: 2,
                    odd_index: 3,
                    twiddle_exponent: 0,
                },
                Fft1Butterfly {
                    stage: 0,
                    span: 2,
                    even_index: 4,
                    odd_index: 5,
                    twiddle_exponent: 0,
                },
                Fft1Butterfly {
                    stage: 0,
                    span: 2,
                    even_index: 6,
                    odd_index: 7,
                    twiddle_exponent: 0,
                },
            ]
        );

        assert_eq!(stages[1].span(), 4);

        assert_eq!(
            stages[1]
                .butterflies()
                .iter()
                .map(|b| { (b.even_index(), b.odd_index(), b.twiddle_exponent(),) })
                .collect::<Vec<_>>(),
            vec![(0, 2, 0), (1, 3, 1), (4, 6, 0), (5, 7, 1),]
        );

        assert_eq!(stages[2].span(), 8);

        assert_eq!(
            stages[2]
                .butterflies()
                .iter()
                .map(|b| { (b.even_index(), b.odd_index(), b.twiddle_exponent(),) })
                .collect::<Vec<_>>(),
            vec![(0, 4, 0), (1, 5, 1), (2, 6, 2), (3, 7, 3),]
        );
    }

    #[test]
    fn plan_counts_match_shape_contract_through_256() {
        for length in [1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
            let shape = Fft1Shape::new(length);
            let plan = Fft1Plan::new(shape);

            assert_eq!(plan.stage_count(), shape.stages() as usize);

            assert_eq!(plan.butterfly_count(), shape.butterflies());

            println!(
                "FFT1_PLAN_CASE=N{} STAGES{} BUTTERFLIES{}",
                length,
                plan.stage_count(),
                plan.butterfly_count(),
            );
        }
    }

    #[test]
    fn planned_forward_execution_matches_fft_and_dft() {
        for &(length, salt) in &[
            (1usize, 31usize),
            (2, 32),
            (4, 33),
            (8, 34),
            (16, 35),
            (32, 36),
            (64, 37),
            (128, 38),
            (256, 39),
        ] {
            let shape = Fft1Shape::new(length);
            let plan = Fft1Plan::new(shape);
            let input = deterministic_complex(length, salt);

            let expected_dft = dft1_reference(shape, FftDirection::Forward, &input);

            let expected_fft = fft1_pp(shape, FftDirection::Forward, &input);

            let actual = execute_fft1_plan(&plan, FftDirection::Forward, &input);

            let vs_fft = max_abs_error(&actual, &expected_fft);

            let vs_dft = max_abs_error(&actual, &expected_dft);

            println!(
                "FFT1_PLAN_FORWARD_CASE=N{} \
                 MAX_ABS_VS_FFT={:.12e} \
                 MAX_ABS_VS_DFT={:.12e}",
                length, vs_fft, vs_dft,
            );

            assert!(
                vs_fft < 1.0e-12,
                "planned FFT differs from fft1_pp by {vs_fft:e}"
            );

            assert!(
                vs_dft < 1.0e-10,
                "planned FFT differs from DFT oracle by {vs_dft:e}"
            );
        }
    }

    #[test]
    fn planned_inverse_execution_matches_fft_and_roundtrip() {
        for &(length, salt) in &[
            (1usize, 41usize),
            (2, 42),
            (4, 43),
            (8, 44),
            (16, 45),
            (32, 46),
            (64, 47),
            (128, 48),
            (256, 49),
        ] {
            let shape = Fft1Shape::new(length);
            let plan = Fft1Plan::new(shape);
            let input = deterministic_complex(length, salt);

            let spectrum = execute_fft1_plan(&plan, FftDirection::Forward, &input);

            let planned_inverse = execute_fft1_plan(&plan, FftDirection::Inverse, &spectrum);

            let direct_inverse = fft1_pp(shape, FftDirection::Inverse, &spectrum);

            let vs_fft = max_abs_error(&planned_inverse, &direct_inverse);

            let roundtrip = max_abs_error(&planned_inverse, &input);

            println!(
                "FFT1_PLAN_INVERSE_CASE=N{} \
                 MAX_ABS_VS_FFT={:.12e} \
                 ROUNDTRIP_MAX_ABS={:.12e}",
                length, vs_fft, roundtrip,
            );

            assert!(vs_fft < 1.0e-12);
            assert!(roundtrip < 1.0e-10);
        }
    }
}
