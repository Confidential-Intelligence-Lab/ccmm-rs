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

use crate::ckks::{
    mod_switch_rns_ckks_to_next, rotate_left_rns_ckks_with_prepared_ntt,
    rotate_right_rns_ckks_with_prepared_ntt, CkksCanonicalEmbedding, PreparedRnsGaloisKey,
    RnsCkksCiphertext, RnsCkksEvaluator,
};
use crate::matrix::RnsCkksCiphertextMatrix;
use crate::ring::{ModulusChain, RnsNttPlan};

use super::level1::{multiply_complex_slots_cp, scale_complex_cp};

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

/// Shape contract for a two-dimensional radix-2 FFT.
///
/// Logical values use row-major layout:
///
/// ```text
/// physical_index = row * cols + col
/// ```
///
/// Both dimensions are transformed independently and must be non-zero
/// powers of two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fft2Shape {
    rows: usize,
    cols: usize,
}

impl Fft2Shape {
    /// Creates a two-dimensional radix-2 FFT shape.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0, "eBLAS FFT2 row count must be positive");
        assert!(cols > 0, "eBLAS FFT2 column count must be positive");
        assert!(
            rows.is_power_of_two(),
            "eBLAS radix-2 FFT2 row count must be a power of two"
        );
        assert!(
            cols.is_power_of_two(),
            "eBLAS radix-2 FFT2 column count must be a power of two"
        );

        rows.checked_mul(cols)
            .expect("eBLAS FFT2 element count overflow");

        Self { rows, cols }
    }

    /// Number of logical matrix rows.
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Number of logical matrix columns.
    pub const fn cols(self) -> usize {
        self.cols
    }

    /// Number of logical complex values.
    pub fn elements(self) -> usize {
        self.rows
            .checked_mul(self.cols)
            .expect("eBLAS FFT2 element count overflow")
    }

    /// Number of radix-2 stages across rows.
    pub fn row_stages(self) -> u32 {
        self.cols.trailing_zeros()
    }

    /// Number of radix-2 stages down columns.
    pub fn column_stages(self) -> u32 {
        self.rows.trailing_zeros()
    }

    /// Total number of separable radix-2 stages.
    pub fn stages(self) -> u32 {
        self.row_stages()
            .checked_add(self.column_stages())
            .expect("eBLAS FFT2 stage count overflow")
    }

    /// Total number of radix-2 butterflies in the separable transform.
    pub fn butterflies(self) -> usize {
        self.elements()
            .checked_mul(self.stages() as usize)
            .expect("eBLAS FFT2 butterfly count overflow")
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

/// Public diagonal representation of one packed radix-2 FFT1 stage.
///
/// For a stage of span `m = 2h`, the packed stage is represented as
///
/// ```text
/// y = D0 .* x + D1 .* rotl(x, h) + D2 .* rotr(x, h)
/// ```
///
/// where all three diagonals are public. This representation preserves the
/// canonical contiguous logical-slot order while masks prevent cyclic
/// rotations from mixing adjacent butterfly blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedFft1StageDiagonals {
    span: usize,
    half: usize,
    direct: Vec<Complex64>,
    rotate_left: Vec<Complex64>,
    rotate_right: Vec<Complex64>,
}

impl PackedFft1StageDiagonals {
    /// Builds the three public diagonals for one radix-2 stage.
    pub fn new(length: usize, span: usize, direction: FftDirection) -> Self {
        assert!(length > 0, "packed FFT1 stage length must be positive");
        assert!(
            length.is_power_of_two(),
            "packed FFT1 stage length must be a power of two"
        );
        assert!(
            span >= 2 && span.is_power_of_two(),
            "packed FFT1 stage span must be a power of two of at least two"
        );
        assert!(
            span <= length,
            "packed FFT1 stage span must not exceed transform length"
        );
        assert_eq!(
            length % span,
            0,
            "packed FFT1 stage span must divide transform length"
        );

        let half = span / 2;
        let sign = match direction {
            FftDirection::Forward => -1.0,
            FftDirection::Inverse => 1.0,
        };

        let mut direct = Vec::with_capacity(length);
        let mut rotate_left = Vec::with_capacity(length);
        let mut rotate_right = Vec::with_capacity(length);

        for index in 0..length {
            let offset = index % span;

            if offset < half {
                let angle = sign * 2.0 * PI * offset as f64 / span as f64;
                let twiddle = Complex64::new(angle.cos(), angle.sin());

                direct.push(Complex64::new(1.0, 0.0));
                rotate_left.push(twiddle);
                rotate_right.push(Complex64::new(0.0, 0.0));
            } else {
                let exponent = offset - half;
                let angle = sign * 2.0 * PI * exponent as f64 / span as f64;
                let twiddle = Complex64::new(angle.cos(), angle.sin());

                direct.push(-twiddle);
                rotate_left.push(Complex64::new(0.0, 0.0));
                rotate_right.push(Complex64::new(1.0, 0.0));
            }
        }

        Self {
            span,
            half,
            direct,
            rotate_left,
            rotate_right,
        }
    }

    pub const fn span(&self) -> usize {
        self.span
    }

    pub const fn rotation(&self) -> usize {
        self.half
    }

    pub fn direct(&self) -> &[Complex64] {
        &self.direct
    }

    pub fn rotate_left(&self) -> &[Complex64] {
        &self.rotate_left
    }

    pub fn rotate_right(&self) -> &[Complex64] {
        &self.rotate_right
    }
}

/// Executes one packed radix-2 stage in cleartext using its diagonal/rotation
/// representation.
///
/// Rotations are cyclic over the complete logical vector. The public
/// diagonals mask cross-block wrap-around so the result is exactly the same
/// stage as the canonical radix-2 butterfly formulation.
pub fn execute_packed_fft1_stage_pp(
    input: &[Complex64],
    diagonals: &PackedFft1StageDiagonals,
) -> Vec<Complex64> {
    let n = input.len();

    assert_eq!(
        diagonals.direct.len(),
        n,
        "packed FFT1 direct diagonal length must match input"
    );
    assert_eq!(
        diagonals.rotate_left.len(),
        n,
        "packed FFT1 left diagonal length must match input"
    );
    assert_eq!(
        diagonals.rotate_right.len(),
        n,
        "packed FFT1 right diagonal length must match input"
    );

    let rotation = diagonals.rotation();

    (0..n)
        .map(|index| {
            let left = input[(index + rotation) % n];
            let right = input[(index + n - rotation) % n];

            diagonals.direct[index] * input[index]
                + diagonals.rotate_left[index] * left
                + diagonals.rotate_right[index] * right
        })
        .collect()
}

/// Executes one packed radix-2 FFT1 stage on encrypted CKKS slots.
///
/// Logical FFT values occupy the first `diagonals.direct().len()` canonical
/// CKKS slots. Remaining physical slots are inactive and are masked to zero.
///
/// The stage realizes
///
/// ```text
/// y = D0 .* x + D1 .* rotl(x, h) + D2 .* rotr(x, h)
/// ```
///
/// using two Galois rotations and three independent ciphertext-plaintext
/// products. All three products begin at the same CKKS level and therefore
/// consume one level in parallel. The final additions consume no further
/// level.
///
/// This operation consumes exactly one CKKS level.
pub fn execute_packed_fft1_stage_cp(
    input: &RnsCkksCiphertext,
    diagonals: &PackedFft1StageDiagonals,
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertext {
    input.assert_matches_chain(chain);

    let logical_length = diagonals.direct().len();
    let slot_count = embedding.slot_count();

    assert!(
        logical_length <= slot_count,
        "packed FFT1 logical length must not exceed CKKS slot count"
    );
    assert_eq!(
        embedding.degree(),
        input.rlwe().degree(),
        "packed FFT1 embedding degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.degree(),
        input.rlwe().degree(),
        "packed FFT1 NTT plan degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.moduli(),
        input.basis().moduli(),
        "packed FFT1 NTT plan basis must match ciphertext basis"
    );

    let rotation = diagonals.rotation();

    assert!(
        rotation < logical_length,
        "packed FFT1 stage rotation must be smaller than logical length"
    );

    let mut direct = vec![Complex64::new(0.0, 0.0); slot_count];
    let mut left = vec![Complex64::new(0.0, 0.0); slot_count];
    let mut right = vec![Complex64::new(0.0, 0.0); slot_count];

    direct[..logical_length].copy_from_slice(diagonals.direct());
    left[..logical_length].copy_from_slice(diagonals.rotate_left());
    right[..logical_length].copy_from_slice(diagonals.rotate_right());

    let rotated_left = evaluator.rotate_left(input, rotation);
    let rotated_right = evaluator.rotate_right(input, rotation);

    let direct_term = multiply_complex_slots_cp(input, &direct, embedding, chain, plan);
    let left_term = multiply_complex_slots_cp(&rotated_left, &left, embedding, chain, plan);
    let right_term = multiply_complex_slots_cp(&rotated_right, &right, embedding, chain, plan);

    let partial = evaluator.add(&direct_term, &left_term);
    evaluator.add(&partial, &right_term)
}

/// Public diagonal representation of one packed radix-2
/// decimation-in-frequency FFT1 stage.
///
/// For a stage of span `m = 2h`, DIF computes
///
/// ```text
/// upper = a + b
/// lower = (a - b) * w
/// ```
///
/// over every butterfly pair separated by `h`. Natural-order input followed
/// by spans `N, N/2, ..., 2` produces the Fourier coefficients in bit-reversed
/// slot order. This is useful for encrypted execution because the permutation
/// can remain a representation mapping rather than an encrypted operation.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedFft1DifStageDiagonals {
    span: usize,
    half: usize,
    direct: Vec<Complex64>,
    rotate_left: Vec<Complex64>,
    rotate_right: Vec<Complex64>,
}

impl PackedFft1DifStageDiagonals {
    pub fn new(length: usize, span: usize, direction: FftDirection) -> Self {
        assert!(length > 0, "packed DIF FFT1 stage length must be positive");
        assert!(
            length.is_power_of_two(),
            "packed DIF FFT1 stage length must be a power of two"
        );
        assert!(
            span >= 2 && span.is_power_of_two(),
            "packed DIF FFT1 stage span must be a power of two of at least two"
        );
        assert!(
            span <= length,
            "packed DIF FFT1 stage span must not exceed transform length"
        );
        assert_eq!(
            length % span,
            0,
            "packed DIF FFT1 stage span must divide transform length"
        );

        let half = span / 2;
        let sign = match direction {
            FftDirection::Forward => -1.0,
            FftDirection::Inverse => 1.0,
        };

        let mut direct = Vec::with_capacity(length);
        let mut rotate_left = Vec::with_capacity(length);
        let mut rotate_right = Vec::with_capacity(length);

        for index in 0..length {
            let offset = index % span;

            if offset < half {
                direct.push(Complex64::new(1.0, 0.0));
                rotate_left.push(Complex64::new(1.0, 0.0));
                rotate_right.push(Complex64::new(0.0, 0.0));
            } else {
                let exponent = offset - half;
                let angle = sign * 2.0 * PI * exponent as f64 / span as f64;
                let twiddle = Complex64::new(angle.cos(), angle.sin());

                direct.push(-twiddle);
                rotate_left.push(Complex64::new(0.0, 0.0));
                rotate_right.push(twiddle);
            }
        }

        Self {
            span,
            half,
            direct,
            rotate_left,
            rotate_right,
        }
    }

    pub const fn span(&self) -> usize {
        self.span
    }

    pub const fn rotation(&self) -> usize {
        self.half
    }

    pub fn direct(&self) -> &[Complex64] {
        &self.direct
    }

    pub fn rotate_left(&self) -> &[Complex64] {
        &self.rotate_left
    }

    pub fn rotate_right(&self) -> &[Complex64] {
        &self.rotate_right
    }
}

/// Executes one packed DIF FFT1 stage in cleartext.
///
/// The same cyclic-rotation convention used by the encrypted packed DIT stage
/// is retained here; public diagonals mask all cross-block contamination.
pub fn execute_packed_fft1_dif_stage_pp(
    input: &[Complex64],
    diagonals: &PackedFft1DifStageDiagonals,
) -> Vec<Complex64> {
    let n = input.len();

    assert_eq!(
        diagonals.direct.len(),
        n,
        "packed DIF FFT1 direct diagonal length must match input"
    );
    assert_eq!(
        diagonals.rotate_left.len(),
        n,
        "packed DIF FFT1 left diagonal length must match input"
    );
    assert_eq!(
        diagonals.rotate_right.len(),
        n,
        "packed DIF FFT1 right diagonal length must match input"
    );

    let rotation = diagonals.rotation();

    (0..n)
        .map(|index| {
            let left = input[(index + rotation) % n];
            let right = input[(index + n - rotation) % n];

            diagonals.direct[index] * input[index]
                + diagonals.rotate_left[index] * left
                + diagonals.rotate_right[index] * right
        })
        .collect()
}

/// Executes a complete packed radix-2 DIF FFT1 in cleartext.
///
/// Input is in natural logical order. The returned vector intentionally remains
/// in bit-reversed physical slot order. For inverse transforms, normalization
/// is applied without changing that representation.
pub fn execute_packed_fft1_dif_pp(
    shape: Fft1Shape,
    direction: FftDirection,
    input: &[Complex64],
) -> Vec<Complex64> {
    assert_eq!(
        input.len(),
        shape.length(),
        "packed DIF FFT1 input length must match transform shape"
    );

    let n = shape.length();
    let mut values = input.to_vec();
    let mut span = n;

    while span >= 2 {
        let diagonals = PackedFft1DifStageDiagonals::new(n, span, direction);
        values = execute_packed_fft1_dif_stage_pp(&values, &diagonals);
        span /= 2;
    }

    if direction == FftDirection::Inverse {
        let scale = n as f64;

        for value in &mut values {
            *value /= scale;
        }
    }

    values
}

/// Executes one packed radix-2 DIF FFT1 stage on encrypted CKKS slots.
///
/// Logical FFT values occupy the first `diagonals.direct().len()` canonical
/// CKKS slots. Remaining physical slots are inactive and are masked to zero.
///
/// The stage realizes
///
/// ```text
/// y = D0 .* x + D1 .* rotl(x, h) + D2 .* rotr(x, h)
/// ```
///
/// for the DIF diagonals implementing
///
/// ```text
/// upper = a + b
/// lower = (a - b) * w
/// ```
///
/// The two rotations preserve CKKS level. The three public slot-vector
/// products execute independently from the same input level, so the complete
/// stage consumes exactly one CKKS level.
pub fn execute_packed_fft1_dif_stage_cp(
    input: &RnsCkksCiphertext,
    diagonals: &PackedFft1DifStageDiagonals,
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertext {
    input.assert_matches_chain(chain);

    let logical_length = diagonals.direct().len();
    let slot_count = embedding.slot_count();

    assert!(
        logical_length <= slot_count,
        "packed DIF FFT1 logical length must not exceed CKKS slot count"
    );
    assert_eq!(
        embedding.degree(),
        input.rlwe().degree(),
        "packed DIF FFT1 embedding degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.degree(),
        input.rlwe().degree(),
        "packed DIF FFT1 NTT plan degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.moduli(),
        input.basis().moduli(),
        "packed DIF FFT1 NTT plan basis must match ciphertext basis"
    );

    let rotation = diagonals.rotation();

    assert!(
        rotation < logical_length,
        "packed DIF FFT1 stage rotation must be smaller than logical length"
    );

    let mut direct = vec![Complex64::new(0.0, 0.0); slot_count];
    let mut left = vec![Complex64::new(0.0, 0.0); slot_count];
    let mut right = vec![Complex64::new(0.0, 0.0); slot_count];

    direct[..logical_length].copy_from_slice(diagonals.direct());
    left[..logical_length].copy_from_slice(diagonals.rotate_left());
    right[..logical_length].copy_from_slice(diagonals.rotate_right());

    let rotated_left = evaluator.rotate_left(input, rotation);
    let rotated_right = evaluator.rotate_right(input, rotation);

    let direct_term = multiply_complex_slots_cp(input, &direct, embedding, chain, plan);
    let left_term = multiply_complex_slots_cp(&rotated_left, &left, embedding, chain, plan);
    let right_term = multiply_complex_slots_cp(&rotated_right, &right, embedding, chain, plan);

    let partial = evaluator.add(&direct_term, &left_term);
    evaluator.add(&partial, &right_term)
}

/// Executes a complete packed radix-2 DIF FFT1 over encrypted CKKS slots.
///
/// The first `shape.length()` canonical CKKS slots contain the logical input
/// in natural order. DIF stages execute with spans `N, N/2, ..., 2`, so the
/// returned ciphertext intentionally stores Fourier coefficients in
/// bit-reversed physical slot order. No encrypted bit-reversal permutation is
/// performed.
///
/// Each generic DIF stage consumes one CKKS level. Forward execution therefore
/// consumes `log2(N)` levels. Inverse execution additionally multiplies all
/// active slots by public `1/N`, consuming one further level for normalization.
///
/// The NTT plan is rebuilt at every stage from the ciphertext's active RNS
/// basis. The evaluator selects level-specific Galois keys automatically from
/// the ciphertext state.
pub fn execute_packed_fft1_dif_cp(
    shape: Fft1Shape,
    direction: FftDirection,
    input: &RnsCkksCiphertext,
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
) -> RnsCkksCiphertext {
    input.assert_matches_chain(chain);

    let logical_length = shape.length();

    assert!(
        logical_length <= embedding.slot_count(),
        "packed encrypted FFT1 length must not exceed CKKS slot count"
    );
    assert_eq!(
        embedding.degree(),
        input.rlwe().degree(),
        "packed encrypted FFT1 embedding degree must match ciphertext ring degree"
    );

    let mut value = input.clone();
    let mut span = logical_length;

    while span >= 2 {
        let diagonals = PackedFft1DifStageDiagonals::new(logical_length, span, direction);
        let plan = RnsNttPlan::new(value.basis().moduli().to_vec(), value.rlwe().degree());

        value = execute_packed_fft1_dif_stage_cp(
            &value, &diagonals, evaluator, embedding, chain, &plan,
        );

        span /= 2;
    }

    if direction == FftDirection::Inverse && logical_length > 1 {
        let plan = RnsNttPlan::new(value.basis().moduli().to_vec(), value.rlwe().degree());
        let mut normalization = vec![Complex64::new(0.0, 0.0); embedding.slot_count()];
        let factor = 1.0 / logical_length as f64;

        normalization[..logical_length].fill(Complex64::new(factor, 0.0));

        value = multiply_complex_slots_cp(&value, &normalization, embedding, chain, &plan);
    }

    value
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

/// Executes one packed radix-2 butterfly across two encrypted CKKS
/// ciphertexts using a public complex twiddle vector.
///
/// For encrypted packed operands `a` and `b`, this computes
///
/// ```text
/// t = twiddles .* b
/// u = a + t
/// v = a - t
/// ```
///
/// `multiply_complex_slots_cp` performs the public slot-vector
/// multiplication and consumes one CKKS level. `a` is modulus-switched
/// to the same next level without changing its scale, after which
/// ciphertext addition and subtraction preserve the aligned CKKS state.
///
/// This primitive is intended for factored/distributed FFT execution in
/// which one logical FFT butterfly spans two ciphertexts. Both operands
/// remain encrypted throughout; only the FFT twiddle vector is public.
///
/// This operation consumes exactly one CKKS level.
pub fn packed_fft_butterfly_cp(
    evaluator: &RnsCkksEvaluator<'_>,
    a: &RnsCkksCiphertext,
    b: &RnsCkksCiphertext,
    twiddles: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> (RnsCkksCiphertext, RnsCkksCiphertext) {
    a.assert_matches_chain(chain);
    b.assert_matches_chain(chain);

    assert_eq!(
        a.rlwe().degree(),
        b.rlwe().degree(),
        "packed FFT butterfly operands must have matching ring degrees"
    );
    assert_eq!(
        a.level(),
        b.level(),
        "packed FFT butterfly operands must have matching levels"
    );
    assert_eq!(
        a.scale(),
        b.scale(),
        "packed FFT butterfly operands must have matching scales"
    );
    assert_eq!(
        a.basis(),
        b.basis(),
        "packed FFT butterfly operands must have matching RNS bases"
    );
    assert_eq!(
        embedding.degree(),
        a.rlwe().degree(),
        "packed FFT butterfly embedding degree must match ciphertext ring degree"
    );
    assert_eq!(
        twiddles.len(),
        embedding.slot_count(),
        "packed FFT butterfly twiddle vector must match CKKS slot count"
    );
    assert_eq!(
        plan.degree(),
        a.rlwe().degree(),
        "packed FFT butterfly NTT plan degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.moduli(),
        a.basis().moduli(),
        "packed FFT butterfly NTT plan basis must match ciphertext basis"
    );
    assert!(
        chain.has_next_level(a.level()),
        "packed FFT butterfly requires another CKKS chain level"
    );

    twiddles.iter().for_each(|twiddle| {
        assert!(
            twiddle.re.is_finite() && twiddle.im.is_finite(),
            "packed FFT butterfly twiddles must be finite"
        );
    });

    let weighted_b = multiply_complex_slots_cp(b, twiddles, embedding, chain, plan);

    let aligned_a = mod_switch_rns_ckks_to_next(a, chain);

    assert_eq!(
        aligned_a.level(),
        weighted_b.level(),
        "packed FFT butterfly aligned operands must have matching levels"
    );
    assert_eq!(
        aligned_a.scale(),
        weighted_b.scale(),
        "packed FFT butterfly aligned operands must have matching scales"
    );
    assert_eq!(
        aligned_a.basis(),
        weighted_b.basis(),
        "packed FFT butterfly aligned operands must have matching bases"
    );

    let upper = evaluator.add(&aligned_a, &weighted_b);
    let lower = evaluator.sub(&aligned_a, &weighted_b);

    (upper, lower)
}

/// Executes one packed decimation-in-frequency radix-2 butterfly across
/// two encrypted CKKS ciphertexts using a public complex twiddle vector.
///
/// For encrypted packed operands `a` and `b`, this computes
///
/// ```text
/// sum  = a + b
/// diff = a - b
/// u    = sum
/// v    = twiddles .* diff
/// ```
///
/// This matches the packed FFT2 DIF convention used by
/// `PackedFft2DifStageDiagonals`. For an upper/lower pair
/// `(a, b)`, the lower branch is `(a - b) * twiddle`.
///
/// The public twiddle multiplication consumes one CKKS level. The sum is
/// modulus-switched to the same next level without changing its scale.
///
/// Both data operands remain encrypted throughout. Only the FFT twiddle
/// vector is public.
///
/// This operation consumes exactly one CKKS level.
pub fn packed_fft_dif_butterfly_cp(
    evaluator: &RnsCkksEvaluator<'_>,
    a: &RnsCkksCiphertext,
    b: &RnsCkksCiphertext,
    twiddles: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> (RnsCkksCiphertext, RnsCkksCiphertext) {
    a.assert_matches_chain(chain);
    b.assert_matches_chain(chain);

    assert_eq!(
        a.rlwe().degree(),
        b.rlwe().degree(),
        "packed DIF butterfly operands must have matching ring degrees"
    );
    assert_eq!(
        a.level(),
        b.level(),
        "packed DIF butterfly operands must have matching levels"
    );
    assert_eq!(
        a.scale(),
        b.scale(),
        "packed DIF butterfly operands must have matching scales"
    );
    assert_eq!(
        a.basis(),
        b.basis(),
        "packed DIF butterfly operands must have matching RNS bases"
    );
    assert_eq!(
        embedding.degree(),
        a.rlwe().degree(),
        "packed DIF butterfly embedding degree must match ciphertext ring degree"
    );
    assert_eq!(
        twiddles.len(),
        embedding.slot_count(),
        "packed DIF butterfly twiddle vector must match CKKS slot count"
    );
    assert_eq!(
        plan.degree(),
        a.rlwe().degree(),
        "packed DIF butterfly NTT plan degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.moduli(),
        a.basis().moduli(),
        "packed DIF butterfly NTT plan basis must match ciphertext basis"
    );
    assert!(
        chain.has_next_level(a.level()),
        "packed DIF butterfly requires another CKKS chain level"
    );

    twiddles.iter().for_each(|twiddle| {
        assert!(
            twiddle.re.is_finite() && twiddle.im.is_finite(),
            "packed DIF butterfly twiddles must be finite"
        );
    });

    let sum = evaluator.add(a, b);
    let diff = evaluator.sub(a, b);

    /*
     * The lower weighted-difference path and upper modulus-switch path are
     * independent once sum and difference have been formed.
     */
    let (weighted_diff, aligned_sum) = std::thread::scope(|scope| {
        let weighted_worker =
            scope.spawn(|| multiply_complex_slots_cp(&diff, twiddles, embedding, chain, plan));

        let aligned_worker = scope.spawn(|| mod_switch_rns_ckks_to_next(&sum, chain));

        (
            weighted_worker
                .join()
                .expect("packed DIF weighted-difference worker panicked"),
            aligned_worker
                .join()
                .expect("packed DIF aligned-sum worker panicked"),
        )
    });

    assert_eq!(
        aligned_sum.level(),
        weighted_diff.level(),
        "packed DIF butterfly aligned outputs must have matching levels"
    );
    assert_eq!(
        aligned_sum.scale(),
        weighted_diff.scale(),
        "packed DIF butterfly aligned outputs must have matching scales"
    );
    assert_eq!(
        aligned_sum.basis(),
        weighted_diff.basis(),
        "packed DIF butterfly aligned outputs must have matching bases"
    );

    (aligned_sum, weighted_diff)
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

/// Executes an explicit FFT1 plan over encrypted data and a public transform.
///
/// The input is one ciphertext matrix per logical FFT value. All operands must
/// begin at the same CKKS level, basis, scale, shape, and ring degree.
///
/// Execution follows the same structural plan as [`execute_fft1_plan`]:
///
/// 1. bit-reversal permutation;
/// 2. staged radix-2 butterflies;
/// 3. optional inverse normalization.
///
/// The first radix-2 stage contains only unit twiddles and therefore executes
/// with ciphertext addition/subtraction only, consuming no CKKS level.
/// Subsequent stages consume one level each because their public twiddle
/// multiplications use ciphertext-plaintext multiplication followed by one
/// rescale. Forward execution therefore consumes `max(stage_count - 1, 0)`
/// levels.
///
/// Twiddle coefficients and FFT topology are public. This executor performs no
/// ciphertext-ciphertext multiplication and requires no relinearization.
pub fn execute_fft1_plan_cp(
    plan: &Fft1Plan,
    direction: FftDirection,
    input: &[RnsCkksCiphertextMatrix],
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
) -> Vec<RnsCkksCiphertextMatrix> {
    let shape = plan.shape();

    assert_eq!(
        input.len(),
        shape.length(),
        "eBLAS encrypted FFT1 input length must match transform shape"
    );

    assert!(
        !input.is_empty(),
        "eBLAS encrypted FFT1 requires at least one ciphertext operand"
    );

    let first = &input[0];

    let butterfly_levels = plan.stage_count().saturating_sub(1);
    let normalization_levels =
        usize::from(direction == FftDirection::Inverse && shape.length() > 1);
    let required_levels = butterfly_levels + normalization_levels;

    assert!(
        first.level() + required_levels <= chain.max_level(),
        "eBLAS encrypted FFT1 modulus chain is too shallow for the planned execution"
    );

    for value in &input[1..] {
        assert_eq!(
            value.rows(),
            first.rows(),
            "eBLAS encrypted FFT1 operand row counts must match"
        );
        assert_eq!(
            value.cols(),
            first.cols(),
            "eBLAS encrypted FFT1 operand column counts must match"
        );
        assert_eq!(
            value.level(),
            first.level(),
            "eBLAS encrypted FFT1 operand levels must match"
        );
        assert_eq!(
            value.scale(),
            first.scale(),
            "eBLAS encrypted FFT1 operand scales must match"
        );
        assert_eq!(
            value.get(0, 0).basis(),
            first.get(0, 0).basis(),
            "eBLAS encrypted FFT1 operand bases must match"
        );
        assert_eq!(
            value.ring_degree(),
            first.ring_degree(),
            "eBLAS encrypted FFT1 operand ring degrees must match"
        );
    }

    let mut values = input.to_vec();

    let n = values.len();
    if n > 2 {
        let bits = n.trailing_zeros();

        for index in 0..n {
            let reversed = index.reverse_bits() >> (usize::BITS - bits);

            if reversed > index {
                values.swap(index, reversed);
            }
        }
    }

    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };

    for stage in plan.stages() {
        let stage_input = values.clone();
        let mut stage_output = values.clone();

        let level_free_unit_stage = stage.span() == 2
            && stage
                .butterflies()
                .iter()
                .all(|butterfly| butterfly.twiddle_exponent() == 0);

        if level_free_unit_stage {
            for butterfly in stage.butterflies() {
                debug_assert_eq!(butterfly.stage(), stage.index());
                debug_assert_eq!(butterfly.span(), stage.span());

                let a = &stage_input[butterfly.even_index()];
                let b = &stage_input[butterfly.odd_index()];

                assert_eq!(
                    a.rows(),
                    b.rows(),
                    "eBLAS FFT1 unit butterfly operand row counts must match"
                );
                assert_eq!(
                    a.cols(),
                    b.cols(),
                    "eBLAS FFT1 unit butterfly operand column counts must match"
                );
                assert_eq!(
                    a.level(),
                    b.level(),
                    "eBLAS FFT1 unit butterfly operand levels must match"
                );
                assert_eq!(
                    a.scale(),
                    b.scale(),
                    "eBLAS FFT1 unit butterfly operand scales must match"
                );
                assert_eq!(
                    a.get(0, 0).basis(),
                    b.get(0, 0).basis(),
                    "eBLAS FFT1 unit butterfly operand bases must match"
                );

                let mut upper = Vec::with_capacity(a.len());
                let mut lower = Vec::with_capacity(a.len());

                for col in 0..a.cols() {
                    for row in 0..a.rows() {
                        upper.push(evaluator.add(a.get(row, col), b.get(row, col)));
                        lower.push(evaluator.sub(a.get(row, col), b.get(row, col)));
                    }
                }

                stage_output[butterfly.even_index()] =
                    RnsCkksCiphertextMatrix::from_vec_column_major(a.rows(), a.cols(), upper);

                stage_output[butterfly.odd_index()] =
                    RnsCkksCiphertextMatrix::from_vec_column_major(a.rows(), a.cols(), lower);
            }
        } else {
            let level = stage_input[0].level();

            assert!(
                chain.has_next_level(level),
                "eBLAS encrypted FFT1 stage requires another CKKS chain level"
            );

            let stage_ntt_plan = RnsNttPlan::new(
                chain.level(level).moduli().to_vec(),
                stage_input[0].ring_degree(),
            );

            for butterfly in stage.butterflies() {
                debug_assert_eq!(butterfly.stage(), stage.index());
                debug_assert_eq!(butterfly.span(), stage.span());

                let angle =
                    sign * 2.0 * PI * butterfly.twiddle_exponent() as f64 / butterfly.span() as f64;

                let twiddle = Complex64::new(angle.cos(), angle.sin());

                let (upper, lower) = fft1_butterfly_cp(
                    evaluator,
                    &stage_input[butterfly.even_index()],
                    &stage_input[butterfly.odd_index()],
                    twiddle,
                    embedding,
                    chain,
                    &stage_ntt_plan,
                );

                stage_output[butterfly.even_index()] = upper;
                stage_output[butterfly.odd_index()] = lower;
            }
        }

        values = stage_output;
    }

    if direction == FftDirection::Inverse && shape.length() > 1 {
        let level = values[0].level();

        assert!(
            chain.has_next_level(level),
            "eBLAS encrypted inverse FFT1 normalization requires another CKKS chain level"
        );

        let normalization_plan = RnsNttPlan::new(
            chain.level(level).moduli().to_vec(),
            values[0].ring_degree(),
        );

        let normalization = Complex64::new(1.0 / shape.length() as f64, 0.0);

        values = values
            .iter()
            .map(|value| {
                scale_complex_cp(value, normalization, embedding, chain, &normalization_plan)
            })
            .collect();
    }

    values
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

/// Dense O(R^2 C^2) two-dimensional DFT reference oracle.
///
/// Input and output use row-major layout. The inverse transform is normalized
/// by `1 / (rows * cols)`.
pub fn dft2_reference(
    shape: Fft2Shape,
    direction: FftDirection,
    input: &[Complex64],
) -> Vec<Complex64> {
    assert_eq!(
        input.len(),
        shape.elements(),
        "eBLAS FFT2 input length must match shape"
    );

    let rows = shape.rows();
    let cols = shape.cols();
    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };
    let normalization = match direction {
        FftDirection::Forward => 1.0,
        FftDirection::Inverse => 1.0 / shape.elements() as f64,
    };

    let mut output = vec![Complex64::new(0.0, 0.0); shape.elements()];

    for kr in 0..rows {
        for kc in 0..cols {
            let mut sum = Complex64::new(0.0, 0.0);

            for nr in 0..rows {
                for nc in 0..cols {
                    let phase = (kr * nr) as f64 / rows as f64 + (kc * nc) as f64 / cols as f64;
                    let angle = sign * 2.0 * PI * phase;
                    let twiddle = Complex64::new(angle.cos(), angle.sin());

                    sum += input[nr * cols + nc] * twiddle;
                }
            }

            output[kr * cols + kc] = sum * normalization;
        }
    }

    output
}

/// Application-facing separable two-dimensional radix-2 FFT.
///
/// The logical matrix is row-major. The implementation first transforms every
/// row and then every column. The result remains in canonical logical
/// row-major order.
pub fn fft2_pp(shape: Fft2Shape, direction: FftDirection, input: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(
        input.len(),
        shape.elements(),
        "eBLAS FFT2 input length must match shape"
    );

    let rows = shape.rows();
    let cols = shape.cols();
    let row_shape = Fft1Shape::new(cols);
    let column_shape = Fft1Shape::new(rows);

    let mut values = input.to_vec();

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let transformed = fft1_pp(row_shape, direction, &values[start..end]);
        values[start..end].copy_from_slice(&transformed);
    }

    let mut column = vec![Complex64::new(0.0, 0.0); rows];

    for col in 0..cols {
        for row in 0..rows {
            column[row] = values[row * cols + col];
        }

        let transformed = fft1_pp(column_shape, direction, &column);

        for row in 0..rows {
            values[row * cols + col] = transformed[row];
        }
    }

    values
}

/// Axis transformed by a packed two-dimensional FFT stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedFft2Axis {
    /// Transform independently across each row.
    Rows,
    /// Transform independently down each column.
    Columns,
}

/// Public diagonal representation of one packed radix-2 DIF FFT2 stage.
///
/// The logical matrix occupies contiguous row-major slots. Row stages use
/// rotations by `half`; column stages use rotations by `half * cols`.
///
/// The three public diagonals realize
///
/// ```text
/// upper: y = x + partner
/// lower: y = (partner - x) * w
/// ```
///
/// where `partner` is selected by the axis-specific cyclic rotation. Public
/// masks prevent cyclic rotations from coupling independent rows, columns, or
/// butterfly blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedFft2DifStageDiagonals {
    shape: Fft2Shape,
    axis: PackedFft2Axis,
    span: usize,
    half: usize,
    rotation: usize,
    direct: Vec<Complex64>,
    rotate_left: Vec<Complex64>,
    rotate_right: Vec<Complex64>,
}

impl PackedFft2DifStageDiagonals {
    /// Constructs one packed FFT2 DIF stage.
    pub fn new(
        shape: Fft2Shape,
        axis: PackedFft2Axis,
        span: usize,
        direction: FftDirection,
    ) -> Self {
        let axis_length = match axis {
            PackedFft2Axis::Rows => shape.cols(),
            PackedFft2Axis::Columns => shape.rows(),
        };

        assert!(
            span >= 2 && span.is_power_of_two(),
            "packed FFT2 DIF span must be a power of two of at least two"
        );
        assert!(
            span <= axis_length,
            "packed FFT2 DIF span must not exceed the transformed axis"
        );
        assert_eq!(
            axis_length % span,
            0,
            "packed FFT2 DIF span must divide the transformed axis"
        );

        let half = span / 2;
        let rotation = match axis {
            PackedFft2Axis::Rows => half,
            PackedFft2Axis::Columns => half
                .checked_mul(shape.cols())
                .expect("packed FFT2 column rotation overflow"),
        };

        let mut direct = vec![Complex64::new(0.0, 0.0); shape.elements()];
        let mut rotate_left = vec![Complex64::new(0.0, 0.0); shape.elements()];
        let mut rotate_right = vec![Complex64::new(0.0, 0.0); shape.elements()];

        let sign = match direction {
            FftDirection::Forward => -1.0,
            FftDirection::Inverse => 1.0,
        };

        for row in 0..shape.rows() {
            for col in 0..shape.cols() {
                let index = row * shape.cols() + col;
                let axis_index = match axis {
                    PackedFft2Axis::Rows => col,
                    PackedFft2Axis::Columns => row,
                };
                let offset = axis_index % span;
                let twiddle_index = offset % half;
                let angle = sign * 2.0 * PI * twiddle_index as f64 / span as f64;
                let twiddle = Complex64::new(angle.cos(), angle.sin());

                if offset < half {
                    direct[index] = Complex64::new(1.0, 0.0);
                    rotate_left[index] = Complex64::new(1.0, 0.0);
                } else {
                    direct[index] = -twiddle;
                    rotate_right[index] = twiddle;
                }
            }
        }

        Self {
            shape,
            axis,
            span,
            half,
            rotation,
            direct,
            rotate_left,
            rotate_right,
        }
    }

    pub const fn shape(&self) -> Fft2Shape {
        self.shape
    }

    pub const fn axis(&self) -> PackedFft2Axis {
        self.axis
    }

    pub const fn span(&self) -> usize {
        self.span
    }

    pub const fn half(&self) -> usize {
        self.half
    }

    pub const fn rotation(&self) -> usize {
        self.rotation
    }

    pub fn direct(&self) -> &[Complex64] {
        &self.direct
    }

    pub fn rotate_left(&self) -> &[Complex64] {
        &self.rotate_left
    }

    pub fn rotate_right(&self) -> &[Complex64] {
        &self.rotate_right
    }
}

/// Public diagonal representation of one DIF stage across fixed-size
/// contiguous tile blocks packed inside one CKKS slot vector.
///
/// Each logical tile occupies `tile_elements` consecutive slots. A stage with
/// `span_tiles` pairs tile blocks separated by `span_tiles / 2` tiles, so the
/// corresponding CKKS rotation is:
///
/// `rotation = (span_tiles / 2) * tile_elements`.
///
/// This is the tile-block analogue of [`PackedFft2DifStageDiagonals`].
#[derive(Debug, Clone, PartialEq)]
pub struct PackedTileDifStageDiagonals {
    tile_elements: usize,
    tile_rows: usize,
    tile_cols: usize,
    packed_tiles: usize,
    span_tiles: usize,
    half_tiles: usize,
    rotation: usize,
    direct: Vec<Complex64>,
    rotate_left: Vec<Complex64>,
    rotate_right: Vec<Complex64>,
}

impl PackedTileDifStageDiagonals {
    /// Constructs one packed global-column DIF stage over tile blocks.
    ///
    /// The packed representation contains `packed_tiles` vertically adjacent
    /// logical tiles. Twiddles reproduce the factored global-column FFT2
    /// schedule:
    ///
    /// `twiddle_index = tile_offset * tile_rows + local_row`.
    pub fn new_column(
        tile_rows: usize,
        tile_cols: usize,
        packed_tiles: usize,
        span_tiles: usize,
        direction: FftDirection,
    ) -> Self {
        assert!(tile_rows > 0 && tile_rows.is_power_of_two());
        assert!(tile_cols > 0 && tile_cols.is_power_of_two());
        assert!(packed_tiles > 0 && packed_tiles.is_power_of_two());
        assert!(
            span_tiles >= 2 && span_tiles.is_power_of_two(),
            "packed tile DIF span must be a power of two of at least two"
        );
        assert!(
            span_tiles <= packed_tiles,
            "packed tile DIF span must not exceed packed tile count"
        );
        assert_eq!(
            packed_tiles % span_tiles,
            0,
            "packed tile DIF span must divide packed tile count"
        );

        let tile_elements = tile_rows
            .checked_mul(tile_cols)
            .expect("packed tile DIF tile element count overflow");

        let half_tiles = span_tiles / 2;

        let rotation = half_tiles
            .checked_mul(tile_elements)
            .expect("packed tile DIF rotation overflow");

        let slot_count = packed_tiles
            .checked_mul(tile_elements)
            .expect("packed tile DIF slot count overflow");

        let mut direct = vec![Complex64::new(0.0, 0.0); slot_count];
        let mut rotate_left = vec![Complex64::new(0.0, 0.0); slot_count];
        let mut rotate_right = vec![Complex64::new(0.0, 0.0); slot_count];

        let sign = match direction {
            FftDirection::Forward => -1.0,
            FftDirection::Inverse => 1.0,
        };

        let global_span = span_tiles
            .checked_mul(tile_rows)
            .expect("packed tile DIF global span overflow");

        for lane in 0..packed_tiles {
            let offset = lane % span_tiles;
            let tile_offset = offset % half_tiles;

            for local_row in 0..tile_rows {
                let twiddle_index = tile_offset * tile_rows + local_row;
                let angle = sign * 2.0 * PI * twiddle_index as f64 / global_span as f64;
                let twiddle = Complex64::new(angle.cos(), angle.sin());

                for local_col in 0..tile_cols {
                    let tile_slot = local_row * tile_cols + local_col;
                    let index = lane * tile_elements + tile_slot;

                    if offset < half_tiles {
                        direct[index] = Complex64::new(1.0, 0.0);
                        rotate_left[index] = Complex64::new(1.0, 0.0);
                    } else {
                        direct[index] = -twiddle;
                        rotate_right[index] = twiddle;
                    }
                }
            }
        }

        Self {
            tile_elements,
            tile_rows,
            tile_cols,
            packed_tiles,
            span_tiles,
            half_tiles,
            rotation,
            direct,
            rotate_left,
            rotate_right,
        }
    }

    pub const fn tile_elements(&self) -> usize {
        self.tile_elements
    }

    pub const fn tile_rows(&self) -> usize {
        self.tile_rows
    }

    pub const fn tile_cols(&self) -> usize {
        self.tile_cols
    }

    pub const fn packed_tiles(&self) -> usize {
        self.packed_tiles
    }

    pub const fn span_tiles(&self) -> usize {
        self.span_tiles
    }

    pub const fn half_tiles(&self) -> usize {
        self.half_tiles
    }

    pub const fn rotation(&self) -> usize {
        self.rotation
    }

    pub fn direct(&self) -> &[Complex64] {
        &self.direct
    }

    pub fn rotate_left(&self) -> &[Complex64] {
        &self.rotate_left
    }

    pub fn rotate_right(&self) -> &[Complex64] {
        &self.rotate_right
    }
}

/// Executes one clear packed tile-block DIF stage using exactly the
/// diagonal-times-rotation algebra required by the encrypted realization.
pub fn execute_packed_tile_dif_stage_pp(
    input: &[Complex64],
    diagonals: &PackedTileDifStageDiagonals,
) -> Vec<Complex64> {
    let expected_len = diagonals
        .packed_tiles()
        .checked_mul(diagonals.tile_elements())
        .expect("packed tile DIF input length overflow");

    assert_eq!(
        input.len(),
        expected_len,
        "packed tile DIF input length must match packed tile layout"
    );

    let left = rotate_left_complex(input, diagonals.rotation());
    let right = rotate_right_complex(input, diagonals.rotation());

    input
        .iter()
        .zip(left.iter())
        .zip(right.iter())
        .enumerate()
        .map(|(index, ((current, left), right))| {
            diagonals.direct()[index] * *current
                + diagonals.rotate_left()[index] * *left
                + diagonals.rotate_right()[index] * *right
        })
        .collect()
}

fn rotate_left_complex(values: &[Complex64], amount: usize) -> Vec<Complex64> {
    if values.is_empty() {
        return Vec::new();
    }

    let amount = amount % values.len();
    (0..values.len())
        .map(|index| values[(index + amount) % values.len()])
        .collect()
}

fn rotate_right_complex(values: &[Complex64], amount: usize) -> Vec<Complex64> {
    if values.is_empty() {
        return Vec::new();
    }

    let amount = amount % values.len();
    (0..values.len())
        .map(|index| values[(index + values.len() - amount) % values.len()])
        .collect()
}

fn expand_repeated_packed_fft2_dif_diagonals(
    diagonals: &PackedFft2DifStageDiagonals,
    tiles_per_vector: usize,
    vector_length: usize,
) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
    assert!(
        tiles_per_vector > 0,
        "repeated packed FFT2 requires at least one logical tile"
    );

    let tile_elements = diagonals.shape().elements();
    let active_elements = tile_elements
        .checked_mul(tiles_per_vector)
        .expect("repeated packed FFT2 active-element count overflow");

    assert!(
        active_elements <= vector_length,
        "repeated packed FFT2 requires {active_elements} slots but vector provides only {vector_length}"
    );

    let mut direct = vec![Complex64::new(0.0, 0.0); vector_length];
    let mut rotate_left = vec![Complex64::new(0.0, 0.0); vector_length];
    let mut rotate_right = vec![Complex64::new(0.0, 0.0); vector_length];

    for packed_tile_index in 0..tiles_per_vector {
        let start = packed_tile_index
            .checked_mul(tile_elements)
            .expect("repeated packed FFT2 tile offset overflow");
        let end = start + tile_elements;

        direct[start..end].copy_from_slice(diagonals.direct());
        rotate_left[start..end].copy_from_slice(diagonals.rotate_left());
        rotate_right[start..end].copy_from_slice(diagonals.rotate_right());
    }

    (direct, rotate_left, rotate_right)
}

/// Executes one FFT2 DIF stage simultaneously across multiple contiguous
/// logical tiles packed into one slot vector.
///
/// The CKKS-style rotations act on the complete slot vector. Repeated public
/// diagonals mask those global rotations so each logical FFT tile remains
/// isolated from neighboring packed tiles.
pub fn execute_repeated_packed_fft2_dif_stage_pp(
    input: &[Complex64],
    diagonals: &PackedFft2DifStageDiagonals,
    tiles_per_vector: usize,
) -> Vec<Complex64> {
    let (direct, left_diagonal, right_diagonal) =
        expand_repeated_packed_fft2_dif_diagonals(diagonals, tiles_per_vector, input.len());

    let left = rotate_left_complex(input, diagonals.rotation());
    let right = rotate_right_complex(input, diagonals.rotation());

    input
        .iter()
        .zip(left.iter())
        .zip(right.iter())
        .enumerate()
        .map(|(index, ((current, left), right))| {
            direct[index] * *current + left_diagonal[index] * *left + right_diagonal[index] * *right
        })
        .collect()
}

/// Executes one packed tile-block DIF stage over encrypted CKKS slots using
/// prepared Galois keys.
///
/// This is the encrypted counterpart of [`execute_packed_tile_dif_stage_pp`].
/// One stage computes
///
/// ```text
/// y = D0 .* x + DL .* rotl(x, r) + DR .* rotr(x, r)
/// ```
///
/// where the public diagonals pair complete contiguous tile blocks. The two
/// rotations preserve level and scale. The three public slot-vector products
/// consume one CKKS level and are evaluated independently before addition.
pub fn execute_packed_tile_dif_stage_cp_prepared(
    input: &RnsCkksCiphertext,
    diagonals: &PackedTileDifStageDiagonals,
    evaluator: &RnsCkksEvaluator<'_>,
    prepared_galois_keys: (&PreparedRnsGaloisKey, &PreparedRnsGaloisKey),
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertext {
    let (left_galois_key, right_galois_key) = prepared_galois_keys;

    input.assert_matches_chain(chain);

    let slot_count = embedding.slot_count();
    let logical_length = diagonals
        .packed_tiles()
        .checked_mul(diagonals.tile_elements())
        .expect("packed tile DIF logical length overflow");

    assert_eq!(
        logical_length, slot_count,
        "packed tile DIF layout must occupy the complete CKKS slot vector"
    );
    assert_eq!(
        embedding.degree(),
        input.rlwe().degree(),
        "packed tile DIF embedding degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.degree(),
        input.rlwe().degree(),
        "packed tile DIF NTT plan degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.moduli(),
        input.basis().moduli(),
        "packed tile DIF NTT plan basis must match ciphertext basis"
    );

    let rotation = diagonals.rotation();

    assert!(
        rotation > 0 && rotation < slot_count,
        "packed tile DIF rotation must lie inside the CKKS slot vector"
    );

    let (rotated_left, rotated_right) = std::thread::scope(|scope| {
        let left_worker = scope.spawn(|| {
            rotate_left_rns_ckks_with_prepared_ntt(input, rotation, left_galois_key, chain, plan)
        });

        let right_worker = scope.spawn(|| {
            rotate_right_rns_ckks_with_prepared_ntt(input, rotation, right_galois_key, chain, plan)
        });

        (
            left_worker
                .join()
                .expect("packed tile DIF prepared left rotation worker panicked"),
            right_worker
                .join()
                .expect("packed tile DIF prepared right rotation worker panicked"),
        )
    });

    let (direct_term, left_term, right_term) = std::thread::scope(|scope| {
        let direct_worker = scope
            .spawn(|| multiply_complex_slots_cp(input, diagonals.direct(), embedding, chain, plan));

        let left_worker = scope.spawn(|| {
            multiply_complex_slots_cp(
                &rotated_left,
                diagonals.rotate_left(),
                embedding,
                chain,
                plan,
            )
        });

        let right_worker = scope.spawn(|| {
            multiply_complex_slots_cp(
                &rotated_right,
                diagonals.rotate_right(),
                embedding,
                chain,
                plan,
            )
        });

        (
            direct_worker
                .join()
                .expect("packed tile DIF prepared direct CP worker panicked"),
            left_worker
                .join()
                .expect("packed tile DIF prepared left CP worker panicked"),
            right_worker
                .join()
                .expect("packed tile DIF prepared right CP worker panicked"),
        )
    });

    let partial = evaluator.add(&direct_term, &left_term);
    evaluator.add(&partial, &right_term)
}

/// Executes one packed cleartext FFT2 DIF stage using the same
/// diagonal-times-rotation algebra required by the encrypted realization.
pub fn execute_packed_fft2_dif_stage_pp(
    input: &[Complex64],
    diagonals: &PackedFft2DifStageDiagonals,
) -> Vec<Complex64> {
    assert_eq!(
        input.len(),
        diagonals.shape().elements(),
        "packed FFT2 DIF input length must match shape"
    );

    let left = rotate_left_complex(input, diagonals.rotation());
    let right = rotate_right_complex(input, diagonals.rotation());

    input
        .iter()
        .zip(left.iter())
        .zip(right.iter())
        .enumerate()
        .map(|(index, ((current, left), right))| {
            diagonals.direct()[index] * *current
                + diagonals.rotate_left()[index] * *left
                + diagonals.rotate_right()[index] * *right
        })
        .collect()
}

/// Executes one packed two-dimensional DIF FFT stage on encrypted CKKS slots.
///
/// The logical matrix occupies the first `shape.elements()` canonical slots in
/// row-major order. Row stages use contiguous rotations by `half`; column
/// stages use strided rotations by `half * cols`.
///
/// The stage computes
///
/// ```text
/// y = D0 .* x + D1 .* rotl(x, r) + D2 .* rotr(x, r)
/// ```
///
/// using three independent public slot-vector multiplications from the same
/// input level. Therefore the complete stage consumes exactly one CKKS level,
/// while the two rotations preserve level and scale.
/// Executes one packed radix-2 DIF FFT2 stage using Galois keys whose
/// key-switch material has already been transformed to the NTT domain.
///
/// This is execution-equivalent to [`execute_packed_fft2_dif_stage_cp`], but
/// avoids repeatedly preparing the same level-specific Galois keys when the
/// stage is evaluated across multiple ciphertexts.
pub fn execute_repeated_packed_fft2_dif_stage_cp_prepared(
    input: &RnsCkksCiphertext,
    diagonals: &PackedFft2DifStageDiagonals,
    evaluator: &RnsCkksEvaluator<'_>,
    prepared_stage: (usize, &PreparedRnsGaloisKey, &PreparedRnsGaloisKey),
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertext {
    let (tiles_per_ciphertext, left_galois_key, right_galois_key) = prepared_stage;

    input.assert_matches_chain(chain);

    let slot_count = embedding.slot_count();

    let (direct, left, right) =
        expand_repeated_packed_fft2_dif_diagonals(diagonals, tiles_per_ciphertext, slot_count);

    assert_eq!(
        embedding.degree(),
        input.rlwe().degree(),
        "repeated packed FFT2 embedding degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.degree(),
        input.rlwe().degree(),
        "repeated packed FFT2 NTT plan degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.moduli(),
        input.basis().moduli(),
        "repeated packed FFT2 NTT plan basis must match ciphertext basis"
    );

    let rotation = diagonals.rotation();

    let (rotated_left, rotated_right) = std::thread::scope(|scope| {
        let left_worker = scope.spawn(|| {
            rotate_left_rns_ckks_with_prepared_ntt(input, rotation, left_galois_key, chain, plan)
        });

        let right_worker = scope.spawn(|| {
            rotate_right_rns_ckks_with_prepared_ntt(input, rotation, right_galois_key, chain, plan)
        });

        (
            left_worker
                .join()
                .expect("repeated packed FFT2 left rotation worker panicked"),
            right_worker
                .join()
                .expect("repeated packed FFT2 right rotation worker panicked"),
        )
    });

    let (direct_term, left_term, right_term) = std::thread::scope(|scope| {
        let direct_worker =
            scope.spawn(|| multiply_complex_slots_cp(input, &direct, embedding, chain, plan));

        let left_worker =
            scope.spawn(|| multiply_complex_slots_cp(&rotated_left, &left, embedding, chain, plan));

        let right_worker = scope
            .spawn(|| multiply_complex_slots_cp(&rotated_right, &right, embedding, chain, plan));

        (
            direct_worker
                .join()
                .expect("repeated packed FFT2 direct CP worker panicked"),
            left_worker
                .join()
                .expect("repeated packed FFT2 left CP worker panicked"),
            right_worker
                .join()
                .expect("repeated packed FFT2 right CP worker panicked"),
        )
    });

    let partial = evaluator.add(&direct_term, &left_term);
    evaluator.add(&partial, &right_term)
}

pub fn execute_packed_fft2_dif_stage_cp_prepared(
    input: &RnsCkksCiphertext,
    diagonals: &PackedFft2DifStageDiagonals,
    evaluator: &RnsCkksEvaluator<'_>,
    prepared_galois_keys: (&PreparedRnsGaloisKey, &PreparedRnsGaloisKey),
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertext {
    let (left_galois_key, right_galois_key) = prepared_galois_keys;
    input.assert_matches_chain(chain);

    let logical_length = diagonals.shape().elements();
    let slot_count = embedding.slot_count();

    assert!(
        logical_length <= slot_count,
        "packed FFT2 logical size must not exceed CKKS slot count"
    );
    assert_eq!(
        embedding.degree(),
        input.rlwe().degree(),
        "packed FFT2 embedding degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.degree(),
        input.rlwe().degree(),
        "packed FFT2 NTT plan degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.moduli(),
        input.basis().moduli(),
        "packed FFT2 NTT plan basis must match ciphertext basis"
    );

    let rotation = diagonals.rotation();

    let mut direct = vec![Complex64::new(0.0, 0.0); slot_count];
    let mut left = vec![Complex64::new(0.0, 0.0); slot_count];
    let mut right = vec![Complex64::new(0.0, 0.0); slot_count];

    direct[..logical_length].copy_from_slice(diagonals.direct());
    left[..logical_length].copy_from_slice(diagonals.rotate_left());
    right[..logical_length].copy_from_slice(diagonals.rotate_right());

    let (rotated_left, rotated_right) = std::thread::scope(|scope| {
        let left_worker = scope.spawn(|| {
            rotate_left_rns_ckks_with_prepared_ntt(input, rotation, left_galois_key, chain, plan)
        });

        let right_worker = scope.spawn(|| {
            rotate_right_rns_ckks_with_prepared_ntt(input, rotation, right_galois_key, chain, plan)
        });

        (
            left_worker
                .join()
                .expect("packed FFT2 prepared left rotation worker panicked"),
            right_worker
                .join()
                .expect("packed FFT2 prepared right rotation worker panicked"),
        )
    });

    let (direct_term, left_term, right_term) = std::thread::scope(|scope| {
        let direct_worker =
            scope.spawn(|| multiply_complex_slots_cp(input, &direct, embedding, chain, plan));

        let left_worker =
            scope.spawn(|| multiply_complex_slots_cp(&rotated_left, &left, embedding, chain, plan));

        let right_worker = scope
            .spawn(|| multiply_complex_slots_cp(&rotated_right, &right, embedding, chain, plan));

        (
            direct_worker
                .join()
                .expect("packed FFT2 prepared direct CP worker panicked"),
            left_worker
                .join()
                .expect("packed FFT2 prepared left CP worker panicked"),
            right_worker
                .join()
                .expect("packed FFT2 prepared right CP worker panicked"),
        )
    });

    let partial = evaluator.add(&direct_term, &left_term);
    evaluator.add(&partial, &right_term)
}

pub fn execute_packed_fft2_dif_stage_cp(
    input: &RnsCkksCiphertext,
    diagonals: &PackedFft2DifStageDiagonals,
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    plan: &RnsNttPlan,
) -> RnsCkksCiphertext {
    input.assert_matches_chain(chain);

    let logical_length = diagonals.shape().elements();
    let slot_count = embedding.slot_count();

    assert!(
        logical_length <= slot_count,
        "packed FFT2 logical size must not exceed CKKS slot count"
    );
    assert_eq!(
        embedding.degree(),
        input.rlwe().degree(),
        "packed FFT2 embedding degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.degree(),
        input.rlwe().degree(),
        "packed FFT2 NTT plan degree must match ciphertext ring degree"
    );
    assert_eq!(
        plan.moduli(),
        input.basis().moduli(),
        "packed FFT2 NTT plan basis must match ciphertext basis"
    );

    let rotation = diagonals.rotation();

    assert!(
        rotation < logical_length,
        "packed FFT2 stage rotation must be smaller than logical size"
    );

    let mut direct = vec![Complex64::new(0.0, 0.0); slot_count];
    let mut left = vec![Complex64::new(0.0, 0.0); slot_count];
    let mut right = vec![Complex64::new(0.0, 0.0); slot_count];

    direct[..logical_length].copy_from_slice(diagonals.direct());
    left[..logical_length].copy_from_slice(diagonals.rotate_left());
    right[..logical_length].copy_from_slice(diagonals.rotate_right());

    let left_exponent = crate::ckks::rotation_exponent_left(input.rlwe().degree(), rotation);
    let right_exponent = crate::ckks::rotation_exponent_right(input.rlwe().degree(), rotation);

    let left_key = evaluator.keys().galois_for(input.state(), left_exponent);
    let right_key = evaluator.keys().galois_for(input.state(), right_exponent);

    /*
     * The two Galois rotations are independent computations over the same
     * stage input. Execute them concurrently while preserving the exact
     * rotation semantics and level state.
     */
    let (rotated_left, rotated_right) = std::thread::scope(|scope| {
        let left_handle = scope.spawn(|| {
            crate::ckks::rotate_left_rns_ckks_with_ntt(input, rotation, left_key, chain, plan)
        });

        let right_handle = scope.spawn(|| {
            crate::ckks::rotate_right_rns_ckks_with_ntt(input, rotation, right_key, chain, plan)
        });

        (
            left_handle
                .join()
                .expect("packed FFT2 left rotation worker panicked"),
            right_handle
                .join()
                .expect("packed FFT2 right rotation worker panicked"),
        )
    });

    /*
     * Once the rotations are available, the direct, left, and right public
     * diagonal products are mutually independent. Execute all three
     * concurrently. Each product performs exactly the same CP multiply and
     * rescale as the serial implementation.
     */
    let (direct_term, left_term, right_term) = std::thread::scope(|scope| {
        let direct_handle =
            scope.spawn(|| multiply_complex_slots_cp(input, &direct, embedding, chain, plan));

        let left_handle =
            scope.spawn(|| multiply_complex_slots_cp(&rotated_left, &left, embedding, chain, plan));

        let right_handle = scope
            .spawn(|| multiply_complex_slots_cp(&rotated_right, &right, embedding, chain, plan));

        (
            direct_handle
                .join()
                .expect("packed FFT2 direct CP worker panicked"),
            left_handle
                .join()
                .expect("packed FFT2 left CP worker panicked"),
            right_handle
                .join()
                .expect("packed FFT2 right CP worker panicked"),
        )
    });

    let partial = evaluator.add(&direct_term, &left_term);
    evaluator.add(&partial, &right_term)
}

/// Executes a complete packed radix-2 DIF FFT2 over encrypted CKKS slots.
///
/// The first `shape.elements()` canonical CKKS slots contain the logical input
/// in natural row-major order. Row DIF stages execute first, followed directly
/// by column DIF stages on the same ciphertext representation.
///
/// No encrypted transpose is performed.
///
/// The physical output has independently bit-reversed row and column
/// coordinates:
///
/// ```text
/// physical(bit_reverse(row), bit_reverse(col)) = logical(row, col)
/// ```
///
/// Every DIF stage consumes one CKKS level. Therefore forward FFT2 consumes
/// `log2(cols) + log2(rows)` levels. Inverse FFT2 additionally performs one
/// public slot-vector multiplication by `1 / (rows * cols)`, consuming one
/// further level.
pub fn execute_packed_fft2_dif_cp(
    shape: Fft2Shape,
    direction: FftDirection,
    input: &RnsCkksCiphertext,
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
) -> RnsCkksCiphertext {
    input.assert_matches_chain(chain);

    assert!(
        shape.elements() <= embedding.slot_count(),
        "packed encrypted FFT2 size must not exceed CKKS slot count"
    );
    assert_eq!(
        embedding.degree(),
        input.rlwe().degree(),
        "packed encrypted FFT2 embedding degree must match ciphertext ring degree"
    );

    let mut value = input.clone();

    let mut span = shape.cols();
    while span >= 2 {
        let diagonals =
            PackedFft2DifStageDiagonals::new(shape, PackedFft2Axis::Rows, span, direction);

        let plan = RnsNttPlan::new(value.basis().moduli().to_vec(), value.rlwe().degree());

        value = execute_packed_fft2_dif_stage_cp(
            &value, &diagonals, evaluator, embedding, chain, &plan,
        );

        span /= 2;
    }

    span = shape.rows();
    while span >= 2 {
        let diagonals =
            PackedFft2DifStageDiagonals::new(shape, PackedFft2Axis::Columns, span, direction);

        let plan = RnsNttPlan::new(value.basis().moduli().to_vec(), value.rlwe().degree());

        value = execute_packed_fft2_dif_stage_cp(
            &value, &diagonals, evaluator, embedding, chain, &plan,
        );

        span /= 2;
    }

    if direction == FftDirection::Inverse && shape.elements() > 1 {
        let plan = RnsNttPlan::new(value.basis().moduli().to_vec(), value.rlwe().degree());

        let mut normalization = vec![Complex64::new(0.0, 0.0); embedding.slot_count()];

        let factor = 1.0 / shape.elements() as f64;

        normalization[..shape.elements()].fill(Complex64::new(factor, 0.0));

        value = multiply_complex_slots_cp(&value, &normalization, embedding, chain, &plan);
    }

    value
}

/// Executes a complete packed cleartext two-dimensional DIF FFT.
///
/// Input uses natural row-major order. The physical output has independently
/// bit-reversed row and column coordinates:
///
/// ```text
/// physical(bit_reverse(row), bit_reverse(col)) = logical(row, col)
/// ```
///
/// No transpose or global permutation is executed between dimensions.
pub fn execute_packed_fft2_dif_pp(
    shape: Fft2Shape,
    direction: FftDirection,
    input: &[Complex64],
) -> Vec<Complex64> {
    assert_eq!(
        input.len(),
        shape.elements(),
        "packed FFT2 DIF input length must match shape"
    );

    let mut values = input.to_vec();

    let mut span = shape.cols();
    while span >= 2 {
        let diagonals =
            PackedFft2DifStageDiagonals::new(shape, PackedFft2Axis::Rows, span, direction);
        values = execute_packed_fft2_dif_stage_pp(&values, &diagonals);
        span /= 2;
    }

    span = shape.rows();
    while span >= 2 {
        let diagonals =
            PackedFft2DifStageDiagonals::new(shape, PackedFft2Axis::Columns, span, direction);
        values = execute_packed_fft2_dif_stage_pp(&values, &diagonals);
        span /= 2;
    }

    if direction == FftDirection::Inverse {
        let normalization = 1.0 / shape.elements() as f64;
        for value in &mut values {
            *value *= normalization;
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
    use proptest::prelude::*;

    fn execute_direct_fft1_stage(
        input: &[Complex64],
        stage: &Fft1Stage,
        direction: FftDirection,
    ) -> Vec<Complex64> {
        let sign = match direction {
            FftDirection::Forward => -1.0,
            FftDirection::Inverse => 1.0,
        };

        let mut output = input.to_vec();

        for butterfly in stage.butterflies() {
            let angle =
                sign * 2.0 * PI * butterfly.twiddle_exponent() as f64 / butterfly.span() as f64;
            let twiddle = Complex64::new(angle.cos(), angle.sin());

            let even = input[butterfly.even_index()];
            let odd = twiddle * input[butterfly.odd_index()];

            output[butterfly.even_index()] = even + odd;
            output[butterfly.odd_index()] = even - odd;
        }

        output
    }

    #[test]
    fn packed_dif_fft_matches_canonical_fft_through_256() {
        for &direction in &[FftDirection::Forward, FftDirection::Inverse] {
            for exponent in 0..=8 {
                let n = 1usize << exponent;
                let shape = Fft1Shape::new(n);

                let input: Vec<Complex64> = (0..n)
                    .map(|index| {
                        let re = ((index * 11 + 7) % 37) as f64 - 18.0;
                        let im = ((index * 17 + 3) % 41) as f64 - 20.0;
                        Complex64::new(re / 23.0, im / 29.0)
                    })
                    .collect();

                let canonical = fft1_pp(shape, direction, &input);
                let packed_physical = execute_packed_fft1_dif_pp(shape, direction, &input);

                let bits = n.trailing_zeros();
                let mut max_abs = 0.0_f64;

                for (logical_index, &expected) in canonical.iter().enumerate() {
                    let physical_index = if n <= 2 {
                        logical_index
                    } else {
                        logical_index.reverse_bits() >> (usize::BITS - bits)
                    };

                    max_abs = max_abs.max((packed_physical[physical_index] - expected).norm());
                }

                println!(
                    "PACKED_FFT1_DIF_CASE=N{} DIRECTION={:?} STAGES={} MAX_ABS={:.12e}",
                    n,
                    direction,
                    shape.stages(),
                    max_abs
                );

                assert!(
                    max_abs < 1.0e-10,
                    "packed DIF FFT1 must match canonical FFT under bit-reversed output mapping"
                );
            }
        }
    }

    #[test]
    fn packed_dif_stage_matches_direct_dif_butterflies_through_256() {
        for &direction in &[FftDirection::Forward, FftDirection::Inverse] {
            let sign = match direction {
                FftDirection::Forward => -1.0,
                FftDirection::Inverse => 1.0,
            };

            for exponent in 1..=8 {
                let n = 1usize << exponent;

                let input: Vec<Complex64> = (0..n)
                    .map(|index| {
                        Complex64::new(
                            (((index * 5 + 1) % 23) as f64 - 11.0) / 13.0,
                            (((index * 9 + 4) % 19) as f64 - 9.0) / 17.0,
                        )
                    })
                    .collect();

                let mut span = n;

                while span >= 2 {
                    let half = span / 2;

                    let diagonals = PackedFft1DifStageDiagonals::new(n, span, direction);

                    let packed = execute_packed_fft1_dif_stage_pp(&input, &diagonals);

                    let mut direct = input.clone();

                    for base in (0..n).step_by(span) {
                        for offset in 0..half {
                            let upper = base + offset;
                            let lower = upper + half;

                            let angle = sign * 2.0 * PI * offset as f64 / span as f64;

                            let twiddle = Complex64::new(angle.cos(), angle.sin());

                            direct[upper] = input[upper] + input[lower];

                            direct[lower] = (input[upper] - input[lower]) * twiddle;
                        }
                    }

                    let max_abs = packed
                        .iter()
                        .zip(&direct)
                        .map(|(actual, expected)| (*actual - *expected).norm())
                        .fold(0.0_f64, f64::max);

                    println!(
                        "PACKED_FFT1_DIF_STAGE_CASE=N{} DIRECTION={:?} SPAN={} ROTATION={} MAX_ABS={:.12e}",
                        n,
                        direction,
                        span,
                        half,
                        max_abs
                    );

                    assert!(
                        max_abs < 1.0e-12,
                        "packed DIF stage must match direct DIF butterflies"
                    );

                    span /= 2;
                }
            }
        }
    }

    #[test]
    fn packed_stage_diagonals_match_direct_butterflies_through_256() {
        for &direction in &[FftDirection::Forward, FftDirection::Inverse] {
            for exponent in 1..=8 {
                let n = 1usize << exponent;
                let plan = Fft1Plan::new(Fft1Shape::new(n));

                let input: Vec<Complex64> = (0..n)
                    .map(|index| {
                        let re = ((index * 7 + 3) % 31) as f64 - 15.0;
                        let im = ((index * 13 + 5) % 29) as f64 - 14.0;
                        Complex64::new(re / 17.0, im / 19.0)
                    })
                    .collect();

                for stage in plan.stages() {
                    let diagonals = PackedFft1StageDiagonals::new(n, stage.span(), direction);

                    assert_eq!(diagonals.span(), stage.span());
                    assert_eq!(diagonals.rotation(), stage.span() / 2);

                    let packed = execute_packed_fft1_stage_pp(&input, &diagonals);
                    let direct = execute_direct_fft1_stage(&input, stage, direction);

                    let max_abs = packed
                        .iter()
                        .zip(&direct)
                        .map(|(lhs, rhs)| (*lhs - *rhs).norm())
                        .fold(0.0_f64, f64::max);

                    println!(
                        "PACKED_FFT1_STAGE_CASE=N{n} DIRECTION={direction:?} SPAN={} ROTATION={} MAX_ABS={max_abs:.12e}",
                        stage.span(),
                        diagonals.rotation()
                    );

                    assert!(
                        max_abs <= 1.0e-12,
                        "packed FFT1 stage mismatch for N={n}, direction={direction:?}, span={}: {max_abs:e}",
                        stage.span()
                    );
                }
            }
        }
    }

    #[test]
    fn packed_stage_masks_expose_expected_n8_span4_structure() {
        let diagonals = PackedFft1StageDiagonals::new(8, 4, FftDirection::Forward);

        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let minus_one = Complex64::new(-1.0, 0.0);
        let minus_i = Complex64::new(0.0, -1.0);
        let plus_i = Complex64::new(0.0, 1.0);

        let expected_direct = [one, one, minus_one, plus_i, one, one, minus_one, plus_i];
        let expected_rotate_left = [one, minus_i, zero, zero, one, minus_i, zero, zero];
        let expected_rotate_right = [zero, zero, one, one, zero, zero, one, one];

        let max_direct_error = diagonals
            .direct()
            .iter()
            .zip(expected_direct)
            .map(|(actual, expected)| (*actual - expected).norm())
            .fold(0.0_f64, f64::max);

        let max_left_error = diagonals
            .rotate_left()
            .iter()
            .zip(expected_rotate_left)
            .map(|(actual, expected)| (*actual - expected).norm())
            .fold(0.0_f64, f64::max);

        let max_right_error = diagonals
            .rotate_right()
            .iter()
            .zip(expected_rotate_right)
            .map(|(actual, expected)| (*actual - expected).norm())
            .fold(0.0_f64, f64::max);

        println!(
            "PACKED_FFT1_STAGE_MASK_CASE=N8 SPAN4 DIRECT_MAX_ABS={max_direct_error:.12e} LEFT_MAX_ABS={max_left_error:.12e} RIGHT_MAX_ABS={max_right_error:.12e}"
        );

        assert!(max_direct_error <= 1.0e-12);
        assert!(max_left_error <= 1.0e-12);
        assert!(max_right_error <= 1.0e-12);
    }

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

    fn bounded_complex_strategy() -> impl Strategy<Value = Complex64> {
        (-1024_i32..=1024_i32, -1024_i32..=1024_i32)
            .prop_map(|(re, im)| Complex64::new(re as f64 / 1024.0, im as f64 / 1024.0))
    }

    fn fft1_property_case() -> impl Strategy<Value = (usize, Vec<Complex64>)> {
        prop_oneof![
            Just(1usize),
            Just(2usize),
            Just(4usize),
            Just(8usize),
            Just(16usize),
            Just(32usize),
        ]
        .prop_flat_map(|n| {
            prop::collection::vec(bounded_complex_strategy(), n).prop_map(move |input| (n, input))
        })
    }

    fn fft2_property_case() -> impl Strategy<Value = (usize, usize, Vec<Complex64>)> {
        (
            prop_oneof![Just(1usize), Just(2usize), Just(4usize), Just(8usize),],
            prop_oneof![Just(1usize), Just(2usize), Just(4usize), Just(8usize),],
        )
            .prop_flat_map(|(rows, cols)| {
                prop::collection::vec(bounded_complex_strategy(), rows * cols)
                    .prop_map(move |input| (rows, cols, input))
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            max_shrink_iters: 4096,
            .. ProptestConfig::default()
        })]

        #[test]
        fn property_fft1_roundtrip_recovers_generated_input(
            (n, input) in fft1_property_case(),
        ) {
            let shape = Fft1Shape::new(n);

            let spectrum =
                fft1_pp(shape, FftDirection::Forward, &input);
            let recovered =
                fft1_pp(shape, FftDirection::Inverse, &spectrum);

            let max_abs = recovered
                .iter()
                .zip(&input)
                .map(|(actual, expected)| {
                    (*actual - *expected).norm()
                })
                .fold(0.0_f64, f64::max);

            prop_assert!(
                max_abs < 1.0e-10,
                "FFT1 roundtrip mismatch N={n}: {max_abs:e}"
            );
        }

        #[test]
        fn property_fft2_roundtrip_recovers_generated_input(
            (rows, cols, input) in fft2_property_case(),
        ) {
            let shape = Fft2Shape::new(rows, cols);

            let spectrum =
                fft2_pp(shape, FftDirection::Forward, &input);
            let recovered =
                fft2_pp(shape, FftDirection::Inverse, &spectrum);

            let max_abs = recovered
                .iter()
                .zip(&input)
                .map(|(actual, expected)| {
                    (*actual - *expected).norm()
                })
                .fold(0.0_f64, f64::max);

            prop_assert!(
                max_abs < 1.0e-10,
                "FFT2 roundtrip mismatch \
                 {rows}x{cols}: {max_abs:e}"
            );
        }

        #[test]
        fn property_packed_fft2_dif_matches_canonical_layout(
            (rows, cols, input) in fft2_property_case(),
            inverse in any::<bool>(),
        ) {
            let shape = Fft2Shape::new(rows, cols);

            let direction = if inverse {
                FftDirection::Inverse
            } else {
                FftDirection::Forward
            };

            let packed =
                execute_packed_fft2_dif_pp(
                    shape,
                    direction,
                    &input,
                );

            let canonical =
                fft2_pp(shape, direction, &input);

            let mut max_abs = 0.0_f64;

            for logical_row in 0..rows {
                for logical_col in 0..cols {
                    let physical_row =
                        bit_reverse_index_for_test(
                            logical_row,
                            rows,
                        );
                    let physical_col =
                        bit_reverse_index_for_test(
                            logical_col,
                            cols,
                        );

                    let physical =
                        physical_row * cols + physical_col;
                    let logical =
                        logical_row * cols + logical_col;

                    max_abs = max_abs.max(
                        (packed[physical] - canonical[logical])
                            .norm(),
                    );
                }
            }

            prop_assert!(
                max_abs < 1.0e-10,
                "packed FFT2 DIF mismatch \
                 {rows}x{cols} {direction:?}: {max_abs:e}"
            );
        }
    }

    fn bit_reverse_index_for_test(index: usize, length: usize) -> usize {
        if length <= 1 {
            return 0;
        }

        let bits = length.trailing_zeros();
        index.reverse_bits() >> (usize::BITS - bits)
    }

    fn direct_fft2_dif_stage_for_test(
        shape: Fft2Shape,
        axis: PackedFft2Axis,
        span: usize,
        direction: FftDirection,
        input: &[Complex64],
    ) -> Vec<Complex64> {
        let mut output = input.to_vec();
        let half = span / 2;
        let sign = match direction {
            FftDirection::Forward => -1.0,
            FftDirection::Inverse => 1.0,
        };

        match axis {
            PackedFft2Axis::Rows => {
                for row in 0..shape.rows() {
                    for base_col in (0..shape.cols()).step_by(span) {
                        for offset in 0..half {
                            let upper = row * shape.cols() + base_col + offset;
                            let lower = upper + half;
                            let angle = sign * 2.0 * PI * offset as f64 / span as f64;
                            let twiddle = Complex64::new(angle.cos(), angle.sin());
                            let a = input[upper];
                            let b = input[lower];

                            output[upper] = a + b;
                            output[lower] = (a - b) * twiddle;
                        }
                    }
                }
            }
            PackedFft2Axis::Columns => {
                for col in 0..shape.cols() {
                    for base_row in (0..shape.rows()).step_by(span) {
                        for offset in 0..half {
                            let upper = (base_row + offset) * shape.cols() + col;
                            let lower = (base_row + offset + half) * shape.cols() + col;
                            let angle = sign * 2.0 * PI * offset as f64 / span as f64;
                            let twiddle = Complex64::new(angle.cos(), angle.sin());
                            let a = input[upper];
                            let b = input[lower];

                            output[upper] = a + b;
                            output[lower] = (a - b) * twiddle;
                        }
                    }
                }
            }
        }

        output
    }

    #[test]
    fn packed_fft2_dif_stages_match_direct_butterflies() {
        let cases = [(2, 2), (2, 4), (4, 2), (4, 4), (4, 8), (8, 4), (8, 8)];
        let mut case_count = 0usize;

        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            for (rows, cols) in cases {
                let shape = Fft2Shape::new(rows, cols);
                let input: Vec<Complex64> = (0..shape.elements())
                    .map(|index| {
                        Complex64::new(
                            (((index * 7 + 3) % 23) as f64 - 11.0) / 16.0,
                            (((index * 11 + 5) % 29) as f64 - 14.0) / 16.0,
                        )
                    })
                    .collect();

                for axis in [PackedFft2Axis::Rows, PackedFft2Axis::Columns] {
                    let axis_length = match axis {
                        PackedFft2Axis::Rows => cols,
                        PackedFft2Axis::Columns => rows,
                    };

                    let mut span = axis_length;
                    while span >= 2 {
                        let diagonals =
                            PackedFft2DifStageDiagonals::new(shape, axis, span, direction);
                        let actual = execute_packed_fft2_dif_stage_pp(&input, &diagonals);
                        let expected =
                            direct_fft2_dif_stage_for_test(shape, axis, span, direction, &input);

                        let max_abs = actual
                            .iter()
                            .zip(&expected)
                            .map(|(a, b)| (*a - *b).norm())
                            .fold(0.0_f64, f64::max);

                        println!(
                            "PACKED_FFT2_DIF_STAGE_CASE=ROWS{rows} COLS{cols} \
                             AXIS={axis:?} SPAN={span} ROTATION={} \
                             DIRECTION={direction:?} MAX_ABS={max_abs:.12e}",
                            diagonals.rotation()
                        );

                        assert!(
                            max_abs < 1.0e-12,
                            "packed FFT2 DIF stage mismatch for \
                             {rows}x{cols} {axis:?} span {span} {direction:?}: {max_abs}"
                        );

                        case_count += 1;
                        span /= 2;
                    }
                }
            }
        }

        println!("PACKED_FFT2_DIF_STAGE_CASE_COUNT={case_count}");
        assert_eq!(case_count, 56);
    }

    #[test]
    fn packed_fft2_dif_matches_canonical_fft2_without_transpose() {
        let cases = [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 4),
            (4, 2),
            (4, 4),
            (4, 8),
            (8, 4),
            (8, 8),
        ];
        let mut case_count = 0usize;

        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            for (rows, cols) in cases {
                let shape = Fft2Shape::new(rows, cols);
                let input: Vec<Complex64> = (0..shape.elements())
                    .map(|index| {
                        Complex64::new(
                            (((index * 13 + 1) % 31) as f64 - 15.0) / 32.0,
                            (((index * 17 + 9) % 37) as f64 - 18.0) / 32.0,
                        )
                    })
                    .collect();

                let packed = execute_packed_fft2_dif_pp(shape, direction, &input);
                let canonical = fft2_pp(shape, direction, &input);

                let mut max_abs = 0.0_f64;

                for logical_row in 0..rows {
                    for logical_col in 0..cols {
                        let physical_row = bit_reverse_index_for_test(logical_row, rows);
                        let physical_col = bit_reverse_index_for_test(logical_col, cols);

                        let physical_index = physical_row * cols + physical_col;
                        let logical_index = logical_row * cols + logical_col;

                        max_abs =
                            max_abs.max((packed[physical_index] - canonical[logical_index]).norm());
                    }
                }

                println!(
                    "PACKED_FFT2_DIF_CASE=ROWS{rows} COLS{cols} \
                     DIRECTION={direction:?} ROW_STAGES={} COLUMN_STAGES={} \
                     TRANSPOSES=0 MAX_ABS={max_abs:.12e}",
                    shape.row_stages(),
                    shape.column_stages()
                );

                assert!(
                    max_abs < 1.0e-11,
                    "packed FFT2 DIF mismatch for \
                     {rows}x{cols} {direction:?}: {max_abs}"
                );

                case_count += 1;
            }
        }

        println!("PACKED_FFT2_DIF_CASE_COUNT={case_count}");
        assert_eq!(case_count, 20);
    }

    #[test]
    fn fft2_shape_exposes_separable_radix2_structure() {
        let shape = Fft2Shape::new(8, 4);

        assert_eq!(shape.rows(), 8);
        assert_eq!(shape.cols(), 4);
        assert_eq!(shape.elements(), 32);
        assert_eq!(shape.row_stages(), 2);
        assert_eq!(shape.column_stages(), 3);
        assert_eq!(shape.stages(), 5);
        assert_eq!(shape.butterflies(), 80);
    }

    #[test]
    #[should_panic(expected = "eBLAS FFT2 row count must be positive")]
    fn fft2_shape_rejects_zero_rows() {
        let _ = Fft2Shape::new(0, 4);
    }

    #[test]
    #[should_panic(expected = "eBLAS FFT2 column count must be positive")]
    fn fft2_shape_rejects_zero_cols() {
        let _ = Fft2Shape::new(4, 0);
    }

    #[test]
    #[should_panic(expected = "eBLAS radix-2 FFT2 row count must be a power of two")]
    fn fft2_shape_rejects_non_power_of_two_rows() {
        let _ = Fft2Shape::new(3, 4);
    }

    #[test]
    #[should_panic(expected = "eBLAS radix-2 FFT2 column count must be a power of two")]
    fn fft2_shape_rejects_non_power_of_two_cols() {
        let _ = Fft2Shape::new(4, 6);
    }

    #[test]
    fn fft2_matches_dense_dft2_reference() {
        let cases = [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 4),
            (4, 2),
            (4, 4),
            (4, 8),
            (8, 4),
            (8, 8),
        ];

        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            for (rows, cols) in cases {
                let shape = Fft2Shape::new(rows, cols);
                let input: Vec<Complex64> = (0..shape.elements())
                    .map(|index| {
                        let re = ((index * 7 + 3) % 23) as f64 - 11.0;
                        let im = ((index * 11 + 5) % 29) as f64 - 14.0;
                        Complex64::new(re / 16.0, im / 16.0)
                    })
                    .collect();

                let actual = fft2_pp(shape, direction, &input);
                let expected = dft2_reference(shape, direction, &input);

                let max_abs = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (*a - *b).norm())
                    .fold(0.0_f64, f64::max);

                println!(
                    "FFT2_PP_DFT_CASE=ROWS{rows} COLS{cols} DIRECTION={direction:?} \
                     ROW_STAGES={} COLUMN_STAGES={} MAX_ABS={max_abs:.12e}",
                    shape.row_stages(),
                    shape.column_stages()
                );

                assert!(
                    max_abs < 1.0e-11,
                    "FFT2 mismatch for {rows}x{cols} {direction:?}: {max_abs}"
                );
            }
        }
    }

    #[test]
    fn fft2_roundtrip_recovers_input() {
        let cases = [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 4),
            (4, 2),
            (4, 4),
            (4, 8),
            (8, 4),
            (8, 8),
        ];

        for (rows, cols) in cases {
            let shape = Fft2Shape::new(rows, cols);
            let input: Vec<Complex64> = (0..shape.elements())
                .map(|index| {
                    let re = ((index * 13 + 1) % 31) as f64 - 15.0;
                    let im = ((index * 17 + 9) % 37) as f64 - 18.0;
                    Complex64::new(re / 32.0, im / 32.0)
                })
                .collect();

            let spectrum = fft2_pp(shape, FftDirection::Forward, &input);
            let recovered = fft2_pp(shape, FftDirection::Inverse, &spectrum);

            let max_abs = recovered
                .iter()
                .zip(&input)
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0_f64, f64::max);

            println!(
                "FFT2_PP_ROUNDTRIP_CASE=ROWS{rows} COLS{cols} \
                 ROW_STAGES={} COLUMN_STAGES={} MAX_ABS={max_abs:.12e}",
                shape.row_stages(),
                shape.column_stages()
            );

            assert!(
                max_abs < 1.0e-11,
                "FFT2 roundtrip mismatch for {rows}x{cols}: {max_abs}"
            );
        }
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

#[cfg(test)]
mod repeated_packed_fft2_tests {
    use super::{
        execute_packed_fft2_dif_stage_pp, execute_repeated_packed_fft2_dif_stage_pp, Fft2Shape,
        FftDirection, PackedFft2Axis, PackedFft2DifStageDiagonals,
    };
    use num_complex::Complex64;

    fn deterministic_tile(tile_index: usize, elements: usize) -> Vec<Complex64> {
        (0..elements)
            .map(|element_index| {
                Complex64::new(
                    (17 * tile_index + element_index % 251) as f64 / 251.0,
                    (13 * tile_index + element_index % 127) as f64 / 127.0,
                )
            })
            .collect()
    }

    fn validate_repeated_stage(axis: PackedFft2Axis, span: usize, tiles_per_vector: usize) {
        let shape = Fft2Shape::new(64, 64);
        let tile_elements = shape.elements();
        let slot_count = 32_768;

        let diagonals = PackedFft2DifStageDiagonals::new(shape, axis, span, FftDirection::Forward);

        let tiles: Vec<Vec<Complex64>> = (0..tiles_per_vector)
            .map(|tile_index| deterministic_tile(tile_index, tile_elements))
            .collect();

        let mut packed = vec![Complex64::new(0.0, 0.0); slot_count];

        for (tile_index, tile) in tiles.iter().enumerate() {
            let start = tile_index * tile_elements;
            let end = start + tile_elements;
            packed[start..end].copy_from_slice(tile);
        }

        let actual =
            execute_repeated_packed_fft2_dif_stage_pp(&packed, &diagonals, tiles_per_vector);

        for (tile_index, tile) in tiles.iter().enumerate() {
            let expected = execute_packed_fft2_dif_stage_pp(tile, &diagonals);

            let start = tile_index * tile_elements;
            let end = start + tile_elements;

            let max_error = actual[start..end]
                .iter()
                .zip(expected.iter())
                .map(|(actual, expected)| (*actual - *expected).norm())
                .fold(0.0_f64, f64::max);

            assert!(
                max_error <= 1.0e-12,
                "repeated packed FFT2 mismatch: axis={axis:?} span={span} tiles={tiles_per_vector} tile={tile_index} max_error={max_error:e}"
            );
        }

        let active_elements = tiles_per_vector * tile_elements;
        let inactive_max = actual[active_elements..]
            .iter()
            .map(|value| value.norm())
            .fold(0.0_f64, f64::max);

        assert!(
            inactive_max <= 1.0e-15,
            "repeated packed FFT2 contaminated inactive slots: axis={axis:?} span={span} tiles={tiles_per_vector} inactive_max={inactive_max:e}"
        );
    }

    #[test]
    fn repeated_packed_fft2_stages_isolate_2_4_8_tiles() {
        for tiles_per_vector in [2, 4, 8] {
            for axis in [PackedFft2Axis::Rows, PackedFft2Axis::Columns] {
                for span in [64, 32, 16, 8, 4, 2] {
                    validate_repeated_stage(axis, span, tiles_per_vector);
                }
            }
        }
    }
}

#[cfg(test)]
mod packed_tile_dif_tests {
    use super::{execute_packed_tile_dif_stage_pp, FftDirection, PackedTileDifStageDiagonals};
    use num_complex::Complex64;
    use std::f64::consts::PI;

    fn direct_reference(
        input: &[Complex64],
        tile_rows: usize,
        tile_cols: usize,
        packed_tiles: usize,
        span_tiles: usize,
    ) -> Vec<Complex64> {
        let tile_elements = tile_rows * tile_cols;
        let half_tiles = span_tiles / 2;
        let global_span = span_tiles * tile_rows;

        let mut output = vec![Complex64::new(0.0, 0.0); input.len()];

        for group_start in (0..packed_tiles).step_by(span_tiles) {
            for tile_offset in 0..half_tiles {
                let upper_lane = group_start + tile_offset;
                let lower_lane = upper_lane + half_tiles;

                let upper_base = upper_lane * tile_elements;
                let lower_base = lower_lane * tile_elements;

                for local_row in 0..tile_rows {
                    let twiddle_index = tile_offset * tile_rows + local_row;

                    let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

                    let w = Complex64::new(angle.cos(), angle.sin());

                    for local_col in 0..tile_cols {
                        let tile_slot = local_row * tile_cols + local_col;

                        let upper_slot = upper_base + tile_slot;
                        let lower_slot = lower_base + tile_slot;

                        let a = input[upper_slot];
                        let b = input[lower_slot];

                        output[upper_slot] = a + b;
                        output[lower_slot] = w * (a - b);
                    }
                }
            }
        }

        output
    }

    #[test]
    fn packed_tile_dif_rotations_match_r17() {
        for (span_tiles, rotation) in [
            (8usize, 16_384usize),
            (4usize, 8_192usize),
            (2usize, 4_096usize),
        ] {
            let diagonals = PackedTileDifStageDiagonals::new_column(
                64,
                64,
                8,
                span_tiles,
                FftDirection::Forward,
            );

            assert_eq!(diagonals.rotation(), rotation);
        }
    }

    #[test]
    fn packed_tile_dif_matches_direct_r17_column_stages() {
        let tile_rows = 64usize;
        let tile_cols = 64usize;
        let packed_tiles = 8usize;
        let tile_elements = tile_rows * tile_cols;

        let mut value = (0..packed_tiles * tile_elements)
            .map(|index| {
                Complex64::new(
                    ((17 * index + 11) % 251) as f64 / 251.0,
                    ((13 * index + 7) % 127) as f64 / 127.0,
                )
            })
            .collect::<Vec<_>>();

        for span_tiles in [8usize, 4, 2] {
            let expected = direct_reference(&value, tile_rows, tile_cols, packed_tiles, span_tiles);

            let diagonals = PackedTileDifStageDiagonals::new_column(
                tile_rows,
                tile_cols,
                packed_tiles,
                span_tiles,
                FftDirection::Forward,
            );

            let actual = execute_packed_tile_dif_stage_pp(&value, &diagonals);

            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!(
                    (*actual - *expected).norm() <= 1.0e-12,
                    "packed tile DIF mismatch"
                );
            }

            value = actual;
        }
    }
}
