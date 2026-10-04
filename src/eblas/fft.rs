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
    mod_switch_rns_ckks_to_next, CkksCanonicalEmbedding, RnsCkksCiphertext, RnsCkksEvaluator,
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
