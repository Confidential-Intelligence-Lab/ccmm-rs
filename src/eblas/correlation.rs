//! eBLAS correlation front-ends.
//!
//! Correlation is an application-facing structured linear-algebra operation.
//! This module defines its semantics independently of the execution mechanism.
//!
//! For a signal `x` of length `N` and a kernel/template `h` of length `K`,
//! valid one-dimensional correlation computes
//!
//! `y[i] = sum_j x[i + j] * h[j]`
//!
//! for `0 <= i < N - K + 1`.
//!
//! Unlike convolution, correlation does not reverse the kernel.

use crate::eblas::{
    gemm_cc_decomposed, gemm_cp_decomposed, gemm_pp, BatchGemmGeometry,
    DecomposedCcmmExecutionContext, DecomposedCpmmExecutionContext, GemmShape, GemmSpec,
    MatrixShape, NhwcShape, PrivacyMode,
};
use crate::matrix::BatchMatrix;

/// Shape contract for valid one-dimensional correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Correlation1dShape {
    signal_length: usize,
    kernel_length: usize,
    output_length: usize,
}

impl Correlation1dShape {
    /// Creates a valid one-dimensional correlation shape.
    pub fn new(signal_length: usize, kernel_length: usize) -> Self {
        assert!(
            signal_length > 0,
            "eBLAS correlation signal length must be positive"
        );
        assert!(
            kernel_length > 0,
            "eBLAS correlation kernel length must be positive"
        );
        assert!(
            kernel_length <= signal_length,
            "eBLAS correlation kernel cannot exceed signal length"
        );

        Self {
            signal_length,
            kernel_length,
            output_length: signal_length - kernel_length + 1,
        }
    }

    /// Number of signal elements.
    pub const fn signal_length(self) -> usize {
        self.signal_length
    }

    /// Number of kernel/template elements.
    pub const fn kernel_length(self) -> usize {
        self.kernel_length
    }

    /// Number of valid correlation positions.
    pub const fn output_length(self) -> usize {
        self.output_length
    }
}

/// Shape contract for valid two-dimensional correlation over NHWC input.
///
/// Input uses canonical `[B,H,W,C]` layout. Filters use logical
/// `[Kh,Kw,C,F]` layout. Valid correlation produces
/// `[B,H-Kh+1,W-Kw+1,F]`.
///
/// The structured operation lowers to one logical GEMM:
///
/// ```text
/// [B*OH*OW, Kh*Kw*C] * [Kh*Kw*C, F]
/// ```
///
/// No kernel reversal is performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Correlation2dShape {
    input: NhwcShape,
    kernel_height: usize,
    kernel_width: usize,
    filters: usize,
    output: NhwcShape,
}

impl Correlation2dShape {
    /// Creates a valid two-dimensional correlation shape.
    pub fn new(
        input: NhwcShape,
        kernel_height: usize,
        kernel_width: usize,
        filters: usize,
    ) -> Self {
        assert!(
            kernel_height > 0,
            "eBLAS 2D correlation kernel height must be positive"
        );
        assert!(
            kernel_width > 0,
            "eBLAS 2D correlation kernel width must be positive"
        );
        assert!(
            filters > 0,
            "eBLAS 2D correlation filter count must be positive"
        );
        assert!(
            kernel_height <= input.height(),
            "eBLAS 2D correlation kernel height cannot exceed input height"
        );
        assert!(
            kernel_width <= input.width(),
            "eBLAS 2D correlation kernel width cannot exceed input width"
        );

        let output_height = input.height() - kernel_height + 1;
        let output_width = input.width() - kernel_width + 1;

        let _ = kernel_height
            .checked_mul(kernel_width)
            .and_then(|value| value.checked_mul(input.channels()))
            .expect("eBLAS 2D correlation inner dimension overflow");

        let output = NhwcShape::new(input.batches(), output_height, output_width, filters);

        Self {
            input,
            kernel_height,
            kernel_width,
            filters,
            output,
        }
    }

    pub const fn input(self) -> NhwcShape {
        self.input
    }

    pub const fn kernel_height(self) -> usize {
        self.kernel_height
    }

    pub const fn kernel_width(self) -> usize {
        self.kernel_width
    }

    pub const fn filters(self) -> usize {
        self.filters
    }

    pub const fn output(self) -> NhwcShape {
        self.output
    }

    pub const fn output_height(self) -> usize {
        self.output.height()
    }

    pub const fn output_width(self) -> usize {
        self.output.width()
    }

    /// Logical GEMM row count: `B * OH * OW`.
    pub fn rows(self) -> usize {
        self.output
            .batches()
            .checked_mul(self.output.height())
            .and_then(|value| value.checked_mul(self.output.width()))
            .expect("eBLAS 2D correlation row count overflow")
    }

    /// Logical GEMM reduction dimension: `Kh * Kw * C`.
    pub fn inner(self) -> usize {
        self.kernel_height
            .checked_mul(self.kernel_width)
            .and_then(|value| value.checked_mul(self.input.channels()))
            .expect("eBLAS 2D correlation inner dimension overflow")
    }

    /// Logical matrix multiplication induced by this correlation.
    pub fn gemm_shape(self) -> GemmShape {
        GemmShape::new(
            MatrixShape::new(self.rows(), self.inner()),
            MatrixShape::new(self.inner(), self.filters),
        )
    }
}

/// Lowers valid one-dimensional correlation to one logical GEMM.
///
/// For signal length `N` and kernel length `K`, the left operand is the
/// `(N - K + 1) x K` sliding-window matrix
///
/// `W[i, j] = signal[i + j]`.
///
/// The kernel remains a `K x 1` column vector, so valid correlation is
/// exactly `W * kernel`. The kernel is not reversed.
fn correlation_1d_gemm_operand(
    shape: Correlation1dShape,
    signal: &BatchMatrix<f64>,
    kernel: &BatchMatrix<f64>,
) -> (GemmShape, BatchMatrix<f64>) {
    assert_eq!(
        signal.batches(),
        1,
        "eBLAS 1D correlation currently requires one signal batch"
    );
    assert_eq!(
        kernel.batches(),
        1,
        "eBLAS 1D correlation currently requires one kernel batch"
    );
    assert_eq!(
        signal.rows(),
        shape.signal_length(),
        "eBLAS correlation signal length does not match its shape"
    );
    assert_eq!(
        signal.cols(),
        1,
        "eBLAS correlation signal must be a column vector"
    );
    assert_eq!(
        kernel.rows(),
        shape.kernel_length(),
        "eBLAS correlation kernel length does not match its shape"
    );
    assert_eq!(
        kernel.cols(),
        1,
        "eBLAS correlation kernel must be a column vector"
    );

    let mut windows = BatchMatrix::<f64>::new(shape.output_length(), shape.kernel_length(), 1);

    for position in 0..shape.output_length() {
        for offset in 0..shape.kernel_length() {
            windows.set(0, position, offset, *signal.get(0, position + offset, 0));
        }
    }

    let gemm_shape = GemmShape::new(
        MatrixShape::new(shape.output_length(), shape.kernel_length()),
        MatrixShape::new(shape.kernel_length(), 1),
    );

    (gemm_shape, windows)
}

/// Computes valid one-dimensional ciphertext/plaintext correlation through
/// decomposed Batch CPMM.
///
/// Correlation is lowered to one logical sliding-window GEMM. Native tiling,
/// SinC lane scheduling, encrypted CPMM execution, boundary handling,
/// reduction accumulation, and logical reconstruction remain owned by the
/// decomposed GEMM substrate.
pub fn correlate_1d_cp(
    shape: Correlation1dShape,
    geometry: BatchGemmGeometry,
    signal: &BatchMatrix<f64>,
    kernel: &BatchMatrix<f64>,
    context: &DecomposedCpmmExecutionContext<'_>,
) -> BatchMatrix<f64> {
    let (gemm_shape, windows) = correlation_1d_gemm_operand(shape, signal, kernel);

    gemm_cp_decomposed(gemm_shape, geometry, &windows, kernel, context)
}

/// Computes valid one-dimensional ciphertext/ciphertext correlation through
/// decomposed Batch CCMM.
///
/// Correlation semantics and sliding-window lowering are identical to PP and
/// CP execution. Both lowered operands cross the encrypted representation
/// boundary; decomposition, SinC scheduling, CCMM execution, and logical
/// reconstruction remain owned by the decomposed GEMM substrate.
pub fn correlate_1d_cc(
    shape: Correlation1dShape,
    geometry: BatchGemmGeometry,
    signal: &BatchMatrix<f64>,
    kernel: &BatchMatrix<f64>,
    context: &DecomposedCcmmExecutionContext<'_>,
) -> BatchMatrix<f64> {
    let (gemm_shape, windows) = correlation_1d_gemm_operand(shape, signal, kernel);

    gemm_cc_decomposed(gemm_shape, geometry, &windows, kernel, context)
}

/// Computes valid one-dimensional plaintext/plaintext correlation.
///
/// `signal` and `kernel` use column-vector representations with shapes
/// `N x 1` and `K x 1`. The result is `(N - K + 1) x 1`.
///
/// The kernel is used in its supplied order; it is not reversed.
pub fn correlate_1d_pp(
    shape: Correlation1dShape,
    signal: &BatchMatrix<f64>,
    kernel: &BatchMatrix<f64>,
) -> BatchMatrix<f64> {
    let (gemm_shape, windows) = correlation_1d_gemm_operand(shape, signal, kernel);

    gemm_pp(GemmSpec::new(gemm_shape, PrivacyMode::Pp), &windows, kernel)
}

/// Lowers valid NHWC correlation and `[Kh,Kw,C,F]` filters to GEMM operands.
///
/// The lhs row order is canonical `[batch, output_row, output_col]`.
/// The reduction order is `[kernel_row, kernel_col, channel]`.
/// The rhs column order is the filter index.
fn correlation_2d_gemm_operands(
    shape: Correlation2dShape,
    input: &[f64],
    filters: &[f64],
) -> (GemmShape, BatchMatrix<f64>, BatchMatrix<f64>) {
    let input_shape = shape.input();

    assert_eq!(
        input.len(),
        input_shape.elements(),
        "eBLAS 2D correlation input length does not match its NHWC shape"
    );

    let expected_filter_elements = shape
        .inner()
        .checked_mul(shape.filters())
        .expect("eBLAS 2D correlation filter element count overflow");

    assert_eq!(
        filters.len(),
        expected_filter_elements,
        "eBLAS 2D correlation filter length does not match [Kh,Kw,C,F]"
    );

    let mut windows = BatchMatrix::<f64>::new(shape.rows(), shape.inner(), 1);

    for batch in 0..input_shape.batches() {
        for output_row in 0..shape.output_height() {
            for output_col in 0..shape.output_width() {
                let logical_row = ((batch * shape.output_height() + output_row)
                    * shape.output_width())
                    + output_col;

                for kernel_row in 0..shape.kernel_height() {
                    for kernel_col in 0..shape.kernel_width() {
                        for channel in 0..input_shape.channels() {
                            let inner = ((kernel_row * shape.kernel_width() + kernel_col)
                                * input_shape.channels())
                                + channel;

                            let input_row = output_row + kernel_row;
                            let input_col = output_col + kernel_col;

                            let input_index = (((batch * input_shape.height() + input_row)
                                * input_shape.width()
                                + input_col)
                                * input_shape.channels())
                                + channel;

                            windows.set(0, logical_row, inner, input[input_index]);
                        }
                    }
                }
            }
        }
    }

    let mut lowered_filters = BatchMatrix::<f64>::new(shape.inner(), shape.filters(), 1);

    for kernel_row in 0..shape.kernel_height() {
        for kernel_col in 0..shape.kernel_width() {
            for channel in 0..input_shape.channels() {
                let inner = ((kernel_row * shape.kernel_width() + kernel_col)
                    * input_shape.channels())
                    + channel;

                for filter in 0..shape.filters() {
                    let filter_index = ((((kernel_row * shape.kernel_width()) + kernel_col)
                        * input_shape.channels()
                        + channel)
                        * shape.filters())
                        + filter;

                    lowered_filters.set(0, inner, filter, filters[filter_index]);
                }
            }
        }
    }

    (shape.gemm_shape(), windows, lowered_filters)
}

/// Restores canonical NHWC storage from the lowered `[B*OH*OW,F]` output.
fn correlation_2d_output_nhwc(shape: Correlation2dShape, matrix: &BatchMatrix<f64>) -> Vec<f64> {
    assert_eq!(
        matrix.batches(),
        1,
        "eBLAS 2D correlation lowered output must contain one matrix batch"
    );
    assert_eq!(
        matrix.rows(),
        shape.rows(),
        "eBLAS 2D correlation lowered output row count mismatch"
    );
    assert_eq!(
        matrix.cols(),
        shape.filters(),
        "eBLAS 2D correlation lowered output filter count mismatch"
    );

    let mut output = Vec::with_capacity(shape.output().elements());

    for batch in 0..shape.output().batches() {
        for output_row in 0..shape.output_height() {
            for output_col in 0..shape.output_width() {
                let logical_row = ((batch * shape.output_height() + output_row)
                    * shape.output_width())
                    + output_col;

                for filter in 0..shape.filters() {
                    output.push(*matrix.get(0, logical_row, filter));
                }
            }
        }
    }

    output
}

/// Computes valid two-dimensional plaintext/plaintext correlation.
///
/// Input storage is canonical NHWC `[B,H,W,C]`. Filter storage is canonical
/// `[Kh,Kw,C,F]`. The returned vector is canonical NHWC
/// `[B,OH,OW,F]`.
///
/// Correlation is implemented through deterministic lowering to one logical
/// GEMM. The filters are used in supplied order and are not reversed.
pub fn correlate_2d_pp(shape: Correlation2dShape, input: &[f64], filters: &[f64]) -> Vec<f64> {
    let (gemm_shape, windows, lowered_filters) =
        correlation_2d_gemm_operands(shape, input, filters);

    let output = gemm_pp(
        GemmSpec::new(gemm_shape, PrivacyMode::Pp),
        &windows,
        &lowered_filters,
    );

    correlation_2d_output_nhwc(shape, &output)
}

/// Computes valid two-dimensional ciphertext/plaintext correlation through
/// decomposed Batch CPMM.
///
/// The NHWC input and `[Kh,Kw,C,F]` filters use the same deterministic
/// lowering as PP execution. The lowered input-window matrix is encrypted;
/// the lowered filter matrix remains plaintext. Native decomposition, SinC
/// packing, Batch CPMM execution, reduction accumulation, and logical output
/// reconstruction remain owned by the decomposed GEMM substrate.
pub fn correlate_2d_cp(
    shape: Correlation2dShape,
    geometry: BatchGemmGeometry,
    input: &[f64],
    filters: &[f64],
    context: &DecomposedCpmmExecutionContext<'_>,
) -> Vec<f64> {
    let (gemm_shape, windows, lowered_filters) =
        correlation_2d_gemm_operands(shape, input, filters);

    let output = gemm_cp_decomposed(gemm_shape, geometry, &windows, &lowered_filters, context);

    correlation_2d_output_nhwc(shape, &output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(values: Vec<f64>) -> BatchMatrix<f64> {
        BatchMatrix::from_vec_column_major(values.len(), 1, 1, values)
    }

    #[test]
    fn shape_exposes_valid_correlation_extent() {
        let shape = Correlation1dShape::new(8, 3);

        assert_eq!(shape.signal_length(), 8);
        assert_eq!(shape.kernel_length(), 3);
        assert_eq!(shape.output_length(), 6);
    }

    #[test]
    fn pp_correlation_matches_exact_reference() {
        let signal = column(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let kernel = column(vec![2.0, -1.0, 3.0]);

        let result = correlate_1d_pp(Correlation1dShape::new(5, 3), &signal, &kernel);

        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 1);
        assert_eq!(*result.get(0, 0, 0), 9.0);
        assert_eq!(*result.get(0, 1, 0), 13.0);
        assert_eq!(*result.get(0, 2, 0), 17.0);
    }

    #[test]
    fn correlation_does_not_reverse_kernel() {
        let signal = column(vec![1.0, 2.0, 4.0]);
        let kernel = column(vec![1.0, 3.0]);

        let result = correlate_1d_pp(Correlation1dShape::new(3, 2), &signal, &kernel);

        assert_eq!(*result.get(0, 0, 0), 7.0);
        assert_eq!(*result.get(0, 1, 0), 14.0);
    }

    #[test]
    fn kernel_equal_to_signal_produces_one_output() {
        let signal = column(vec![1.0, 2.0, 3.0]);
        let kernel = column(vec![4.0, 5.0, 6.0]);

        let result = correlate_1d_pp(Correlation1dShape::new(3, 3), &signal, &kernel);

        assert_eq!(result.rows(), 1);
        assert_eq!(*result.get(0, 0, 0), 32.0);
    }

    #[test]
    #[should_panic(expected = "kernel cannot exceed signal length")]
    fn shape_rejects_kernel_longer_than_signal() {
        let _ = Correlation1dShape::new(3, 4);
    }

    const CPMM_DIMENSION: usize = 64;
    const CPMM_SCALAR_DEGREE: usize = 128;
    const CPMM_LARGE_DEGREE: usize = 4096;

    fn cpmm_moduli() -> Vec<crate::ring::Modulus> {
        vec![
            crate::ring::Modulus::new(68_712_923_137),
            crate::ring::Modulus::new(268_238_849),
        ]
    }

    fn cpmm_scale() -> f64 {
        2.0_f64.powf(27.993302092216055)
    }

    fn deterministic_column(length: usize, salt: usize) -> BatchMatrix<f64> {
        let values = (0..length)
            .map(|index| {
                let x = (index * 17 + salt * 19) % 41;
                (x as f64 - 20.0) / 128.0
            })
            .collect::<Vec<_>>();

        BatchMatrix::from_vec_column_major(length, 1, 1, values)
    }

    fn assert_close(
        actual: &BatchMatrix<f64>,
        expected: &BatchMatrix<f64>,
        tolerance: f64,
        metric_prefix: &str,
    ) {
        assert_eq!(actual.rows(), expected.rows());
        assert_eq!(actual.cols(), expected.cols());
        assert_eq!(actual.batches(), expected.batches());

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for row in 0..actual.rows() {
            let observed = *actual.get(0, row, 0);
            let reference = *expected.get(0, row, 0);
            let error = observed - reference;

            squared_error += error * error;
            squared_reference += reference * reference;
            max_abs = max_abs.max(error.abs());
        }

        let rel_l2 = if squared_reference > 0.0 {
            (squared_error / squared_reference).sqrt()
        } else {
            squared_error.sqrt()
        };

        println!("CORRELATION_{metric_prefix}_REL_L2={rel_l2:.12e} CORRELATION_{metric_prefix}_MAX_ABS={max_abs:.12e}");

        assert!(
            rel_l2 <= tolerance,
            "encrypted CP correlation relative L2 error {rel_l2:e} exceeds tolerance {tolerance:e}"
        );
        assert!(
            max_abs <= tolerance,
            "encrypted CP correlation max abs error {max_abs:e} exceeds tolerance {tolerance:e}"
        );
    }

    fn run_cp_case(
        signal_length: usize,
        kernel_length: usize,
        signal_salt: usize,
        kernel_salt: usize,
        seed: u64,
    ) {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        let shape = Correlation1dShape::new(signal_length, kernel_length);
        let signal = deterministic_column(signal_length, signal_salt);
        let kernel = deterministic_column(kernel_length, kernel_salt);

        let expected = correlate_1d_pp(shape, &signal, &kernel);

        let basis = crate::ring::ModulusBasis::new(cpmm_moduli());
        let chain = crate::ring::ModulusChain::from_top_basis(basis.clone());

        let large_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), CPMM_LARGE_DEGREE);
        let scalar_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), CPMM_SCALAR_DEGREE);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4350_434f_5252_0000 ^ seed);

        let mut secret: Vec<i8> = (0..CPMM_LARGE_DEGREE)
            .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
            .collect();

        if secret.iter().all(|&value| value == 0) {
            secret[0] = 1;
        }

        let geometry = BatchGemmGeometry::new(
            crate::eblas::BatchGemmMechanism::Cpmm,
            CPMM_DIMENSION,
            CPMM_SCALAR_DEGREE,
            CPMM_LARGE_DEGREE,
        );

        let context = DecomposedCpmmExecutionContext {
            basis: &basis,
            chain: &chain,
            large_plan: &large_plan,
            scalar_plan: &scalar_plan,
            secret: &secret,
            scale: cpmm_scale(),
            encryption_seed: seed ^ 0x0045_4e43_5259_5054,
        };

        let actual = correlate_1d_cp(shape, geometry, &signal, &kernel, &context);

        println!(
            "CORRELATION_CP_CASE=N{} K{} OUT{}",
            signal_length,
            kernel_length,
            shape.output_length()
        );

        assert_close(&actual, &expected, 1.0e-3, "CP");
    }

    #[test]
    fn cp_small_semantic_case_matches_pp() {
        // W is 6 x 3, so the complete correlation occupies one native product.
        run_cp_case(8, 3, 1, 7, 0x5238_1001);
    }

    #[test]
    fn cp_output_boundary_decomposition_matches_pp() {
        // N=70, K=5 -> output length 66.
        //
        // Logical GEMM:
        //
        //     66 x 5  *  5 x 1
        //
        // With native d=64 this creates two output-row products and therefore
        // exercises boundary padding plus logical output reconstruction.
        run_cp_case(70, 5, 2, 11, 0x5238_1002);
    }

    #[test]
    fn cp_reduction_decomposition_matches_pp() {
        // N=130, K=67 -> output length 64.
        //
        // Logical GEMM:
        //
        //     64 x 67  *  67 x 1
        //
        // With native d=64 this creates two K products contributing to the
        // same output tile, directly exercising encrypted reduction
        // accumulation for the correlation semantic operation.
        run_cp_case(130, 67, 3, 13, 0x5238_1003);
    }

    const CCMM_DIMENSION: usize = 64;
    const CCMM_SCALAR_DEGREE: usize = 128;
    const CCMM_LARGE_DEGREE: usize = 8192;

    fn ccmm_moduli() -> Vec<crate::ring::Modulus> {
        vec![
            crate::ring::Modulus::new(68_712_923_137),
            crate::ring::Modulus::new(268_238_849),
        ]
    }

    fn ccmm_scale() -> f64 {
        2.0_f64.powf(27.993302092216055)
    }

    fn run_cc_case(
        signal_length: usize,
        kernel_length: usize,
        signal_salt: usize,
        kernel_salt: usize,
        seed: u64,
    ) {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        let shape = Correlation1dShape::new(signal_length, kernel_length);
        let signal = deterministic_column(signal_length, signal_salt);
        let kernel = deterministic_column(kernel_length, kernel_salt);

        let expected = correlate_1d_pp(shape, &signal, &kernel);

        let basis = crate::ring::ModulusBasis::new(ccmm_moduli());
        let chain = crate::ring::ModulusChain::from_top_basis(basis.clone());

        let large_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), CCMM_LARGE_DEGREE);
        let prepared_large_plan = crate::ring::PreparedRnsNttPlan::new(&large_plan);

        let scalar_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), CCMM_SCALAR_DEGREE);
        let prepared_scalar_plan = crate::ring::PreparedRnsNttPlan::new(&scalar_plan);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4343_434f_5252_0000 ^ seed);

        let mut secret: Vec<i8> = (0..CCMM_LARGE_DEGREE)
            .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
            .collect();

        if secret.iter().all(|&value| value == 0) {
            secret[0] = 1;
        }

        let galois_layout = crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let galois_keys: Vec<crate::ckks::RnsGaloisKey> = (1..CCMM_DIMENSION)
            .map(|index| 2 * CCMM_SCALAR_DEGREE * index + 1)
            .map(|exponent| {
                let mut rng =
                    ChaCha20Rng::seed_from_u64(0x434d_5400_0000_0000 ^ exponent as u64 ^ seed);

                crate::ckks::RnsGaloisKey::generate_with_ntt_rng(
                    crate::grafting::RnsKeygenConfig {
                        degree: CCMM_LARGE_DEGREE,
                        plaintext_modulus: 2,
                        noise_bound: 0,
                        layout: galois_layout.clone(),
                        plan: &large_plan,
                    },
                    &secret,
                    exponent,
                    &mut rng,
                )
            })
            .collect();

        let prepared_galois_keys: Vec<crate::ckks::PreparedRnsGaloisKey> = galois_keys
            .iter()
            .map(|key| crate::ckks::PreparedRnsGaloisKey::new(key, &large_plan))
            .collect();

        let multiplication_layout =
            crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let mut multiplication_key_rng = ChaCha20Rng::seed_from_u64(0x4343_4d4d_434f_5200 ^ seed);

        let multiplication_key = crate::grafting::RnsMultiplicationKey::generate_with_ntt_rng(
            crate::grafting::RnsKeygenConfig {
                degree: CCMM_LARGE_DEGREE,
                plaintext_modulus: 2,
                noise_bound: 0,
                layout: multiplication_layout,
                plan: &large_plan,
            },
            &secret,
            &mut multiplication_key_rng,
        );

        let prepared_multiplication_key =
            crate::grafting::PreparedRnsMultiplicationKey::new(&multiplication_key, &large_plan);

        let execution = crate::ccmm::batch::BatchCcmmExecutionContext {
            scalar_plan: &scalar_plan,
            prepared_scalar_plan: &prepared_scalar_plan,
            prepared_large_plan: &prepared_large_plan,
            prepared_galois_keys: &prepared_galois_keys,
            prepared_multiplication_key: &prepared_multiplication_key,
            chain: &chain,
            scale: ccmm_scale(),
        };

        let context = DecomposedCcmmExecutionContext {
            basis: &basis,
            large_plan: &large_plan,
            secret: &secret,
            execution: &execution,
            encryption_seed: 0x4343_454e_4352_5950 ^ seed,
        };

        let geometry = BatchGemmGeometry::new(
            crate::eblas::BatchGemmMechanism::Ccmm,
            CCMM_DIMENSION,
            CCMM_SCALAR_DEGREE,
            CCMM_LARGE_DEGREE,
        );

        let actual = correlate_1d_cc(shape, geometry, &signal, &kernel, &context);

        println!(
            "CORRELATION_CC_CASE=N{} K{} OUT{}",
            signal_length,
            kernel_length,
            shape.output_length()
        );

        assert_close(&actual, &expected, 1.0e-3, "CC");
    }

    #[test]
    fn cc_small_semantic_case_matches_pp() {
        run_cc_case(8, 3, 21, 31, 0x5238_2001);
    }

    #[test]
    fn cc_output_boundary_decomposition_matches_pp() {
        // 66 x 5 * 5 x 1: two native output-row products at d=64.
        run_cc_case(70, 5, 22, 32, 0x5238_2002);
    }

    #[test]
    fn cc_reduction_decomposition_matches_pp() {
        // 64 x 67 * 67 x 1: two K products accumulated into one output tile.
        run_cc_case(130, 67, 23, 33, 0x5238_2003);
    }

    #[test]
    fn correlation_2d_shape_exposes_valid_geometry() {
        let shape = Correlation2dShape::new(NhwcShape::new(1, 4, 5, 2), 2, 3, 4);

        assert_eq!(shape.output(), NhwcShape::new(1, 3, 3, 4));
        assert_eq!(shape.rows(), 9);
        assert_eq!(shape.inner(), 12);
        assert_eq!(
            shape.gemm_shape(),
            GemmShape::new(MatrixShape::new(9, 12), MatrixShape::new(12, 4),)
        );
    }

    #[test]
    fn correlation_2d_pp_matches_exact_single_channel_reference() {
        let shape = Correlation2dShape::new(NhwcShape::new(1, 3, 3, 1), 2, 2, 1);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // [Kh=2, Kw=2, C=1, F=1], supplied without reversal:
        //
        // [1 2]
        // [3 4]
        let filters = vec![1.0, 2.0, 3.0, 4.0];

        let output = correlate_2d_pp(shape, &input, &filters);

        assert_eq!(shape.output(), NhwcShape::new(1, 2, 2, 1));
        assert_eq!(output, vec![37.0, 47.0, 67.0, 77.0]);
    }

    #[test]
    fn correlation_2d_pp_preserves_channels_and_multiple_filters() {
        let shape = Correlation2dShape::new(NhwcShape::new(1, 2, 2, 2), 1, 1, 2);

        // Four NHWC pixels:
        // [1,10], [2,20], [3,30], [4,40].
        let input = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];

        // [Kh=1,Kw=1,C=2,F=2]:
        //
        // filter 0 = [1,2]
        // filter 1 = [10,20]
        let filters = vec![1.0, 10.0, 2.0, 20.0];

        let output = correlate_2d_pp(shape, &input, &filters);

        assert_eq!(
            output,
            vec![21.0, 210.0, 42.0, 420.0, 63.0, 630.0, 84.0, 840.0,]
        );
    }

    #[test]
    fn correlation_2d_pp_flattens_multiple_nhwc_batches_into_logical_rows() {
        let shape = Correlation2dShape::new(NhwcShape::new(2, 1, 1, 2), 1, 1, 1);

        // batch 0 pixel [1,2], batch 1 pixel [3,4].
        let input = vec![1.0, 2.0, 3.0, 4.0];

        // One 1x1 two-channel filter [10,1].
        let filters = vec![10.0, 1.0];

        let output = correlate_2d_pp(shape, &input, &filters);

        assert_eq!(shape.rows(), 2);
        assert_eq!(output, vec![12.0, 34.0]);
    }

    #[test]
    #[should_panic(expected = "kernel height cannot exceed input height")]
    fn correlation_2d_shape_rejects_tall_kernel() {
        let _ = Correlation2dShape::new(NhwcShape::new(1, 2, 3, 1), 3, 1, 1);
    }

    #[test]
    #[should_panic(expected = "kernel width cannot exceed input width")]
    fn correlation_2d_shape_rejects_wide_kernel() {
        let _ = Correlation2dShape::new(NhwcShape::new(1, 3, 2, 1), 1, 3, 1);
    }

    fn deterministic_2d_values(length: usize, salt: usize) -> Vec<f64> {
        (0..length)
            .map(|index| {
                let x = (index * 17 + salt * 19) % 41;
                (x as f64 - 20.0) / 128.0
            })
            .collect()
    }

    fn assert_slice_close(actual: &[f64], expected: &[f64], tolerance: f64, metric_prefix: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "2D correlation output lengths must match"
        );

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for (&observed, &reference) in actual.iter().zip(expected) {
            let error = observed - reference;
            squared_error += error * error;
            squared_reference += reference * reference;
            max_abs = max_abs.max(error.abs());
        }

        let rel_l2 = if squared_reference > 0.0 {
            (squared_error / squared_reference).sqrt()
        } else {
            squared_error.sqrt()
        };

        println!(
            "CORRELATION_2D_{metric_prefix}_REL_L2={rel_l2:.12e} \
             CORRELATION_2D_{metric_prefix}_MAX_ABS={max_abs:.12e}"
        );

        assert!(
            rel_l2 <= tolerance,
            "encrypted 2D correlation relative L2 error {rel_l2:e} exceeds tolerance {tolerance:e}"
        );
        assert!(
            max_abs <= tolerance,
            "encrypted 2D correlation max abs error {max_abs:e} exceeds tolerance {tolerance:e}"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn run_cp_2d_case(
        input_height: usize,
        input_width: usize,
        channels: usize,
        kernel_height: usize,
        kernel_width: usize,
        filters: usize,
        input_salt: usize,
        filter_salt: usize,
        seed: u64,
    ) {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        let shape = Correlation2dShape::new(
            NhwcShape::new(1, input_height, input_width, channels),
            kernel_height,
            kernel_width,
            filters,
        );

        let input = deterministic_2d_values(shape.input().elements(), input_salt);

        let filter_values = deterministic_2d_values(
            shape
                .inner()
                .checked_mul(shape.filters())
                .expect("2D correlation filter element count overflow"),
            filter_salt,
        );

        let expected = correlate_2d_pp(shape, &input, &filter_values);

        let basis = crate::ring::ModulusBasis::new(cpmm_moduli());
        let chain = crate::ring::ModulusChain::from_top_basis(basis.clone());

        let large_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), CPMM_LARGE_DEGREE);

        let scalar_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), CPMM_SCALAR_DEGREE);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4350_3244_434f_5200 ^ seed);

        let mut secret: Vec<i8> = (0..CPMM_LARGE_DEGREE)
            .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
            .collect();

        if secret.iter().all(|&value| value == 0) {
            secret[0] = 1;
        }

        let geometry = BatchGemmGeometry::new(
            crate::eblas::BatchGemmMechanism::Cpmm,
            CPMM_DIMENSION,
            CPMM_SCALAR_DEGREE,
            CPMM_LARGE_DEGREE,
        );

        let context = DecomposedCpmmExecutionContext {
            basis: &basis,
            chain: &chain,
            large_plan: &large_plan,
            scalar_plan: &scalar_plan,
            secret: &secret,
            scale: cpmm_scale(),
            encryption_seed: seed ^ 0x3244_4350_454e_4352,
        };

        let actual = correlate_2d_cp(shape, geometry, &input, &filter_values, &context);

        println!(
            "CORRELATION_2D_CP_CASE=\
             H{} W{} C{} KH{} KW{} F{} M{} K{} N{}",
            input_height,
            input_width,
            channels,
            kernel_height,
            kernel_width,
            filters,
            shape.rows(),
            shape.inner(),
            shape.filters(),
        );

        assert_slice_close(&actual, &expected, 1.0e-3, "CP");
    }

    #[test]
    fn cp_2d_semantic_channels_filters_matches_pp() {
        // Logical GEMM:
        //
        //   M = 2*2 = 4
        //   K = 1*1*2 = 2
        //   N = 2
        //
        // Exercises channels and multiple filters without decomposition.
        run_cp_2d_case(2, 2, 2, 1, 1, 2, 41, 51, 0x5238_3001);
    }

    #[test]
    fn cp_2d_m_decomposition_matches_pp() {
        // Valid 1x1 correlation:
        //
        //   input = 66 x 1 x 1
        //   M = 66
        //   K = 1
        //   N = 1
        //
        // d=64 therefore forces two output-row tiles.
        run_cp_2d_case(66, 1, 1, 1, 1, 1, 42, 52, 0x5238_3002);
    }

    #[test]
    fn cp_2d_k_decomposition_matches_pp() {
        // One valid spatial position with a 9x9 kernel:
        //
        //   M = 1
        //   K = 9*9*1 = 81
        //   N = 1
        //
        // d=64 therefore forces two reduction products.
        run_cp_2d_case(9, 9, 1, 9, 9, 1, 43, 53, 0x5238_3003);
    }

    #[test]
    fn cp_2d_n_decomposition_matches_pp() {
        // One input value evaluated against 65 filters:
        //
        //   M = 1
        //   K = 1
        //   N = 65
        //
        // d=64 therefore forces two output-column tiles.
        run_cp_2d_case(1, 1, 1, 1, 1, 65, 44, 54, 0x5238_3004);
    }
}
