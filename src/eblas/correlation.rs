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
    gemm_cp_decomposed, gemm_pp, BatchGemmGeometry, DecomposedCpmmExecutionContext, GemmShape,
    GemmSpec, MatrixShape, PrivacyMode,
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

    fn assert_close(actual: &BatchMatrix<f64>, expected: &BatchMatrix<f64>, tolerance: f64) {
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

        println!("CORRELATION_CP_REL_L2={rel_l2:.12e} CORRELATION_CP_MAX_ABS={max_abs:.12e}");

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

        assert_close(&actual, &expected, 1.0e-3);
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
}
