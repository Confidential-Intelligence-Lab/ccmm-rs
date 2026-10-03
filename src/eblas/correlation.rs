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

    let mut output = BatchMatrix::<f64>::new(shape.output_length(), 1, 1);

    for position in 0..shape.output_length() {
        let mut sum = 0.0_f64;

        for offset in 0..shape.kernel_length() {
            sum += *signal.get(0, position + offset, 0) * *kernel.get(0, offset, 0);
        }

        output.set(0, position, 0, sum);
    }

    output
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
}
