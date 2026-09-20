//! Tensor-to-matrix mapping contracts for eBLAS.
//!
//! These mappings define index/shape transformations only. They do not imply
//! CKKS SIMD packing, ciphertext rotations, rescaling, encryption, or any
//! other cryptographic operation.

use crate::matrix::BatchMatrix;

/// Logical rank-3 tensor shape `[batch, rows, cols]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorBatchShape {
    batches: usize,
    rows: usize,
    cols: usize,
}

impl TensorBatchShape {
    /// Creates a non-empty rank-3 tensor shape.
    pub fn new(batches: usize, rows: usize, cols: usize) -> Self {
        assert!(batches > 0, "tensor batch count must be positive");
        assert!(rows > 0, "tensor row count must be positive");
        assert!(cols > 0, "tensor column count must be positive");
        let _ = batches
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(cols))
            .expect("tensor element count overflow");
        Self {
            batches,
            rows,
            cols,
        }
    }

    pub const fn batches(self) -> usize {
        self.batches
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn cols(self) -> usize {
        self.cols
    }

    pub fn elements(self) -> usize {
        self.batches
            .checked_mul(self.rows)
            .and_then(|v| v.checked_mul(self.cols))
            .expect("tensor element count overflow")
    }
}

/// Logical rank-4 NHWC tensor shape `[batch, height, width, channels]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NhwcShape {
    batches: usize,
    height: usize,
    width: usize,
    channels: usize,
}

impl NhwcShape {
    /// Creates a non-empty NHWC tensor shape.
    pub fn new(batches: usize, height: usize, width: usize, channels: usize) -> Self {
        assert!(batches > 0, "NHWC batch count must be positive");
        assert!(height > 0, "NHWC height must be positive");
        assert!(width > 0, "NHWC width must be positive");
        assert!(channels > 0, "NHWC channel count must be positive");
        let _ = batches
            .checked_mul(height)
            .and_then(|v| v.checked_mul(width))
            .and_then(|v| v.checked_mul(channels))
            .expect("NHWC element count overflow");
        Self {
            batches,
            height,
            width,
            channels,
        }
    }

    pub const fn batches(self) -> usize {
        self.batches
    }

    pub const fn height(self) -> usize {
        self.height
    }

    pub const fn width(self) -> usize {
        self.width
    }

    pub const fn channels(self) -> usize {
        self.channels
    }

    pub fn spatial_rows(self) -> usize {
        self.batches
            .checked_mul(self.height)
            .and_then(|v| v.checked_mul(self.width))
            .expect("NHWC flattened row count overflow")
    }

    pub fn elements(self) -> usize {
        self.spatial_rows()
            .checked_mul(self.channels)
            .expect("NHWC element count overflow")
    }
}

/// Shape descriptor for same-shape im2col lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Im2ColShape {
    input: NhwcShape,
    kernel_height: usize,
    kernel_width: usize,
    filters: usize,
}

impl Im2ColShape {
    pub fn new(
        input: NhwcShape,
        kernel_height: usize,
        kernel_width: usize,
        filters: usize,
    ) -> Self {
        assert!(kernel_height > 0, "kernel height must be positive");
        assert!(kernel_width > 0, "kernel width must be positive");
        assert!(filters > 0, "filter count must be positive");

        let _ = kernel_height
            .checked_mul(kernel_width)
            .and_then(|v| v.checked_mul(input.channels()))
            .expect("im2col inner dimension overflow");

        Self {
            input,
            kernel_height,
            kernel_width,
            filters,
        }
    }

    pub const fn input(self) -> NhwcShape {
        self.input
    }

    pub fn rows(self) -> usize {
        self.input.spatial_rows()
    }

    pub fn inner(self) -> usize {
        self.kernel_height
            .checked_mul(self.kernel_width)
            .and_then(|v| v.checked_mul(self.input.channels()))
            .expect("im2col inner dimension overflow")
    }

    pub const fn filters(self) -> usize {
        self.filters
    }

    /// Output shape for same-spatial-size lowering.
    pub fn output(self) -> NhwcShape {
        NhwcShape::new(
            self.input.batches(),
            self.input.height(),
            self.input.width(),
            self.filters,
        )
    }
}

/// Maps a logical `[B,M,N]` tensor directly to the existing batch-matrix
/// representation. This is layout-preserving and performs no data movement.
pub fn tensor3_to_batch_matrix<T>(shape: TensorBatchShape, data: Vec<T>) -> BatchMatrix<T> {
    assert_eq!(
        data.len(),
        shape.elements(),
        "tensor data length does not match shape"
    );
    BatchMatrix::from_vec_column_major(shape.rows(), shape.cols(), shape.batches(), data)
}

/// Converts a batch matrix back to logical `[B,M,N]` shape and storage.
///
/// The returned vector preserves the repository's canonical batch-major,
/// column-major-within-batch storage.
pub fn batch_matrix_to_tensor3<T: Clone>(matrix: &BatchMatrix<T>) -> (TensorBatchShape, Vec<T>) {
    (
        TensorBatchShape::new(matrix.batches(), matrix.rows(), matrix.cols()),
        matrix.raw().to_vec(),
    )
}

/// Flattens NHWC storage to a single matrix `[B*H*W, C]`.
///
/// Input data is interpreted in canonical NHWC logical order:
/// `(((b * H + h) * W + w) * C + c)`.
/// The returned matrix is stored column-major as required by eBLAS.
pub fn flatten_nhwc_to_matrix<T: Clone>(shape: NhwcShape, nhwc: &[T]) -> BatchMatrix<T> {
    assert_eq!(
        nhwc.len(),
        shape.elements(),
        "NHWC data length does not match shape"
    );

    let rows = shape.spatial_rows();
    let cols = shape.channels();
    let mut data = Vec::with_capacity(nhwc.len());

    for col in 0..cols {
        for row in 0..rows {
            let b = row / (shape.height() * shape.width());
            let rem = row % (shape.height() * shape.width());
            let h = rem / shape.width();
            let w = rem % shape.width();
            let index = (((b * shape.height() + h) * shape.width() + w) * cols) + col;
            data.push(nhwc[index].clone());
        }
    }

    BatchMatrix::from_vec_column_major(rows, cols, 1, data)
}

/// Restores canonical NHWC storage from a `[B*H*W, C]` matrix.
pub fn unflatten_matrix_to_nhwc<T: Clone>(shape: NhwcShape, matrix: &BatchMatrix<T>) -> Vec<T> {
    assert_eq!(
        matrix.batches(),
        1,
        "flattened NHWC matrix must have one batch"
    );
    assert_eq!(
        matrix.rows(),
        shape.spatial_rows(),
        "flattened NHWC row count mismatch"
    );
    assert_eq!(
        matrix.cols(),
        shape.channels(),
        "flattened NHWC column count mismatch"
    );

    let mut out: Vec<Option<T>> = vec![None; shape.elements()];

    for col in 0..matrix.cols() {
        for row in 0..matrix.rows() {
            let b = row / (shape.height() * shape.width());
            let rem = row % (shape.height() * shape.width());
            let h = rem / shape.width();
            let w = rem % shape.width();
            let index = (((b * shape.height() + h) * shape.width() + w) * shape.channels()) + col;
            out[index] = Some(matrix.get(0, row, col).clone());
        }
    }

    out.into_iter()
        .map(|value| value.expect("NHWC unflatten left an unmapped element"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor3_roundtrip_preserves_shape_and_storage() {
        let shape = TensorBatchShape::new(2, 2, 3);
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let matrix = tensor3_to_batch_matrix(shape, data.clone());
        let (decoded_shape, decoded) = batch_matrix_to_tensor3(&matrix);
        assert_eq!(decoded_shape, shape);
        assert_eq!(decoded, data);
    }

    #[test]
    fn nhwc_flatten_roundtrip_is_exact() {
        let shape = NhwcShape::new(2, 2, 2, 3);
        let data: Vec<i64> = (0..shape.elements() as i64).collect();

        let matrix = flatten_nhwc_to_matrix(shape, &data);
        assert_eq!(matrix.rows(), 8);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.batches(), 1);

        let recovered = unflatten_matrix_to_nhwc(shape, &matrix);
        assert_eq!(recovered, data);
    }

    #[test]
    fn nhwc_flatten_uses_expected_feature_rows() {
        let shape = NhwcShape::new(1, 2, 2, 2);
        // NHWC rows: [10,11], [20,21], [30,31], [40,41]
        let data = vec![10, 11, 20, 21, 30, 31, 40, 41];
        let matrix = flatten_nhwc_to_matrix(shape, &data);

        // Column-major 4x2 matrix:
        // first feature column [10,20,30,40], then [11,21,31,41].
        assert_eq!(matrix.raw(), &[10, 20, 30, 40, 11, 21, 31, 41]);
    }

    #[test]
    fn im2col_shape_matches_existing_cleartext_workload_convention() {
        let input = NhwcShape::new(1, 32, 32, 3);
        let mapping = Im2ColShape::new(input, 3, 3, 64);

        assert_eq!(mapping.rows(), 1024);
        assert_eq!(mapping.inner(), 27);
        assert_eq!(mapping.filters(), 64);
        assert_eq!(mapping.output(), NhwcShape::new(1, 32, 32, 64));
    }

    #[test]
    fn pointwise_convolution_lowering_shape_is_matrix_gemm() {
        let input = NhwcShape::new(2, 4, 4, 8);
        assert_eq!(input.spatial_rows(), 32);

        // A 1x1 convolution lowers to [B*H*W,C] * [C,F].
        let lowered_lhs = (input.spatial_rows(), input.channels());
        let filters = 16;
        let lowered_rhs = (input.channels(), filters);
        let lowered_out = (input.spatial_rows(), filters);

        assert_eq!(lowered_lhs, (32, 8));
        assert_eq!(lowered_rhs, (8, 16));
        assert_eq!(lowered_out, (32, 16));
    }

    #[test]
    #[should_panic(expected = "tensor data length")]
    fn tensor3_rejects_wrong_data_length() {
        let _ = tensor3_to_batch_matrix(TensorBatchShape::new(2, 2, 2), vec![1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "NHWC data length")]
    fn nhwc_flatten_rejects_wrong_data_length() {
        let _ = flatten_nhwc_to_matrix(NhwcShape::new(1, 2, 2, 3), &[1, 2, 3]);
    }
}
