//! Cleartext execution of decomposed eBLAS GEMM.
//!
//! This module composes the deterministic decomposition, native tile mapping,
//! cleartext native representation, reference native GEMM, reduction
//! accumulation, and logical output assembly.
//!
//! It is intentionally representation-neutral with respect to encrypted
//! execution: no SinC packing, CKKS encoding, CPMM, or CCMM is performed here.

use crate::eblas::{
    crop_native_output_tile, gemm_pp, pad_native_lhs_tile, pad_native_rhs_tile,
    GemmDecompositionPlan, GemmShape, GemmSpec, MatrixShape, NativeGemmTileMapping, PrivacyMode,
};
use crate::matrix::BatchMatrix;

/// Executes one plaintext/plaintext GEMM through square native decomposition.
///
/// Every logical tile product is extracted and zero-padded to
/// `tile_dimension x tile_dimension`, evaluated through the existing reference
/// GEMM path, cropped back to its logical contribution, accumulated across the
/// reduction dimension, and assembled into the full logical output.
pub fn gemm_pp_decomposed(
    shape: GemmShape,
    tile_dimension: usize,
    lhs: &BatchMatrix<f64>,
    rhs: &BatchMatrix<f64>,
) -> BatchMatrix<f64> {
    assert_eq!(
        lhs.batches(),
        1,
        "eBLAS decomposed GEMM PP currently requires one lhs batch"
    );
    assert_eq!(
        rhs.batches(),
        1,
        "eBLAS decomposed GEMM PP currently requires one rhs batch"
    );

    assert_eq!(lhs.rows(), shape.lhs().rows());
    assert_eq!(lhs.cols(), shape.lhs().cols());
    assert_eq!(rhs.rows(), shape.rhs().rows());
    assert_eq!(rhs.cols(), shape.rhs().cols());

    let plan = GemmDecompositionPlan::new(shape, tile_dimension);

    let mut output = BatchMatrix::<f64>::new(shape.output().rows(), shape.output().cols(), 1);

    for output_row in 0..plan.row_tiles() {
        for output_col in 0..plan.col_tiles() {
            for reduction in 0..plan.reduction_tiles() {
                let product = plan.product(output_row, reduction, output_col);
                let mapping = NativeGemmTileMapping::new(product, tile_dimension);

                let native_lhs = pad_native_lhs_tile(mapping, lhs);
                let native_rhs = pad_native_rhs_tile(mapping, rhs);

                let native_shape = GemmShape::new(
                    MatrixShape::new(tile_dimension, tile_dimension),
                    MatrixShape::new(tile_dimension, tile_dimension),
                );

                let native_output = gemm_pp(
                    GemmSpec::new(native_shape, PrivacyMode::Pp),
                    &native_lhs,
                    &native_rhs,
                );

                let contribution = crop_native_output_tile(mapping, &native_output);

                let row_start = output_row
                    .checked_mul(tile_dimension)
                    .expect("eBLAS decomposed GEMM output row offset overflow");

                let col_start = output_col
                    .checked_mul(tile_dimension)
                    .expect("eBLAS decomposed GEMM output column offset overflow");

                for col in 0..contribution.cols() {
                    for row in 0..contribution.rows() {
                        let output_row_index = row_start + row;
                        let output_col_index = col_start + col;

                        let accumulated = *output.get(0, output_row_index, output_col_index)
                            + *contribution.get(0, row, col);

                        output.set(0, output_row_index, output_col_index, accumulated);
                    }
                }
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, cols: usize, bias: f64) -> BatchMatrix<f64> {
        let mut matrix = BatchMatrix::<f64>::new(rows, cols, 1);

        for col in 0..cols {
            for row in 0..rows {
                matrix.set(
                    0,
                    row,
                    col,
                    bias + row as f64 * 0.125 - col as f64 * 0.0625 + (row * col) as f64 * 0.03125,
                );
            }
        }

        matrix
    }

    fn direct(
        shape: GemmShape,
        lhs: &BatchMatrix<f64>,
        rhs: &BatchMatrix<f64>,
    ) -> BatchMatrix<f64> {
        gemm_pp(GemmSpec::new(shape, PrivacyMode::Pp), lhs, rhs)
    }

    fn assert_close(actual: &BatchMatrix<f64>, expected: &BatchMatrix<f64>, tolerance: f64) {
        assert_eq!(actual.rows(), expected.rows());
        assert_eq!(actual.cols(), expected.cols());
        assert_eq!(actual.batches(), expected.batches());

        for col in 0..actual.cols() {
            for row in 0..actual.rows() {
                let a = *actual.get(0, row, col);
                let e = *expected.get(0, row, col);
                let error = (a - e).abs();

                assert!(
                    error <= tolerance,
                    "decomposed GEMM mismatch at ({row}, {col}): \
                     actual={a}, expected={e}, error={error}"
                );
            }
        }
    }

    #[test]
    fn exact_single_tile_matches_direct_gemm() {
        let shape = GemmShape::new(MatrixShape::new(4, 4), MatrixShape::new(4, 4));

        let lhs = matrix(4, 4, 0.5);
        let rhs = matrix(4, 4, -0.25);

        let actual = gemm_pp_decomposed(shape, 4, &lhs, &rhs);
        let expected = direct(shape, &lhs, &rhs);

        assert_eq!(actual, expected);
    }

    #[test]
    fn rectangular_multi_tile_matches_direct_gemm() {
        let shape = GemmShape::new(MatrixShape::new(6, 7), MatrixShape::new(7, 5));

        let lhs = matrix(6, 7, 0.75);
        let rhs = matrix(7, 5, -1.0);

        let actual = gemm_pp_decomposed(shape, 4, &lhs, &rhs);
        let expected = direct(shape, &lhs, &rhs);

        assert_close(&actual, &expected, 1.0e-12);
    }

    #[test]
    fn irregular_multi_tile_matches_direct_gemm() {
        let shape = GemmShape::new(MatrixShape::new(13, 11), MatrixShape::new(11, 9));

        let lhs = matrix(13, 11, -0.375);
        let rhs = matrix(11, 9, 0.625);

        let actual = gemm_pp_decomposed(shape, 4, &lhs, &rhs);
        let expected = direct(shape, &lhs, &rhs);

        assert_close(&actual, &expected, 1.0e-12);
    }

    #[test]
    fn reduction_only_decomposition_matches_direct_gemm() {
        let shape = GemmShape::new(MatrixShape::new(3, 10), MatrixShape::new(10, 2));

        let lhs = matrix(3, 10, 1.25);
        let rhs = matrix(10, 2, -0.75);

        let actual = gemm_pp_decomposed(shape, 4, &lhs, &rhs);
        let expected = direct(shape, &lhs, &rhs);

        assert_close(&actual, &expected, 1.0e-12);
    }
}
