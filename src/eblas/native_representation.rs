//! Cleartext native-tile representation helpers for eBLAS GEMM.
//!
//! This module materializes the shape-only [`NativeGemmTileMapping`] contract
//! using ordinary [`BatchMatrix<f64>`] values. It performs extraction,
//! upper-left zero padding, and logical-output cropping only.
//!
//! No SinC packing, encryption, CKKS encoding, or Batch CPMM/CCMM execution is
//! performed here.

use crate::eblas::NativeGemmTileMapping;
use crate::matrix::BatchMatrix;

/// Extracts and zero-pads the logical left-hand tile into one native square
/// matrix.
///
/// `lhs` is the complete logical GEMM left operand. The tile location is
/// determined by the decomposition coordinates stored in `mapping`.
pub fn pad_native_lhs_tile(
    mapping: NativeGemmTileMapping,
    lhs: &BatchMatrix<f64>,
) -> BatchMatrix<f64> {
    assert_eq!(
        lhs.batches(),
        1,
        "eBLAS native GEMM representation currently requires one lhs batch"
    );

    let product = mapping.product();
    let logical = mapping.lhs_logical();
    let d = mapping.native_dimension();

    let row_start = product
        .output_row()
        .checked_mul(d)
        .expect("eBLAS native GEMM lhs row offset overflow");

    let col_start = product
        .reduction()
        .checked_mul(d)
        .expect("eBLAS native GEMM lhs column offset overflow");

    assert!(
        row_start + logical.rows() <= lhs.rows(),
        "eBLAS native GEMM lhs tile exceeds logical matrix rows"
    );
    assert!(
        col_start + logical.cols() <= lhs.cols(),
        "eBLAS native GEMM lhs tile exceeds logical matrix columns"
    );

    let mut native = BatchMatrix::<f64>::new(d, d, 1);

    for col in 0..logical.cols() {
        for row in 0..logical.rows() {
            native.set(0, row, col, *lhs.get(0, row_start + row, col_start + col));
        }
    }

    native
}

/// Extracts and zero-pads the logical right-hand tile into one native square
/// matrix.
///
/// `rhs` is the complete logical GEMM right operand. The tile location is
/// determined by the decomposition coordinates stored in `mapping`.
pub fn pad_native_rhs_tile(
    mapping: NativeGemmTileMapping,
    rhs: &BatchMatrix<f64>,
) -> BatchMatrix<f64> {
    assert_eq!(
        rhs.batches(),
        1,
        "eBLAS native GEMM representation currently requires one rhs batch"
    );

    let product = mapping.product();
    let logical = mapping.rhs_logical();
    let d = mapping.native_dimension();

    let row_start = product
        .reduction()
        .checked_mul(d)
        .expect("eBLAS native GEMM rhs row offset overflow");

    let col_start = product
        .output_col()
        .checked_mul(d)
        .expect("eBLAS native GEMM rhs column offset overflow");

    assert!(
        row_start + logical.rows() <= rhs.rows(),
        "eBLAS native GEMM rhs tile exceeds logical matrix rows"
    );
    assert!(
        col_start + logical.cols() <= rhs.cols(),
        "eBLAS native GEMM rhs tile exceeds logical matrix columns"
    );

    let mut native = BatchMatrix::<f64>::new(d, d, 1);

    for col in 0..logical.cols() {
        for row in 0..logical.rows() {
            native.set(0, row, col, *rhs.get(0, row_start + row, col_start + col));
        }
    }

    native
}

/// Crops one native square output contribution back to its logical tile shape.
///
/// The returned matrix contains one batch and starts at logical tile-local
/// coordinate `(0, 0)`. Placement into the full logical output belongs to the
/// later assembly layer.
pub fn crop_native_output_tile(
    mapping: NativeGemmTileMapping,
    native_output: &BatchMatrix<f64>,
) -> BatchMatrix<f64> {
    assert_eq!(
        native_output.batches(),
        1,
        "eBLAS native GEMM output crop currently requires one batch"
    );

    let d = mapping.native_dimension();

    assert_eq!(
        native_output.rows(),
        d,
        "eBLAS native GEMM output row count must match native dimension"
    );
    assert_eq!(
        native_output.cols(),
        d,
        "eBLAS native GEMM output column count must match native dimension"
    );

    let logical = mapping.output_logical();

    let mut cropped = BatchMatrix::<f64>::new(logical.rows(), logical.cols(), 1);

    for col in 0..logical.cols() {
        for row in 0..logical.rows() {
            cropped.set(0, row, col, *native_output.get(0, row, col));
        }
    }

    cropped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eblas::{
        gemm_pp, GemmDecompositionPlan, GemmShape, GemmSpec, MatrixShape, NativeGemmTileMapping,
        PrivacyMode,
    };

    fn logical_matrix(rows: usize, cols: usize, bias: f64) -> BatchMatrix<f64> {
        let mut matrix = BatchMatrix::<f64>::new(rows, cols, 1);

        for col in 0..cols {
            for row in 0..rows {
                matrix.set(0, row, col, bias + row as f64 * 0.25 + col as f64 * 0.5);
            }
        }

        matrix
    }

    fn direct_tile_reference(
        lhs: &BatchMatrix<f64>,
        rhs: &BatchMatrix<f64>,
        mapping: NativeGemmTileMapping,
    ) -> BatchMatrix<f64> {
        let product = mapping.product();
        let d = mapping.native_dimension();

        let lhs_row_start = product.output_row() * d;
        let lhs_col_start = product.reduction() * d;

        let rhs_row_start = product.reduction() * d;
        let rhs_col_start = product.output_col() * d;

        let lhs_shape = mapping.lhs_logical();
        let rhs_shape = mapping.rhs_logical();

        let mut lhs_tile = BatchMatrix::<f64>::new(lhs_shape.rows(), lhs_shape.cols(), 1);

        let mut rhs_tile = BatchMatrix::<f64>::new(rhs_shape.rows(), rhs_shape.cols(), 1);

        for col in 0..lhs_shape.cols() {
            for row in 0..lhs_shape.rows() {
                lhs_tile.set(
                    0,
                    row,
                    col,
                    *lhs.get(0, lhs_row_start + row, lhs_col_start + col),
                );
            }
        }

        for col in 0..rhs_shape.cols() {
            for row in 0..rhs_shape.rows() {
                rhs_tile.set(
                    0,
                    row,
                    col,
                    *rhs.get(0, rhs_row_start + row, rhs_col_start + col),
                );
            }
        }

        gemm_pp(
            GemmSpec::new(product.gemm(), PrivacyMode::Pp),
            &lhs_tile,
            &rhs_tile,
        )
    }

    #[test]
    fn exact_native_tiles_roundtrip_without_padding_effect() {
        let shape = GemmShape::new(MatrixShape::new(4, 4), MatrixShape::new(4, 4));

        let plan = GemmDecompositionPlan::new(shape, 4);
        let mapping = NativeGemmTileMapping::new(plan.product(0, 0, 0), 4);

        let lhs = logical_matrix(4, 4, 1.0);
        let rhs = logical_matrix(4, 4, -0.5);

        let native_lhs = pad_native_lhs_tile(mapping, &lhs);
        let native_rhs = pad_native_rhs_tile(mapping, &rhs);

        assert_eq!(native_lhs, lhs);
        assert_eq!(native_rhs, rhs);

        let native_result = gemm_pp(
            GemmSpec::new(
                GemmShape::new(MatrixShape::new(4, 4), MatrixShape::new(4, 4)),
                PrivacyMode::Pp,
            ),
            &native_lhs,
            &native_rhs,
        );

        let cropped = crop_native_output_tile(mapping, &native_result);

        let expected = direct_tile_reference(&lhs, &rhs, mapping);

        assert_eq!(cropped, expected);
    }

    #[test]
    fn boundary_tile_padding_preserves_logical_product() {
        let shape = GemmShape::new(MatrixShape::new(6, 7), MatrixShape::new(7, 5));

        let plan = GemmDecompositionPlan::new(shape, 4);
        let mapping = NativeGemmTileMapping::new(plan.product(1, 1, 1), 4);

        assert_eq!(mapping.lhs_logical(), MatrixShape::new(2, 3));
        assert_eq!(mapping.rhs_logical(), MatrixShape::new(3, 1));
        assert_eq!(mapping.output_logical(), MatrixShape::new(2, 1));

        let lhs = logical_matrix(6, 7, 0.75);
        let rhs = logical_matrix(7, 5, -1.25);

        let native_lhs = pad_native_lhs_tile(mapping, &lhs);
        let native_rhs = pad_native_rhs_tile(mapping, &rhs);

        assert_eq!(native_lhs.rows(), 4);
        assert_eq!(native_lhs.cols(), 4);
        assert_eq!(native_rhs.rows(), 4);
        assert_eq!(native_rhs.cols(), 4);

        for col in 0..4 {
            for row in 0..4 {
                if row >= 2 || col >= 3 {
                    assert_eq!(*native_lhs.get(0, row, col), 0.0);
                }

                if row >= 3 || col >= 1 {
                    assert_eq!(*native_rhs.get(0, row, col), 0.0);
                }
            }
        }

        let native_result = gemm_pp(
            GemmSpec::new(
                GemmShape::new(MatrixShape::new(4, 4), MatrixShape::new(4, 4)),
                PrivacyMode::Pp,
            ),
            &native_lhs,
            &native_rhs,
        );

        let cropped = crop_native_output_tile(mapping, &native_result);

        let expected = direct_tile_reference(&lhs, &rhs, mapping);

        assert_eq!(cropped, expected);
    }

    #[test]
    fn different_reduction_tiles_extract_different_regions() {
        let shape = GemmShape::new(MatrixShape::new(3, 6), MatrixShape::new(6, 2));

        let plan = GemmDecompositionPlan::new(shape, 4);

        let lhs = logical_matrix(3, 6, 0.0);
        let rhs = logical_matrix(6, 2, 1.0);

        let first = NativeGemmTileMapping::new(plan.product(0, 0, 0), 4);
        let second = NativeGemmTileMapping::new(plan.product(0, 1, 0), 4);

        let first_lhs = pad_native_lhs_tile(first, &lhs);
        let second_lhs = pad_native_lhs_tile(second, &lhs);

        assert_ne!(first_lhs.get(0, 0, 0), second_lhs.get(0, 0, 0));

        let first_rhs = pad_native_rhs_tile(first, &rhs);
        let second_rhs = pad_native_rhs_tile(second, &rhs);

        assert_ne!(first_rhs.get(0, 0, 0), second_rhs.get(0, 0, 0));
    }
}
