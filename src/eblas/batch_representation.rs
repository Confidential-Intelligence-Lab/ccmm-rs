//! Structural representation mapping for scheduled packed eBLAS GEMM work.
//!
//! This layer materializes one [`BatchGemmWorkGroup`] into the physical lane
//! representation required by Batch CPMM or Batch CCMM.
//!
//! Responsibilities:
//!
//! - extract and zero-pad scheduled logical tiles into native d x d matrices;
//! - preserve the deterministic R6 physical lane assignment;
//! - zero-fill unused physical lanes;
//! - apply mechanism-specific structural representation mapping;
//! - perform SinC encoding where that encoding remains independent of CKKS
//!   modulus, scale, keys, and encryption.
//!
//! This layer deliberately performs no RNS quantization, encryption, key
//! preparation, or cryptographic execution. In particular, the CPMM plaintext
//! right operand remains a physical batch of real matrices; its CKKS/RNS
//! plaintext encoding belongs to the later execution layer.

use num_complex::Complex64;

use crate::ccmm::batch::{as_half_row, sinc_encode_batch, SinCBatchPlaintext};
use crate::eblas::{
    pad_native_lhs_tile, pad_native_rhs_tile, BatchGemmGeometry, BatchGemmMechanism,
    BatchGemmWorkGroup,
};
use crate::matrix::BatchMatrix;

/// Structural physical representation of one scheduled Batch GEMM work group.
///
/// Both variants retain the complete physical lane batch. Unused lanes are
/// represented by exact zero matrices rather than synthetic logical products.
#[derive(Debug, Clone)]
pub enum BatchGemmRepresentation {
    /// Ciphertext/plaintext Batch CPMM representation.
    ///
    /// The encrypted lhs is structurally SinC encoded using the authors'
    /// half-row representation. The public rhs remains a real physical matrix
    /// batch until the later execution layer supplies CKKS scale and RNS basis.
    Cpmm {
        geometry: BatchGemmGeometry,
        group_index: usize,
        active_lanes: usize,
        lhs: SinCBatchPlaintext,
        rhs: BatchMatrix<f64>,
    },

    /// Ciphertext/ciphertext Batch CCMM representation.
    ///
    /// Both operands use full-row complex SinC representation.
    Ccmm {
        geometry: BatchGemmGeometry,
        group_index: usize,
        active_lanes: usize,
        lhs: SinCBatchPlaintext,
        rhs: SinCBatchPlaintext,
    },
}

impl BatchGemmRepresentation {
    /// Packed execution geometry represented by this work group.
    pub const fn geometry(&self) -> BatchGemmGeometry {
        match self {
            Self::Cpmm { geometry, .. } | Self::Ccmm { geometry, .. } => *geometry,
        }
    }

    /// Stable R6 work-group index.
    pub const fn group_index(&self) -> usize {
        match self {
            Self::Cpmm { group_index, .. } | Self::Ccmm { group_index, .. } => *group_index,
        }
    }

    /// Number of physical lanes carrying logical work.
    pub const fn active_lanes(&self) -> usize {
        match self {
            Self::Cpmm { active_lanes, .. } | Self::Ccmm { active_lanes, .. } => *active_lanes,
        }
    }

    /// Number of physical lanes available in this representation.
    pub const fn capacity(&self) -> usize {
        self.geometry().batch_count()
    }

    /// Number of exact-zero inactive physical lanes.
    pub const fn unused_lanes(&self) -> usize {
        self.capacity() - self.active_lanes()
    }

    /// CPMM SinC-encoded lhs, when this is a CPMM representation.
    pub fn cpmm_lhs(&self) -> Option<&SinCBatchPlaintext> {
        match self {
            Self::Cpmm { lhs, .. } => Some(lhs),
            Self::Ccmm { .. } => None,
        }
    }

    /// CPMM physical real rhs batch, when this is a CPMM representation.
    pub fn cpmm_rhs(&self) -> Option<&BatchMatrix<f64>> {
        match self {
            Self::Cpmm { rhs, .. } => Some(rhs),
            Self::Ccmm { .. } => None,
        }
    }

    /// CCMM SinC-encoded lhs and rhs, when this is a CCMM representation.
    pub fn ccmm_operands(&self) -> Option<(&SinCBatchPlaintext, &SinCBatchPlaintext)> {
        match self {
            Self::Ccmm { lhs, rhs, .. } => Some((lhs, rhs)),
            Self::Cpmm { .. } => None,
        }
    }
}

fn physical_real_batch(
    group: &BatchGemmWorkGroup,
    lhs: &BatchMatrix<f64>,
    rhs: &BatchMatrix<f64>,
) -> (BatchMatrix<f64>, BatchMatrix<f64>) {
    assert!(
        !group.is_empty(),
        "eBLAS Batch GEMM representation requires non-empty scheduled work"
    );

    let first = group.products()[0];
    let geometry = first.binding().geometry();
    let d = geometry.dimension();
    let capacity = geometry.batch_count();

    assert_eq!(
        group.capacity(),
        capacity,
        "eBLAS Batch GEMM work-group capacity must match geometry"
    );

    let mut physical_lhs = BatchMatrix::<f64>::new(d, d, capacity);
    let mut physical_rhs = BatchMatrix::<f64>::new(d, d, capacity);

    for scheduled in group.products() {
        let binding = scheduled.binding();

        assert_eq!(
            binding.geometry(),
            geometry,
            "eBLAS Batch GEMM work group must use one geometry"
        );

        assert_eq!(
            binding.privacy(),
            geometry.privacy(),
            "eBLAS Batch GEMM representation privacy must match geometry"
        );

        let lane = scheduled.lane();

        assert!(
            lane < capacity,
            "eBLAS Batch GEMM scheduled lane exceeds physical capacity"
        );

        let mapping = binding.mapping();
        let native_lhs = pad_native_lhs_tile(mapping, lhs);
        let native_rhs = pad_native_rhs_tile(mapping, rhs);

        for col in 0..d {
            for row in 0..d {
                physical_lhs.set(lane, row, col, *native_lhs.get(0, row, col));
                physical_rhs.set(lane, row, col, *native_rhs.get(0, row, col));
            }
        }
    }

    (physical_lhs, physical_rhs)
}

fn real_batch_to_nested(batch: &BatchMatrix<f64>) -> Vec<Vec<Vec<f64>>> {
    (0..batch.batches())
        .map(|lane| {
            (0..batch.rows())
                .map(|row| {
                    (0..batch.cols())
                        .map(|col| *batch.get(lane, row, col))
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn real_batch_to_complex(batch: &BatchMatrix<f64>) -> Vec<Vec<Vec<Complex64>>> {
    (0..batch.batches())
        .map(|lane| {
            (0..batch.rows())
                .map(|row| {
                    (0..batch.cols())
                        .map(|col| Complex64::new(*batch.get(lane, row, col), 0.0))
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Materializes one R6 work group into its mechanism-specific physical
/// representation.
///
/// The returned object preserves the R6 group index and active-lane count.
/// Physical lanes not assigned by R6 remain exact zeros.
pub fn represent_batch_gemm_work_group(
    group: &BatchGemmWorkGroup,
    lhs: &BatchMatrix<f64>,
    rhs: &BatchMatrix<f64>,
) -> BatchGemmRepresentation {
    let first = group
        .products()
        .first()
        .expect("eBLAS Batch GEMM representation requires non-empty scheduled work");

    let geometry = first.binding().geometry();

    let (physical_lhs, physical_rhs) = physical_real_batch(group, lhs, rhs);

    match geometry.mechanism() {
        BatchGemmMechanism::Cpmm => {
            let lhs_real = real_batch_to_nested(&physical_lhs);

            let lhs_half: Vec<Vec<Vec<Complex64>>> =
                lhs_real.iter().map(|matrix| as_half_row(matrix)).collect();

            let lhs_sinc = sinc_encode_batch(&lhs_half, geometry.scalar_degree());

            BatchGemmRepresentation::Cpmm {
                geometry,
                group_index: group.group_index(),
                active_lanes: group.len(),
                lhs: lhs_sinc,
                rhs: physical_rhs,
            }
        }
        BatchGemmMechanism::Ccmm => {
            let lhs_complex = real_batch_to_complex(&physical_lhs);
            let rhs_complex = real_batch_to_complex(&physical_rhs);

            let lhs_sinc = sinc_encode_batch(&lhs_complex, geometry.scalar_degree());
            let rhs_sinc = sinc_encode_batch(&rhs_complex, geometry.scalar_degree());

            BatchGemmRepresentation::Ccmm {
                geometry,
                group_index: group.group_index(),
                active_lanes: group.len(),
                lhs: lhs_sinc,
                rhs: rhs_sinc,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccmm::batch::{as_double_row, sinc_decode_batch};
    use crate::eblas::{
        batch_gemm_work_groups, GemmDecompositionPlan, GemmShape, MatrixShape, PrivacyMode,
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

    const STRUCTURAL_TOLERANCE: f64 = 1.0e-12;

    fn assert_real_matrix_eq(
        actual: &[Vec<f64>],
        expected: &BatchMatrix<f64>,
        expected_batch: usize,
    ) {
        assert_eq!(actual.len(), expected.rows());

        for (row, actual_row) in actual.iter().enumerate() {
            assert_eq!(actual_row.len(), expected.cols());

            for (col, &actual_value) in actual_row.iter().enumerate() {
                let expected_value = *expected.get(expected_batch, row, col);
                assert!(
                    (actual_value - expected_value).abs() <= STRUCTURAL_TOLERANCE,
                    "SinC structural round-trip mismatch at ({row}, {col}):                      actual={actual_value:.17e}, expected={expected_value:.17e}"
                );
            }
        }
    }

    fn assert_complex_matrix_eq(
        actual: &[Vec<Complex64>],
        expected: &BatchMatrix<f64>,
        expected_batch: usize,
    ) {
        assert_eq!(actual.len(), expected.rows());

        for (row, actual_row) in actual.iter().enumerate() {
            assert_eq!(actual_row.len(), expected.cols());

            for (col, &actual_value) in actual_row.iter().enumerate() {
                let expected_value = *expected.get(expected_batch, row, col);

                assert!(
                    (actual_value.re - expected_value).abs() <= STRUCTURAL_TOLERANCE,
                    "SinC structural real mismatch at ({row}, {col}):                      actual={:.17e}, expected={expected_value:.17e}",
                    actual_value.re
                );

                assert!(
                    actual_value.im.abs() <= STRUCTURAL_TOLERANCE,
                    "SinC structural imaginary residual at ({row}, {col}): {:.17e}",
                    actual_value.im
                );
            }
        }
    }

    #[test]
    fn cpmm_partial_group_preserves_lanes_and_zero_fills_tail() {
        let geometry = BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 4, 8, 16);
        let shape = GemmShape::new(MatrixShape::new(4, 4), MatrixShape::new(4, 4));
        let plan = GemmDecompositionPlan::new(shape, 4);
        let groups = batch_gemm_work_groups(plan, geometry, PrivacyMode::Cp);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 1);
        assert_eq!(groups[0].capacity(), 4);

        let lhs = logical_matrix(4, 4, 1.0);
        let rhs = logical_matrix(4, 4, -0.5);

        let representation = represent_batch_gemm_work_group(&groups[0], &lhs, &rhs);

        assert_eq!(representation.group_index(), 0);
        assert_eq!(representation.active_lanes(), 1);
        assert_eq!(representation.unused_lanes(), 3);

        let lhs_decoded = sinc_decode_batch(representation.cpmm_lhs().unwrap());

        assert_eq!(lhs_decoded.len(), 4);

        let lane0 = as_double_row(&lhs_decoded[0]);
        assert_real_matrix_eq(&lane0, &lhs, 0);

        for lane in lhs_decoded.iter().skip(1) {
            let unpacked = as_double_row(lane);
            assert!(
                unpacked
                    .iter()
                    .flatten()
                    .all(|&value| value.abs() <= STRUCTURAL_TOLERANCE),
                "inactive CPMM lhs lanes must decode to structural zero"
            );
        }

        let rhs_physical = representation.cpmm_rhs().unwrap();

        assert_eq!(rhs_physical.batches(), 4);
        assert_real_matrix_eq(
            &(0..4)
                .map(|row| (0..4).map(|col| *rhs_physical.get(0, row, col)).collect())
                .collect::<Vec<Vec<f64>>>(),
            &rhs,
            0,
        );

        for lane in 1..4 {
            assert!(
                (0..4).all(|row| (0..4).all(|col| *rhs_physical.get(lane, row, col) == 0.0)),
                "inactive CPMM rhs lanes must remain exact zero"
            );
        }
    }

    #[test]
    fn ccmm_partial_group_preserves_both_operands_and_zero_fills_tail() {
        let geometry = BatchGemmGeometry::new(BatchGemmMechanism::Ccmm, 4, 4, 16);
        let shape = GemmShape::new(MatrixShape::new(4, 4), MatrixShape::new(4, 4));
        let plan = GemmDecompositionPlan::new(shape, 4);
        let groups = batch_gemm_work_groups(plan, geometry, PrivacyMode::Cc);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 1);
        assert_eq!(groups[0].capacity(), 2);

        let lhs = logical_matrix(4, 4, 0.75);
        let rhs = logical_matrix(4, 4, -1.25);

        let representation = represent_batch_gemm_work_group(&groups[0], &lhs, &rhs);

        assert_eq!(representation.active_lanes(), 1);
        assert_eq!(representation.unused_lanes(), 1);

        let (lhs_sinc, rhs_sinc) = representation.ccmm_operands().unwrap();
        let lhs_decoded = sinc_decode_batch(lhs_sinc);
        let rhs_decoded = sinc_decode_batch(rhs_sinc);

        assert_complex_matrix_eq(&lhs_decoded[0], &lhs, 0);
        assert_complex_matrix_eq(&rhs_decoded[0], &rhs, 0);

        for matrix in [&lhs_decoded[1], &rhs_decoded[1]] {
            assert!(
                matrix.iter().flatten().all(|value| {
                    value.re.abs() <= STRUCTURAL_TOLERANCE && value.im.abs() <= STRUCTURAL_TOLERANCE
                }),
                "inactive CCMM lanes must decode to structural zero"
            );
        }
    }

    #[test]
    fn boundary_tiles_survive_cpmm_representation_roundtrip() {
        let geometry = BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 4, 8, 16);
        let shape = GemmShape::new(MatrixShape::new(6, 7), MatrixShape::new(7, 5));
        let plan = GemmDecompositionPlan::new(shape, 4);
        let groups = batch_gemm_work_groups(plan, geometry, PrivacyMode::Cp);

        let lhs = logical_matrix(6, 7, 0.25);
        let rhs = logical_matrix(7, 5, -0.75);

        let representation = represent_batch_gemm_work_group(&groups[0], &lhs, &rhs);
        let decoded = sinc_decode_batch(representation.cpmm_lhs().unwrap());

        for scheduled in groups[0].products() {
            let lane = scheduled.lane();
            let expected = pad_native_lhs_tile(scheduled.binding().mapping(), &lhs);
            let actual = as_double_row(&decoded[lane]);

            assert_real_matrix_eq(&actual, &expected, 0);

            let rhs_physical = representation.cpmm_rhs().unwrap();

            for col in 0..geometry.dimension() {
                for row in 0..geometry.dimension() {
                    assert_eq!(
                        *rhs_physical.get(lane, row, col),
                        *pad_native_rhs_tile(scheduled.binding().mapping(), &rhs).get(0, row, col)
                    );
                }
            }
        }
    }

    #[test]
    fn full_cpmm_group_preserves_scheduled_lane_identity() {
        let geometry = BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 4, 8, 16);

        // 2 row tiles x 2 column tiles x 1 reduction tile = 4 products = B.
        let shape = GemmShape::new(MatrixShape::new(8, 4), MatrixShape::new(4, 8));
        let plan = GemmDecompositionPlan::new(shape, 4);
        let groups = batch_gemm_work_groups(plan, geometry, PrivacyMode::Cp);

        assert_eq!(groups.len(), 1);
        assert!(groups[0].is_full());

        let lhs = logical_matrix(8, 4, 2.0);
        let rhs = logical_matrix(4, 8, -2.0);

        let representation = represent_batch_gemm_work_group(&groups[0], &lhs, &rhs);
        let decoded = sinc_decode_batch(representation.cpmm_lhs().unwrap());

        for scheduled in groups[0].products() {
            let expected = pad_native_lhs_tile(scheduled.binding().mapping(), &lhs);
            let actual = as_double_row(&decoded[scheduled.lane()]);
            assert_real_matrix_eq(&actual, &expected, 0);
        }

        assert_eq!(representation.unused_lanes(), 0);
    }

    #[test]
    fn authors_scale_geometries_produce_expected_structural_dimensions() {
        let cases = [
            BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 64, 256, 8192),
            BatchGemmGeometry::new(BatchGemmMechanism::Cpmm, 128, 128, 8192),
            BatchGemmGeometry::new(BatchGemmMechanism::Ccmm, 64, 128, 8192),
            BatchGemmGeometry::new(BatchGemmMechanism::Ccmm, 128, 64, 8192),
        ];

        for geometry in cases {
            let d = geometry.dimension();
            let shape = GemmShape::new(MatrixShape::new(d, d), MatrixShape::new(d, d));
            let plan = GemmDecompositionPlan::new(shape, d);
            let groups = batch_gemm_work_groups(plan, geometry, geometry.privacy());

            let lhs = logical_matrix(d, d, 0.5);
            let rhs = logical_matrix(d, d, -0.25);

            let representation = represent_batch_gemm_work_group(&groups[0], &lhs, &rhs);

            assert_eq!(representation.geometry(), geometry);
            assert_eq!(representation.capacity(), geometry.batch_count());
            assert_eq!(representation.active_lanes(), 1);

            match representation {
                BatchGemmRepresentation::Cpmm { lhs, rhs, .. } => {
                    assert_eq!(lhs.scalar_degree(), geometry.scalar_degree());
                    assert_eq!(lhs.dimension(), d / 2);
                    assert_eq!(lhs.large_degree(), geometry.large_degree());
                    assert_eq!(lhs.num_columns(), d);
                    assert_eq!(rhs.rows(), d);
                    assert_eq!(rhs.cols(), d);
                    assert_eq!(rhs.batches(), geometry.batch_count());
                }
                BatchGemmRepresentation::Ccmm { lhs, rhs, .. } => {
                    for sinc in [&lhs, &rhs] {
                        assert_eq!(sinc.scalar_degree(), geometry.scalar_degree());
                        assert_eq!(sinc.dimension(), d);
                        assert_eq!(sinc.large_degree(), geometry.large_degree());
                        assert_eq!(sinc.num_columns(), d);
                    }
                }
            }
        }
    }
}
