//! Decomposed ciphertext/plaintext eBLAS GEMM through Batch CPMM.
//!
//! This module closes the decomposed CP execution path:
//!
//! logical GEMM -> native tiles -> Batch scheduling -> SinC representation
//! -> CKKS/RNS preparation -> Batch CPMM -> SinC recovery -> logical assembly.
//!
//! Decomposition and scheduling remain independent of cryptographic
//! representation. Each decoded physical lane is routed back through its
//! original `BatchGemmTileBinding`; no logical tile coordinates are
//! reconstructed from physical lane numbers.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::ccmm::batch::{
    as_double_row, decode_cpmm_large_columns, encode_cpmm_real_batch,
    quantize_sinc_batch_plaintext, sinc_decode_batch,
};
use crate::eblas::{
    batch_gemm_cpmm, batch_gemm_work_groups, crop_native_output_tile,
    represent_batch_gemm_work_group, BatchGemmGeometry, BatchGemmMechanism,
    BatchGemmRepresentation, GemmDecompositionPlan, GemmShape, PrivacyMode,
};
use crate::matrix::BatchMatrix;

/// Cryptographic execution parameters for decomposed Batch CPMM.
///
/// Key generation is deliberately outside eBLAS. The caller supplies the
/// large-ring secret and all CKKS/RNS execution objects so that logical
/// decomposition does not own cryptographic policy.
pub struct DecomposedCpmmExecutionContext<'a> {
    pub basis: &'a crate::ring::ModulusBasis,
    pub chain: &'a crate::ring::ModulusChain,
    pub large_plan: &'a crate::ring::RnsNttPlan,
    pub scalar_plan: &'a crate::ring::RnsNttPlan,
    pub secret: &'a [i8],
    pub scale: f64,
    /// Base seed used only by this reproducible research execution driver.
    /// Each physical work group and SinC column receives a distinct derived
    /// seed.
    pub encryption_seed: u64,
}

fn physical_real_batch_to_nested(batch: &BatchMatrix<f64>) -> Vec<Vec<Vec<f64>>> {
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

fn decoded_cpmm_lane(
    decoded: &[Vec<Vec<num_complex::Complex64>>],
    lane: usize,
) -> BatchMatrix<f64> {
    let real = as_double_row(&decoded[lane]);
    let dimension = real.len();

    assert!(dimension > 0, "decoded Batch CPMM lane must be nonempty");
    assert!(
        real.iter().all(|row| row.len() == dimension),
        "decoded Batch CPMM lane must be square"
    );

    let mut native = BatchMatrix::<f64>::new(dimension, dimension, 1);

    for (row, values) in real.iter().enumerate() {
        for (col, &value) in values.iter().enumerate() {
            native.set(0, row, col, value);
        }
    }

    native
}

/// Executes one logical CP GEMM through deterministic decomposition and Batch
/// CPMM.
///
/// `lhs` and `rhs` are clear logical inputs to this end-to-end research
/// execution driver. The lhs is SinC-packed, quantized, and encrypted before
/// Batch CPMM execution; the rhs is represented as packed CKKS/RNS plaintext.
/// The returned matrix is decrypted/decoded for validation and application
/// integration.
pub fn gemm_cp_decomposed(
    shape: GemmShape,
    geometry: BatchGemmGeometry,
    lhs: &BatchMatrix<f64>,
    rhs: &BatchMatrix<f64>,
    context: &DecomposedCpmmExecutionContext<'_>,
) -> BatchMatrix<f64> {
    assert_eq!(
        geometry.mechanism(),
        BatchGemmMechanism::Cpmm,
        "decomposed CP GEMM requires the CPMM mechanism"
    );
    assert_eq!(
        geometry.privacy(),
        PrivacyMode::Cp,
        "decomposed CP GEMM requires CP privacy"
    );

    assert_eq!(
        lhs.batches(),
        1,
        "decomposed CP GEMM currently requires one logical lhs batch"
    );
    assert_eq!(
        rhs.batches(),
        1,
        "decomposed CP GEMM currently requires one logical rhs batch"
    );

    assert_eq!(lhs.rows(), shape.lhs().rows());
    assert_eq!(lhs.cols(), shape.lhs().cols());
    assert_eq!(rhs.rows(), shape.rhs().rows());
    assert_eq!(rhs.cols(), shape.rhs().cols());

    assert_eq!(
        context.large_plan.degree(),
        geometry.large_degree(),
        "decomposed CP GEMM large NTT degree must match Batch geometry"
    );
    assert_eq!(
        context.scalar_plan.degree(),
        geometry.scalar_degree(),
        "decomposed CP GEMM scalar NTT degree must match Batch geometry"
    );
    assert_eq!(
        context.secret.len(),
        geometry.large_degree(),
        "decomposed CP GEMM secret degree must match Batch geometry"
    );
    assert!(
        context.scale.is_finite() && context.scale > 0.0,
        "decomposed CP GEMM CKKS scale must be positive"
    );

    let plan = GemmDecompositionPlan::new(shape, geometry.dimension());
    let groups = batch_gemm_work_groups(plan, geometry, PrivacyMode::Cp);

    let mut output = BatchMatrix::<f64>::new(shape.output().rows(), shape.output().cols(), 1);

    for group in &groups {
        let representation = represent_batch_gemm_work_group(group, lhs, rhs);

        let (lhs_sinc, rhs_real) = match &representation {
            BatchGemmRepresentation::Cpmm {
                geometry: represented_geometry,
                group_index,
                active_lanes,
                lhs,
                rhs,
            } => {
                assert_eq!(*represented_geometry, geometry);
                assert_eq!(*group_index, group.group_index());
                assert_eq!(*active_lanes, group.len());
                (lhs, rhs)
            }
            BatchGemmRepresentation::Ccmm { .. } => {
                panic!("decomposed CP GEMM produced a CCMM representation")
            }
        };

        let lhs_plain = quantize_sinc_batch_plaintext(lhs_sinc, context.basis, context.scale);

        assert_eq!(
            lhs_plain.len(),
            geometry.dimension(),
            "Batch CPMM lhs must contain one large-ring plaintext per matrix column"
        );

        let lhs_ciphertexts: Vec<crate::ckks::RnsCkksCiphertext> = lhs_plain
            .iter()
            .enumerate()
            .map(|(column, plaintext)| {
                let seed =
                    context.encryption_seed ^ ((group.group_index() as u64) << 32) ^ column as u64;
                let mut rng = ChaCha20Rng::seed_from_u64(seed);

                let rlwe = crate::grafting::encrypt_rns_raw_with_distribution_ntt_rng(
                    plaintext,
                    2,
                    crate::rlwe::ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                    context.secret,
                    context.large_plan,
                    &mut rng,
                );

                crate::ckks::RnsCkksCiphertext::new(
                    rlwe,
                    crate::ckks::CkksChainState::top(context.chain, context.scale),
                    context.chain,
                )
            })
            .collect();

        let rhs_nested = physical_real_batch_to_nested(rhs_real);
        let rhs_plain = encode_cpmm_real_batch(
            &rhs_nested,
            geometry.scalar_degree(),
            context.basis,
            context.scale,
        );

        let encrypted_output = batch_gemm_cpmm(
            geometry,
            &lhs_ciphertexts,
            &rhs_plain,
            context.scalar_plan,
            context.chain,
            context.scale,
        );

        let output_sinc = decode_cpmm_large_columns(
            &encrypted_output,
            context.secret,
            geometry.scalar_degree(),
            geometry.dimension() / 2,
        );
        let decoded = sinc_decode_batch(&output_sinc);

        assert_eq!(
            decoded.len(),
            geometry.batch_count(),
            "decoded Batch CPMM lane count must match physical capacity"
        );

        for scheduled in group.products() {
            let lane = scheduled.lane();

            assert!(
                lane < decoded.len(),
                "scheduled Batch CPMM lane must exist in decoded output"
            );

            let mapping = scheduled.binding().mapping();
            let product = mapping.product();
            let native_output = decoded_cpmm_lane(&decoded, lane);
            let contribution = crop_native_output_tile(mapping, &native_output);

            let row_start = product
                .output_row()
                .checked_mul(geometry.dimension())
                .expect("decomposed CP GEMM output row offset overflow");
            let col_start = product
                .output_col()
                .checked_mul(geometry.dimension())
                .expect("decomposed CP GEMM output column offset overflow");

            for col in 0..contribution.cols() {
                for row in 0..contribution.rows() {
                    let output_row = row_start + row;
                    let output_col = col_start + col;
                    let accumulated =
                        *output.get(0, output_row, output_col) + *contribution.get(0, row, col);

                    output.set(0, output_row, output_col, accumulated);
                }
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn matrix(rows: usize, cols: usize, bias: f64) -> BatchMatrix<f64> {
        let mut matrix = BatchMatrix::<f64>::new(rows, cols, 1);

        for col in 0..cols {
            for row in 0..rows {
                matrix.set(
                    0,
                    row,
                    col,
                    bias + row as f64 * 0.03125 - col as f64 * 0.015625
                        + (row * col) as f64 * 0.00390625,
                );
            }
        }

        matrix
    }

    fn clear_gemm(
        shape: GemmShape,
        lhs: &BatchMatrix<f64>,
        rhs: &BatchMatrix<f64>,
    ) -> BatchMatrix<f64> {
        crate::eblas::gemm_pp(
            crate::eblas::GemmSpec::new(shape, PrivacyMode::Pp),
            lhs,
            rhs,
        )
    }

    fn assert_close(actual: &BatchMatrix<f64>, expected: &BatchMatrix<f64>, tolerance: f64) {
        assert_eq!(actual.rows(), expected.rows());
        assert_eq!(actual.cols(), expected.cols());
        assert_eq!(actual.batches(), expected.batches());

        let mut squared_error = 0.0;
        let mut squared_reference = 0.0;
        let mut max_error = 0.0_f64;

        for col in 0..actual.cols() {
            for row in 0..actual.rows() {
                let a = *actual.get(0, row, col);
                let e = *expected.get(0, row, col);
                let error = (a - e).abs();

                max_error = max_error.max(error);
                squared_error += error * error;
                squared_reference += e * e;
            }
        }

        let relative_l2 = squared_error.sqrt() / squared_reference.sqrt().max(f64::EPSILON);

        println!("DECOMPOSED_CPMM_REL_L2={relative_l2:.12e}");
        println!("DECOMPOSED_CPMM_MAX_ABS={max_error:.12e}");

        assert!(
            relative_l2 <= tolerance,
            "decomposed CPMM relative L2 error {relative_l2} exceeds tolerance {tolerance}"
        );
    }

    fn run_case(m: usize, k: usize, n: usize, seed: u64) {
        // Authors-scale Batch CPMM d=64 geometry:
        //
        // d  = 64
        // Ns = 256
        // N  = (d/2) * Ns = 8192
        // physical real lanes = Ns/2 = 128
        //
        // Logical matrices may be smaller or larger than d. R4 zero padding,
        // R6 scheduling, and R7 representation map them onto this fixed
        // cryptographic execution geometry.
        const DIMENSION: usize = 64;
        const SCALAR_DEGREE: usize = 256;
        const LARGE_DEGREE: usize = 8192;

        let shape = GemmShape::new(
            crate::eblas::MatrixShape::new(m, k),
            crate::eblas::MatrixShape::new(k, n),
        );

        let lhs = matrix(m, k, 0.125);
        let rhs = matrix(k, n, -0.0625);
        let expected = clear_gemm(shape, &lhs, &rhs);

        // Same modulus chain and scale as the validated authors-equivalent
        // Batch CPMM capability tests.
        let moduli = vec![
            crate::ring::Modulus::new(68_712_923_137),
            crate::ring::Modulus::new(268_238_849),
        ];
        let scale = 2.0_f64.powf(27.993302092216055);

        let basis = crate::ring::ModulusBasis::new(moduli.clone());
        let chain = crate::ring::ModulusChain::from_top_basis(basis.clone());
        let large_plan = crate::ring::RnsNttPlan::new(moduli.clone(), LARGE_DEGREE);
        let scalar_plan = crate::ring::RnsNttPlan::new(moduli, SCALAR_DEGREE);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(seed ^ 0x5345_4352_4554);
        let mut secret: Vec<i8> = (0..LARGE_DEGREE)
            .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
            .collect();

        if secret.iter().all(|&value| value == 0) {
            secret[0] = 1;
        }

        let geometry = BatchGemmGeometry::new(
            BatchGemmMechanism::Cpmm,
            DIMENSION,
            SCALAR_DEGREE,
            LARGE_DEGREE,
        );

        let context = DecomposedCpmmExecutionContext {
            basis: &basis,
            chain: &chain,
            large_plan: &large_plan,
            scalar_plan: &scalar_plan,
            secret: &secret,
            scale,
            encryption_seed: seed ^ 0x0045_4E43_5259_5054,
        };

        let actual = gemm_cp_decomposed(shape, geometry, &lhs, &rhs, &context);

        assert_close(&actual, &expected, 1.0e-3);
    }

    #[test]
    fn exact_single_native_product_matches_clear_reference() {
        // One logical product occupies one physical lane. The remaining
        // authors-scale CPMM lanes are inactive and structurally zero.
        run_case(4, 4, 4, 0x5238_0001);
    }

    #[test]
    fn row_and_column_decomposition_preserves_lane_identity() {
        // ceil(67/64) * ceil(69/64) * ceil(5/64) = 4 products.
        //
        // Exercises simultaneous M/N decomposition, boundary padding, four
        // active physical lanes, and output-tile routing without K reduction.
        run_case(67, 5, 69, 0x5238_0002);
    }

    #[test]
    fn reduction_across_native_products_matches_clear_reference() {
        // ceil(5/64) * ceil(4/64) * ceil(67/64) = 2 products.
        //
        // Both products contribute to the same logical output tile, directly
        // exercising K-tile accumulation after cryptographic execution.
        run_case(5, 67, 4, 0x5238_0003);
    }

    #[test]
    fn multidimensional_decomposition_preserves_product_provenance() {
        // ceil(65/64)^3 = 8 products.
        //
        // Exercises M, K, and N decomposition simultaneously. Products are
        // recovered from distinct physical lanes and routed through their
        // original scheduled bindings before K contributions are accumulated.
        run_case(65, 65, 65, 0x5238_0004);
    }
}
