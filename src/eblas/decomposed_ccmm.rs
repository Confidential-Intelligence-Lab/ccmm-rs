//! Decomposed ciphertext/ciphertext eBLAS GEMM through Batch CCMM.
//!
//! This module closes the decomposed CC execution path:
//!
//! logical GEMM -> native tiles -> Batch scheduling -> SinC representation
//! -> CKKS/RNS preparation -> Batch CCMM -> SinC recovery -> logical assembly.
//!
//! Decomposition and scheduling remain independent of cryptographic
//! representation. Each decoded physical lane is routed back through its
//! original `BatchGemmTileBinding`; no logical tile coordinates are
//! reconstructed from physical lane numbers.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::ccmm::batch::{
    decode_sinc_large_columns, quantize_sinc_batch_plaintext, sinc_decode_batch,
    BatchCcmmExecutionContext,
};
use crate::eblas::{
    batch_gemm_ccmm, batch_gemm_work_groups, crop_native_output_tile,
    represent_batch_gemm_work_group, BatchGemmGeometry, BatchGemmMechanism,
    BatchGemmRepresentation, GemmDecompositionPlan, GemmShape, PrivacyMode,
};
use crate::matrix::BatchMatrix;

/// Cryptographic execution parameters for decomposed Batch CCMM.
///
/// Evaluation-key generation and preparation are deliberately outside eBLAS.
/// The caller supplies the validated Batch CCMM execution context together
/// with the large-ring secret and encryption resources used by this
/// reproducible end-to-end research driver.
pub struct DecomposedCcmmExecutionContext<'a> {
    pub basis: &'a crate::ring::ModulusBasis,
    pub large_plan: &'a crate::ring::RnsNttPlan,
    pub secret: &'a [i8],
    pub execution: &'a BatchCcmmExecutionContext<'a>,
    /// Base seed used only by this reproducible research execution driver.
    /// Each physical work group, operand, and SinC column receives a distinct
    /// derived seed.
    pub encryption_seed: u64,
}

fn decoded_ccmm_lane(
    decoded: &[Vec<Vec<num_complex::Complex64>>],
    lane: usize,
) -> BatchMatrix<f64> {
    let complex = &decoded[lane];
    let dimension = complex.len();

    assert!(dimension > 0, "decoded Batch CCMM lane must be nonempty");
    assert!(
        complex.iter().all(|row| row.len() == dimension),
        "decoded Batch CCMM lane must be square"
    );

    let mut native = BatchMatrix::<f64>::new(dimension, dimension, 1);

    for (row, values) in complex.iter().enumerate() {
        for (col, value) in values.iter().enumerate() {
            assert!(
                value.re.is_finite() && value.im.is_finite(),
                "decoded Batch CCMM value must be finite"
            );
            native.set(0, row, col, value.re);
        }
    }

    native
}

/// Executes one logical CC GEMM through deterministic decomposition and Batch
/// CCMM.
///
/// `lhs` and `rhs` are clear logical inputs to this end-to-end research
/// execution driver. Both operands are SinC-packed, quantized, and encrypted
/// before Batch CCMM execution. The returned matrix is decrypted/decoded for
/// validation and application integration.
pub fn gemm_cc_decomposed(
    shape: GemmShape,
    geometry: BatchGemmGeometry,
    lhs: &BatchMatrix<f64>,
    rhs: &BatchMatrix<f64>,
    context: &DecomposedCcmmExecutionContext<'_>,
) -> BatchMatrix<f64> {
    assert_eq!(
        geometry.mechanism(),
        BatchGemmMechanism::Ccmm,
        "decomposed CC GEMM requires the CCMM mechanism"
    );
    assert_eq!(
        geometry.privacy(),
        PrivacyMode::Cc,
        "decomposed CC GEMM requires CC privacy"
    );

    assert_eq!(
        lhs.batches(),
        1,
        "decomposed CC GEMM currently requires one logical lhs batch"
    );
    assert_eq!(
        rhs.batches(),
        1,
        "decomposed CC GEMM currently requires one logical rhs batch"
    );

    assert_eq!(lhs.rows(), shape.lhs().rows());
    assert_eq!(lhs.cols(), shape.lhs().cols());
    assert_eq!(rhs.rows(), shape.rhs().rows());
    assert_eq!(rhs.cols(), shape.rhs().cols());

    assert_eq!(
        context.large_plan.degree(),
        geometry.large_degree(),
        "decomposed CC GEMM large NTT degree must match Batch geometry"
    );
    assert_eq!(
        context.execution.scalar_plan.degree(),
        geometry.scalar_degree(),
        "decomposed CC GEMM scalar NTT degree must match Batch geometry"
    );
    assert_eq!(
        context.execution.prepared_large_plan.degree(),
        geometry.large_degree(),
        "decomposed CC GEMM prepared large NTT degree must match Batch geometry"
    );
    assert_eq!(
        context.secret.len(),
        geometry.large_degree(),
        "decomposed CC GEMM secret degree must match Batch geometry"
    );
    assert!(
        context.execution.scale.is_finite() && context.execution.scale > 0.0,
        "decomposed CC GEMM CKKS scale must be positive"
    );

    let plan = GemmDecompositionPlan::new(shape, geometry.dimension());
    let groups = batch_gemm_work_groups(plan, geometry, PrivacyMode::Cc);

    let mut output = BatchMatrix::<f64>::new(shape.output().rows(), shape.output().cols(), 1);

    for group in &groups {
        let representation = represent_batch_gemm_work_group(group, lhs, rhs);

        let (lhs_sinc, rhs_sinc) = match &representation {
            BatchGemmRepresentation::Ccmm {
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
            BatchGemmRepresentation::Cpmm { .. } => {
                panic!("decomposed CC GEMM produced a CPMM representation")
            }
        };

        let encrypt_columns = |plaintext: &crate::ccmm::batch::SinCBatchPlaintext,
                               operand_seed: u64| {
            quantize_sinc_batch_plaintext(plaintext, context.basis, context.execution.scale)
                .iter()
                .enumerate()
                .map(|(column, polynomial)| {
                    let seed = context.encryption_seed
                        ^ ((group.group_index() as u64) << 32)
                        ^ operand_seed
                        ^ column as u64;
                    let mut rng = ChaCha20Rng::seed_from_u64(seed);

                    crate::grafting::encrypt_rns_raw_with_distribution_ntt_rng(
                        polynomial,
                        2,
                        crate::rlwe::ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                        context.secret,
                        context.large_plan,
                        &mut rng,
                    )
                })
                .collect::<Vec<_>>()
        };

        let lhs_ciphertexts = encrypt_columns(lhs_sinc, 0x4c48_5300);
        let rhs_ciphertexts = encrypt_columns(rhs_sinc, 0x5248_5300);

        assert_eq!(
            lhs_ciphertexts.len(),
            geometry.dimension(),
            "Batch CCMM lhs must contain one large-ring ciphertext per matrix column"
        );
        assert_eq!(
            rhs_ciphertexts.len(),
            geometry.dimension(),
            "Batch CCMM rhs must contain one large-ring ciphertext per matrix column"
        );

        let encrypted_output = batch_gemm_ccmm(
            geometry,
            &lhs_ciphertexts,
            &rhs_ciphertexts,
            context.execution,
        );

        let output_sinc = decode_sinc_large_columns(
            &encrypted_output,
            context.secret,
            geometry.scalar_degree(),
            geometry.dimension(),
        );
        let decoded = sinc_decode_batch(&output_sinc);

        assert_eq!(
            decoded.len(),
            geometry.batch_count(),
            "decoded Batch CCMM lane count must match physical capacity"
        );

        for scheduled in group.products() {
            let lane = scheduled.lane();

            assert!(
                lane < decoded.len(),
                "scheduled Batch CCMM lane must exist in decoded output"
            );

            let mapping = scheduled.binding().mapping();
            let product = mapping.product();
            let native_output = decoded_ccmm_lane(&decoded, lane);
            let contribution = crop_native_output_tile(mapping, &native_output);

            let row_start = product
                .output_row()
                .checked_mul(geometry.dimension())
                .expect("decomposed CC GEMM output row offset overflow");
            let col_start = product
                .output_col()
                .checked_mul(geometry.dimension())
                .expect("decomposed CC GEMM output column offset overflow");

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

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    const DIMENSION: usize = 64;
    const SCALAR_DEGREE: usize = 128;
    const LARGE_DEGREE: usize = 8192;

    fn moduli() -> Vec<crate::ring::Modulus> {
        vec![
            crate::ring::Modulus::new(68_712_923_137),
            crate::ring::Modulus::new(268_238_849),
        ]
    }

    fn scale() -> f64 {
        2.0_f64.powf(27.993302092216055)
    }

    fn deterministic_matrix(rows: usize, cols: usize, seed: u64) -> BatchMatrix<f64> {
        let mut matrix = BatchMatrix::<f64>::new(rows, cols, 1);

        for row in 0..rows {
            for col in 0..cols {
                let index = row
                    .checked_mul(cols)
                    .and_then(|value| value.checked_add(col))
                    .expect("deterministic matrix index overflow");

                let centered = ((index as u64)
                    .wrapping_mul(17)
                    .wrapping_add(seed.wrapping_mul(13))
                    % 29) as i64
                    - 14;

                matrix.set(0, row, col, centered as f64 / 64.0);
            }
        }

        matrix
    }

    fn clear_gemm(lhs: &BatchMatrix<f64>, rhs: &BatchMatrix<f64>) -> BatchMatrix<f64> {
        assert_eq!(lhs.batches(), 1);
        assert_eq!(rhs.batches(), 1);
        assert_eq!(lhs.cols(), rhs.rows());

        let mut output = BatchMatrix::<f64>::new(lhs.rows(), rhs.cols(), 1);

        for row in 0..lhs.rows() {
            for col in 0..rhs.cols() {
                let mut sum = 0.0_f64;

                for inner in 0..lhs.cols() {
                    sum += *lhs.get(0, row, inner) * *rhs.get(0, inner, col);
                }

                output.set(0, row, col, sum);
            }
        }

        output
    }

    fn error_metrics(actual: &BatchMatrix<f64>, expected: &BatchMatrix<f64>) -> (f64, f64) {
        assert_eq!(actual.rows(), expected.rows());
        assert_eq!(actual.cols(), expected.cols());
        assert_eq!(actual.batches(), expected.batches());

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for row in 0..actual.rows() {
            for col in 0..actual.cols() {
                let observed = *actual.get(0, row, col);
                let reference = *expected.get(0, row, col);
                let error = observed - reference;

                squared_error += error * error;
                squared_reference += reference * reference;
                max_abs = max_abs.max(error.abs());
            }
        }

        let rel_l2 = if squared_reference > 0.0 {
            (squared_error / squared_reference).sqrt()
        } else {
            squared_error.sqrt()
        };

        (rel_l2, max_abs)
    }

    fn run_case(m: usize, k: usize, n: usize, seed: u64) {
        let lhs = deterministic_matrix(m, k, seed);
        let rhs = deterministic_matrix(k, n, seed ^ 0x5a5a);
        let expected = clear_gemm(&lhs, &rhs);

        let shape = GemmShape::new(
            crate::eblas::MatrixShape::new(m, k),
            crate::eblas::MatrixShape::new(k, n),
        );
        let geometry = BatchGemmGeometry::new(
            BatchGemmMechanism::Ccmm,
            DIMENSION,
            SCALAR_DEGREE,
            LARGE_DEGREE,
        );

        let basis = crate::ring::ModulusBasis::new(moduli());
        let chain = crate::ring::ModulusChain::from_top_basis(basis.clone());

        let large_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), LARGE_DEGREE);
        let prepared_large_plan = crate::ring::PreparedRnsNttPlan::new(&large_plan);

        let scalar_plan = crate::ring::RnsNttPlan::new(basis.moduli().to_vec(), SCALAR_DEGREE);
        let prepared_scalar_plan = crate::ring::PreparedRnsNttPlan::new(&scalar_plan);

        let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4445_4343_4d4d_0000 ^ seed);

        let mut secret: Vec<i8> = (0..LARGE_DEGREE)
            .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
            .collect();

        if secret.iter().all(|&value| value == 0) {
            secret[0] = 1;
        }

        let galois_layout = crate::grafting::RnsGadgetLayout::new(basis.clone(), vec![1, 1]);

        let galois_keys: Vec<crate::ckks::RnsGaloisKey> = (1..DIMENSION)
            .map(|index| 2 * SCALAR_DEGREE * index + 1)
            .map(|exponent| {
                let mut rng = ChaCha20Rng::seed_from_u64(0x434d_5400_0000_0000 ^ exponent as u64);

                crate::ckks::RnsGaloisKey::generate_with_ntt_rng(
                    crate::grafting::RnsKeygenConfig {
                        degree: LARGE_DEGREE,
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

        let mut multiplication_key_rng = ChaCha20Rng::seed_from_u64(0x4343_4d4d_524c_4b00);

        let multiplication_key = crate::grafting::RnsMultiplicationKey::generate_with_ntt_rng(
            crate::grafting::RnsKeygenConfig {
                degree: LARGE_DEGREE,
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

        let execution = BatchCcmmExecutionContext {
            scalar_plan: &scalar_plan,
            prepared_scalar_plan: &prepared_scalar_plan,
            prepared_large_plan: &prepared_large_plan,
            prepared_galois_keys: &prepared_galois_keys,
            prepared_multiplication_key: &prepared_multiplication_key,
            chain: &chain,
            scale: scale(),
        };

        let context = DecomposedCcmmExecutionContext {
            basis: &basis,
            large_plan: &large_plan,
            secret: &secret,
            execution: &execution,
            encryption_seed: 0x4445_4343_454e_4300 ^ seed,
        };

        let actual = gemm_cc_decomposed(shape, geometry, &lhs, &rhs, &context);
        let (rel_l2, max_abs) = error_metrics(&actual, &expected);

        println!(
            "DECOMPOSED_CCMM_CASE={}x{}x{} REL_L2={:.12e} MAX_ABS={:.12e}",
            m, k, n, rel_l2, max_abs
        );

        assert!(
            rel_l2 < 1.0e-3,
            "decomposed Batch CCMM relative L2 error too large: {rel_l2:e}"
        );
        assert!(
            max_abs < 1.0e-3,
            "decomposed Batch CCMM maximum absolute error too large: {max_abs:e}"
        );
    }

    #[test]
    fn exact_single_native_product_matches_clear_reference() {
        run_case(4, 4, 4, 0x01);
    }

    #[test]
    fn row_and_column_decomposition_preserves_lane_identity() {
        run_case(67, 5, 69, 0x02);
    }

    #[test]
    fn reduction_across_native_products_matches_clear_reference() {
        run_case(5, 67, 4, 0x03);
    }

    #[test]
    fn multidimensional_decomposition_preserves_product_provenance() {
        run_case(65, 65, 65, 0x04);
    }
}
