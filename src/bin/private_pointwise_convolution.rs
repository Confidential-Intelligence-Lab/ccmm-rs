use std::time::Instant;

use ccmm_rs::application_support::ckks::{
    decode_matrix, encode_plain_scalar, encrypt_scalar, EncryptionContext,
};
use ccmm_rs::ckks::{research_profile_4096, CkksCanonicalEmbedding};
use ccmm_rs::eblas::{
    flatten_nhwc_to_matrix, gemm_cp, unflatten_matrix_to_nhwc, GemmShape, GemmSpec, MatrixShape,
    NhwcShape, PrivacyMode,
};
use ccmm_rs::matrix::{BatchMatrix, RnsCkksCiphertextMatrix, RnsCkksPlaintextMatrix};
use ccmm_rs::ring::RnsNttPlan;
use ccmm_rs::rlwe::ErrorDistribution;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn main() {
    const SIGMA: f64 = 3.19;
    const TOLERANCE: f64 = 2.0e-3;

    let input_shape = NhwcShape::new(1, 2, 2, 3);
    let output_shape = NhwcShape::new(1, 2, 2, 2);

    let input_nhwc = vec![
        0.20, -0.10, 0.30, 0.40, 0.50, -0.20, -0.30, 0.10, 0.60, 0.70, -0.40, 0.20,
    ];
    let weights = [0.50, -0.25, 0.75, -0.20, 0.40, 0.10];

    let input_matrix = flatten_nhwc_to_matrix(input_shape, &input_nhwc);
    assert_eq!((input_matrix.rows(), input_matrix.cols()), (4, 3));

    let mut reference_matrix = BatchMatrix::new(4, 2, 1);
    for row in 0..4 {
        for filter in 0..2 {
            let mut sum = 0.0;
            for channel in 0..3 {
                sum += *input_matrix.get(0, row, channel) * weights[channel + filter * 3];
            }
            reference_matrix.set(0, row, filter, sum);
        }
    }
    let reference_nhwc = unflatten_matrix_to_nhwc(output_shape, &reference_matrix);

    let profile = research_profile_4096();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x360D_0000);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&v| v == 0) {
        secret[0] = 1;
    }

    let context = EncryptionContext {
        embedding: &embedding,
        basis: &basis,
        scale,
        secret: &secret,
        plan: &plan,
        chain: &chain,
        distribution,
    };

    let encryption_start = Instant::now();
    let encrypted_input = RnsCkksCiphertextMatrix::from_vec_column_major(
        4,
        3,
        (0..3)
            .flat_map(|col| (0..4).map(move |row| (row, col)))
            .enumerate()
            .map(|(index, (row, col))| {
                encrypt_scalar(
                    *input_matrix.get(0, row, col),
                    0x360D_1000 + index as u64,
                    &context,
                )
            })
            .collect(),
    );
    let encryption_us = encryption_start.elapsed().as_secs_f64() * 1.0e6;

    let weight_encoding_start = Instant::now();
    let plaintext_weights = RnsCkksPlaintextMatrix::from_vec_column_major(
        3,
        2,
        scale,
        weights
            .iter()
            .map(|&v| encode_plain_scalar(v, &embedding, &basis, scale))
            .collect(),
    );
    let weight_encoding_us = weight_encoding_start.elapsed().as_secs_f64() * 1.0e6;

    let spec = GemmSpec::new(
        GemmShape::new(MatrixShape::new(4, 3), MatrixShape::new(3, 2)),
        PrivacyMode::Cp,
    );

    let eval_start = Instant::now();
    let encrypted_output = gemm_cp(spec, &encrypted_input, &plaintext_weights, &chain, &plan);
    let eval_us = eval_start.elapsed().as_secs_f64() * 1.0e6;

    assert_eq!(encrypted_output.level(), 1);
    let (decoded_column_major, max_imag) = decode_matrix(&encrypted_output, &secret, &embedding);
    let decoded_matrix = BatchMatrix::from_vec_column_major(4, 2, 1, decoded_column_major);
    let observed_nhwc = unflatten_matrix_to_nhwc(output_shape, &decoded_matrix);

    let mut max_error = 0.0_f64;
    for (index, (&observed, &expected)) in observed_nhwc.iter().zip(&reference_nhwc).enumerate() {
        let error = (observed - expected).abs();
        max_error = max_error.max(error);
        println!(
            "R3_6D_OUTPUT_{index}_EXPECTED={expected:.12e} OBSERVED={observed:.12e} ABS_ERROR={error:.12e}"
        );
    }

    println!("R3_6D_PRIVATE_POINTWISE_CONVOLUTION_VERSION=1");
    println!("APPLICATION=private-pointwise-convolution");
    println!("OPERATION=1x1-convolution");
    println!("TENSOR_LAYOUT=NHWC");
    println!("INPUT_SHAPE=[1,2,2,3]");
    println!("KERNEL_SHAPE=[1,1,3,2]");
    println!("OUTPUT_SHAPE=[1,2,2,2]");
    println!("LOWERED_LHS_SHAPE=[4,3]");
    println!("LOWERED_RHS_SHAPE=[3,2]");
    println!("LOWERED_OUTPUT_SHAPE=[4,2]");
    println!("EBLAS_OPERATION=GEMM");
    println!("EBLAS_PRIVACY_MODE=CP");
    println!("EBLAS_BACKEND=CpDirect");
    println!("PROFILE={}", profile.name());
    println!("PARAMETER_SECURITY_VALIDATED=false");
    println!("LEVEL_IN=0");
    println!("LEVEL_OUT={}", encrypted_output.level());
    println!("SIMD_PACKING_USED=NO");
    println!("NEW_CRYPTO_KERNEL_USED=NO");
    println!("CIPHERTEXT_ERROR_SIGMA={SIGMA}");
    println!("EVALUATION_KEY_REQUIRED=false");
    println!("HARDENED_SAMPLER=false");
    println!("ENCRYPTION_12_VALUES_US={encryption_us:.3}");
    println!("WEIGHT_ENCODING_6_VALUES_US={weight_encoding_us:.3}");
    println!("POINTWISE_CONVOLUTION_GEMM_US={eval_us:.3}");
    println!("MAX_OUTPUT_ERROR={max_error:.12e}");
    println!("MAX_IMAGINARY_RESIDUAL={max_imag:.12e}");
    println!("TOLERANCE={TOLERANCE:.12e}");

    assert!(max_error <= TOLERANCE);
    println!("R3_6D_PRIVATE_POINTWISE_CONVOLUTION_STATUS=PASS");
}
