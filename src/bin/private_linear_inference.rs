use std::time::Instant;

use ccmm_rs::application_support::ckks::{
    decode_scalar, encode_plain_scalar, encrypt_scalar, EncryptionContext,
};
use ccmm_rs::ckks::{research_4096_security_model, research_profile_4096, CkksCanonicalEmbedding};
use ccmm_rs::eblas::{gemm_cp, GemmShape, GemmSpec, MatrixShape, PrivacyMode};
use ccmm_rs::matrix::{RnsCkksCiphertextMatrix, RnsCkksPlaintextMatrix};
use ccmm_rs::ring::RnsNttPlan;
use ccmm_rs::rlwe::ErrorDistribution;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn main() {
    const SIGMA: f64 = 3.19;
    const TOLERANCE: f64 = 2.0e-3;

    let profile = research_profile_4096();
    let security_model = research_4096_security_model();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x320B_0000);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let encryption_context = EncryptionContext {
        embedding: &embedding,
        basis: &basis,
        scale,
        secret: &secret,
        plan: &plan,
        chain: &chain,
        distribution,
    };

    // One encrypted feature vector:
    // x = [0.2, -0.4, 0.6, 0.1]
    //
    // Plaintext 4x3 weight matrix:
    //
    //      [ 0.50, -0.20,  0.30]
    // W =  [-0.25,  0.40,  0.20]
    //      [ 0.75,  0.10, -0.40]
    //      [ 0.10, -0.50,  0.80]
    //
    // y = xW = [0.66, -0.19, -0.18]
    let input_values = [0.2, -0.4, 0.6, 0.1];

    // Column-major 4x3.
    let weight_values = [
        0.50, -0.25, 0.75, 0.10, -0.20, 0.40, 0.10, -0.50, 0.30, 0.20, -0.40, 0.80,
    ];

    let expected = [0.66, -0.19, -0.18];

    let encryption_start = Instant::now();

    let encrypted_input = RnsCkksCiphertextMatrix::from_vec_column_major(
        1,
        4,
        input_values
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                encrypt_scalar(value, 0x320B_1000 + index as u64, &encryption_context)
            })
            .collect(),
    );

    let encryption_us = encryption_start.elapsed().as_secs_f64() * 1.0e6;

    let encoding_start = Instant::now();

    let plaintext_weights = RnsCkksPlaintextMatrix::from_vec_column_major(
        4,
        3,
        scale,
        weight_values
            .iter()
            .map(|&value| encode_plain_scalar(value, &embedding, &basis, scale))
            .collect(),
    );

    let weight_encoding_us = encoding_start.elapsed().as_secs_f64() * 1.0e6;

    let gemm_spec = GemmSpec::new(
        GemmShape::new(MatrixShape::new(1, 4), MatrixShape::new(4, 3)),
        PrivacyMode::Cp,
    );

    let inference_start = Instant::now();
    let encrypted_output = gemm_cp(
        gemm_spec,
        &encrypted_input,
        &plaintext_weights,
        &chain,
        &plan,
    );
    let inference_us = inference_start.elapsed().as_secs_f64() * 1.0e6;

    assert_eq!(encrypted_output.rows(), 1);
    assert_eq!(encrypted_output.cols(), 3);
    assert_eq!(encrypted_output.level(), 1);

    let mut max_error = 0.0_f64;
    let mut max_imag = 0.0_f64;

    for (col, &expected_value) in expected.iter().enumerate() {
        let (observed, imag) = decode_scalar(encrypted_output.get(0, col), &secret, &embedding);

        let error = (observed - expected_value).abs();
        max_error = max_error.max(error);
        max_imag = max_imag.max(imag);

        println!(
            "R3_2B_OUTPUT_{}_EXPECTED={:.12e} OBSERVED={:.12e} ABS_ERROR={:.12e}",
            col, expected_value, observed, error
        );
    }

    println!("R3_2B_PRIVATE_LINEAR_INFERENCE_VERSION=1");
    println!("APPLICATION=private-linear-inference");
    println!("EBLAS_OPERATION=GEMM");
    println!("EBLAS_PRIVACY_MODE=CP");
    println!("EBLAS_BACKEND=CpDirect");
    println!("PROFILE={}", profile.name());
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={}", profile.slot_count());
    println!(
        "UNDERLYING_RLWE_TARGET_SECURITY_BITS={}",
        security_model.classical_security_bits
    );
    println!("ENCRYPTED_FEATURES=4");
    println!("PLAINTEXT_OUTPUTS=3");
    println!("CIPHERTEXT_PLAINTEXT_MULTIPLIES=12");
    println!("CIPHERTEXT_ADDITIONS=9");
    println!("OUTPUT_RESCALES=3");
    println!("EVALUATION_KEY_REQUIRED=false");
    println!("CIPHERTEXT_ERROR_SIGMA={SIGMA}");
    println!("INPUT_SCALE={scale:.17e}");
    println!("OUTPUT_SCALE={:.17e}", encrypted_output.scale());
    println!("ENCRYPTION_4_FEATURES_US={encryption_us:.3}");
    println!("WEIGHT_ENCODING_12_VALUES_US={weight_encoding_us:.3}");
    println!("PRIVATE_LINEAR_INFERENCE_US={inference_us:.3}");
    println!("MAX_OUTPUT_ERROR={max_error:.12e}");
    println!("MAX_IMAGINARY_RESIDUAL={max_imag:.12e}");
    println!("TOLERANCE={TOLERANCE:.12e}");
    println!("CIRCULAR_KDM_ASSUMPTION_FOR_APPLICATION=false");
    println!("HARDENED_SAMPLER=false");

    assert!(
        max_error <= TOLERANCE,
        "private linear inference exceeded tolerance: max_error={max_error:e}"
    );

    println!("R3_2B_PRIVATE_LINEAR_INFERENCE_STATUS=PASS");
}
