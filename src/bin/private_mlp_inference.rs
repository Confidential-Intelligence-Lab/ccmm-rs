use std::time::Instant;

use ccmm_rs::application_support::ckks::{
    decode_matrix, encode_plain_scalar, encrypt_scalar, square_matrix_bounded_with_ntt,
    EncryptionContext,
};
use ccmm_rs::ckks::{research_profile_8192, CkksCanonicalEmbedding};
use ccmm_rs::eblas::{gemm_cp, GemmShape, GemmSpec, MatrixShape, PrivacyMode};
use ccmm_rs::grafting::{BoundedGadgetLayout, BoundedRnsKeygenConfig, BoundedRnsMultiplicationKey};
use ccmm_rs::matrix::{RnsCkksCiphertextMatrix, RnsCkksPlaintextMatrix};
use ccmm_rs::ring::RnsNttPlan;
use ccmm_rs::rlwe::ErrorDistribution;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn plaintext_matrix(
    rows: usize,
    cols: usize,
    values: &[f64],
    embedding: &CkksCanonicalEmbedding,
    basis: &ccmm_rs::ring::ModulusBasis,
    scale: f64,
) -> RnsCkksPlaintextMatrix {
    assert_eq!(values.len(), rows * cols);

    RnsCkksPlaintextMatrix::from_vec_column_major(
        rows,
        cols,
        scale,
        values
            .iter()
            .map(|&value| encode_plain_scalar(value, embedding, basis, scale))
            .collect(),
    )
}

fn main() {
    const BASE_LOG: u32 = 20;
    const SIGMA: f64 = 3.19;
    const TOLERANCE: f64 = 5.0e-3;

    let profile = research_profile_8192();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let basis = chain.top().clone();
    let top_plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    assert!(
        chain.max_level() >= 3,
        "private MLP requires at least three CKKS level transitions"
    );

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x360B_0000);
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
        plan: &top_plan,
        chain: &chain,
        distribution,
    };

    // Input:
    //
    // x = [0.2, -0.4, 0.6, 0.1]
    //
    // First layer:
    //
    //      [ 0.50, -0.20,  0.30]
    // W1 = [-0.25,  0.40,  0.20]
    //      [ 0.75,  0.10, -0.40]
    //      [ 0.10, -0.50,  0.80]
    //
    // x W1 = [0.66, -0.19, -0.18]
    //
    // Activation:
    //
    // h = (x W1)^2 = [0.4356, 0.0361, 0.0324]
    //
    // Second layer:
    //
    //      [ 0.50, -0.30]
    // W2 = [ 0.20,  0.40]
    //      [-0.10,  0.60]
    //
    // y = h W2 = [0.22178, -0.09680]

    let input = [0.2, -0.4, 0.6, 0.1];

    // Column-major 4x3.
    let w1 = [
        0.50, -0.25, 0.75, 0.10, -0.20, 0.40, 0.10, -0.50, 0.30, 0.20, -0.40, 0.80,
    ];

    // Column-major 3x2.
    let w2 = [0.50, 0.20, -0.10, -0.30, 0.40, 0.60];

    let expected = [0.22178, -0.09680];

    let encryption_start = Instant::now();

    let encrypted_input = RnsCkksCiphertextMatrix::from_vec_column_major(
        1,
        4,
        input
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                encrypt_scalar(value, 0x360B_1000 + index as u64, &encryption_context)
            })
            .collect(),
    );

    let encryption_us = encryption_start.elapsed().as_secs_f64() * 1.0e6;

    // Layer 1: CP GEMM at level 0 -> level 1.
    let w1_encoding_start = Instant::now();

    let plaintext_w1 = plaintext_matrix(4, 3, &w1, &embedding, &basis, scale);

    let w1_encoding_us = w1_encoding_start.elapsed().as_secs_f64() * 1.0e6;

    let layer1_spec = GemmSpec::new(
        GemmShape::new(MatrixShape::new(1, 4), MatrixShape::new(4, 3)),
        PrivacyMode::Cp,
    );

    let layer1_start = Instant::now();

    let hidden_linear = gemm_cp(
        layer1_spec,
        &encrypted_input,
        &plaintext_w1,
        &chain,
        &top_plan,
    );

    let layer1_us = layer1_start.elapsed().as_secs_f64() * 1.0e6;

    assert_eq!(hidden_linear.level(), 1);
    assert_eq!(hidden_linear.rows(), 1);
    assert_eq!(hidden_linear.cols(), 3);

    // Square activation at level 1 -> level 2.
    let activation_basis = hidden_linear.get(0, 0).basis().clone();
    let activation_plan = RnsNttPlan::new(activation_basis.moduli().to_vec(), degree);
    let activation_layout = BoundedGadgetLayout::new(activation_basis.clone(), BASE_LOG);

    let activation_keygen_start = Instant::now();
    let mut activation_key_rng = ChaCha20Rng::seed_from_u64(0x360B_2000);

    let activation_key = BoundedRnsMultiplicationKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: activation_layout.clone(),
            plan: &activation_plan,
        },
        &secret,
        distribution,
        &mut activation_key_rng,
    );

    let activation_keygen_us = activation_keygen_start.elapsed().as_secs_f64() * 1.0e6;

    let activation_start = Instant::now();

    let hidden_activated =
        square_matrix_bounded_with_ntt(&hidden_linear, &activation_key, &chain, &activation_plan);

    let activation_us = activation_start.elapsed().as_secs_f64() * 1.0e6;

    assert_eq!(hidden_activated.level(), 2);
    assert_eq!(hidden_activated.rows(), 1);
    assert_eq!(hidden_activated.cols(), 3);

    // Layer 2: encode weights against the active level-2 basis and scale,
    // then CP GEMM level 2 -> level 3.
    let layer2_basis = hidden_activated.get(0, 0).basis().clone();
    let layer2_scale = hidden_activated.scale();
    let layer2_plan = RnsNttPlan::new(layer2_basis.moduli().to_vec(), degree);

    let w2_encoding_start = Instant::now();

    let plaintext_w2 = plaintext_matrix(3, 2, &w2, &embedding, &layer2_basis, layer2_scale);

    let w2_encoding_us = w2_encoding_start.elapsed().as_secs_f64() * 1.0e6;

    let layer2_spec = GemmSpec::new(
        GemmShape::new(MatrixShape::new(1, 3), MatrixShape::new(3, 2)),
        PrivacyMode::Cp,
    );

    let layer2_start = Instant::now();

    let encrypted_output = gemm_cp(
        layer2_spec,
        &hidden_activated,
        &plaintext_w2,
        &chain,
        &layer2_plan,
    );

    let layer2_us = layer2_start.elapsed().as_secs_f64() * 1.0e6;

    assert_eq!(encrypted_output.level(), 3);
    assert_eq!(encrypted_output.rows(), 1);
    assert_eq!(encrypted_output.cols(), 2);

    let (observed, max_imag) = decode_matrix(&encrypted_output, &secret, &embedding);

    assert_eq!(observed.len(), expected.len());

    let mut max_error = 0.0_f64;

    for (index, (&actual, &reference)) in observed.iter().zip(expected.iter()).enumerate() {
        let error = (actual - reference).abs();
        max_error = max_error.max(error);

        println!(
            "R3_6B_OUTPUT_{index}_EXPECTED={reference:.12e} \
             OBSERVED={actual:.12e} ABS_ERROR={error:.12e}"
        );
    }

    println!("R3_6B_PRIVATE_MLP_INFERENCE_VERSION=1");
    println!("APPLICATION=private-mlp-inference");
    println!("PROFILE={}", profile.name());
    println!("PARAMETER_SECURITY_VALIDATED=false");
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={}", profile.slot_count());
    println!("CHAIN_LEVELS={}", chain.len());
    println!("MAX_CHAIN_LEVEL={}", chain.max_level());
    println!("TOTAL_MODULUS_BITS={}", profile.total_modulus_bits());
    println!("INITIAL_SCALE={scale:.17e}");

    println!("INPUT_FEATURES=4");
    println!("HIDDEN_UNITS=3");
    println!("OUTPUT_UNITS=2");

    println!("LAYER1_OPERATION=EBLAS_GEMM_CP");
    println!("LAYER1_LEVEL_IN=0");
    println!("LAYER1_LEVEL_OUT={}", hidden_linear.level());

    println!("ACTIVATION=square");
    println!("ACTIVATION_OPERATION=BOUNDED_CC_ELEMENTWISE");
    println!("ACTIVATION_LEVEL_IN={}", hidden_linear.level());
    println!("ACTIVATION_LEVEL_OUT={}", hidden_activated.level());
    println!("ACTIVATION_RELINEARIZATIONS=3");
    println!("ACTIVATION_RESCALES=3");

    println!("LAYER2_OPERATION=EBLAS_GEMM_CP");
    println!("LAYER2_LEVEL_IN={}", hidden_activated.level());
    println!("LAYER2_LEVEL_OUT={}", encrypted_output.level());

    println!("MULTIPLICATIVE_LEVELS_CONSUMED=3");

    println!("BOUNDED_BASE_LOG={BASE_LOG}");
    println!("BOUNDED_BASE={}", activation_layout.base());
    println!("BOUNDED_DIGITS={}", activation_layout.digit_count());

    println!("CIPHERTEXT_ERROR_SIGMA={SIGMA}");
    println!("EVALUATION_KEY_ERROR_SIGMA={SIGMA}");
    println!("CIRCULAR_KDM_ASSUMPTION=required");
    println!("HARDENED_SAMPLER=false");

    println!("ENCRYPTION_4_FEATURES_US={encryption_us:.3}");
    println!("LAYER1_WEIGHT_ENCODING_US={w1_encoding_us:.3}");
    println!("LAYER1_EBLAS_GEMM_US={layer1_us:.3}");
    println!("ACTIVATION_KEYGEN_US={activation_keygen_us:.3}");
    println!("ACTIVATION_SQUARE_US={activation_us:.3}");
    println!("LAYER2_WEIGHT_ENCODING_US={w2_encoding_us:.3}");
    println!("LAYER2_EBLAS_GEMM_US={layer2_us:.3}");

    println!("OUTPUT_SCALE={:.17e}", encrypted_output.scale());
    println!("MAX_OUTPUT_ERROR={max_error:.12e}");
    println!("MAX_IMAGINARY_RESIDUAL={max_imag:.12e}");
    println!("TOLERANCE={TOLERANCE:.12e}");

    assert!(
        max_error <= TOLERANCE,
        "private MLP inference exceeded tolerance: max_error={max_error:e}"
    );

    println!("R3_6B_PRIVATE_MLP_INFERENCE_STATUS=PASS");
}
