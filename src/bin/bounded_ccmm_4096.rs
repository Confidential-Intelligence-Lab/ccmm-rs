use std::time::Instant;

use ccmm_rs::ckks::{
    research_4096_security_model, research_profile_4096, CkksCanonicalEmbedding, CkksChainState,
    RnsCkksCiphertext,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig, BoundedRnsMultiplicationKey,
};
use ccmm_rs::matrix::RnsCkksCiphertextMatrix;
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn encode_rns(
    slots: &[Complex64],
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
) -> RnsPolynomial {
    let coefficients = embedding.slots_to_coefficients(slots);
    let residues = basis
        .moduli()
        .iter()
        .copied()
        .map(|modulus| {
            let q = i128::from(modulus.value());
            Polynomial::new(
                modulus,
                coefficients
                    .iter()
                    .map(|&value| ((value * scale).round() as i128).rem_euclid(q) as u64)
                    .collect(),
            )
        })
        .collect();
    RnsPolynomial::from_residues(residues)
}

fn centered(value: u128, modulus: u128) -> i128 {
    if value > modulus / 2 {
        value as i128 - modulus as i128
    } else {
        value as i128
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn encrypt_scalar(
    value: f64,
    seed: u64,
    embedding: &CkksCanonicalEmbedding,
    basis: &ModulusBasis,
    scale: f64,
    secret: &[i8],
    plan: &RnsNttPlan,
    chain: &ccmm_rs::ring::ModulusChain,
    distribution: ErrorDistribution,
) -> RnsCkksCiphertext {
    let slots = vec![Complex64::new(value, 0.0); secret.len() / 2];
    let plaintext = encode_rns(&slots, embedding, basis, scale);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        distribution,
        secret,
        plan,
        &mut rng,
    );
    RnsCkksCiphertext::new(rlwe, CkksChainState::top(chain, scale), chain)
}

fn decode_scalar(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> (f64, f64) {
    let plan = RnsNttPlan::new(
        ciphertext.basis().moduli().to_vec(),
        ciphertext.rlwe().degree(),
    );
    let plaintext = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);
    let modulus = plaintext.composite_modulus();

    let coefficients: Vec<f64> = plaintext
        .reconstruct_coefficients()
        .into_iter()
        .map(|value| centered(value, modulus) as f64 / ciphertext.scale())
        .collect();

    let slots = embedding.coefficients_to_slots(&coefficients);
    let mean = slots.iter().map(|slot| slot.re).sum::<f64>() / slots.len() as f64;
    let max_imag = slots
        .iter()
        .map(|slot| slot.im.abs())
        .fold(0.0_f64, f64::max);

    (mean, max_imag)
}

fn main() {
    const BASE_LOG: u32 = 20;
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

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x31E3_0000);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();
    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let layout = BoundedGadgetLayout::new(basis.clone(), BASE_LOG);

    let keygen_start = Instant::now();
    let mut key_rng = ChaCha20Rng::seed_from_u64(0x31E3_1000);
    let multiplication_key = BoundedRnsMultiplicationKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: layout.clone(),
            plan: &plan,
        },
        &secret,
        distribution,
        &mut key_rng,
    );
    let keygen_us = keygen_start.elapsed().as_secs_f64() * 1.0e6;

    let lhs_values = [0.50, 0.75, -0.25, 0.125];
    let rhs_values = [0.25, -0.50, 0.50, 0.25];

    let encryption_start = Instant::now();

    let lhs = RnsCkksCiphertextMatrix::from_vec_column_major(
        2,
        2,
        lhs_values
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                encrypt_scalar(
                    value,
                    0x31E3_2000 + index as u64,
                    &embedding,
                    &basis,
                    scale,
                    &secret,
                    &plan,
                    &chain,
                    distribution,
                )
            })
            .collect(),
    );

    let rhs = RnsCkksCiphertextMatrix::from_vec_column_major(
        2,
        2,
        rhs_values
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                encrypt_scalar(
                    value,
                    0x31E3_3000 + index as u64,
                    &embedding,
                    &basis,
                    scale,
                    &secret,
                    &plan,
                    &chain,
                    distribution,
                )
            })
            .collect(),
    );

    let encryption_us = encryption_start.elapsed().as_secs_f64() * 1.0e6;

    let ccmm_start = Instant::now();
    let result = lhs.matmul_bounded_with_ntt(&rhs, &multiplication_key, &chain, &plan);
    let ccmm_us = ccmm_start.elapsed().as_secs_f64() * 1.0e6;

    assert_eq!(result.rows(), 2);
    assert_eq!(result.cols(), 2);
    assert_eq!(result.level(), 1);

    let expected = [[0.25, 0.1875], [0.125, 0.40625]];

    let mut max_error = 0.0_f64;
    let mut max_imag = 0.0_f64;

    for (row, expected_row) in expected.iter().enumerate() {
        for (col, &expected_value) in expected_row.iter().enumerate() {
            let (observed, imag) = decode_scalar(result.get(row, col), &secret, &embedding);
            let error = (observed - expected_value).abs();
            max_error = max_error.max(error);
            max_imag = max_imag.max(imag);

            println!(
                "R3_1E3_CELL_{}_{}_EXPECTED={:.12e} OBSERVED={:.12e} ABS_ERROR={:.12e}",
                row, col, expected_value, observed, error
            );
        }
    }

    println!("R3_1E3_BOUNDED_CCMM_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={}", profile.slot_count());
    println!(
        "UNDERLYING_RLWE_TARGET_SECURITY_BITS={}",
        security_model.classical_security_bits
    );
    println!("MATRIX_ROWS=2");
    println!("MATRIX_INNER=2");
    println!("MATRIX_COLS=2");
    println!("SCALAR_CKKS_MULTIPLIES=8");
    println!("CIPHERTEXT_ADDITIONS=4");
    println!("INPUT_SCALE={scale:.17e}");
    println!("OUTPUT_SCALE={:.17e}", result.scale());
    println!("BOUNDED_BASE_LOG={BASE_LOG}");
    println!("BOUNDED_BASE={}", layout.base());
    println!("BOUNDED_DIGITS={}", layout.digit_count());
    println!("CIPHERTEXT_ERROR_SIGMA={SIGMA}");
    println!("EVALUATION_KEY_ERROR_SIGMA={SIGMA}");
    println!("KEYGEN_US={keygen_us:.3}");
    println!("ENCRYPTION_8_CIPHERTEXTS_US={encryption_us:.3}");
    println!("CCMM_US={ccmm_us:.3}");
    println!("MAX_MATRIX_ERROR={max_error:.12e}");
    println!("MAX_IMAGINARY_RESIDUAL={max_imag:.12e}");
    println!("TOLERANCE={TOLERANCE:.12e}");
    println!("CIRCULAR_KDM_ASSUMPTION=required");
    println!("HARDENED_SAMPLER=false");

    assert!(
        max_error <= TOLERANCE,
        "bounded 2x2 CCMM exceeded tolerance: max_error={max_error:e}"
    );

    println!("R3_1E3_BOUNDED_CCMM_STATUS=PASS");
}
