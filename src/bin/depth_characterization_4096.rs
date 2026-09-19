use std::time::Instant;

use ccmm_rs::ckks::{
    multiply_relinearize_rescale_rns_ckks_bounded_with_ntt, research_4096_security_model,
    research_profile_4096, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig, BoundedRnsMultiplicationKey,
};
use ccmm_rs::ring::{ModulusBasis, Polynomial, RnsNttPlan, RnsPolynomial};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

#[derive(Debug, Clone, Copy)]
struct ErrorStats {
    max: f64,
    mean: f64,
    rms: f64,
    max_imag: f64,
}

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
                    .map(|&value| {
                        let signed = (value * scale).round() as i128;
                        signed.rem_euclid(q) as u64
                    })
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

fn decode_slots(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
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

    embedding.coefficients_to_slots(&coefficients)
}

fn error_stats(actual: &[Complex64], expected: &[Complex64]) -> ErrorStats {
    assert_eq!(actual.len(), expected.len());

    let mut max = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut max_imag = 0.0_f64;

    for (&observed, &reference) in actual.iter().zip(expected) {
        let error = (observed - reference).norm();
        max = max.max(error);
        sum += error;
        sum_sq += error * error;
        max_imag = max_imag.max(observed.im.abs());
    }

    let count = actual.len() as f64;

    ErrorStats {
        max,
        mean: sum / count,
        rms: (sum_sq / count).sqrt(),
        max_imag,
    }
}

fn generate_bounded_key(
    basis: &ModulusBasis,
    degree: usize,
    secret: &[i8],
    distribution: ErrorDistribution,
    seed: u64,
    base_log: u32,
) -> (BoundedRnsMultiplicationKey, RnsNttPlan, f64) {
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let layout = BoundedGadgetLayout::new(basis.clone(), base_log);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let start = Instant::now();

    let key = BoundedRnsMultiplicationKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout,
            plan: &plan,
        },
        secret,
        distribution,
        &mut rng,
    );

    let elapsed_us = start.elapsed().as_secs_f64() * 1.0e6;

    (key, plan, elapsed_us)
}

fn main() {
    const BASE_LOG: u32 = 20;
    const SIGMA: f64 = 3.19;
    const TOLERANCE: f64 = 2.0e-3;

    let profile = research_profile_4096();
    let security_model = research_4096_security_model();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let slot_count = profile.slot_count();
    let scale = profile.initial_scale();
    let top_basis = chain.top().clone();
    let top_plan = RnsNttPlan::new(top_basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x330A_0000);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    // Exercise all logical slots with values in approximately [-0.25, 0.25].
    // Keeping magnitudes below one lets repeated squaring characterize CKKS
    // numerical depth without overflow dominating the experiment.
    let input_slots: Vec<Complex64> = (0..slot_count)
        .map(|index| {
            let fraction = index as f64 / (slot_count - 1) as f64;
            Complex64::new(-0.25 + 0.50 * fraction, 0.0)
        })
        .collect();

    let expected_depth_1: Vec<Complex64> = input_slots.iter().map(|&value| value * value).collect();

    let expected_depth_2: Vec<Complex64> = expected_depth_1
        .iter()
        .map(|&value| value * value)
        .collect();

    let plaintext = encode_rns(&input_slots, &embedding, &top_basis, scale);

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x330A_1000);
    let encryption_start = Instant::now();

    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        distribution,
        &secret,
        &top_plan,
        &mut encryption_rng,
    );

    let encryption_us = encryption_start.elapsed().as_secs_f64() * 1.0e6;

    let depth_0 = RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, scale), &chain);

    let (level_0_key, level_0_plan, level_0_keygen_us) = generate_bounded_key(
        chain.level(0),
        degree,
        &secret,
        distribution,
        0x330A_2000,
        BASE_LOG,
    );

    let depth_1_start = Instant::now();
    let depth_1 = multiply_relinearize_rescale_rns_ckks_bounded_with_ntt(
        &depth_0,
        &depth_0,
        &level_0_key,
        &chain,
        &level_0_plan,
    );
    let depth_1_us = depth_1_start.elapsed().as_secs_f64() * 1.0e6;

    let observed_depth_1 = decode_slots(&depth_1, &secret, &embedding);
    let depth_1_stats = error_stats(&observed_depth_1, &expected_depth_1);

    let (level_1_key, level_1_plan, level_1_keygen_us) = generate_bounded_key(
        chain.level(1),
        degree,
        &secret,
        distribution,
        0x330A_3000,
        BASE_LOG,
    );

    let depth_2_start = Instant::now();
    let depth_2 = multiply_relinearize_rescale_rns_ckks_bounded_with_ntt(
        &depth_1,
        &depth_1,
        &level_1_key,
        &chain,
        &level_1_plan,
    );
    let depth_2_us = depth_2_start.elapsed().as_secs_f64() * 1.0e6;

    let observed_depth_2 = decode_slots(&depth_2, &secret, &embedding);
    let depth_2_stats = error_stats(&observed_depth_2, &expected_depth_2);

    println!("R3_3A_DEPTH_CHARACTERIZATION_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={slot_count}");
    println!(
        "UNDERLYING_RLWE_TARGET_SECURITY_BITS={}",
        security_model.classical_security_bits
    );
    println!("CHAIN_LEVELS={}", chain.len());
    println!("MAX_CHAIN_LEVEL={}", chain.max_level());
    println!("INITIAL_SCALE={scale:.17e}");
    println!("BOUNDED_BASE_LOG={BASE_LOG}");
    println!("BOUNDED_BASE={}", 1_u128 << BASE_LOG);
    println!("CIPHERTEXT_ERROR_SIGMA={SIGMA}");
    println!("EVALUATION_KEY_ERROR_SIGMA={SIGMA}");
    println!("ENCRYPTION_US={encryption_us:.3}");

    println!("R3_3A_DEPTH=1");
    println!("R3_3A_DEPTH_1_LEVEL={}", depth_1.level());
    println!("R3_3A_DEPTH_1_SCALE={:.17e}", depth_1.scale());
    println!("R3_3A_DEPTH_1_KEYGEN_US={level_0_keygen_us:.3}");
    println!("R3_3A_DEPTH_1_EVAL_US={depth_1_us:.3}");
    println!("R3_3A_DEPTH_1_MAX_SLOT_ERROR={:.12e}", depth_1_stats.max);
    println!("R3_3A_DEPTH_1_MEAN_SLOT_ERROR={:.12e}", depth_1_stats.mean);
    println!("R3_3A_DEPTH_1_RMS_SLOT_ERROR={:.12e}", depth_1_stats.rms);
    println!(
        "R3_3A_DEPTH_1_MAX_IMAGINARY_RESIDUAL={:.12e}",
        depth_1_stats.max_imag
    );

    println!("R3_3A_DEPTH=2");
    println!("R3_3A_DEPTH_2_LEVEL={}", depth_2.level());
    println!("R3_3A_DEPTH_2_SCALE={:.17e}", depth_2.scale());
    println!("R3_3A_DEPTH_2_KEYGEN_US={level_1_keygen_us:.3}");
    println!("R3_3A_DEPTH_2_EVAL_US={depth_2_us:.3}");
    println!("R3_3A_DEPTH_2_MAX_SLOT_ERROR={:.12e}", depth_2_stats.max);
    println!("R3_3A_DEPTH_2_MEAN_SLOT_ERROR={:.12e}", depth_2_stats.mean);
    println!("R3_3A_DEPTH_2_RMS_SLOT_ERROR={:.12e}", depth_2_stats.rms);
    println!(
        "R3_3A_DEPTH_2_MAX_IMAGINARY_RESIDUAL={:.12e}",
        depth_2_stats.max_imag
    );

    println!(
        "R3_3A_DEPTH_2_CAN_RESCALE={}",
        depth_2.state().can_rescale(&chain)
    );
    println!("R3_3A_MAX_SUPPORTED_MULTIPLICATIVE_DEPTH=2");
    println!("R3_3A_NEXT_MULTIPLICATION_REQUIRES_DEEPER_CHAIN=true");
    println!("TOLERANCE={TOLERANCE:.12e}");
    println!("CIRCULAR_KDM_ASSUMPTION=required");
    println!("HARDENED_SAMPLER=false");

    assert_eq!(depth_1.level(), 1);
    assert_eq!(depth_2.level(), 2);
    assert!(!depth_2.state().can_rescale(&chain));

    assert!(
        depth_1_stats.max <= TOLERANCE,
        "depth-1 error exceeded tolerance: {}",
        depth_1_stats.max
    );

    assert!(
        depth_2_stats.max <= TOLERANCE,
        "depth-2 error exceeded tolerance: {}",
        depth_2_stats.max
    );

    println!("R3_3A_DEPTH_CHARACTERIZATION_STATUS=PASS");
}
