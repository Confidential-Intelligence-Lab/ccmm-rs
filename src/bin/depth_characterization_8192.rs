use std::time::Instant;

use ccmm_rs::ckks::{
    multiply_relinearize_rescale_rns_ckks_bounded_with_ntt, research_profile_8192,
    CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig, BoundedRnsMultiplicationKey,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, ModulusBasis,
    Polynomial, RnsNttPlan, RnsPolynomial,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
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
    let modulus = composite_modulus_big(plaintext.basis());

    let coefficients: Vec<f64> = reconstruct_coefficients_big(&plaintext)
        .iter()
        .map(|value| {
            centered_representative_big(value, &modulus)
                .to_f64()
                .expect("centered CKKS coefficient must convert to f64")
                / ciphertext.scale()
        })
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

    let profile = research_profile_8192();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let slot_count = profile.slot_count();
    let scale = profile.initial_scale();
    let top_basis = chain.top().clone();
    let top_plan = RnsNttPlan::new(top_basis.moduli().to_vec(), degree);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let distribution = ErrorDistribution::DiscreteGaussian { sigma: SIGMA };

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x330B_0000);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    // Keep the signal large enough to remain meaningful through x^(16),
    // while remaining comfortably inside the CKKS dynamic range.
    let input_slots: Vec<Complex64> = (0..slot_count)
        .map(|index| {
            let fraction = index as f64 / (slot_count - 1) as f64;
            Complex64::new(0.70 + 0.20 * fraction, 0.0)
        })
        .collect();

    let plaintext = encode_rns(&input_slots, &embedding, &top_basis, scale);

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x330B_1000);
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

    let mut current = RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, scale), &chain);

    let mut expected = input_slots.clone();
    let mut total_keygen_us = 0.0_f64;
    let mut total_eval_us = 0.0_f64;
    let mut worst_error = 0.0_f64;

    println!("R3_3B_DEPTH_CHARACTERIZATION_VERSION=1");
    println!("PROFILE={}", profile.name());
    println!("RING_DEGREE={degree}");
    println!("SLOT_COUNT={slot_count}");
    println!("PARAMETER_SECURITY_VALIDATED=false");
    println!("CHAIN_LEVELS={}", chain.len());
    println!("MAX_CHAIN_LEVEL={}", chain.max_level());
    println!("TOTAL_MODULUS_BITS={}", profile.total_modulus_bits());
    println!("INITIAL_SCALE={scale:.17e}");
    println!("BOUNDED_BASE_LOG={BASE_LOG}");
    println!("BOUNDED_BASE={}", 1_u128 << BASE_LOG);
    println!("CIPHERTEXT_ERROR_SIGMA={SIGMA}");
    println!("EVALUATION_KEY_ERROR_SIGMA={SIGMA}");
    println!("ENCRYPTION_US={encryption_us:.3}");

    for depth in 1..=chain.max_level() {
        let level = current.level();
        let basis = chain.level(level);

        let (key, plan, keygen_us) = generate_bounded_key(
            basis,
            degree,
            &secret,
            distribution,
            0x330B_2000 + depth as u64,
            BASE_LOG,
        );

        let eval_start = Instant::now();

        let next = multiply_relinearize_rescale_rns_ckks_bounded_with_ntt(
            &current, &current, &key, &chain, &plan,
        );

        let eval_us = eval_start.elapsed().as_secs_f64() * 1.0e6;

        expected = expected.iter().map(|&value| value * value).collect();

        let observed = decode_slots(&next, &secret, &embedding);
        let stats = error_stats(&observed, &expected);

        total_keygen_us += keygen_us;
        total_eval_us += eval_us;
        worst_error = worst_error.max(stats.max);

        println!("R3_3B_DEPTH={depth}");
        println!("R3_3B_DEPTH_{depth}_LEVEL={}", next.level());
        println!("R3_3B_DEPTH_{depth}_ACTIVE_LIMBS={}", next.basis().len());
        println!("R3_3B_DEPTH_{depth}_SCALE={:.17e}", next.scale());
        println!("R3_3B_DEPTH_{depth}_KEYGEN_US={keygen_us:.3}");
        println!("R3_3B_DEPTH_{depth}_EVAL_US={eval_us:.3}");
        println!("R3_3B_DEPTH_{depth}_MAX_SLOT_ERROR={:.12e}", stats.max);
        println!("R3_3B_DEPTH_{depth}_MEAN_SLOT_ERROR={:.12e}", stats.mean);
        println!("R3_3B_DEPTH_{depth}_RMS_SLOT_ERROR={:.12e}", stats.rms);
        println!(
            "R3_3B_DEPTH_{depth}_MAX_IMAGINARY_RESIDUAL={:.12e}",
            stats.max_imag
        );

        assert_eq!(next.level(), depth);
        assert!(
            stats.max <= TOLERANCE,
            "depth-{depth} error exceeded tolerance: {}",
            stats.max
        );

        current = next;
    }

    println!(
        "R3_3B_TERMINAL_CAN_RESCALE={}",
        current.state().can_rescale(&chain)
    );
    println!(
        "R3_3B_MAX_SUPPORTED_MULTIPLICATIVE_DEPTH={}",
        chain.max_level()
    );
    println!("R3_3B_NEXT_MULTIPLICATION_REQUIRES_DEEPER_CHAIN=true");
    println!("R3_3B_TOTAL_KEYGEN_US={total_keygen_us:.3}");
    println!("R3_3B_TOTAL_EVAL_US={total_eval_us:.3}");
    println!("R3_3B_WORST_MAX_SLOT_ERROR={worst_error:.12e}");
    println!("TOLERANCE={TOLERANCE:.12e}");
    println!("CIRCULAR_KDM_ASSUMPTION=required");
    println!("HARDENED_SAMPLER=false");

    assert_eq!(current.level(), chain.max_level());
    assert!(!current.state().can_rescale(&chain));

    println!("R3_3B_DEPTH_CHARACTERIZATION_STATUS=PASS");
}
