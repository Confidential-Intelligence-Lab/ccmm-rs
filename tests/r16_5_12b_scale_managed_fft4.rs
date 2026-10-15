// R16.5.12b: deferred CKKS rescale feasibility.
// Two consecutive public complex multiplications, compared against
// the conventional two-rescale implementation.
//
// This is NOT yet an encrypted FFT4 or a security validation.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next, research_profile_16384,
    CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
};
use ccmm_rs::eblas::level1::{multiply_complex_slots_cp_prepared, PreparedComplexSlotsCp};
use ccmm_rs::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
    PreparedRnsNttPlan, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

fn decode(
    ct: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let plan = RnsNttPlan::new(ct.basis().moduli().to_vec(), embedding.degree());
    let decrypted = decrypt_rns_raw_with_ntt(ct.rlwe(), secret, &plan);
    let modulus = composite_modulus_big(decrypted.basis());
    let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("coefficient conversion")
                / ct.scale()
        })
        .collect();
    embedding.coefficients_to_slots(&coefficients)
}

fn metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    let mut error_sq = 0.0;
    let mut reference_sq = 0.0;
    let mut max_abs = 0.0_f64;

    for (&a, &e) in actual.iter().zip(expected.iter()) {
        let error = (a - e).norm();
        error_sq += error * error;
        reference_sq += e.norm_sqr();
        max_abs = max_abs.max(error);
    }

    ((error_sq / reference_sq).sqrt(), max_abs)
}

#[test]
fn r16_5_12b_deferred_rescale_feasibility() {
    let total = Instant::now();

    let profile = research_profile_16384();
    let degree = profile.degree();
    let chain = profile.modulus_chain();
    let scale = profile.initial_scale();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    assert!(chain.max_level() >= 2);

    let basis0 = chain.top().clone();
    let basis1 = chain.level(1).clone();

    let plan0 = RnsNttPlan::new(basis0.moduli().to_vec(), degree);
    let plan1 = RnsNttPlan::new(basis1.moduli().to_vec(), degree);

    let prepared0 = PreparedRnsNttPlan::new(&plan0);
    let prepared1 = PreparedRnsNttPlan::new(&plan1);

    // Deterministic test-only secret and randomness.
    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x1651_120B);

    let input: Vec<Complex64> = (0..slots)
        .map(|i| {
            let x = i as f64;
            Complex64::new(0.005 * (0.013 * x).sin(), 0.005 * (0.017 * x).cos())
        })
        .collect();

    let a = Complex64::new(0.7, 0.2);
    let b = Complex64::new(0.9, -0.1);

    let coeff_a = vec![a; slots];
    let coeff_b = vec![b; slots];

    let expected: Vec<Complex64> = input.iter().map(|&x| x * a * b).collect();

    let plaintext = encode_rns(&input, &embedding, &basis0, scale);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        &secret,
        &plan0,
        &mut rng,
    );

    let ct = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    // Path A: conventional, two rescale operations.
    let baseline_start = Instant::now();

    let pa = PreparedComplexSlotsCp::new(&ct, &coeff_a, &embedding, &chain, &plan0);

    let baseline1 = multiply_complex_slots_cp_prepared(&ct, &pa, &chain, &prepared0);

    let pb = PreparedComplexSlotsCp::new(&baseline1, &coeff_b, &embedding, &chain, &plan1);

    let baseline2 = multiply_complex_slots_cp_prepared(&baseline1, &pb, &chain, &prepared1);

    let baseline_ms = baseline_start.elapsed().as_secs_f64() * 1000.0;

    // Path B: two raw products, one rescale.
    // 2^22 * 2^23 = 2^45.
    let deferred_start = Instant::now();

    let plain_scale_a = 2.0_f64.powi(22);
    let plain_scale_b = 2.0_f64.powi(23);

    let plain_a = encode_rns(&coeff_a, &embedding, &basis0, plain_scale_a);
    let plain_b = encode_rns(&coeff_b, &embedding, &basis0, plain_scale_b);

    let prepared_a = ccmm_rs::ckks::PreparedRnsPlaintextNtt::new(&plain_a, &prepared0);

    let prepared_b = ccmm_rs::ckks::PreparedRnsPlaintextNtt::new(&plain_b, &prepared0);

    let deferred1 = multiply_plain_rns_ckks_with_prepared_ntt(
        &ct,
        &prepared_a,
        plain_scale_a,
        &chain,
        &prepared0,
    );

    let deferred2 = multiply_plain_rns_ckks_with_prepared_ntt(
        &deferred1,
        &prepared_b,
        plain_scale_b,
        &chain,
        &prepared0,
    );

    let deferred3 = rescale_rns_ckks_to_next(&deferred2, &chain);

    let deferred_ms = deferred_start.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(baseline2.level(), 2);
    assert_eq!(deferred3.level(), 1);

    let baseline_values = decode(&baseline2, &secret, &embedding);
    let deferred_values = decode(&deferred3, &secret, &embedding);

    let (baseline_rel, baseline_max) = metrics(&baseline_values, &expected);

    let (deferred_rel, deferred_max) = metrics(&deferred_values, &expected);

    println!("R16_SCALE_MANAGED_DEGREE={degree}");
    println!("R16_SCALE_MANAGED_SLOTS={slots}");
    println!("R16_SCALE_MANAGED_BASELINE_LEVEL={}", baseline2.level());
    println!("R16_SCALE_MANAGED_DEFERRED_LEVEL={}", deferred3.level());
    println!(
        "R16_SCALE_MANAGED_BASELINE_SCALE={:.12e}",
        baseline2.scale()
    );
    println!(
        "R16_SCALE_MANAGED_DEFERRED_SCALE={:.12e}",
        deferred3.scale()
    );
    println!("R16_SCALE_MANAGED_RAW_SCALE={:.12e}", deferred2.scale());
    println!("R16_SCALE_MANAGED_BASELINE_REL_L2={baseline_rel:.12e}");
    println!("R16_SCALE_MANAGED_BASELINE_MAX_ABS={baseline_max:.12e}");
    println!("R16_SCALE_MANAGED_DEFERRED_REL_L2={deferred_rel:.12e}");
    println!("R16_SCALE_MANAGED_DEFERRED_MAX_ABS={deferred_max:.12e}");
    println!("R16_SCALE_MANAGED_BASELINE_MS={baseline_ms:.3}");
    println!("R16_SCALE_MANAGED_DEFERRED_MS={deferred_ms:.3}");
    println!(
        "R16_SCALE_MANAGED_TOTAL_MS={:.3}",
        total.elapsed().as_secs_f64() * 1000.0
    );

    let pass = baseline_rel.is_finite()
        && deferred_rel.is_finite()
        && baseline_rel < 1e-3
        && deferred_rel < 1e-3
        && deferred_max < 1e-3;

    println!(
        "R16_SCALE_MANAGED_STATUS={}",
        if pass { "PASS" } else { "FAIL" }
    );

    assert!(pass, "deferred rescale numerical validation failed");
}
