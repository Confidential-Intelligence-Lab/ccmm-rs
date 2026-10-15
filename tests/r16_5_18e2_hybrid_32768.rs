//! R16.5.18e.2: UHD extraction-scale hybrid Galois rotation.
//!
//! Research parameters only. Exact BigInt modulus-down.
//! Explicit opt-in required before expensive allocations.

use std::time::Instant;

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    apply_hybrid_rns_galois_automorphism_with_prepared_ntt, research_profile_32768_for_levels,
    rotation_exponent_left, CkksCanonicalEmbedding, PreparedHybridRnsGaloisKey,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Modulus,
    ModulusBasis, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;

use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const DEGREE: usize = 32768;
const HALF: usize = 4096;
const SIGMA: f64 = 3.19;
const BASE_LOG: u32 = 8;

fn decode(
    ciphertext: &ccmm_rs::grafting::RnsRlweCiphertext,
    secret: &[i8],
    plan: &RnsNttPlan,
    embedding: &CkksCanonicalEmbedding,
    scale: f64,
) -> Vec<Complex64> {
    let decrypted = decrypt_rns_raw_with_ntt(ciphertext, secret, plan);
    let modulus = composite_modulus_big(decrypted.basis());

    let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("coefficient conversion")
                / scale
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

fn error_metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());

    let mut error_squared = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (&value, &reference) in actual.iter().zip(expected) {
        let error = (value - reference).norm();
        error_squared += error * error;
        reference_squared += reference.norm_sqr();
        max_abs = max_abs.max(error);
    }

    (
        max_abs,
        (error_squared / reference_squared.max(f64::MIN_POSITIVE)).sqrt(),
    )
}

#[test]
#[ignore = "explicit high-memory degree-32768 experiment"]
fn r16_5_18e2_hybrid_32768() {
    let profile = research_profile_32768_for_levels(13, 3);
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let q_basis = chain.top().clone();
    let scale = profile.initial_scale();

    assert_eq!(degree, DEGREE);
    assert_eq!(q_basis.moduli().len(), 16);

    let layout = BoundedGadgetLayout::new(q_basis.clone(), BASE_LOG);
    assert_eq!(layout.digit_count(), 101);

    // Two distinct 50-bit NTT-compatible auxiliary primes are
    // selected in a separate preflight before execution.
    let p_values: Vec<u64> = std::env::var("R16_18E2_P_PRIMES")
        .expect("R16_18E2_P_PRIMES must contain two comma-separated primes")
        .split(',')
        .map(|value| value.trim().parse::<u64>().expect("invalid P prime"))
        .collect();

    assert_eq!(p_values.len(), 2);
    assert_ne!(p_values[0], p_values[1]);

    let q_values: Vec<u64> = q_basis.moduli().iter().map(|m| m.value()).collect();

    for &p in &p_values {
        assert!(p > (1_u64 << 49));
        assert!(p < (1_u64 << 51));
        assert_eq!(p % (2 * DEGREE as u64), 1);
        assert!(!q_values.contains(&p));
    }

    let p_basis = ModulusBasis::new(p_values.into_iter().map(Modulus::new).collect());

    let mut qp_moduli = q_basis.moduli().to_vec();
    qp_moduli.extend_from_slice(p_basis.moduli());

    let key_bytes = layout.digit_count() as u128 * 2 * qp_moduli.len() as u128 * degree as u128 * 8;

    println!("R16_5_18E2_DEGREE={degree}");
    println!("R16_5_18E2_Q_LIMBS={}", q_basis.moduli().len());
    println!("R16_5_18E2_P_LIMBS={}", p_basis.moduli().len());
    println!("R16_5_18E2_GADGET_DIGITS={}", layout.digit_count());
    println!(
        "R16_5_18E2_KEY_STORAGE_GIB={:.6}",
        key_bytes as f64 / 1073741824.0
    );

    assert_eq!(
        std::env::var("R16_18E2_AUTHORIZE").ok().as_deref(),
        Some("YES"),
        "explicit high-memory authorization required"
    );

    let started = Instant::now();
    let q_plan = RnsNttPlan::new(q_basis.moduli().to_vec(), degree);
    let qp_plan = RnsNttPlan::new(qp_moduli, degree);

    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();
    assert_eq!(slots, 16384);

    let input: Vec<Complex64> = (0..slots)
        .map(|index| {
            let x = index as f64;
            Complex64::new(
                0.35 * (0.013 * x).sin() + 0.12 * (0.037 * x).cos(),
                0.21 * (0.019 * x).sin() - 0.08 * (0.041 * x).cos(),
            )
        })
        .collect();

    let encoded = encode_rns(&input, &embedding, &q_basis, scale);

    let secret: Vec<i8> = (0..degree)
        .map(|index| match index % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x18e2_0001);

    let encryption_started = Instant::now();

    let ciphertext = encrypt_rns_raw_with_distribution_ntt_rng(
        &encoded,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &q_plan,
        &mut encryption_rng,
    );

    println!(
        "R16_5_18E2_ENCRYPT_SECONDS={:.6}",
        encryption_started.elapsed().as_secs_f64()
    );

    let original = decode(&ciphertext, &secret, &q_plan, &embedding, scale);

    let (input_max_abs, input_rel_l2) = error_metrics(&original, &input);

    println!("R16_5_18E2_INPUT_MAX_ABS={input_max_abs:.9e}");
    println!("R16_5_18E2_INPUT_REL_L2={input_rel_l2:.9e}");

    let exponent = rotation_exponent_left(degree, HALF);
    println!("R16_5_18E2_GALOIS_EXPONENT={exponent}");

    let mut key_rng = ChaCha20Rng::seed_from_u64(0x18e2_0002);

    let keygen_started = Instant::now();

    let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
        layout,
        p_basis,
        &secret,
        exponent,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        2,
        &qp_plan,
        &mut key_rng,
    );

    println!(
        "R16_5_18E2_KEYGEN_SECONDS={:.6}",
        keygen_started.elapsed().as_secs_f64()
    );

    let rotation_started = Instant::now();

    let rotated =
        apply_hybrid_rns_galois_automorphism_with_prepared_ntt(&ciphertext, &key, &qp_plan);

    println!(
        "R16_5_18E2_ROTATION_SECONDS={:.6}",
        rotation_started.elapsed().as_secs_f64()
    );

    let actual = decode(&rotated, &secret, &q_plan, &embedding, scale);

    let expected: Vec<Complex64> = (0..slots)
        .map(|index| input[(index + HALF) % slots])
        .collect();

    let (max_abs, rel_l2) = error_metrics(&actual, &expected);

    println!("R16_5_18E2_MAX_ABS={max_abs:.9e}");
    println!("R16_5_18E2_REL_L2={rel_l2:.9e}");
    println!(
        "R16_5_18E2_TOTAL_SECONDS={:.6}",
        started.elapsed().as_secs_f64()
    );

    assert!(max_abs.is_finite());
    assert!(rel_l2.is_finite());
    assert!(
        rel_l2 < 0.01,
        "degree-32768 hybrid rotation exceeds diagnostic precision threshold"
    );

    println!("R16_5_18E2_STATUS=PASS");
    println!("R16_5_18E2_SECURITY_STATUS=RESEARCH_PARAMETERS_ONLY");
}
