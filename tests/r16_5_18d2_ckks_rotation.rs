//! R16.5.18d.2b: degree-4096 CKKS hybrid Gaussian rotation.
//!
//! Research parameters only. Exact BigInt modulus-down reference.
//! This test is intentionally ignored unless explicitly requested.

use std::time::Instant;

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    apply_bounded_rns_galois_automorphism_with_ntt,
    apply_hybrid_rns_galois_automorphism_with_prepared_ntt, research_profile_4096,
    rotation_exponent_left, BoundedRnsGaloisKey, CkksCanonicalEmbedding,
    PreparedHybridRnsGaloisKey,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig,
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
                .expect("coefficient fits f64")
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
#[ignore = "explicit degree-4096 hybrid CKKS experiment"]
fn r16_5_18d2_ckks_rotation() {
    let base_log: u32 = std::env::var("FHE_HYBRID_GADGET_BASE_LOG")
        .ok()
        .map(|value| value.parse().expect("invalid gadget base"))
        .unwrap_or(BASE_LOG);
    assert!([8, 16, 32].contains(&base_log));

    let started = Instant::now();

    let profile = research_profile_4096();
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let q_basis = chain.top().clone();
    let scale = 2.0_f64.powi(35);

    assert_eq!(degree, 4096);

    let p_basis = ModulusBasis::new(vec![Modulus::new(0x800004001), Modulus::new(0x800008001)]);

    let mut qp_moduli = q_basis.moduli().to_vec();
    qp_moduli.extend_from_slice(p_basis.moduli());

    assert_eq!(qp_moduli.len(), 5);
    assert!(qp_moduli
        .iter()
        .all(|m| m.value() % (2 * degree as u64) == 1));

    let q_plan = RnsNttPlan::new(q_basis.moduli().to_vec(), degree);
    let qp_plan = RnsNttPlan::new(qp_moduli, degree);

    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

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
            _ => 1,
        })
        .collect();

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x18d2_0001);

    let encrypt_started = Instant::now();

    let ciphertext = encrypt_rns_raw_with_distribution_ntt_rng(
        &encoded,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &q_plan,
        &mut encryption_rng,
    );

    let encryption_seconds = encrypt_started.elapsed().as_secs_f64();

    let original = decode(&ciphertext, &secret, &q_plan, &embedding, scale);

    let (input_max_abs, input_rel_l2) = error_metrics(&original, &input);

    println!("R16_5_18D2_INPUT_MAX_ABS={input_max_abs:.9e}");
    println!("R16_5_18D2_INPUT_REL_L2={input_rel_l2:.9e}");
    println!("R16_5_18D2_ENCRYPT_SECONDS={encryption_seconds:.6}");

    let exponent = rotation_exponent_left(degree, 1);

    let mut key_rng = ChaCha20Rng::seed_from_u64(0x18d2_0002);

    let keygen_started = Instant::now();

    let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedGadgetLayout::new(q_basis.clone(), base_log),
        p_basis,
        &secret,
        exponent,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        2,
        &qp_plan,
        &mut key_rng,
    );

    let keygen_seconds = keygen_started.elapsed().as_secs_f64();

    println!("R16_5_18D2_GALOIS_EXPONENT={exponent}");
    println!("R16_5_18D2_GADGET_BASE_LOG={base_log}");
    println!("R16_5_18D2_GADGET_DIGITS={}", key.digit_count());
    println!("R16_5_18D2_KEYGEN_SECONDS={keygen_seconds:.6}");

    let rotation_started = Instant::now();

    let rotated =
        apply_hybrid_rns_galois_automorphism_with_prepared_ntt(&ciphertext, &key, &qp_plan);

    let rotation_seconds = rotation_started.elapsed().as_secs_f64();

    println!("R16_5_18D2_ROTATION_SECONDS={rotation_seconds:.6}");

    let actual = decode(&rotated, &secret, &q_plan, &embedding, scale);

    let expected: Vec<Complex64> = (0..slots).map(|index| input[(index + 1) % slots]).collect();

    let (max_abs, rel_l2) = error_metrics(&actual, &expected);

    println!("R16_5_18D2_MAX_ABS={max_abs:.9e}");
    println!("R16_5_18D2_REL_L2={rel_l2:.9e}");
    println!(
        "R16_5_18D2_TOTAL_SECONDS={:.6}",
        started.elapsed().as_secs_f64()
    );

    assert!(max_abs.is_finite());
    assert!(rel_l2.is_finite());

    // Initial diagnostic threshold. Tighten after observing actual
    // CKKS precision at degree 4096.
    assert!(
        rel_l2 < 0.01,
        "hybrid Gaussian rotation did not meet initial CKKS precision target"
    );

    // R16.5.18d.2c: matched Q-only Gaussian comparison.
    // Both switching mechanisms use the same encrypted input,
    // secret, automorphism exponent, and gadget base.
    let mut qonly_rng = ChaCha20Rng::seed_from_u64(0x18d2_0003);

    let qonly_keygen_started = Instant::now();

    let qonly_key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: BoundedGadgetLayout::new(q_basis.clone(), base_log),
            plan: &q_plan,
        },
        &secret,
        exponent,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &mut qonly_rng,
    );

    let qonly_keygen_seconds = qonly_keygen_started.elapsed().as_secs_f64();

    let qonly_rotation_started = Instant::now();

    let qonly_rotated =
        apply_bounded_rns_galois_automorphism_with_ntt(&ciphertext, &qonly_key, &q_plan);

    let qonly_rotation_seconds = qonly_rotation_started.elapsed().as_secs_f64();

    let qonly_actual = decode(&qonly_rotated, &secret, &q_plan, &embedding, scale);

    let (qonly_max_abs, qonly_rel_l2) = error_metrics(&qonly_actual, &expected);

    println!("R16_5_18D2_QONLY_KEYGEN_SECONDS={qonly_keygen_seconds:.6}");
    println!("R16_5_18D2_QONLY_ROTATION_SECONDS={qonly_rotation_seconds:.6}");
    println!("R16_5_18D2_QONLY_MAX_ABS={qonly_max_abs:.9e}");
    println!("R16_5_18D2_QONLY_REL_L2={qonly_rel_l2:.9e}");

    assert!(qonly_max_abs.is_finite());
    assert!(qonly_rel_l2.is_finite());

    println!(
        "R16_5_18D2_REL_L2_RATIO_QONLY_OVER_HYBRID={:.6e}",
        qonly_rel_l2 / rel_l2.max(f64::MIN_POSITIVE)
    );

    println!("R16_5_18D2_COMPARISON_STATUS=PASS");

    println!("R16_5_18D2_STATUS=PASS");
    println!("R16_5_18D2_SECURITY_STATUS=RESEARCH_PARAMETERS_ONLY");
}
