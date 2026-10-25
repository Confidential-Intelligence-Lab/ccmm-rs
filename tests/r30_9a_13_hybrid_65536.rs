//! R30.9A.13: degree-65536 Q16-stage-0 hybrid-QP rotation.
//!
//! Research parameters only. Exact BigInt modulus-down reference.
//! This test is intentionally ignored unless explicitly requested.

use std::time::Instant;

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    apply_hybrid_rns_galois_automorphism_with_prepared_ntt, research_profile_65536_for_levels,
    rotation_exponent_left, CkksCanonicalEmbedding, PreparedHybridRnsGaloisKey,
    PreparedRnsGaloisKey, RnsGaloisKey,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    RnsGadgetLayout, RnsKeygenConfig,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, ModulusBasis,
    RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;

use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const SIGMA: f64 = 3.19;
const BASE_LOG: u32 = 32;

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
#[ignore = "explicit degree-65536 hybrid-QP experiment"]
fn r30_9a_13_hybrid_65536() {
    let base_log: u32 = std::env::var("FHE_HYBRID_GADGET_BASE_LOG")
        .ok()
        .map(|value| value.parse().expect("invalid gadget base"))
        .unwrap_or(BASE_LOG);
    assert!([8, 16, 32].contains(&base_log));

    let started = Instant::now();

    let profile = research_profile_65536_for_levels(13, 3);
    let chain = profile.modulus_chain();
    let degree = profile.degree();
    let q_basis = chain.level(3).clone();
    let scale = profile.initial_scale();

    assert_eq!(degree, 65536);
    assert_eq!(q_basis.moduli().len(), 13);

    let extended = research_profile_65536_for_levels(15, 3);
    let extended_moduli = extended.modulus_chain().top().moduli().to_vec();
    assert_eq!(extended_moduli.len(), 18);
    assert_eq!(&extended_moduli[..16], chain.top().moduli());
    let p_basis = ModulusBasis::new(extended_moduli[16..18].to_vec());

    let mut qp_moduli = q_basis.moduli().to_vec();
    qp_moduli.extend_from_slice(p_basis.moduli());

    assert_eq!(qp_moduli.len(), 15);
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

    println!("R30_9A_13_INPUT_MAX_ABS={input_max_abs:.9e}");
    println!("R30_9A_13_INPUT_REL_L2={input_rel_l2:.9e}");
    println!("R30_9A_13_ENCRYPT_SECONDS={encryption_seconds:.6}");

    let exponent = rotation_exponent_left(degree, 1);

    // R30_21B_RNS_HYBRID_COMPARISON
    // Reference key: one RNS gadget block per Q modulus.
    let rns_layout = RnsGadgetLayout::new(q_basis.clone(), vec![1; q_basis.moduli().len()]);
    let mut rns_rng = ChaCha20Rng::seed_from_u64(0x18d2_0003);
    let rns_keygen_started = Instant::now();
    let rns_key = RnsGaloisKey::generate_with_ntt_rng(
        RnsKeygenConfig {
            degree,
            plaintext_modulus: 2,
            noise_bound: 0,
            layout: rns_layout,
            plan: &q_plan,
        },
        &secret,
        exponent,
        &mut rns_rng,
    );
    let prepared_rns_key = PreparedRnsGaloisKey::new(&rns_key, &q_plan);
    let rns_keygen_seconds = rns_keygen_started.elapsed().as_secs_f64();

    println!("R30_21B_RNS_KEYGEN_SECONDS={rns_keygen_seconds:.6}");

    let rns_rotation_started = Instant::now();
    let rns_rotated = ccmm_rs::ckks::apply_rns_galois_automorphism_with_prepared_ntt(
        &ciphertext,
        &prepared_rns_key,
        &q_plan,
    );
    let rns_rotation_seconds = rns_rotation_started.elapsed().as_secs_f64();

    println!("R30_21B_RNS_ROTATION_SECONDS={rns_rotation_seconds:.6}");

    let rns_actual = decode(&rns_rotated, &secret, &q_plan, &embedding, scale);
    let rns_expected: Vec<Complex64> = (0..slots).map(|index| input[(index + 1) % slots]).collect();
    let (rns_max_abs, rns_rel_l2) = error_metrics(&rns_actual, &rns_expected);

    println!("R30_21B_RNS_MAX_ABS={rns_max_abs:.9e}");
    println!("R30_21B_RNS_REL_L2={rns_rel_l2:.9e}");

    assert!(rns_rel_l2.is_finite() && rns_rel_l2 < 0.01);

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

    println!("R30_9A_13_GALOIS_EXPONENT={exponent}");
    println!("R30_9A_13_GADGET_BASE_LOG={base_log}");
    println!("R30_9A_13_GADGET_DIGITS={}", key.digit_count());
    println!("R30_9A_13_KEYGEN_SECONDS={keygen_seconds:.6}");

    let rotation_started = Instant::now();

    let rotated =
        apply_hybrid_rns_galois_automorphism_with_prepared_ntt(&ciphertext, &key, &qp_plan);

    let rotation_seconds = rotation_started.elapsed().as_secs_f64();

    println!("R30_9A_13_ROTATION_SECONDS={rotation_seconds:.6}");
    println!(
        "R30_21B_HYBRID_OVER_RNS_ROTATION_RATIO={:.6}",
        rotation_seconds / rns_rotation_seconds
    );
    println!(
        "R30_21B_RNS_OVER_HYBRID_ROTATION_RATIO={:.6}",
        rns_rotation_seconds / rotation_seconds
    );

    let actual = decode(&rotated, &secret, &q_plan, &embedding, scale);

    let expected: Vec<Complex64> = (0..slots).map(|index| input[(index + 1) % slots]).collect();

    let (max_abs, rel_l2) = error_metrics(&actual, &expected);

    println!("R30_9A_13_MAX_ABS={max_abs:.9e}");
    println!("R30_9A_13_REL_L2={rel_l2:.9e}");
    println!(
        "R30_9A_13_TOTAL_SECONDS={:.6}",
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
}
