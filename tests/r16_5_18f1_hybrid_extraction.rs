//! R16.5.18f.1: UHD extraction-scale hybrid Galois rotation.
//!
//! Research parameters only. Exact BigInt modulus-down.
//! Explicit opt-in required before expensive allocations.

use std::time::Instant;

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_32768_for_levels, rotate_left_hybrid_rns_ckks_with_prepared_ntt,
    rotation_exponent_left, CkksCanonicalEmbedding, CkksChainState, PreparedHybridRnsGaloisKey,
    RnsCkksCiphertext,
};
use ccmm_rs::eblas::{multiply_complex_slots_cp_prepared, PreparedComplexSlotsCp};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Modulus,
    ModulusBasis, PreparedRnsNttPlan, RnsNttPlan,
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
fn r16_5_18f1_hybrid_extraction() {
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
    let p_values: Vec<u64> = std::env::var("R16_18F1_P_PRIMES")
        .expect("R16_18F1_P_PRIMES must contain two comma-separated primes")
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

    println!("R16_5_18F1_DEGREE={degree}");
    println!("R16_5_18F1_Q_LIMBS={}", q_basis.moduli().len());
    println!("R16_5_18F1_P_LIMBS={}", p_basis.moduli().len());
    println!("R16_5_18F1_GADGET_DIGITS={}", layout.digit_count());
    println!(
        "R16_5_18F1_KEY_STORAGE_GIB={:.6}",
        key_bytes as f64 / 1073741824.0
    );

    assert_eq!(
        std::env::var("R16_18F1_AUTHORIZE").ok().as_deref(),
        Some("YES"),
        "explicit high-memory authorization required"
    );

    let started = Instant::now();
    let q_plan = RnsNttPlan::new(q_basis.moduli().to_vec(), degree);
    let qp_plan = RnsNttPlan::new(qp_moduli, degree);

    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();
    assert_eq!(slots, 16384);

    // Match the horizontal FFT's two-tile physical packing.
    fn reverse_base4_4(mut value: usize) -> usize {
        let mut reversed = 0usize;
        for _ in 0..4 {
            reversed = 4 * reversed + value % 4;
            value /= 4;
        }
        reversed
    }

    fn sample(row: usize, col: usize) -> Complex64 {
        let y = row as f64;
        let x = col as f64;
        Complex64::new(
            0.35 * (0.013 * x + 0.019 * y).sin() + 0.12 * (0.037 * x - 0.011 * y).cos(),
            0.21 * (0.019 * x + 0.023 * y).sin() - 0.08 * (0.041 * x - 0.017 * y).cos(),
        )
    }

    let mut input = vec![Complex64::new(0.0, 0.0); slots];

    for half in 0..2 {
        for packed_row in 0..16 {
            let natural_row = (packed_row % 4) * 4 + packed_row / 4;
            for packed_col in 0..256 {
                let natural_col = reverse_base4_4(packed_col);
                let slot = half * HALF + packed_row * 256 + packed_col;
                input[slot] = sample(natural_row, half * 256 + natural_col);
            }
        }
    }

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
        "R16_5_18F1_ENCRYPT_SECONDS={:.6}",
        encryption_started.elapsed().as_secs_f64()
    );

    let original = decode(&ciphertext, &secret, &q_plan, &embedding, scale);

    let (input_max_abs, input_rel_l2) = error_metrics(&original, &input);

    println!("R16_5_18F1_INPUT_MAX_ABS={input_max_abs:.9e}");
    println!("R16_5_18F1_INPUT_REL_L2={input_rel_l2:.9e}");

    let exponent = rotation_exponent_left(degree, HALF);
    println!("R16_5_18F1_GALOIS_EXPONENT={exponent}");

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
        "R16_5_18F1_KEYGEN_SECONDS={:.6}",
        keygen_started.elapsed().as_secs_f64()
    );

    let rotation_started = Instant::now();

    let encrypted = RnsCkksCiphertext::new(
        ciphertext.clone(),
        CkksChainState::top(&chain, scale),
        &chain,
    );

    let shifted =
        rotate_left_hybrid_rns_ckks_with_prepared_ntt(&encrypted, HALF, &key, &chain, &qp_plan);

    let rotated = shifted.rlwe();

    println!(
        "R16_5_18F1_ROTATION_SECONDS={:.6}",
        rotation_started.elapsed().as_secs_f64()
    );

    let actual = decode(rotated, &secret, &q_plan, &embedding, scale);

    let expected_rotated: Vec<Complex64> = (0..slots)
        .map(|index| input[(index + HALF) % slots])
        .collect();

    let (rotation_max_abs, rotation_rel_l2) = error_metrics(&actual, &expected_rotated);

    println!("R16_5_18F1_ROTATION_MAX_ABS={rotation_max_abs:.9e}");
    println!("R16_5_18F1_ROTATION_REL_L2={rotation_rel_l2:.9e}");

    let low_mask: Vec<Complex64> = (0..slots)
        .map(|slot| Complex64::new(if slot < HALF { 1.0 } else { 0.0 }, 0.0))
        .collect();

    let prepared_ntt = PreparedRnsNttPlan::new(&q_plan);

    let mask_started = Instant::now();

    let prepared_mask =
        PreparedComplexSlotsCp::new(&shifted, &low_mask, &embedding, &chain, &q_plan);

    let extracted =
        multiply_complex_slots_cp_prepared(&shifted, &prepared_mask, &chain, &prepared_ntt);

    println!(
        "R16_5_18F1_MASK_SECONDS={:.6}",
        mask_started.elapsed().as_secs_f64()
    );
    println!("R16_5_18F1_OUTPUT_LEVEL={}", extracted.level());
    println!("R16_5_18F1_OUTPUT_SCALE={:.9e}", extracted.scale());

    assert_eq!(extracted.level(), 1);

    let level1_plan = RnsNttPlan::new(chain.level(1).moduli().to_vec(), degree);

    let decoded = decode(
        extracted.rlwe(),
        &secret,
        &level1_plan,
        &embedding,
        extracted.scale(),
    );

    let mut active_error_sq = 0.0_f64;
    let mut active_reference_sq = 0.0_f64;
    let mut active_max_abs = 0.0_f64;
    let mut inactive_error_sq = 0.0_f64;
    let mut inactive_max_abs = 0.0_f64;

    for (slot, value) in decoded.iter().enumerate().take(slots) {
        if slot < HALF {
            let packed_row = slot / 256;
            let packed_col = slot % 256;
            let natural_row = (packed_row % 4) * 4 + packed_row / 4;
            let natural_col = reverse_base4_4(packed_col);

            // Odd tile: source columns 256..511.
            let expected = sample(natural_row, 256 + natural_col);
            let error = (*value - expected).norm();

            active_error_sq += error * error;
            active_reference_sq += expected.norm_sqr();
            active_max_abs = active_max_abs.max(error);
        } else {
            let magnitude = value.norm();
            inactive_error_sq += magnitude * magnitude;
            inactive_max_abs = inactive_max_abs.max(magnitude);
        }
    }

    let active_rel_l2 = (active_error_sq / active_reference_sq.max(f64::MIN_POSITIVE)).sqrt();

    let inactive_rms = (inactive_error_sq / (slots - HALF) as f64).sqrt();

    println!("R16_5_18F1_ACTIVE_REL_L2={active_rel_l2:.12e}");
    println!("R16_5_18F1_ACTIVE_MAX_ABS={active_max_abs:.12e}");
    println!("R16_5_18F1_INACTIVE_RMS={inactive_rms:.12e}");
    println!("R16_5_18F1_INACTIVE_MAX_ABS={inactive_max_abs:.12e}");
    println!(
        "R16_5_18F1_TOTAL_SECONDS={:.6}",
        started.elapsed().as_secs_f64()
    );

    assert!(rotation_rel_l2.is_finite());
    assert!(active_rel_l2.is_finite());
    assert!(inactive_rms.is_finite());

    assert!(
        rotation_rel_l2 < 1.0e-6,
        "hybrid rotation precision regression"
    );
    assert!(
        active_rel_l2 < 1.0e-3,
        "hybrid extraction active-half precision regression"
    );
    assert!(
        inactive_rms < 1.0e-3,
        "hybrid extraction inactive-half leakage regression"
    );

    println!("R16_5_18F1_STATUS=PASS");
    println!("R16_5_18F1_SECURITY_STATUS=RESEARCH_PARAMETERS_ONLY");
}
