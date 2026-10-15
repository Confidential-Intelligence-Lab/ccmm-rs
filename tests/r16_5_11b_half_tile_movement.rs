// R16.5.11b: encrypted UHD half-tile extraction and repacking.
// Research correctness experiment; not a cryptographic security claim.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_16384, rotate_left_bounded_rns_ckks_with_prepared_execution,
    rotate_right_bounded_rns_ckks_with_prepared_execution, rotation_exponent_left,
    rotation_exponent_right, BoundedRnsGaloisKey, CkksCanonicalEmbedding, CkksChainState,
    PreparedBoundedRnsGaloisKey, RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::level1::{multiply_complex_slots_cp_prepared, PreparedComplexSlotsCp};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig,
};
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

const HALF: usize = 4096;
const SIGMA: f64 = 3.19;

fn decode(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let plan = RnsNttPlan::new(ciphertext.basis().moduli().to_vec(), embedding.degree());

    let decrypted = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);

    let modulus = composite_modulus_big(decrypted.basis());

    let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("CKKS coefficient must fit f64")
                / ciphertext.scale()
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

fn metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());

    let mut error_squared = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (actual, expected) in actual.iter().zip(expected) {
        let error = (*actual - *expected).norm();

        error_squared += error * error;
        reference_squared += expected.norm_sqr();
        max_abs = max_abs.max(error);
    }

    ((error_squared / reference_squared).sqrt(), max_abs)
}

#[test]
fn r16_5_11b_encrypted_half_tile_movement() {
    let total_start = Instant::now();

    let profile = research_profile_16384();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let basis0 = chain.top().clone();

    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    assert_eq!(degree, 16384);
    assert_eq!(slots, HALF * 2);

    let plan0 = RnsNttPlan::new(basis0.moduli().to_vec(), degree);
    let prepared0 = PreparedRnsNttPlan::new(&plan0);

    let basis1 = chain.level(1);
    let plan1 = RnsNttPlan::new(basis1.moduli().to_vec(), degree);
    let prepared1 = PreparedRnsNttPlan::new(&plan1);

    let secret: Vec<i8> = (0..degree)
        .map(|index| match index % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let input: Vec<Complex64> = (0..slots)
        .map(|slot| {
            if slot < HALF {
                Complex64::new(
                    0.05 + 0.01 * (slot as f64 * 0.013).sin(),
                    0.02 * (slot as f64 * 0.017).cos(),
                )
            } else {
                Complex64::new(
                    -0.04 + 0.01 * (slot as f64 * 0.011).cos(),
                    -0.03 * (slot as f64 * 0.019).sin(),
                )
            }
        })
        .collect();

    let low_mask: Vec<Complex64> = (0..slots)
        .map(|slot| {
            if slot < HALF {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();

    let expected_a: Vec<Complex64> = (0..slots)
        .map(|slot| {
            if slot < HALF {
                input[slot]
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();

    let expected_b: Vec<Complex64> = (0..slots)
        .map(|slot| {
            if slot < HALF {
                input[slot + HALF]
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x1651_1B00);

    let plaintext = encode_rns(&input, &embedding, &basis0, scale);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &plan0,
        &mut rng,
    );

    let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    // Generate both rotation keys independently.
    let keygen_start = Instant::now();

    let key_left = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: BoundedGadgetLayout::new(basis0.clone(), 8),
            plan: &plan0,
        },
        &secret,
        rotation_exponent_left(degree, HALF),
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &mut rng,
    );

    let key_right = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedRnsKeygenConfig {
            plaintext_modulus: 2,
            layout: BoundedGadgetLayout::new(basis1.clone(), 8),
            plan: &plan1,
        },
        &secret,
        rotation_exponent_right(degree, HALF),
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &mut rng,
    );

    let keygen_ms = keygen_start.elapsed().as_secs_f64() * 1000.0;

    let preparation_start = Instant::now();

    let prepared_left = PreparedBoundedRnsGaloisKey::new(&key_left, &plan0);

    let prepared_right = PreparedBoundedRnsGaloisKey::new(&key_right, &plan1);

    let prepared_mask =
        PreparedComplexSlotsCp::new(&ciphertext, &low_mask, &embedding, &chain, &plan0);

    let preparation_ms = preparation_start.elapsed().as_secs_f64() * 1000.0;

    let rotation_start = Instant::now();

    let rotated_left = rotate_left_bounded_rns_ckks_with_prepared_execution(
        &ciphertext,
        HALF,
        &prepared_left,
        &chain,
        &plan0,
        &prepared0,
    );

    let rotation_left_ms = rotation_start.elapsed().as_secs_f64() * 1000.0;

    let masking_start = Instant::now();

    let extracted_a =
        multiply_complex_slots_cp_prepared(&ciphertext, &prepared_mask, &chain, &prepared0);

    let extracted_b =
        multiply_complex_slots_cp_prepared(&rotated_left, &prepared_mask, &chain, &prepared0);

    let masking_ms = masking_start.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(extracted_a.level(), 1);
    assert_eq!(extracted_b.level(), 1);
    assert_eq!(extracted_a.scale(), scale);
    assert_eq!(extracted_b.scale(), scale);

    let rotation_right_start = Instant::now();

    let shifted_b = rotate_right_bounded_rns_ckks_with_prepared_execution(
        &extracted_b,
        HALF,
        &prepared_right,
        &chain,
        &plan1,
        &prepared1,
    );

    let rotation_right_ms = rotation_right_start.elapsed().as_secs_f64() * 1000.0;

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let repack_start = Instant::now();

    let repacked = evaluator.add(&extracted_a, &shifted_b);

    let repack_ms = repack_start.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(repacked.level(), 1);
    assert_eq!(repacked.scale(), scale);

    let actual_a = decode(&extracted_a, &secret, &embedding);
    let actual_b = decode(&extracted_b, &secret, &embedding);
    let actual_repacked = decode(&repacked, &secret, &embedding);

    let (a_rel_l2, a_max_abs) = metrics(&actual_a, &expected_a);
    let (b_rel_l2, b_max_abs) = metrics(&actual_b, &expected_b);
    let (repack_rel_l2, repack_max_abs) = metrics(&actual_repacked, &input);

    let inactive_leakage = actual_a[HALF..]
        .iter()
        .chain(actual_b[HALF..].iter())
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);

    println!("R16_MOVEMENT_DEGREE={degree}");
    println!("R16_MOVEMENT_SLOTS={slots}");
    println!("R16_MOVEMENT_KEYGEN_MS={keygen_ms:.3}");
    println!("R16_MOVEMENT_PREPARATION_MS={preparation_ms:.3}");
    println!("R16_MOVEMENT_ROTATE_LEFT_MS={rotation_left_ms:.3}");
    println!("R16_MOVEMENT_MASKING_MS={masking_ms:.3}");
    println!("R16_MOVEMENT_ROTATE_RIGHT_MS={rotation_right_ms:.3}");
    println!("R16_MOVEMENT_REPACK_ADD_MS={repack_ms:.3}");
    println!("R16_MOVEMENT_A_REL_L2={a_rel_l2:.12e}");
    println!("R16_MOVEMENT_A_MAX_ABS={a_max_abs:.12e}");
    println!("R16_MOVEMENT_B_REL_L2={b_rel_l2:.12e}");
    println!("R16_MOVEMENT_B_MAX_ABS={b_max_abs:.12e}");
    println!("R16_MOVEMENT_INACTIVE_LEAKAGE={inactive_leakage:.12e}");
    println!("R16_MOVEMENT_REPACK_REL_L2={repack_rel_l2:.12e}");
    println!("R16_MOVEMENT_REPACK_MAX_ABS={repack_max_abs:.12e}");
    println!(
        "R16_MOVEMENT_TOTAL_MS={:.3}",
        total_start.elapsed().as_secs_f64() * 1000.0
    );

    let pass = a_rel_l2 < 1.0e-3
        && b_rel_l2 < 1.0e-3
        && repack_rel_l2 < 1.0e-3
        && inactive_leakage < 1.0e-3;

    println!("R16_MOVEMENT_STATUS={}", if pass { "PASS" } else { "FAIL" });

    assert!(pass, "Encrypted half-tile movement validation failed");
}
