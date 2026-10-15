use std::time::Instant;

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    apply_bounded_rns_galois_automorphism_with_prepared_ntt_profiled,
    research_profile_65536_for_levels, rotate_left_bounded_rns_ckks_with_ntt,
    rotate_left_bounded_rns_ckks_with_prepared_ntt, rotation_exponent_left, BoundedRnsGaloisKey,
    CkksCanonicalEmbedding, CkksChainState, PreparedBoundedRnsGaloisKey, RnsCkksCiphertext,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const DEGREE: usize = 65536;
const ROTATION: usize = 32;
const SIGMA: f64 = 3.19;
const TARGET_REL_L2: f64 = 1.0e-3;

fn rotate_left_clear(values: &[Complex64], amount: usize) -> Vec<Complex64> {
    let amount = amount % values.len();
    (0..values.len())
        .map(|index| values[(index + amount) % values.len()])
        .collect()
}

fn error_metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());

    let mut error_sq = 0.0_f64;
    let mut reference_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (actual, expected) in actual.iter().zip(expected) {
        let error = *actual - *expected;
        error_sq += error.norm_sqr();
        reference_sq += expected.norm_sqr();
        max_abs = max_abs.max(error.norm());
    }

    (
        error_sq.sqrt() / reference_sq.sqrt().max(f64::MIN_POSITIVE),
        max_abs,
    )
}

fn decrypt_slots(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let basis = ciphertext.basis();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), ciphertext.rlwe().degree());
    let decrypted = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);
    let modulus = composite_modulus_big(basis);
    let coefficients = reconstruct_coefficients_big(&decrypted);
    let scale = ciphertext.state().scale();

    let decoded_coefficients = coefficients
        .iter()
        .map(|value| {
            centered_representative_big(value, &modulus)
                .to_f64()
                .expect("centered CKKS coefficient must fit f64")
                / scale
        })
        .collect::<Vec<_>>();

    embedding.coefficients_to_slots(&decoded_coefficients)
}

fn main() {
    println!("R22_5B_BEGIN");

    let profile = research_profile_65536_for_levels(14, 0);
    let chain = profile.modulus_chain();
    let basis = chain.top().clone();
    let scale = profile.initial_scale();
    let embedding = CkksCanonicalEmbedding::new(DEGREE);
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), DEGREE);

    assert_eq!(profile.degree(), DEGREE);

    let input = (0..embedding.slot_count())
        .map(|index| {
            Complex64::new(
                ((index % 257) as f64) / 257.0,
                ((index % 131) as f64) / 131.0,
            )
        })
        .collect::<Vec<_>>();

    let expected = rotate_left_clear(&input, ROTATION);
    let plaintext = encode_rns(&input, &embedding, &basis, scale);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x223C_0001);
    let mut secret: Vec<i8> = (0..DEGREE)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x223C_0002);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &plan,
        &mut encryption_rng,
    );

    let ciphertext = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    let baseline_decoded = decrypt_slots(&ciphertext, &secret, &embedding);
    let (baseline_rel_l2, baseline_max_abs) = error_metrics(&baseline_decoded, &input);

    println!("R22_5B_DEGREE={DEGREE}");
    println!("R22_5B_SLOTS={}", embedding.slot_count());
    println!("R22_5B_LIMBS={}", basis.len());
    println!("R22_5B_SCALE={scale:.12e}");
    println!("R22_5B_ROTATION={ROTATION}");
    println!("R22_5B_BASELINE rel_l2={baseline_rel_l2:.12e} max_abs={baseline_max_abs:.12e}");

    let exponent = rotation_exponent_left(DEGREE, ROTATION);
    let mut all_pass = baseline_rel_l2 < TARGET_REL_L2;

    {
        let base_log = 8_u32;
        let layout = BoundedGadgetLayout::new(basis.clone(), base_log);

        let mut key_rng = ChaCha20Rng::seed_from_u64(0x223C_1000 + u64::from(base_log));

        let keygen_start = Instant::now();

        let key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
            BoundedRnsKeygenConfig {
                plaintext_modulus: 2,
                layout,
                plan: &plan,
            },
            &secret,
            exponent,
            ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
            &mut key_rng,
        );

        let keygen_ms = keygen_start.elapsed().as_secs_f64() * 1_000.0;

        let rotation_start = Instant::now();

        let rotated =
            rotate_left_bounded_rns_ckks_with_ntt(&ciphertext, ROTATION, &key, &chain, &plan);

        let rotation_ms = rotation_start.elapsed().as_secs_f64() * 1_000.0;

        let preparation_start = Instant::now();
        let prepared_key = PreparedBoundedRnsGaloisKey::new(&key, &plan);
        let preparation_ms = preparation_start.elapsed().as_secs_f64() * 1_000.0;

        let prepared_start = Instant::now();
        let prepared_rotated = rotate_left_bounded_rns_ckks_with_prepared_ntt(
            &ciphertext,
            ROTATION,
            &prepared_key,
            &chain,
            &plan,
        );

        let profile_start = Instant::now();

        let (profiled_rlwe, profile) =
            apply_bounded_rns_galois_automorphism_with_prepared_ntt_profiled(
                ciphertext.rlwe(),
                &prepared_key,
                &plan,
            );

        let profiled_rotation_ms = profile_start.elapsed().as_secs_f64() * 1_000.0;

        assert_eq!(
            &profiled_rlwe,
            prepared_rotated.rlwe(),
            "profiled and unprofiled prepared rotations differ"
        );

        println!(
            "R22_5B_PROFILE automorphism_ms={:.3} decomposition_ms={:.3} base_forward_ms={:.3} digit_prepare_ms={:.3} digit_forward_ms={:.3} mac_ms={:.3} inverse_ms={:.3} total_ms={:.3}",
            profile.automorphism_seconds * 1_000.0,
            profile.decomposition_seconds * 1_000.0,
            profile.base_forward_seconds * 1_000.0,
            profile.digit_prepare_seconds * 1_000.0,
            profile.digit_forward_seconds * 1_000.0,
            profile.mac_seconds * 1_000.0,
            profile.inverse_seconds * 1_000.0,
            profiled_rotation_ms,
        );

        let prepared_rotation_ms = prepared_start.elapsed().as_secs_f64() * 1_000.0;

        let exact_equal = prepared_rotated.rlwe() == rotated.rlwe();

        println!(
            "R22_5B_PREPARED preparation_ms={:.3} rotation_ms={:.3} speedup={:.3} exact_equal={}",
            preparation_ms,
            prepared_rotation_ms,
            rotation_ms / prepared_rotation_ms,
            exact_equal,
        );

        assert!(
            exact_equal,
            "prepared and reference rotations differ at degree 65536"
        );

        let decoded = decrypt_slots(&rotated, &secret, &embedding);
        let (rel_l2, max_abs) = error_metrics(&decoded, &expected);

        let key_bytes =
            (key.entries().len() as u128) * (basis.len() as u128) * 2 * (DEGREE as u128) * 8;

        let pass = rel_l2.is_finite() && rel_l2 < TARGET_REL_L2;
        all_pass &= pass;

        println!(
            "R22_5B_CASE base_log={} digits={} key_mib={:.3} keygen_ms={:.3} rotation_ms={:.3} rel_l2={:.12e} max_abs={:.12e} status={}",
            base_log,
            key.entries().len(),
            key_bytes as f64 / (1024.0 * 1024.0),
            keygen_ms,
            rotation_ms,
            rel_l2,
            max_abs,
            if pass { "PASS" } else { "FAIL" },
        );
    }

    println!("R22_5B_STATUS={}", if all_pass { "PASS" } else { "FAIL" });

    if !all_pass {
        std::process::exit(1);
    }
}
