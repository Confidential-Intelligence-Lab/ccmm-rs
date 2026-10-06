use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    mod_switch_rns_ckks_to_next, research_profile_65536_for_levels, rotation_exponent_left,
    rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState, PreparedRnsGaloisKey,
    RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{
    execute_packed_tile_dif_stage_cp_prepared, execute_packed_tile_dif_stage_pp, FftDirection,
    PackedTileDifStageDiagonals,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
    RnsKeygenConfig,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

const SIGMA: f64 = 3.19;

const REQUIRED_LEVELS: usize = 20;
const TERMINAL_GUARD_LIMBS: usize = 2;

const TILE_ROWS: usize = 64;
const TILE_COLS: usize = 64;
const TILE_ELEMENTS: usize = TILE_ROWS * TILE_COLS;

const PACKED_TILES: usize = 8;
const SLOT_COUNT: usize = TILE_ELEMENTS * PACKED_TILES;

const FIRST_TILE_STAGE_LEVEL: usize = 11;

fn block_sizes(limb_count: usize) -> Vec<usize> {
    vec![1; limb_count]
}

fn deterministic_packed_input() -> Vec<Complex64> {
    (0..SLOT_COUNT)
        .map(|index| {
            let tile = index / TILE_ELEMENTS;
            let local = index % TILE_ELEMENTS;

            let real = ((37 * tile + 17 * local + 11) % 251) as f64 / 251.0;

            let imag = ((23 * tile + 13 * local + 7) % 127) as f64 / 127.0;

            Complex64::new(real, imag)
        })
        .collect()
}

fn error_metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());

    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (actual, expected) in actual.iter().zip(expected) {
        let error = *actual - *expected;

        squared_error += error.norm_sqr();
        squared_reference += expected.norm_sqr();
        max_abs = max_abs.max(error.norm());
    }

    let rel_l2 = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else {
        squared_error.sqrt()
    };

    (rel_l2, max_abs)
}

fn decrypt_slots(
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
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("centered CKKS coefficient must fit f64")
                / ciphertext.scale()
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

fn main() {
    let profile = research_profile_65536_for_levels(REQUIRED_LEVELS, TERMINAL_GUARD_LIMBS);

    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let top_basis = chain.top().clone();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let top_plan = profile.rns_ntt_plan();

    assert_eq!(degree, 65_536);
    assert_eq!(embedding.slot_count(), SLOT_COUNT);
    assert_eq!(
        top_basis.len(),
        REQUIRED_LEVELS + TERMINAL_GUARD_LIMBS,
        "R17 tile validation requires the requested execution depth plus terminal guard limbs"
    );
    assert!(
        chain.max_level() >= REQUIRED_LEVELS,
        "R17 tile validation requires {REQUIRED_LEVELS} levels but profile supports only {}",
        chain.max_level()
    );

    println!("R17_TILE_ENCRYPTED_BEGIN");
    println!("R17_TILE_PROFILE={}", profile.name());
    println!("R17_TILE_RING_DEGREE={degree}");
    println!("R17_TILE_SLOT_COUNT={}", embedding.slot_count());
    println!("R17_TILE_TOP_LIMBS={}", top_basis.len());
    println!("R17_TILE_REQUIRED_LEVELS={REQUIRED_LEVELS}");
    println!("R17_TILE_FIRST_EXECUTION_LEVEL={FIRST_TILE_STAGE_LEVEL}");

    let packed_input = deterministic_packed_input();

    let plaintext = encode_rns(&packed_input, &embedding, &top_basis, scale);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5231_3754_494c_4553);

    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let encryption_start = Instant::now();

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x5231_3745_4e43_5259);

    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &top_plan,
        &mut encryption_rng,
    );

    let mut value = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    println!(
        "R17_TILE_ENCRYPTION_MS={}",
        encryption_start.elapsed().as_millis()
    );

    /*
     * Move the fresh ciphertext to exactly level 11, corresponding to:
     *
     *   levels 0..3   global row
     *   levels 4..9   local row
     *   level 10      global column span 16
     *
     * We intentionally modulus-switch here rather than execute those
     * already-validated operations because this validator isolates only
     * the new R17 tile-block primitive.
     */
    let align_start = Instant::now();

    while value.level() < FIRST_TILE_STAGE_LEVEL {
        value = mod_switch_rns_ckks_to_next(&value, &chain);
    }

    assert_eq!(value.level(), FIRST_TILE_STAGE_LEVEL);

    println!(
        "R17_TILE_ALIGNMENT level={} limbs={} elapsed_ms={}",
        value.level(),
        value.basis().len(),
        align_start.elapsed().as_millis()
    );

    /*
     * Generate exactly the three level-specific Galois key pairs needed
     * for rotations 16384, 8192, and 4096.
     */
    let stages = [(11usize, 8usize), (12usize, 4usize), (13usize, 2usize)];

    let keygen_start = Instant::now();
    let mut evaluation_keys = RnsCkksEvaluationKeys::new();

    for (level, span_tiles) in stages {
        let diagonals = PackedTileDifStageDiagonals::new_column(
            TILE_ROWS,
            TILE_COLS,
            PACKED_TILES,
            span_tiles,
            FftDirection::Forward,
        );

        let rotation = diagonals.rotation();

        let left_exponent = rotation_exponent_left(degree, rotation);

        let right_exponent = rotation_exponent_right(degree, rotation);

        let level_basis = chain.level(level).clone();

        let level_layout =
            RnsGadgetLayout::new(level_basis.clone(), block_sizes(level_basis.len()));

        let level_plan = RnsNttPlan::new(level_basis.moduli().to_vec(), degree);

        let mut level_keys = RnsCkksLevelKeys::new(level, level_basis);

        println!(
            "R17_TILE_KEYGEN_BEGIN level={} span_tiles={} rotation={} limbs={}",
            level,
            span_tiles,
            rotation,
            level_layout.full_basis().len()
        );

        let level_start = Instant::now();
        if left_exponent == right_exponent {
            let start = Instant::now();

            let mut rng = ChaCha20Rng::seed_from_u64(0x5231_374b_4559_0000 ^ ((level as u64) << 8));

            let key = RnsGaloisKey::generate_with_ntt_rng(
                RnsKeygenConfig {
                    degree,
                    plaintext_modulus: 2,
                    noise_bound: 0,
                    layout: level_layout.clone(),
                    plan: &level_plan,
                },
                &secret,
                left_exponent,
                &mut rng,
            );

            println!(
                "R17_TILE_KEYGEN_KEY level={} direction=shared exponent={} elapsed_ms={}",
                level,
                left_exponent,
                start.elapsed().as_millis()
            );

            level_keys.insert_galois_key(key);
        } else {
            let ((left_key, left_ms), (right_key, right_ms)) = std::thread::scope(|scope| {
                let left_handle = scope.spawn(|| {
                    let start = Instant::now();

                    let mut rng =
                        ChaCha20Rng::seed_from_u64(0x5231_374b_4559_0000 ^ ((level as u64) << 8));

                    let key = RnsGaloisKey::generate_with_ntt_rng(
                        RnsKeygenConfig {
                            degree,
                            plaintext_modulus: 2,
                            noise_bound: 0,
                            layout: level_layout.clone(),
                            plan: &level_plan,
                        },
                        &secret,
                        left_exponent,
                        &mut rng,
                    );

                    (key, start.elapsed().as_millis())
                });

                let right_handle = scope.spawn(|| {
                    let start = Instant::now();

                    let mut rng =
                        ChaCha20Rng::seed_from_u64(0x5231_374b_4559_0001 ^ ((level as u64) << 8));

                    let key = RnsGaloisKey::generate_with_ntt_rng(
                        RnsKeygenConfig {
                            degree,
                            plaintext_modulus: 2,
                            noise_bound: 0,
                            layout: level_layout.clone(),
                            plan: &level_plan,
                        },
                        &secret,
                        right_exponent,
                        &mut rng,
                    );

                    (key, start.elapsed().as_millis())
                });

                (
                    left_handle
                        .join()
                        .expect("R17 tile left keygen worker panicked"),
                    right_handle
                        .join()
                        .expect("R17 tile right keygen worker panicked"),
                )
            });

            println!(
                "R17_TILE_KEYGEN_KEY level={} direction=left exponent={} elapsed_ms={}",
                level, left_exponent, left_ms
            );

            println!(
                "R17_TILE_KEYGEN_KEY level={} direction=right exponent={} elapsed_ms={}",
                level, right_exponent, right_ms
            );

            level_keys.insert_galois_key(left_key);
            level_keys.insert_galois_key(right_key);
        }

        evaluation_keys.insert_level(level_keys);

        println!(
            "R17_TILE_KEYGEN_END level={} elapsed_ms={}",
            level,
            level_start.elapsed().as_millis()
        );
    }

    println!(
        "R17_TILE_KEYGEN_TOTAL_MS={}",
        keygen_start.elapsed().as_millis()
    );

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let mut clear_state = packed_input;

    let execution_start = Instant::now();

    for (expected_level, span_tiles) in stages {
        assert_eq!(value.level(), expected_level);

        let diagonals = PackedTileDifStageDiagonals::new_column(
            TILE_ROWS,
            TILE_COLS,
            PACKED_TILES,
            span_tiles,
            FftDirection::Forward,
        );

        clear_state = execute_packed_tile_dif_stage_pp(&clear_state, &diagonals);

        let rotation = diagonals.rotation();

        let left_exponent = rotation_exponent_left(degree, rotation);

        let right_exponent = rotation_exponent_right(degree, rotation);

        let stage_plan = RnsNttPlan::new(chain.level(expected_level).moduli().to_vec(), degree);

        let state = value.state();

        let prepared_left = PreparedRnsGaloisKey::new(
            evaluator.keys().galois_for(state, left_exponent),
            &stage_plan,
        );

        let prepared_right = PreparedRnsGaloisKey::new(
            evaluator.keys().galois_for(state, right_exponent),
            &stage_plan,
        );

        let stage_start = Instant::now();

        value = execute_packed_tile_dif_stage_cp_prepared(
            &value,
            &diagonals,
            &evaluator,
            (&prepared_left, &prepared_right),
            &embedding,
            &chain,
            &stage_plan,
        );

        let stage_ms = stage_start.elapsed().as_millis();

        let decoded = decrypt_slots(&value, &secret, &embedding);

        let (rel_l2, max_abs) = error_metrics(&decoded, &clear_state);

        println!(
            "R17_TILE_STAGE_CHECK input_level={} output_level={} span_tiles={} rotation={} limbs={} rel_l2={:.12e} max_abs={:.12e} elapsed_ms={}",
            expected_level,
            value.level(),
            span_tiles,
            rotation,
            value.basis().len(),
            rel_l2,
            max_abs,
            stage_ms
        );

        assert!(
            rel_l2 <= 5.0e-3,
            "R17 encrypted tile-block stage exceeded tolerance"
        );
    }

    let total_execution_ms = execution_start.elapsed().as_millis();

    let decoded = decrypt_slots(&value, &secret, &embedding);

    let (final_rel_l2, final_max_abs) = error_metrics(&decoded, &clear_state);

    let status = if final_rel_l2 <= 5.0e-3 {
        "PASS"
    } else {
        "FAIL"
    };

    println!("R17_TILE_RESULT_BEGIN");
    println!("R17_TILE_FINAL_LEVEL={}", value.level());
    println!("R17_TILE_FINAL_LIMBS={}", value.basis().len());
    println!("R17_TILE_EXECUTION_MS={total_execution_ms}");
    println!("R17_TILE_REL_L2={final_rel_l2:.12e}");
    println!("R17_TILE_MAX_ABS={final_max_abs:.12e}");
    println!("R17_TILE_STATUS={status}");
    println!("R17_TILE_RESULT_END");

    assert_eq!(status, "PASS");
}
