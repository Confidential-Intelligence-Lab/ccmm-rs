use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_65536, rotate_left_rns_ckks_with_prepared_ntt,
    rotate_right_rns_ckks_with_prepared_ntt, rotation_exponent_left, rotation_exponent_right,
    CkksCanonicalEmbedding, CkksChainState, PreparedRnsGaloisKey, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{
    execute_repeated_packed_fft2_dif_stage_cp_prepared, execute_repeated_packed_fft2_dif_stage_pp,
    fft2_pp, Fft2Shape, FftDirection, PackedFft2Axis, PackedFft2DifStageDiagonals,
    PreparedRepeatedPackedFft2DifStage,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
    RnsKeygenConfig,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
    PreparedRnsNttPlan, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

const SIGMA: f64 = 3.19;
const TILES_PER_CIPHERTEXT: usize = 8;

fn block_sizes(limb_count: usize) -> Vec<usize> {
    vec![1; limb_count]
}

fn deterministic_tile(tile_index: usize, elements: usize) -> Vec<Complex64> {
    (0..elements)
        .map(|index| {
            let real = ((37 * tile_index + 17 * index + 11) % 251) as f64 / 251.0;
            let imag = ((29 * tile_index + 13 * index + 7) % 127) as f64 / 127.0;

            Complex64::new(real, imag)
        })
        .collect()
}

fn error_metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());

    let error_sq = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>();

    let reference_sq = expected.iter().map(|value| value.norm_sqr()).sum::<f64>();

    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0_f64, f64::max);

    (
        if reference_sq == 0.0 {
            error_sq.sqrt()
        } else {
            (error_sq / reference_sq).sqrt()
        },
        max_abs,
    )
}

fn rotate_left_clear(values: &[Complex64], amount: usize) -> Vec<Complex64> {
    let amount = amount % values.len();

    (0..values.len())
        .map(|index| values[(index + amount) % values.len()])
        .collect()
}

fn rotate_right_clear(values: &[Complex64], amount: usize) -> Vec<Complex64> {
    let amount = amount % values.len();

    (0..values.len())
        .map(|index| values[(index + values.len() - amount) % values.len()])
        .collect()
}

fn bit_reverse(index: usize, length: usize) -> usize {
    if length <= 2 {
        return index;
    }

    let bits = length.trailing_zeros();
    index.reverse_bits() >> (usize::BITS - bits)
}

fn packed_physical_to_logical(physical: &[Complex64], shape: Fft2Shape) -> Vec<Complex64> {
    assert!(physical.len() >= shape.elements());

    let mut logical = vec![Complex64::new(0.0, 0.0); shape.elements()];

    for row in 0..shape.rows() {
        for col in 0..shape.cols() {
            let logical_index = row * shape.cols() + col;

            let physical_row = bit_reverse(row, shape.rows());
            let physical_col = bit_reverse(col, shape.cols());

            let physical_index = physical_row * shape.cols() + physical_col;

            logical[logical_index] = physical[physical_index];
        }
    }

    logical
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

    let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
        .iter()
        .map(|value| {
            centered_representative_big(value, &modulus)
                .to_f64()
                .expect("decoded coefficient must fit f64")
                / ciphertext.state().scale()
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

fn main() {
    let shape = Fft2Shape::new(64, 64);
    let tile_elements = shape.elements();

    /*
     * Twelve local FFT2 levels are required for a 64x64 transform.
     * Use the already validated N=65536 research profile so the packed
     * experiment has the same slot geometry as the large-image executor.
     */
    let profile = research_profile_65536();

    let degree = profile.degree();
    let chain = profile.modulus_chain();
    let top_basis = chain.top().clone();
    let scale = profile.initial_scale();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slot_count = embedding.slot_count();
    let top_plan = profile.rns_ntt_plan();

    let active_slots = tile_elements * TILES_PER_CIPHERTEXT;

    assert_eq!(tile_elements, 4096);
    assert_eq!(slot_count, 32768);
    assert!(active_slots <= slot_count);

    println!("MULTI_TILE_FFT_BEGIN");
    println!("MULTI_TILE_FFT_TILE_DIMENSION=64x64");
    println!("MULTI_TILE_FFT_TILES_PER_CIPHERTEXT={TILES_PER_CIPHERTEXT}");
    println!("MULTI_TILE_FFT_ACTIVE_SLOTS={active_slots}");
    println!("MULTI_TILE_FFT_SLOT_COUNT={slot_count}");
    println!(
        "MULTI_TILE_FFT_SLOT_UTILIZATION={:.6}",
        active_slots as f64 / slot_count as f64
    );
    println!("MULTI_TILE_FFT_RING_DEGREE={degree}");
    println!(
        "MULTI_TILE_FFT_TOP_LIMBS={}",
        profile.modulus_values().len()
    );
    println!("MULTI_TILE_FFT_LOCAL_LEVELS=12");

    let logical_tiles: Vec<Vec<Complex64>> = (0..TILES_PER_CIPHERTEXT)
        .map(|tile_index| deterministic_tile(tile_index, tile_elements))
        .collect();

    let clear_outputs: Vec<Vec<Complex64>> = logical_tiles
        .iter()
        .map(|tile| fft2_pp(shape, FftDirection::Forward, tile))
        .collect();

    let mut packed_input = vec![Complex64::new(0.0, 0.0); slot_count];

    for (tile_index, tile) in logical_tiles.iter().enumerate() {
        let start = tile_index * tile_elements;
        let end = start + tile_elements;
        packed_input[start..end].copy_from_slice(tile);
    }

    let plaintext = encode_rns(&packed_input, &embedding, &top_basis, scale);

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x4d55_4c54_4954_494c);

    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x4d55_4c54_4945_4e43);

    let encrypt_start = Instant::now();

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
        "MULTI_TILE_FFT_ENCRYPTION_MS={}",
        encrypt_start.elapsed().as_millis()
    );

    /*
     * Generate exactly the level-specific rotation keys required by the
     * twelve local FFT stages.
     */
    let keygen_total_start = Instant::now();

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();
    let mut execution_level = 0usize;

    for axis in [PackedFft2Axis::Rows, PackedFft2Axis::Columns] {
        let mut span = 64usize;

        while span >= 2 {
            let diagonals =
                PackedFft2DifStageDiagonals::new(shape, axis, span, FftDirection::Forward);

            let rotation = diagonals.rotation();

            let left_exponent = rotation_exponent_left(degree, rotation);
            let right_exponent = rotation_exponent_right(degree, rotation);

            let level_basis = chain.level(execution_level).clone();

            let level_layout =
                RnsGadgetLayout::new(level_basis.clone(), block_sizes(level_basis.len()));

            let level_plan = RnsNttPlan::new(level_basis.moduli().to_vec(), degree);

            let mut level_keys = RnsCkksLevelKeys::new(execution_level, level_basis);

            println!(
                "MULTI_TILE_FFT_KEYGEN_LEVEL_BEGIN level={} axis={:?} span={} rotation={} limbs={}",
                execution_level,
                axis,
                span,
                rotation,
                level_layout.full_basis().len()
            );

            let level_start = Instant::now();

            let ((left_key, left_ms), (right_key, right_ms)) = std::thread::scope(|scope| {
                let left_handle = scope.spawn(|| {
                    let start = Instant::now();

                    let mut rng = ChaCha20Rng::seed_from_u64(
                        0x4d55_4c54_494b_4559 ^ ((execution_level as u64) << 8),
                    );

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

                    let mut rng = ChaCha20Rng::seed_from_u64(
                        0x4d55_4c54_494b_4559 ^ ((execution_level as u64) << 8) ^ 1_u64,
                    );

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
                        .expect("multi-tile left Galois-key worker panicked"),
                    right_handle
                        .join()
                        .expect("multi-tile right Galois-key worker panicked"),
                )
            });

            println!(
                "MULTI_TILE_FFT_KEYGEN_KEY level={} direction=left exponent={} elapsed_ms={}",
                execution_level, left_exponent, left_ms
            );

            println!(
                "MULTI_TILE_FFT_KEYGEN_KEY level={} direction=right exponent={} elapsed_ms={}",
                execution_level, right_exponent, right_ms
            );

            level_keys.insert_galois_key(left_key);
            level_keys.insert_galois_key(right_key);

            evaluation_keys.insert_level(level_keys);

            println!(
                "MULTI_TILE_FFT_KEYGEN_LEVEL_END level={} elapsed_ms={}",
                execution_level,
                level_start.elapsed().as_millis()
            );

            execution_level += 1;
            span /= 2;
        }
    }

    assert_eq!(execution_level, 12);

    println!(
        "MULTI_TILE_FFT_KEYGEN_TOTAL_MS={}",
        keygen_total_start.elapsed().as_millis()
    );
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    execution_level = 0;

    let mut clear_packed_state = packed_input.clone();

    let fft_start = Instant::now();

    for axis in [PackedFft2Axis::Rows, PackedFft2Axis::Columns] {
        let mut span = 64usize;

        while span >= 2 {
            let diagonals =
                PackedFft2DifStageDiagonals::new(shape, axis, span, FftDirection::Forward);

            clear_packed_state = execute_repeated_packed_fft2_dif_stage_pp(
                &clear_packed_state,
                &diagonals,
                TILES_PER_CIPHERTEXT,
            );

            let rotation = diagonals.rotation();

            let left_exponent = rotation_exponent_left(degree, rotation);
            let right_exponent = rotation_exponent_right(degree, rotation);

            let stage_plan =
                RnsNttPlan::new(chain.level(execution_level).moduli().to_vec(), degree);

            let state = value.state();

            let prepared_left = PreparedRnsGaloisKey::new(
                evaluator.keys().galois_for(state, left_exponent),
                &stage_plan,
            );

            let prepared_right = PreparedRnsGaloisKey::new(
                evaluator.keys().galois_for(state, right_exponent),
                &stage_plan,
            );

            let prepared_plan = PreparedRnsNttPlan::new(&stage_plan);
            let prepared_fft_stage = PreparedRepeatedPackedFft2DifStage::new(
                &value,
                &diagonals,
                TILES_PER_CIPHERTEXT,
                &embedding,
                &chain,
                &stage_plan,
            );

            if execution_level == 0 {
                let rotated_left = rotate_left_rns_ckks_with_prepared_ntt(
                    &value,
                    rotation,
                    &prepared_left,
                    &chain,
                    &stage_plan,
                );

                let rotated_right = rotate_right_rns_ckks_with_prepared_ntt(
                    &value,
                    rotation,
                    &prepared_right,
                    &chain,
                    &stage_plan,
                );

                let decoded_left = decrypt_slots(&rotated_left, &secret, &embedding);

                let decoded_right = decrypt_slots(&rotated_right, &secret, &embedding);

                let clear_left = rotate_left_clear(&packed_input, rotation);

                let clear_right = rotate_right_clear(&packed_input, rotation);

                let (left_rel_l2, left_max_abs) = error_metrics(&decoded_left, &clear_left);

                let (right_rel_l2, right_max_abs) = error_metrics(&decoded_right, &clear_right);

                println!(
                    "MULTI_TILE_FFT_ROTATION_LEFT_CHECK rotation={} rel_l2={:.12e} max_abs={:.12e}",
                    rotation, left_rel_l2, left_max_abs,
                );

                println!(
                    "MULTI_TILE_FFT_ROTATION_RIGHT_CHECK rotation={} rel_l2={:.12e} max_abs={:.12e}",
                    rotation,
                    right_rel_l2,
                    right_max_abs,
                );
            }

            let stage_start = Instant::now();

            value = execute_repeated_packed_fft2_dif_stage_cp_prepared(
                &value,
                &prepared_fft_stage,
                &evaluator,
                (&prepared_left, &prepared_right),
                &chain,
                &stage_plan,
                &prepared_plan,
            );

            execution_level += 1;

            let decoded_stage = decrypt_slots(&value, &secret, &embedding);

            let (stage_rel_l2, stage_max_abs) = error_metrics(
                &decoded_stage[..active_slots],
                &clear_packed_state[..active_slots],
            );

            let inactive_stage_max_abs = decoded_stage[active_slots..]
                .iter()
                .map(|value| value.norm())
                .fold(0.0_f64, f64::max);

            println!(
                "MULTI_TILE_FFT_STAGE_CHECK level={} axis={:?} span={} rel_l2={:.12e} max_abs={:.12e} inactive_max_abs={:.12e}",
                execution_level,
                axis,
                span,
                stage_rel_l2,
                stage_max_abs,
                inactive_stage_max_abs,
            );

            println!(
                "MULTI_TILE_FFT_STAGE level={} axis={:?} span={} limbs={} elapsed_ms={}",
                execution_level,
                axis,
                span,
                value.basis().moduli().len(),
                stage_start.elapsed().as_millis()
            );

            span /= 2;
        }
    }

    let fft_ms = fft_start.elapsed().as_millis();

    assert_eq!(execution_level, 12);

    let decrypt_start = Instant::now();

    let decoded = decrypt_slots(&value, &secret, &embedding);

    let decrypt_ms = decrypt_start.elapsed().as_millis();

    let (packed_physical_rel_l2, packed_physical_max_abs) = error_metrics(
        &decoded[..active_slots],
        &clear_packed_state[..active_slots],
    );

    println!("MULTI_TILE_FFT_PACKED_PHYSICAL_REL_L2={packed_physical_rel_l2:.12e}");
    println!("MULTI_TILE_FFT_PACKED_PHYSICAL_MAX_ABS={packed_physical_max_abs:.12e}");

    let mut all_pass = true;

    for (tile_index, expected) in clear_outputs.iter().enumerate() {
        let start = tile_index * tile_elements;
        let end = start + tile_elements;

        let actual_physical = &decoded[start..end];

        let actual_logical = packed_physical_to_logical(actual_physical, shape);

        let (rel_l2, max_abs) = error_metrics(&actual_logical, expected);

        println!("MULTI_TILE_FFT_TILE_{tile_index}_REL_L2={rel_l2:.12e}");
        println!("MULTI_TILE_FFT_TILE_{tile_index}_MAX_ABS={max_abs:.12e}");

        /*
         * Match the numerical scale of the existing FFT validation
         * rather than imposing an unnecessarily strict cleartext bound.
         */
        if rel_l2 > 5.0e-3 {
            all_pass = false;
        }
    }

    let inactive_max_abs = decoded[active_slots..]
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);

    println!("MULTI_TILE_FFT_INACTIVE_MAX_ABS={inactive_max_abs:.12e}");
    println!("MULTI_TILE_FFT_EXECUTION_MS={fft_ms}");
    println!("MULTI_TILE_FFT_DECRYPT_MS={decrypt_ms}");
    println!(
        "MULTI_TILE_FFT_STATUS={}",
        if all_pass { "PASS" } else { "FAIL" }
    );
    println!("MULTI_TILE_FFT_END");

    assert!(all_pass, "multi-tile encrypted FFT validation failed");
}
