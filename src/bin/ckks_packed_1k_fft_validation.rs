use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    apply_rns_galois_automorphism_with_prepared_dynamic_ntt_profiled,
    research_profile_65536_for_levels, rotation_exponent_left, rotation_exponent_right,
    CkksCanonicalEmbedding, CkksChainState, PreparedRnsGaloisKey, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{
    execute_packed_tile_dif_stage_cp_prepared, execute_repeated_packed_fft2_dif_stage_cp_prepared,
    fft2_pp, packed_fft_dif_butterfly_cp_prepared, Fft2Shape, FftDirection, PackedFft2Axis,
    PackedFft2DifStageDiagonals, PackedTileDifStageDiagonals, PreparedRepeatedPackedFft2DifStage,
};
use ccmm_rs::eblas::PreparedComplexSlotsCp;
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
use std::f64::consts::PI;
use std::io::{self, Write};
use std::time::Instant;

const SIGMA: f64 = 3.19;
const TOLERANCE: f64 = 5.0e-3;

const IMAGE_DIM: usize = 1024;
const TILE_DIM: usize = 64;
const TILE_ELEMENTS: usize = TILE_DIM * TILE_DIM;
const TILES_PER_AXIS: usize = IMAGE_DIM / TILE_DIM;

const TILES_PER_CIPHERTEXT: usize = 8;
const SLOT_COUNT: usize = TILE_ELEMENTS * TILES_PER_CIPHERTEXT;

const ROW_BLOCKS: usize = TILES_PER_AXIS / TILES_PER_CIPHERTEXT;
const CIPHERTEXTS_PER_ROW_BLOCK: usize = TILES_PER_AXIS;
const CIPHERTEXT_COUNT: usize = ROW_BLOCKS * CIPHERTEXTS_PER_ROW_BLOCK;

const REQUIRED_LEVELS: usize = 20;
const TERMINAL_GUARD_LIMBS: usize = 2;

const DEFAULT_LOCAL_OUTER_PARALLELISM: usize = 2;
const DEFAULT_INTER_CT_PARALLELISM: usize = 4;

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => {
            let parsed = value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be a positive integer"));

            assert!(parsed > 0, "{name} must be greater than zero");
            parsed
        }
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("failed to read {name}: {error}"),
    }
}

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => panic!("failed to read {name}: {error}"),
    }
}

fn block_sizes(limb_count: usize) -> Vec<usize> {
    vec![1; limb_count]
}

fn bit_reverse(index: usize, length: usize) -> usize {
    if length <= 2 {
        return index;
    }

    let bits = length.trailing_zeros();
    index.reverse_bits() >> (usize::BITS - bits)
}

fn packed_physical_to_logical(physical: &[Complex64], shape: Fft2Shape) -> Vec<Complex64> {
    assert_eq!(physical.len(), shape.elements());

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

fn deterministic_image() -> Vec<Complex64> {
    (0..IMAGE_DIM)
        .flat_map(|row| {
            (0..IMAGE_DIM).map(move |col| {
                let real = ((17 * row + 29 * col + 11) % 251) as f64 / 251.0;

                let imag = ((13 * row + 19 * col + 7) % 127) as f64 / 127.0;

                Complex64::new(real, imag)
            })
        })
        .collect()
}

fn packed_location(tile_row: usize, tile_col: usize) -> (usize, usize) {
    let row_block = tile_row / TILES_PER_CIPHERTEXT;
    let lane = tile_row % TILES_PER_CIPHERTEXT;

    let packed_index = row_block * CIPHERTEXTS_PER_ROW_BLOCK + tile_col;

    (packed_index, lane)
}

fn pack_image_tiles(image: &[Complex64]) -> Vec<Vec<Complex64>> {
    assert_eq!(image.len(), IMAGE_DIM * IMAGE_DIM);

    let mut packed = vec![vec![Complex64::new(0.0, 0.0); SLOT_COUNT]; CIPHERTEXT_COUNT];

    for tile_row in 0..TILES_PER_AXIS {
        for tile_col in 0..TILES_PER_AXIS {
            let (packed_index, lane) = packed_location(tile_row, tile_col);

            let lane_base = lane * TILE_ELEMENTS;

            for local_row in 0..TILE_DIM {
                for local_col in 0..TILE_DIM {
                    let global_row = tile_row * TILE_DIM + local_row;
                    let global_col = tile_col * TILE_DIM + local_col;

                    let global_index = global_row * IMAGE_DIM + global_col;

                    let tile_slot = local_row * TILE_DIM + local_col;

                    packed[packed_index][lane_base + tile_slot] = image[global_index];
                }
            }
        }
    }

    packed
}

fn reconstruct_physical_image(packed: &[Vec<Complex64>]) -> Vec<Complex64> {
    assert_eq!(packed.len(), CIPHERTEXT_COUNT);

    let mut physical = vec![Complex64::new(0.0, 0.0); IMAGE_DIM * IMAGE_DIM];

    for tile_row in 0..TILES_PER_AXIS {
        for tile_col in 0..TILES_PER_AXIS {
            let (packed_index, lane) = packed_location(tile_row, tile_col);

            let lane_base = lane * TILE_ELEMENTS;

            for local_row in 0..TILE_DIM {
                for local_col in 0..TILE_DIM {
                    let tile_slot = local_row * TILE_DIM + local_col;

                    let global_row = tile_row * TILE_DIM + local_row;
                    let global_col = tile_col * TILE_DIM + local_col;

                    let global_index = global_row * IMAGE_DIM + global_col;

                    physical[global_index] = packed[packed_index][lane_base + tile_slot];
                }
            }
        }
    }

    physical
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

fn decrypt_packed_tensor(
    values: &[RnsCkksCiphertext],
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Vec<Complex64>> {
    values
        .iter()
        .map(|value| decrypt_slots(value, secret, embedding))
        .collect()
}

fn packed_tensor_metrics(actual: &[Vec<Complex64>], expected: &[Vec<Complex64>]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());

    let actual_flat = actual
        .iter()
        .flat_map(|value| value.iter().copied())
        .collect::<Vec<_>>();

    let expected_flat = expected
        .iter()
        .flat_map(|value| value.iter().copied())
        .collect::<Vec<_>>();

    error_metrics(&actual_flat, &expected_flat)
}

fn print_checkpoint(
    level: usize,
    encrypted: &[RnsCkksCiphertext],
    clear: &[Vec<Complex64>],
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) {
    let start = Instant::now();

    let decoded = decrypt_packed_tensor(encrypted, secret, embedding);

    let (rel_l2, max_abs) = packed_tensor_metrics(&decoded, clear);

    println!(
        "R17_1K_CHECK level={} limbs={} rel_l2={:.12e} max_abs={:.12e} elapsed_ms={}",
        level,
        encrypted[0].basis().len(),
        rel_l2,
        max_abs,
        start.elapsed().as_millis()
    );

    io::stdout()
        .flush()
        .expect("R17 checkpoint output must flush");
}

fn clear_global_row_stage(values: Vec<Vec<Complex64>>, span_tiles: usize) -> Vec<Vec<Complex64>> {
    let half_tiles = span_tiles / 2;
    let global_span = span_tiles * TILE_DIM;

    let mut output = vec![vec![Complex64::new(0.0, 0.0); SLOT_COUNT]; CIPHERTEXT_COUNT];

    for row_block in 0..ROW_BLOCKS {
        for group_start in (0..TILES_PER_AXIS).step_by(span_tiles) {
            for tile_offset in 0..half_tiles {
                let upper_col = group_start + tile_offset;
                let lower_col = upper_col + half_tiles;

                let upper_index = row_block * TILES_PER_AXIS + upper_col;
                let lower_index = row_block * TILES_PER_AXIS + lower_col;

                for lane in 0..TILES_PER_CIPHERTEXT {
                    let lane_base = lane * TILE_ELEMENTS;

                    for local_row in 0..TILE_DIM {
                        for local_col in 0..TILE_DIM {
                            let tile_slot = local_row * TILE_DIM + local_col;

                            let slot = lane_base + tile_slot;

                            let twiddle_index = tile_offset * TILE_DIM + local_col;

                            let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

                            let w = Complex64::new(angle.cos(), angle.sin());

                            let a = values[upper_index][slot];
                            let b = values[lower_index][slot];

                            output[upper_index][slot] = a + b;
                            output[lower_index][slot] = w * (a - b);
                        }
                    }
                }
            }
        }
    }

    output
}

fn clear_global_column_inter_stage(values: Vec<Vec<Complex64>>) -> Vec<Vec<Complex64>> {
    let span_tiles = 16usize;
    let global_span = span_tiles * TILE_DIM;

    let mut output = vec![vec![Complex64::new(0.0, 0.0); SLOT_COUNT]; CIPHERTEXT_COUNT];

    for tile_col in 0..TILES_PER_AXIS {
        let upper_index = tile_col;
        let lower_index = TILES_PER_AXIS + tile_col;

        for lane in 0..TILES_PER_CIPHERTEXT {
            let lane_base = lane * TILE_ELEMENTS;
            let tile_offset = lane;

            for local_row in 0..TILE_DIM {
                let twiddle_index = tile_offset * TILE_DIM + local_row;

                let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

                let w = Complex64::new(angle.cos(), angle.sin());

                for local_col in 0..TILE_DIM {
                    let tile_slot = local_row * TILE_DIM + local_col;

                    let slot = lane_base + tile_slot;

                    let a = values[upper_index][slot];
                    let b = values[lower_index][slot];

                    output[upper_index][slot] = a + b;
                    output[lower_index][slot] = w * (a - b);
                }
            }
        }
    }

    output
}

fn key_schedule() -> Vec<(usize, usize)> {
    let mut schedule = Vec::new();

    /*
     * Local row stages, levels 4..9.
     */
    let mut level = 4usize;
    let mut rotation = TILE_DIM / 2;

    while rotation > 0 {
        schedule.push((level, rotation));
        level += 1;
        rotation /= 2;
    }

    assert_eq!(level, 10);

    /*
     * Level 10 is the inter-CT global-column span-16 stage.
     *
     * Levels 11..13 are the new tile-block stages.
     */
    schedule.push((11, 16_384));
    schedule.push((12, 8_192));
    schedule.push((13, 4_096));

    /*
     * Local column stages, levels 14..19.
     *
     * For a 64x64 tile the six column-stage rotations are:
     * 2048, 1024, 512, 256, 128, 64.
     */
    level = 14;
    rotation = (TILE_DIM / 2) * TILE_DIM;

    while rotation >= TILE_DIM {
        schedule.push((level, rotation));
        level += 1;
        rotation /= 2;
    }

    assert_eq!(level, 20);
    assert_eq!(schedule.len(), 15);

    schedule
}

fn main() {
    assert_eq!(TILES_PER_AXIS, 16);
    assert_eq!(ROW_BLOCKS, 2);
    assert_eq!(CIPHERTEXT_COUNT, 32);
    assert_eq!(SLOT_COUNT, 32_768);

    let benchmark_mode = env_flag("R17_1K_BENCHMARK");

    let local_outer_parallelism =
        env_usize("R17_1K_LOCAL_PARALLELISM", DEFAULT_LOCAL_OUTER_PARALLELISM);

    let inter_ct_parallelism =
        env_usize("R17_1K_INTER_CT_PARALLELISM", DEFAULT_INTER_CT_PARALLELISM);

    let profile = research_profile_65536_for_levels(REQUIRED_LEVELS, TERMINAL_GUARD_LIMBS);

    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let top_basis = chain.top().clone();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let top_plan = profile.rns_ntt_plan();

    assert_eq!(degree, 65_536);
    assert_eq!(embedding.slot_count(), SLOT_COUNT);
    assert_eq!(top_basis.len(), REQUIRED_LEVELS + TERMINAL_GUARD_LIMBS);
    assert!(chain.max_level() >= REQUIRED_LEVELS);

    let image_shape = Fft2Shape::new(IMAGE_DIM, IMAGE_DIM);

    let tile_shape = Fft2Shape::new(TILE_DIM, TILE_DIM);

    println!("R17_1K_BEGIN");
    println!("R17_1K_IMAGE_DIMENSION=1024x1024");
    println!("R17_1K_TILE_DIMENSION=64x64");
    println!("R17_1K_TILE_GRID=16x16");
    println!("R17_1K_PACKING=8x1");
    println!("R17_1K_TILES_PER_CIPHERTEXT={TILES_PER_CIPHERTEXT}");
    println!("R17_1K_CIPHERTEXT_COUNT={CIPHERTEXT_COUNT}");
    println!("R17_1K_SLOT_COUNT={SLOT_COUNT}");
    println!("R17_1K_SLOT_UTILIZATION=1.000000");
    println!("R17_1K_PROFILE={}", profile.name());
    println!("R17_1K_RING_DEGREE={degree}");
    println!("R17_1K_TOP_LIMBS={}", top_basis.len());
    println!("R17_1K_REQUIRED_LEVELS={REQUIRED_LEVELS}");
    println!("R17_1K_BENCHMARK_MODE={benchmark_mode}");
    println!("R17_1K_LOCAL_OUTER_PARALLELISM={local_outer_parallelism}");
    println!("R17_1K_INTER_CT_PARALLELISM={inter_ct_parallelism}");

    io::stdout().flush().expect("R17 header output must flush");

    let total_start = Instant::now();

    /*
     * ---------------------------------------------------------------
     * Clear oracle + packed clear state.
     * ---------------------------------------------------------------
     */
    let image = deterministic_image();

    let oracle_start = Instant::now();

    let oracle = fft2_pp(image_shape, FftDirection::Forward, &image);

    println!("R17_1K_ORACLE_MS={}", oracle_start.elapsed().as_millis());

    let mut clear_values = pack_image_tiles(&image);

    /*
     * ---------------------------------------------------------------
     * Encode and encrypt 32 fully occupied packed vectors.
     * ---------------------------------------------------------------
     */
    let encode_start = Instant::now();

    let plaintexts = clear_values
        .iter()
        .map(|slots| encode_rns(slots, &embedding, &top_basis, scale))
        .collect::<Vec<_>>();

    println!("R17_1K_ENCODING_MS={}", encode_start.elapsed().as_millis());

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5231_3746_554c_4c31);

    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let encryption_start = Instant::now();

    let mut encrypted = Vec::with_capacity(CIPHERTEXT_COUNT);

    for batch_start in (0..CIPHERTEXT_COUNT).step_by(local_outer_parallelism) {
        let batch_end = (batch_start + local_outer_parallelism).min(CIPHERTEXT_COUNT);

        let mut batch = std::thread::scope(|scope| {
            let mut workers = Vec::new();

            for (index, plaintext) in plaintexts
                .iter()
                .enumerate()
                .take(batch_end)
                .skip(batch_start)
            {
                let secret = &secret;
                let top_plan = &top_plan;
                let chain = &chain;

                workers.push((
                    index,
                    scope.spawn(move || {
                        let mut rng =
                            ChaCha20Rng::seed_from_u64(0x5231_3745_4e43_0000 ^ index as u64);

                        let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                            plaintext,
                            2,
                            ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                            secret,
                            top_plan,
                            &mut rng,
                        );

                        RnsCkksCiphertext::new(rlwe, CkksChainState::top(chain, scale), chain)
                    }),
                ));
            }

            workers
                .into_iter()
                .map(|(index, worker)| {
                    (
                        index,
                        worker.join().expect("R17 encryption worker panicked"),
                    )
                })
                .collect::<Vec<_>>()
        });

        encrypted.append(&mut batch);

        println!(
            "R17_1K_ENCRYPT_PROGRESS completed={}/{}",
            batch_end, CIPHERTEXT_COUNT
        );

        io::stdout()
            .flush()
            .expect("R17 encryption progress must flush");
    }

    encrypted.sort_by_key(|(index, _)| *index);

    let mut values = encrypted
        .into_iter()
        .map(|(_, ciphertext)| ciphertext)
        .collect::<Vec<_>>();

    let encryption_ms = encryption_start.elapsed().as_millis();

    println!("R17_1K_ENCRYPTION_MS={encryption_ms}");

    /*
     * ---------------------------------------------------------------
     * Evaluation keys.
     * ---------------------------------------------------------------
     */
    let schedule = key_schedule();

    println!("R17_1K_GALOIS_LEVEL_COUNT={}", schedule.len());

    let keygen_start = Instant::now();

    let mut evaluation_keys = RnsCkksEvaluationKeys::new();

    let mut distinct_key_count = 0usize;

    for &(level, rotation) in &schedule {
        let level_basis = chain.level(level).clone();

        let level_layout =
            RnsGadgetLayout::new(level_basis.clone(), block_sizes(level_basis.len()));

        let level_plan = RnsNttPlan::new(level_basis.moduli().to_vec(), degree);

        let left_exponent = rotation_exponent_left(degree, rotation);

        let right_exponent = rotation_exponent_right(degree, rotation);

        let mut level_keys = RnsCkksLevelKeys::new(level, level_basis);

        println!(
            "R17_1K_KEYGEN_BEGIN level={} rotation={} limbs={} left_exponent={} right_exponent={}",
            level,
            rotation,
            level_layout.full_basis().len(),
            left_exponent,
            right_exponent
        );

        io::stdout()
            .flush()
            .expect("R17 keygen progress must flush");

        let level_start = Instant::now();

        if left_exponent == right_exponent {
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

            level_keys.insert_galois_key(key);
            distinct_key_count += 1;

            println!(
                "R17_1K_KEYGEN_KEY level={} direction=shared exponent={}",
                level, left_exponent
            );
        } else {
            let ((left_key, left_ms), (right_key, right_ms)) = std::thread::scope(|scope| {
                let left_worker = scope.spawn(|| {
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

                let right_worker = scope.spawn(|| {
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
                    left_worker.join().expect("R17 left keygen worker panicked"),
                    right_worker
                        .join()
                        .expect("R17 right keygen worker panicked"),
                )
            });

            println!(
                "R17_1K_KEYGEN_KEY level={} direction=left exponent={} elapsed_ms={}",
                level, left_exponent, left_ms
            );

            println!(
                "R17_1K_KEYGEN_KEY level={} direction=right exponent={} elapsed_ms={}",
                level, right_exponent, right_ms
            );

            level_keys.insert_galois_key(left_key);
            level_keys.insert_galois_key(right_key);

            distinct_key_count += 2;
        }

        evaluation_keys.insert_level(level_keys);

        println!(
            "R17_1K_KEYGEN_END level={} elapsed_ms={}",
            level,
            level_start.elapsed().as_millis()
        );
    }

    let keygen_ms = keygen_start.elapsed().as_millis();

    println!("R17_1K_GALOIS_DISTINCT_KEY_COUNT={distinct_key_count}");
    println!("R17_1K_KEYGEN_TOTAL_MS={keygen_ms}");

    assert_eq!(distinct_key_count, 29);

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    /*
     * ---------------------------------------------------------------
     * Encrypted FFT execution.
     * ---------------------------------------------------------------
     */
    let fft_start = Instant::now();
    let mut level = 0usize;

    /*
     * Global row stages: levels 0..3.
     */
    let mut span_tiles = 16usize;

    while span_tiles >= 2 {
        let stage_start = Instant::now();

        let half_tiles = span_tiles / 2;
        let global_span = span_tiles * TILE_DIM;

        /*
         * Public row twiddles depend only on the current global span and
         * tile offset. Build each distinct vector once per stage and share
         * it across all ciphertext-pair workers.
         */
        let row_twiddle_cache = (0..half_tiles)
            .map(|tile_offset| {
                let mut twiddles = vec![Complex64::new(0.0, 0.0); SLOT_COUNT];

                for lane in 0..TILES_PER_CIPHERTEXT {
                    let lane_base = lane * TILE_ELEMENTS;

                    for local_row in 0..TILE_DIM {
                        for local_col in 0..TILE_DIM {
                            let tile_slot = local_row * TILE_DIM + local_col;

                            let slot = lane_base + tile_slot;

                            let twiddle_index = tile_offset * TILE_DIM + local_col;

                            let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

                            twiddles[slot] = Complex64::new(angle.cos(), angle.sin());
                        }
                    }
                }

                twiddles
            })
            .collect::<Vec<_>>();

        let stage_plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), degree);
        let prepared_plan = PreparedRnsNttPlan::new(&stage_plan);

        let prepared_row_twiddle_cache = row_twiddle_cache
            .iter()
            .map(|twiddles| {
                PreparedComplexSlotsCp::new(&values[0], twiddles, &embedding, &chain, &stage_plan)
            })
            .collect::<Vec<_>>();

        let stage_input = values;
        let mut stage_output: Vec<Option<RnsCkksCiphertext>> =
            (0..CIPHERTEXT_COUNT).map(|_| None).collect();

        let mut jobs = Vec::new();

        for row_block in 0..ROW_BLOCKS {
            for group_start in (0..TILES_PER_AXIS).step_by(span_tiles) {
                for tile_offset in 0..half_tiles {
                    let upper_col = group_start + tile_offset;
                    let lower_col = upper_col + half_tiles;

                    let upper_index = row_block * TILES_PER_AXIS + upper_col;
                    let lower_index = row_block * TILES_PER_AXIS + lower_col;

                    jobs.push((upper_index, lower_index, tile_offset));
                }
            }
        }

        for batch_start in (0..jobs.len()).step_by(inter_ct_parallelism) {
            let batch_end = (batch_start + inter_ct_parallelism).min(jobs.len());

            let batch = std::thread::scope(|scope| {
                let mut workers = Vec::new();

                for &(upper_index, lower_index, tile_offset) in &jobs[batch_start..batch_end] {
                    let a = &stage_input[upper_index];
                    let b = &stage_input[lower_index];

                    let evaluator_ref = &evaluator;
                    let chain_ref = &chain;
                    let prepared_plan_ref = &prepared_plan;
                    let prepared_twiddles = &prepared_row_twiddle_cache[tile_offset];

                    workers.push((
                        upper_index,
                        lower_index,
                        scope.spawn(move || {
                            packed_fft_dif_butterfly_cp_prepared(
                                evaluator_ref,
                                a,
                                b,
                                prepared_twiddles,
                                chain_ref,
                                prepared_plan_ref,
                            )
                        }),
                    ));
                }

                workers
                    .into_iter()
                    .map(|(upper_index, lower_index, worker)| {
                        let (upper, lower) = worker.join().expect("R17 global-row worker panicked");

                        (upper_index, lower_index, upper, lower)
                    })
                    .collect::<Vec<_>>()
            });

            for (upper_index, lower_index, upper, lower) in batch {
                stage_output[upper_index] = Some(upper);
                stage_output[lower_index] = Some(lower);
            }
        }

        values = stage_output
            .into_iter()
            .map(|value| value.expect("R17 global-row output missing"))
            .collect();

        clear_values = clear_global_row_stage(clear_values, span_tiles);

        level += 1;

        println!(
            "R17_1K_STAGE level={} kind=global-row span_tiles={} limbs={} elapsed_ms={}",
            level,
            span_tiles,
            values[0].basis().len(),
            stage_start.elapsed().as_millis()
        );

        io::stdout().flush().expect("R17 stage output must flush");

        span_tiles /= 2;
    }

    assert_eq!(level, 4);

    if !benchmark_mode {
        print_checkpoint(level, &values, &clear_values, &secret, &embedding);
    }

    /*
     * Local row stages: levels 4..9.
     */
    let mut span = TILE_DIM;

    while span >= 2 {
        let stage_start = Instant::now();

        let diagonals = PackedFft2DifStageDiagonals::new(
            tile_shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        let rotation = diagonals.rotation();

        let left_exponent = rotation_exponent_left(degree, rotation);

        let right_exponent = rotation_exponent_right(degree, rotation);

        let stage_plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), degree);

        let state = values[0].state();

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
            &values[0],
            &diagonals,
            TILES_PER_CIPHERTEXT,
            &embedding,
            &chain,
            &stage_plan,
        );

        let stage_input = values;

        if env_flag("R17_1K_ROTATION_PROFILE") && level == 4 && span == TILE_DIM {
            let prepared_stage_plan = PreparedRnsNttPlan::new(&stage_plan);
            let input = &stage_input[0];

            println!("R17_1K_ROTATION_PROFILE_BEGIN");
            println!("R17_1K_ROTATION_PROFILE_LEVEL={}", level);
            println!("R17_1K_ROTATION_PROFILE_SPAN={}", span);
            println!("R17_1K_ROTATION_PROFILE_ROTATION={}", rotation);
            println!(
                "R17_1K_ROTATION_PROFILE_LIMBS={}",
                input.basis().moduli().len()
            );

            let profile_one = |name: &str, key: &PreparedRnsGaloisKey| {
                let total_start = Instant::now();

                let (_output, automorphism_seconds, profile) =
                    apply_rns_galois_automorphism_with_prepared_dynamic_ntt_profiled(
                        input.rlwe(),
                        key,
                        &prepared_stage_plan,
                    );

                let total_seconds = total_start.elapsed().as_secs_f64();

                let accounted_key_switch_seconds = profile.decompose_seconds
                    + profile.base_forward_seconds
                    + profile.digit_prepare_seconds
                    + profile.digit_forward_seconds
                    + profile.mac_seconds
                    + profile.inverse_seconds;

                println!(
                    "R17_1K_ROTATION_PROFILE_{}_AUTOMORPHISM_MS={:.3}",
                    name,
                    automorphism_seconds * 1.0e3
                );
                println!(
                    "R17_1K_ROTATION_PROFILE_{}_DECOMPOSE_MS={:.3}",
                    name,
                    profile.decompose_seconds * 1.0e3
                );
                println!(
                    "R17_1K_ROTATION_PROFILE_{}_BASE_FORWARD_MS={:.3}",
                    name,
                    profile.base_forward_seconds * 1.0e3
                );
                println!(
                    "R17_1K_ROTATION_PROFILE_{}_DIGIT_PREPARE_MS={:.3}",
                    name,
                    profile.digit_prepare_seconds * 1.0e3
                );
                println!(
                    "R17_1K_ROTATION_PROFILE_{}_DIGIT_FORWARD_MS={:.3}",
                    name,
                    profile.digit_forward_seconds * 1.0e3
                );
                println!(
                    "R17_1K_ROTATION_PROFILE_{}_MAC_MS={:.3}",
                    name,
                    profile.mac_seconds * 1.0e3
                );
                println!(
                    "R17_1K_ROTATION_PROFILE_{}_INVERSE_MS={:.3}",
                    name,
                    profile.inverse_seconds * 1.0e3
                );
                println!(
                    "R17_1K_ROTATION_PROFILE_{}_KEY_SWITCH_ACCOUNTED_MS={:.3}",
                    name,
                    accounted_key_switch_seconds * 1.0e3
                );
                println!(
                    "R17_1K_ROTATION_PROFILE_{}_TOTAL_MS={:.3}",
                    name,
                    total_seconds * 1.0e3
                );
                println!(
                    "R17_1K_ROTATION_PROFILE_{}_BASE_FORWARD_COUNT={}",
                    name, profile.base_forward_count
                );
                println!(
                    "R17_1K_ROTATION_PROFILE_{}_DIGIT_FORWARD_COUNT={}",
                    name, profile.digit_forward_count
                );
                println!(
                    "R17_1K_ROTATION_PROFILE_{}_INVERSE_COUNT={}",
                    name, profile.inverse_count
                );
            };

            profile_one("LEFT", &prepared_left);
            profile_one("RIGHT", &prepared_right);

            println!("R17_1K_ROTATION_PROFILE_END");
            return;
        }

        let mut stage_output = Vec::with_capacity(CIPHERTEXT_COUNT);

        for batch_start in (0..CIPHERTEXT_COUNT).step_by(local_outer_parallelism) {
            let batch_end = (batch_start + local_outer_parallelism).min(CIPHERTEXT_COUNT);

            let batch = std::thread::scope(|scope| {
                let mut workers = Vec::new();

                for input in stage_input.iter().take(batch_end).skip(batch_start) {
                    workers.push(scope.spawn(|| {
                        execute_repeated_packed_fft2_dif_stage_cp_prepared(
                            input,
                            &prepared_fft_stage,
                            &evaluator,
                            (&prepared_left, &prepared_right),
                            &chain,
                            &stage_plan,
                            &prepared_plan,
                        )
                    }));
                }

                workers
                    .into_iter()
                    .map(|worker| worker.join().expect("R17 local-row worker panicked"))
                    .collect::<Vec<_>>()
            });

            stage_output.extend(batch);
        }

        values = stage_output;

        clear_values = clear_values
            .iter()
            .map(|value| {
                ccmm_rs::eblas::fft::execute_repeated_packed_fft2_dif_stage_pp(
                    value,
                    &diagonals,
                    TILES_PER_CIPHERTEXT,
                )
            })
            .collect();

        level += 1;

        println!(
            "R17_1K_STAGE level={} kind=local-row span={} rotation={} limbs={} elapsed_ms={}",
            level,
            span,
            rotation,
            values[0].basis().len(),
            stage_start.elapsed().as_millis()
        );

        io::stdout().flush().expect("R17 stage output must flush");

        span /= 2;
    }

    assert_eq!(level, 10);

    if !benchmark_mode {
        print_checkpoint(level, &values, &clear_values, &secret, &embedding);
    }

    /*
     * Global column span 16: level 10.
     */
    {
        let stage_start = Instant::now();

        let global_span = 16usize * TILE_DIM;

        /*
         * The level-10 inter-ciphertext column stage uses the same public
         * twiddle vector for every ciphertext pair. Construct it once and
         * share it across all workers.
         */
        let mut column_twiddles = vec![Complex64::new(0.0, 0.0); SLOT_COUNT];

        for lane in 0..TILES_PER_CIPHERTEXT {
            let lane_base = lane * TILE_ELEMENTS;

            for local_row in 0..TILE_DIM {
                let twiddle_index = lane * TILE_DIM + local_row;

                let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

                let w = Complex64::new(angle.cos(), angle.sin());

                for local_col in 0..TILE_DIM {
                    let tile_slot = local_row * TILE_DIM + local_col;

                    column_twiddles[lane_base + tile_slot] = w;
                }
            }
        }

        let stage_plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), degree);
        let prepared_plan = PreparedRnsNttPlan::new(&stage_plan);
        let prepared_column_twiddles = PreparedComplexSlotsCp::new(
            &values[0],
            &column_twiddles,
            &embedding,
            &chain,
            &stage_plan,
        );

        let stage_input = values;
        let mut stage_output: Vec<Option<RnsCkksCiphertext>> =
            (0..CIPHERTEXT_COUNT).map(|_| None).collect();

        let jobs = (0..TILES_PER_AXIS)
            .map(|tile_col| (tile_col, TILES_PER_AXIS + tile_col))
            .collect::<Vec<_>>();

        for batch_start in (0..jobs.len()).step_by(inter_ct_parallelism) {
            let batch_end = (batch_start + inter_ct_parallelism).min(jobs.len());

            let batch = std::thread::scope(|scope| {
                let mut workers = Vec::new();

                for &(upper_index, lower_index) in &jobs[batch_start..batch_end] {
                    let a = &stage_input[upper_index];
                    let b = &stage_input[lower_index];

                    workers.push((
                        upper_index,
                        lower_index,
                        scope.spawn(|| {
                            packed_fft_dif_butterfly_cp_prepared(
                                &evaluator,
                                a,
                                b,
                                &prepared_column_twiddles,
                                &chain,
                                &prepared_plan,
                            )
                        }),
                    ));
                }

                workers
                    .into_iter()
                    .map(|(upper_index, lower_index, worker)| {
                        let (upper, lower) =
                            worker.join().expect("R17 global-column worker panicked");

                        (upper_index, lower_index, upper, lower)
                    })
                    .collect::<Vec<_>>()
            });

            for (upper_index, lower_index, upper, lower) in batch {
                stage_output[upper_index] = Some(upper);
                stage_output[lower_index] = Some(lower);
            }
        }

        values = stage_output
            .into_iter()
            .map(|value| value.expect("R17 global-column output missing"))
            .collect();

        clear_values = clear_global_column_inter_stage(clear_values);

        level += 1;

        println!(
            "R17_1K_STAGE level={} kind=global-column span_tiles=16 mode=inter-ct limbs={} elapsed_ms={}",
            level,
            values[0].basis().len(),
            stage_start.elapsed().as_millis()
        );
    }

    assert_eq!(level, 11);

    if !benchmark_mode {
        print_checkpoint(level, &values, &clear_values, &secret, &embedding);
    }

    /*
     * Global column intra-CT tile-block stages: levels 11..13.
     */
    span_tiles = 8;

    while span_tiles >= 2 {
        let stage_start = Instant::now();

        let diagonals = PackedTileDifStageDiagonals::new_column(
            TILE_DIM,
            TILE_DIM,
            TILES_PER_CIPHERTEXT,
            span_tiles,
            FftDirection::Forward,
        );

        let rotation = diagonals.rotation();

        let left_exponent = rotation_exponent_left(degree, rotation);

        let right_exponent = rotation_exponent_right(degree, rotation);

        let stage_plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), degree);

        let state = values[0].state();

        let prepared_left = PreparedRnsGaloisKey::new(
            evaluator.keys().galois_for(state, left_exponent),
            &stage_plan,
        );

        let prepared_right = PreparedRnsGaloisKey::new(
            evaluator.keys().galois_for(state, right_exponent),
            &stage_plan,
        );

        let stage_input = values;
        let mut stage_output = Vec::with_capacity(CIPHERTEXT_COUNT);

        for batch_start in (0..CIPHERTEXT_COUNT).step_by(local_outer_parallelism) {
            let batch_end = (batch_start + local_outer_parallelism).min(CIPHERTEXT_COUNT);

            let batch = std::thread::scope(|scope| {
                let mut workers = Vec::new();

                for input in stage_input.iter().take(batch_end).skip(batch_start) {
                    workers.push(scope.spawn(|| {
                        execute_packed_tile_dif_stage_cp_prepared(
                            input,
                            &diagonals,
                            &evaluator,
                            (&prepared_left, &prepared_right),
                            &embedding,
                            &chain,
                            &stage_plan,
                        )
                    }));
                }

                workers
                    .into_iter()
                    .map(|worker| worker.join().expect("R17 tile-block worker panicked"))
                    .collect::<Vec<_>>()
            });

            stage_output.extend(batch);
        }

        values = stage_output;

        clear_values = clear_values
            .iter()
            .map(|value| ccmm_rs::eblas::fft::execute_packed_tile_dif_stage_pp(value, &diagonals))
            .collect();

        level += 1;

        println!(
            "R17_1K_STAGE level={} kind=global-column span_tiles={} mode=intra-ct rotation={} limbs={} elapsed_ms={}",
            level,
            span_tiles,
            rotation,
            values[0].basis().len(),
            stage_start.elapsed().as_millis()
        );

        io::stdout().flush().expect("R17 stage output must flush");

        span_tiles /= 2;
    }

    assert_eq!(level, 14);

    if !benchmark_mode {
        print_checkpoint(level, &values, &clear_values, &secret, &embedding);
    }

    /*
     * Local column stages: levels 14..19.
     */
    span = TILE_DIM;

    while span >= 2 {
        let stage_start = Instant::now();

        let diagonals = PackedFft2DifStageDiagonals::new(
            tile_shape,
            PackedFft2Axis::Columns,
            span,
            FftDirection::Forward,
        );

        let rotation = diagonals.rotation();

        let left_exponent = rotation_exponent_left(degree, rotation);

        let right_exponent = rotation_exponent_right(degree, rotation);

        let stage_plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), degree);

        let state = values[0].state();

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
            &values[0],
            &diagonals,
            TILES_PER_CIPHERTEXT,
            &embedding,
            &chain,
            &stage_plan,
        );

        let stage_input = values;
        let mut stage_output = Vec::with_capacity(CIPHERTEXT_COUNT);

        for batch_start in (0..CIPHERTEXT_COUNT).step_by(local_outer_parallelism) {
            let batch_end = (batch_start + local_outer_parallelism).min(CIPHERTEXT_COUNT);

            let batch = std::thread::scope(|scope| {
                let mut workers = Vec::new();

                for input in stage_input.iter().take(batch_end).skip(batch_start) {
                    workers.push(scope.spawn(|| {
                        execute_repeated_packed_fft2_dif_stage_cp_prepared(
                            input,
                            &prepared_fft_stage,
                            &evaluator,
                            (&prepared_left, &prepared_right),
                            &chain,
                            &stage_plan,
                            &prepared_plan,
                        )
                    }));
                }

                workers
                    .into_iter()
                    .map(|worker| worker.join().expect("R17 local-column worker panicked"))
                    .collect::<Vec<_>>()
            });

            stage_output.extend(batch);
        }

        values = stage_output;

        clear_values = clear_values
            .iter()
            .map(|value| {
                ccmm_rs::eblas::fft::execute_repeated_packed_fft2_dif_stage_pp(
                    value,
                    &diagonals,
                    TILES_PER_CIPHERTEXT,
                )
            })
            .collect();

        level += 1;

        println!(
            "R17_1K_STAGE level={} kind=local-column span={} rotation={} limbs={} elapsed_ms={}",
            level,
            span,
            rotation,
            values[0].basis().len(),
            stage_start.elapsed().as_millis()
        );

        io::stdout().flush().expect("R17 stage output must flush");

        span /= 2;
    }

    assert_eq!(level, REQUIRED_LEVELS);

    let fft_ms = fft_start.elapsed().as_millis();

    if !benchmark_mode {
        print_checkpoint(level, &values, &clear_values, &secret, &embedding);
    }

    /*
     * ---------------------------------------------------------------
     * Final decrypt, reconstruct, canonical comparison.
     * ---------------------------------------------------------------
     */
    let decrypt_start = Instant::now();

    let decoded = decrypt_packed_tensor(&values, &secret, &embedding);

    let decrypt_ms = decrypt_start.elapsed().as_millis();

    let physical = reconstruct_physical_image(&decoded);

    let actual = packed_physical_to_logical(&physical, image_shape);

    let (rel_l2, max_abs) = error_metrics(&actual, &oracle);

    let status = if rel_l2 <= TOLERANCE { "PASS" } else { "FAIL" };

    let total_ms = total_start.elapsed().as_millis();

    println!("R17_1K_RESULT_BEGIN");
    println!("R17_1K_IMAGE_DIMENSION=1024x1024");
    println!("R17_1K_TILE_DIMENSION=64x64");
    println!("R17_1K_LOGICAL_TILE_COUNT=256");
    println!("R17_1K_CIPHERTEXT_COUNT=32");
    println!("R17_1K_TILES_PER_CIPHERTEXT=8");
    println!("R17_1K_SLOT_UTILIZATION=1.000000");
    println!("R17_1K_TOTAL_LEVELS={level}");
    println!("R17_1K_FINAL_LIMBS={}", values[0].basis().len());
    println!("R17_1K_GALOIS_KEY_COUNT={distinct_key_count}");
    println!("R17_1K_ENCRYPTION_MS={encryption_ms}");
    println!("R17_1K_KEYGEN_MS={keygen_ms}");
    println!("R17_1K_FFT_MS={fft_ms}");
    println!("R17_1K_DECRYPT_MS={decrypt_ms}");
    println!("R17_1K_TOTAL_MS={total_ms}");
    println!(
        "R17_1K_ELEMENTS_PER_SECOND={:.6}",
        (IMAGE_DIM * IMAGE_DIM) as f64 / (fft_ms as f64 / 1000.0)
    );
    println!("R17_1K_REL_L2={rel_l2:.12e}");
    println!("R17_1K_MAX_ABS={max_abs:.12e}");
    println!("R17_1K_TOLERANCE={TOLERANCE:.12e}");
    println!("R17_1K_STATUS={status}");
    println!("R17_1K_RESULT_END");

    assert_eq!(status, "PASS");
}
