use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_65536_for_levels, rotation_exponent_left, rotation_exponent_right,
    CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext, RnsCkksEvaluationKeys,
    RnsCkksEvaluator, RnsCkksLevelKeys, RnsGaloisKey,
};
use ccmm_rs::eblas::fft::{
    execute_packed_1k_fft_columns_cp, execute_packed_1k_fft_rows_cp, fft2_pp,
    packed_fft_dif_butterfly_cp_prepared, Fft2Shape, FftDirection,
};
use ccmm_rs::eblas::PreparedComplexSlotsCp;
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, RnsGadgetLayout,
    RnsKeygenConfig,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, ModulusChain,
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

const IMAGE_DIM: usize = 4096;
const MACRO_DIM: usize = 1024;
const MACROS_PER_AXIS: usize = 4;
const MACRO_COUNT: usize = 16;

const TILE_DIM: usize = 64;
const TILE_ELEMENTS: usize = TILE_DIM * TILE_DIM;
const TILES_PER_CIPHERTEXT: usize = 8;
const SLOT_COUNT: usize = TILE_ELEMENTS * TILES_PER_CIPHERTEXT;

const CT_PER_MACRO: usize = 32;
const TOTAL_CT: usize = MACRO_COUNT * CT_PER_MACRO;

const REQUIRED_LEVELS: usize = 24;
const TERMINAL_GUARD_LIMBS: usize = 2;

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

fn deterministic_black_white_image() -> Vec<Complex64> {
    (0..IMAGE_DIM)
        .flat_map(|row| {
            (0..IMAGE_DIM).map(move |col| {
                let white = ((row / 32) + (col / 32)) % 2 == 0;
                Complex64::new(if white { 1.0 } else { 0.0 }, 0.0)
            })
        })
        .collect()
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

fn reconstruct_decrypted_physical(
    macrotiles: &[Option<Vec<RnsCkksCiphertext>>],
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let tiles_per_axis = MACRO_DIM / TILE_DIM;

    assert_eq!(macrotiles.len(), MACRO_COUNT);

    let mut physical = vec![Complex64::new(0.0, 0.0); IMAGE_DIM * IMAGE_DIM];

    for macro_row in 0..MACROS_PER_AXIS {
        for macro_col in 0..MACROS_PER_AXIS {
            let macro_index = macro_row * MACROS_PER_AXIS + macro_col;

            let ciphertexts = macrotiles[macro_index]
                .as_ref()
                .expect("4K final macrotile missing");

            assert_eq!(ciphertexts.len(), CT_PER_MACRO);

            for (ct_index, ciphertext) in ciphertexts.iter().enumerate() {
                let decoded = decrypt_slots(ciphertext, secret, embedding);

                assert_eq!(decoded.len(), SLOT_COUNT);

                let row_block = ct_index / tiles_per_axis;
                let tile_col = ct_index % tiles_per_axis;

                for lane in 0..TILES_PER_CIPHERTEXT {
                    let tile_row = row_block * TILES_PER_CIPHERTEXT + lane;

                    let lane_base = lane * TILE_ELEMENTS;

                    for local_row in 0..TILE_DIM {
                        for local_col in 0..TILE_DIM {
                            let global_row =
                                macro_row * MACRO_DIM + tile_row * TILE_DIM + local_row;

                            let global_col =
                                macro_col * MACRO_DIM + tile_col * TILE_DIM + local_col;

                            let global_index = global_row * IMAGE_DIM + global_col;

                            let slot = lane_base + local_row * TILE_DIM + local_col;

                            physical[global_index] = decoded[slot];
                        }
                    }
                }
            }

            println!(
                "R20_4K_DECRYPT_PROGRESS macrotiles={}/16 ciphertexts={}/512",
                macro_index + 1,
                (macro_index + 1) * CT_PER_MACRO
            );

            io::stdout().flush().unwrap();
        }
    }

    physical
}

fn error_metrics_physical_vs_oracle(physical: &[Complex64], oracle: &[Complex64]) -> (f64, f64) {
    assert_eq!(physical.len(), IMAGE_DIM * IMAGE_DIM);
    assert_eq!(oracle.len(), IMAGE_DIM * IMAGE_DIM);

    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for row in 0..IMAGE_DIM {
        let physical_row = bit_reverse(row, IMAGE_DIM);

        for col in 0..IMAGE_DIM {
            let physical_col = bit_reverse(col, IMAGE_DIM);

            let logical_index = row * IMAGE_DIM + col;
            let physical_index = physical_row * IMAGE_DIM + physical_col;

            let actual = physical[physical_index];
            let expected = oracle[logical_index];

            let error = actual - expected;

            squared_error += error.norm_sqr();
            squared_reference += expected.norm_sqr();
            max_abs = max_abs.max(error.norm());
        }
    }

    let rel_l2 = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else {
        squared_error.sqrt()
    };

    (rel_l2, max_abs)
}

fn key_schedule_4k() -> Vec<(usize, usize)> {
    let mut schedule = Vec::new();

    // 1K row engine starts at level 2.
    let mut level = 6usize;
    let mut rotation = TILE_DIM / 2;

    while rotation > 0 {
        schedule.push((level, rotation));
        level += 1;
        rotation /= 2;
    }

    assert_eq!(level, 12);

    // 1K column engine starts at level 14.
    schedule.push((15, 16_384));
    schedule.push((16, 8_192));
    schedule.push((17, 4_096));

    level = 18;
    rotation = (TILE_DIM / 2) * TILE_DIM;

    while rotation >= TILE_DIM {
        schedule.push((level, rotation));
        level += 1;
        rotation /= 2;
    }

    assert_eq!(level, 24);
    assert_eq!(schedule.len(), 15);

    schedule
}

fn fill_black_white_slots(macro_row: usize, macro_col: usize, ct_index: usize) -> Vec<Complex64> {
    let row_block = ct_index / 16;
    let tile_col = ct_index % 16;

    let mut slots = vec![Complex64::new(0.0, 0.0); SLOT_COUNT];

    for lane in 0..TILES_PER_CIPHERTEXT {
        let tile_row = row_block * TILES_PER_CIPHERTEXT + lane;
        let lane_base = lane * TILE_ELEMENTS;

        for local_row in 0..TILE_DIM {
            for local_col in 0..TILE_DIM {
                let global_row = macro_row * MACRO_DIM + tile_row * TILE_DIM + local_row;
                let global_col = macro_col * MACRO_DIM + tile_col * TILE_DIM + local_col;

                // Deterministic high-resolution black/white structure.
                let white = ((global_row / 32) + (global_col / 32)) % 2 == 0;

                let slot = lane_base + local_row * TILE_DIM + local_col;

                slots[slot] = Complex64::new(if white { 1.0 } else { 0.0 }, 0.0);
            }
        }
    }

    slots
}

fn row_macro_twiddles(macro_offset: usize, ct_index: usize, span_macros: usize) -> Vec<Complex64> {
    let tile_col = ct_index % 16;
    let global_span = span_macros * MACRO_DIM;

    let mut twiddles = vec![Complex64::new(0.0, 0.0); SLOT_COUNT];

    for lane in 0..TILES_PER_CIPHERTEXT {
        let lane_base = lane * TILE_ELEMENTS;

        for local_row in 0..TILE_DIM {
            for local_col in 0..TILE_DIM {
                let local_x = tile_col * TILE_DIM + local_col;
                let twiddle_index = macro_offset * MACRO_DIM + local_x;

                let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

                twiddles[lane_base + local_row * TILE_DIM + local_col] =
                    Complex64::new(angle.cos(), angle.sin());
            }
        }
    }

    twiddles
}

fn column_macro_twiddles(
    macro_offset: usize,
    ct_index: usize,
    span_macros: usize,
) -> Vec<Complex64> {
    let row_block = ct_index / 16;
    let global_span = span_macros * MACRO_DIM;

    let mut twiddles = vec![Complex64::new(0.0, 0.0); SLOT_COUNT];

    for lane in 0..TILES_PER_CIPHERTEXT {
        let lane_base = lane * TILE_ELEMENTS;

        for local_row in 0..TILE_DIM {
            let local_y = row_block * TILES_PER_CIPHERTEXT * TILE_DIM + lane * TILE_DIM + local_row;

            let twiddle_index = macro_offset * MACRO_DIM + local_y;

            let angle = -2.0 * PI * twiddle_index as f64 / global_span as f64;

            let w = Complex64::new(angle.cos(), angle.sin());

            for local_col in 0..TILE_DIM {
                twiddles[lane_base + local_row * TILE_DIM + local_col] = w;
            }
        }
    }

    twiddles
}

fn execute_macro_row_stage(
    macrotiles: &mut [Option<Vec<RnsCkksCiphertext>>],
    span_macros: usize,
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    inter_ct_parallelism: usize,
) {
    let half = span_macros / 2;

    let sample = &macrotiles
        .iter()
        .find_map(|value| value.as_ref())
        .expect("4K macro-row stage requires ciphertext input")[0];

    let level = sample.level();
    let degree = embedding.degree();

    let stage_plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), degree);
    let prepared_plan = PreparedRnsNttPlan::new(&stage_plan);

    let mut prepared = Vec::with_capacity(half);

    for macro_offset in 0..half {
        let mut by_tile_col = Vec::with_capacity(16);

        for tile_col in 0..16 {
            let twiddles = row_macro_twiddles(macro_offset, tile_col, span_macros);

            by_tile_col.push(PreparedComplexSlotsCp::new(
                sample,
                &twiddles,
                embedding,
                chain,
                &stage_plan,
            ));
        }

        prepared.push(by_tile_col);
    }

    for macro_row in 0..MACROS_PER_AXIS {
        for group_start in (0..MACROS_PER_AXIS).step_by(span_macros) {
            for (macro_offset, prepared_by_tile_col) in prepared.iter().enumerate().take(half) {
                let upper_macro = macro_row * MACROS_PER_AXIS + group_start + macro_offset;
                let lower_macro = upper_macro + half;

                let upper = macrotiles[upper_macro]
                    .take()
                    .expect("4K upper row macrotile missing");

                let lower = macrotiles[lower_macro]
                    .take()
                    .expect("4K lower row macrotile missing");

                let mut upper_out: Vec<Option<RnsCkksCiphertext>> =
                    (0..CT_PER_MACRO).map(|_| None).collect();
                let mut lower_out: Vec<Option<RnsCkksCiphertext>> =
                    (0..CT_PER_MACRO).map(|_| None).collect();

                for batch_start in (0..CT_PER_MACRO).step_by(inter_ct_parallelism) {
                    let batch_end = (batch_start + inter_ct_parallelism).min(CT_PER_MACRO);

                    let batch = std::thread::scope(|scope| {
                        let mut workers = Vec::new();

                        for ct_index in batch_start..batch_end {
                            let a = &upper[ct_index];
                            let b = &lower[ct_index];
                            let tile_col = ct_index % 16;
                            let twiddle = &prepared_by_tile_col[tile_col];

                            workers.push((
                                ct_index,
                                scope.spawn(|| {
                                    packed_fft_dif_butterfly_cp_prepared(
                                        evaluator,
                                        a,
                                        b,
                                        twiddle,
                                        chain,
                                        &prepared_plan,
                                    )
                                }),
                            ));
                        }

                        workers
                            .into_iter()
                            .map(|(index, worker)| {
                                let (u, l) = worker.join().expect("4K macro-row worker panicked");
                                (index, u, l)
                            })
                            .collect::<Vec<_>>()
                    });

                    for (index, u, l) in batch {
                        upper_out[index] = Some(u);
                        lower_out[index] = Some(l);
                    }
                }

                macrotiles[upper_macro] = Some(
                    upper_out
                        .into_iter()
                        .map(|v| v.expect("4K macro-row upper output missing"))
                        .collect(),
                );

                macrotiles[lower_macro] = Some(
                    lower_out
                        .into_iter()
                        .map(|v| v.expect("4K macro-row lower output missing"))
                        .collect(),
                );
            }
        }
    }
}

fn execute_macro_column_stage(
    macrotiles: &mut [Option<Vec<RnsCkksCiphertext>>],
    span_macros: usize,
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    inter_ct_parallelism: usize,
) {
    let half = span_macros / 2;

    let sample = &macrotiles
        .iter()
        .find_map(|value| value.as_ref())
        .expect("4K macro-column stage requires ciphertext input")[0];

    let level = sample.level();
    let degree = embedding.degree();

    let stage_plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), degree);
    let prepared_plan = PreparedRnsNttPlan::new(&stage_plan);

    let mut prepared = Vec::with_capacity(half);

    for macro_offset in 0..half {
        let mut by_row_block = Vec::with_capacity(2);

        for row_block in 0..2 {
            let representative_ct = row_block * 16;
            let twiddles = column_macro_twiddles(macro_offset, representative_ct, span_macros);

            by_row_block.push(PreparedComplexSlotsCp::new(
                sample,
                &twiddles,
                embedding,
                chain,
                &stage_plan,
            ));
        }

        prepared.push(by_row_block);
    }

    for macro_col in 0..MACROS_PER_AXIS {
        for group_start in (0..MACROS_PER_AXIS).step_by(span_macros) {
            for (macro_offset, prepared_by_row_block) in prepared.iter().enumerate().take(half) {
                let upper_macro = (group_start + macro_offset) * MACROS_PER_AXIS + macro_col;

                let lower_macro = (group_start + macro_offset + half) * MACROS_PER_AXIS + macro_col;

                let upper = macrotiles[upper_macro]
                    .take()
                    .expect("4K upper column macrotile missing");

                let lower = macrotiles[lower_macro]
                    .take()
                    .expect("4K lower column macrotile missing");

                let mut upper_out: Vec<Option<RnsCkksCiphertext>> =
                    (0..CT_PER_MACRO).map(|_| None).collect();
                let mut lower_out: Vec<Option<RnsCkksCiphertext>> =
                    (0..CT_PER_MACRO).map(|_| None).collect();

                for batch_start in (0..CT_PER_MACRO).step_by(inter_ct_parallelism) {
                    let batch_end = (batch_start + inter_ct_parallelism).min(CT_PER_MACRO);

                    let batch = std::thread::scope(|scope| {
                        let mut workers = Vec::new();

                        for ct_index in batch_start..batch_end {
                            let a = &upper[ct_index];
                            let b = &lower[ct_index];
                            let row_block = ct_index / 16;
                            let twiddle = &prepared_by_row_block[row_block];

                            workers.push((
                                ct_index,
                                scope.spawn(|| {
                                    packed_fft_dif_butterfly_cp_prepared(
                                        evaluator,
                                        a,
                                        b,
                                        twiddle,
                                        chain,
                                        &prepared_plan,
                                    )
                                }),
                            ));
                        }

                        workers
                            .into_iter()
                            .map(|(index, worker)| {
                                let (u, l) =
                                    worker.join().expect("4K macro-column worker panicked");
                                (index, u, l)
                            })
                            .collect::<Vec<_>>()
                    });

                    for (index, u, l) in batch {
                        upper_out[index] = Some(u);
                        lower_out[index] = Some(l);
                    }
                }

                macrotiles[upper_macro] = Some(
                    upper_out
                        .into_iter()
                        .map(|v| v.expect("4K macro-column upper output missing"))
                        .collect(),
                );

                macrotiles[lower_macro] = Some(
                    lower_out
                        .into_iter()
                        .map(|v| v.expect("4K macro-column lower output missing"))
                        .collect(),
                );
            }
        }
    }
}

fn main() {
    let local_parallelism = env_usize("R20_4K_LOCAL_PARALLELISM", 4);

    let inter_ct_parallelism = env_usize("R20_4K_INTER_CT_PARALLELISM", 2);

    let profile = research_profile_65536_for_levels(REQUIRED_LEVELS, TERMINAL_GUARD_LIMBS);

    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let top_basis = chain.top().clone();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let top_plan = profile.rns_ntt_plan();

    assert_eq!(degree, 65_536);
    assert_eq!(embedding.slot_count(), SLOT_COUNT);
    assert_eq!(top_basis.len(), 26);
    assert!(chain.max_level() >= REQUIRED_LEVELS);

    println!("R20_4K_BEGIN");
    println!("R20_4K_IMAGE_DIMENSION={}x{}", IMAGE_DIM, IMAGE_DIM);
    println!("R20_4K_MACRO_DIMENSION={}x{}", MACRO_DIM, MACRO_DIM);
    println!("R20_4K_MACRO_GRID=4x4");
    println!("R20_4K_MACRO_COUNT={MACRO_COUNT}");
    println!("R20_4K_CT_PER_MACRO={CT_PER_MACRO}");
    println!("R20_4K_CIPHERTEXT_COUNT={TOTAL_CT}");
    println!("R20_4K_SLOT_COUNT={SLOT_COUNT}");
    println!("R20_4K_SLOT_UTILIZATION=1.000000");
    println!("R20_4K_REQUIRED_LEVELS={REQUIRED_LEVELS}");
    println!("R20_4K_TOP_LIMBS={}", top_basis.len());
    println!("R20_4K_SECURITY_BEARING=false");

    io::stdout().flush().unwrap();

    let total_start = Instant::now();

    println!("R20_4K_ORACLE_BEGIN");
    io::stdout().flush().unwrap();

    let oracle_start = Instant::now();

    let oracle_input = deterministic_black_white_image();

    let oracle = fft2_pp(
        Fft2Shape::new(IMAGE_DIM, IMAGE_DIM),
        FftDirection::Forward,
        &oracle_input,
    );

    drop(oracle_input);

    println!("R20_4K_ORACLE_MS={}", oracle_start.elapsed().as_millis());
    println!("R20_4K_ORACLE_END");
    io::stdout().flush().unwrap();

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x5232_304b_5345_4352);

    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&value| value == 0) {
        secret[0] = 1;
    }

    let encryption_start = Instant::now();

    let mut macrotiles: Vec<Option<Vec<RnsCkksCiphertext>>> = Vec::with_capacity(MACRO_COUNT);

    for macro_row in 0..MACROS_PER_AXIS {
        for macro_col in 0..MACROS_PER_AXIS {
            let macro_index = macro_row * MACROS_PER_AXIS + macro_col;

            let mut ciphertexts = Vec::with_capacity(CT_PER_MACRO);

            for ct_index in 0..CT_PER_MACRO {
                let slots = fill_black_white_slots(macro_row, macro_col, ct_index);

                let plaintext = encode_rns(&slots, &embedding, &top_basis, scale);

                let mut rng = ChaCha20Rng::seed_from_u64(
                    0x5232_304b_454e_4300 ^ ((macro_index as u64) << 8) ^ ct_index as u64,
                );

                let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                    &plaintext,
                    2,
                    ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                    &secret,
                    &top_plan,
                    &mut rng,
                );

                ciphertexts.push(RnsCkksCiphertext::new(
                    rlwe,
                    CkksChainState::top(&chain, scale),
                    &chain,
                ));
            }

            macrotiles.push(Some(ciphertexts));

            println!(
                "R20_4K_ENCRYPT_PROGRESS macrotiles={}/16 ciphertexts={}/512",
                macro_index + 1,
                (macro_index + 1) * CT_PER_MACRO
            );
            io::stdout().flush().unwrap();
        }
    }

    println!(
        "R20_4K_ENCRYPTION_MS={}",
        encryption_start.elapsed().as_millis()
    );

    /*
     * Generate only the Galois keys needed by the shifted 1K engines.
     * The four new macro stages are inter-ciphertext butterflies and
     * require no Galois rotations.
     */
    let schedule = key_schedule_4k();
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
            "R20_4K_KEYGEN_BEGIN level={} rotation={} limbs={}",
            level,
            rotation,
            level_layout.full_basis().len()
        );
        io::stdout().flush().unwrap();

        if left_exponent == right_exponent {
            let mut rng = ChaCha20Rng::seed_from_u64(0x5232_304b_4b45_5900 ^ ((level as u64) << 8));

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
        } else {
            let (left_key, right_key) = std::thread::scope(|scope| {
                let left_worker = scope.spawn(|| {
                    let mut rng =
                        ChaCha20Rng::seed_from_u64(0x5232_304b_4b45_5900 ^ ((level as u64) << 8));

                    RnsGaloisKey::generate_with_ntt_rng(
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
                    )
                });

                let right_worker = scope.spawn(|| {
                    let mut rng =
                        ChaCha20Rng::seed_from_u64(0x5232_304b_4b45_5901 ^ ((level as u64) << 8));

                    RnsGaloisKey::generate_with_ntt_rng(
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
                    )
                });

                (
                    left_worker.join().expect("4K left keygen panicked"),
                    right_worker.join().expect("4K right keygen panicked"),
                )
            });

            level_keys.insert_galois_key(left_key);
            level_keys.insert_galois_key(right_key);
            distinct_key_count += 2;
        }

        evaluation_keys.insert_level(level_keys);

        println!("R20_4K_KEYGEN_END level={level}");
        io::stdout().flush().unwrap();
    }

    println!("R20_4K_GALOIS_KEY_COUNT={distinct_key_count}");
    println!("R20_4K_KEYGEN_MS={}", keygen_start.elapsed().as_millis());

    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let fft_start = Instant::now();

    /*
     * Levels 0..1: two 4K macro-row stages.
     */
    for span_macros in [4usize, 2usize] {
        let start = Instant::now();

        execute_macro_row_stage(
            &mut macrotiles,
            span_macros,
            &evaluator,
            &embedding,
            &chain,
            inter_ct_parallelism,
        );

        let level = macrotiles[0].as_ref().unwrap()[0].level();

        println!(
            "R20_4K_STAGE level={} kind=macro-row span_macros={} limbs={} elapsed_ms={}",
            level,
            span_macros,
            macrotiles[0].as_ref().unwrap()[0].basis().len(),
            start.elapsed().as_millis()
        );

        io::stdout().flush().unwrap();
    }

    /*
     * Levels 2..11: one reusable 1K row phase per macrotile.
     */
    for (macro_index, macrotile) in macrotiles.iter_mut().enumerate().take(MACRO_COUNT) {
        let input = macrotile.take().expect("4K row-phase macrotile missing");

        let start = Instant::now();

        let output = execute_packed_1k_fft_rows_cp(
            input,
            &evaluator,
            &embedding,
            &chain,
            inter_ct_parallelism,
            local_parallelism,
        );

        println!(
            "R20_4K_MACRO_ROW_PROGRESS macro={}/16 level={} elapsed_ms={}",
            macro_index + 1,
            output[0].level(),
            start.elapsed().as_millis()
        );

        *macrotile = Some(output);
        io::stdout().flush().unwrap();
    }

    /*
     * Levels 12..13: two 4K macro-column stages.
     */
    for span_macros in [4usize, 2usize] {
        let start = Instant::now();

        execute_macro_column_stage(
            &mut macrotiles,
            span_macros,
            &evaluator,
            &embedding,
            &chain,
            inter_ct_parallelism,
        );

        let level = macrotiles[0].as_ref().unwrap()[0].level();

        println!(
            "R20_4K_STAGE level={} kind=macro-column span_macros={} limbs={} elapsed_ms={}",
            level,
            span_macros,
            macrotiles[0].as_ref().unwrap()[0].basis().len(),
            start.elapsed().as_millis()
        );

        io::stdout().flush().unwrap();
    }

    /*
     * Levels 14..23: one reusable 1K column phase per macrotile.
     */
    for (macro_index, macrotile) in macrotiles.iter_mut().enumerate().take(MACRO_COUNT) {
        let input = macrotile.take().expect("4K column-phase macrotile missing");

        let start = Instant::now();

        let output = execute_packed_1k_fft_columns_cp(
            input,
            &evaluator,
            &embedding,
            &chain,
            inter_ct_parallelism,
            local_parallelism,
        );

        println!(
            "R20_4K_MACRO_COLUMN_PROGRESS macro={}/16 level={} elapsed_ms={}",
            macro_index + 1,
            output[0].level(),
            start.elapsed().as_millis()
        );

        *macrotile = Some(output);
        io::stdout().flush().unwrap();
    }

    let final_ct = &macrotiles[0].as_ref().unwrap()[0];

    assert_eq!(
        final_ct.level(),
        REQUIRED_LEVELS,
        "4K encrypted FFT must consume exactly 24 levels"
    );

    assert_eq!(
        final_ct.basis().len(),
        TERMINAL_GUARD_LIMBS,
        "4K encrypted FFT must retain two terminal guard limbs"
    );

    let fft_ms = fft_start.elapsed().as_millis();

    println!("R20_4K_DECRYPT_BEGIN");
    io::stdout().flush().unwrap();

    let decrypt_start = Instant::now();

    let physical = reconstruct_decrypted_physical(&macrotiles, &secret, &embedding);

    let decrypt_ms = decrypt_start.elapsed().as_millis();

    println!("R20_4K_DECRYPT_MS={decrypt_ms}");

    let validation_start = Instant::now();

    let (rel_l2, max_abs) = error_metrics_physical_vs_oracle(&physical, &oracle);

    let validation_ms = validation_start.elapsed().as_millis();

    let status = if rel_l2 <= TOLERANCE { "PASS" } else { "FAIL" };

    let total_ms = total_start.elapsed().as_millis();

    println!("R20_4K_RESULT_BEGIN");
    println!("R20_4K_IMAGE_DIMENSION={}x{}", IMAGE_DIM, IMAGE_DIM);
    println!("R20_4K_CIPHERTEXT_COUNT={TOTAL_CT}");
    println!("R20_4K_TOTAL_LEVELS={}", final_ct.level());
    println!("R20_4K_FINAL_LIMBS={}", final_ct.basis().len());
    println!("R20_4K_FFT_MS={fft_ms}");
    println!("R20_4K_DECRYPT_MS={decrypt_ms}");
    println!("R20_4K_VALIDATION_MS={validation_ms}");
    println!("R20_4K_REL_L2={rel_l2:.12e}");
    println!("R20_4K_MAX_ABS={max_abs:.12e}");
    println!("R20_4K_TOLERANCE={TOLERANCE:.12e}");
    println!("R20_4K_TOTAL_MS={total_ms}");
    println!("R20_4K_STATUS={status}");
    println!("R20_4K_RESULT_END");

    assert_eq!(
        status, "PASS",
        "encrypted hierarchical 4K FFT must match the canonical clear FFT"
    );
}
