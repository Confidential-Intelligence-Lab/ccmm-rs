use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
const LANE_SLOTS: usize = 4096;
const SLOTS: usize = 32768;
use ccmm_rs::ckks::{research_profile_65536_for_levels, PreparedHybridRnsGaloisKey};
use ccmm_rs::eblas::prepared_radix4_fft256::StageContext as PackedStageContext;
// R16.5.13c: all 15 real ciphertext FFT15 -> fused twiddle FFT256 branches.
// Full 3840-frequency horizontal spectrum for 16 independent physical rows.
// Research-only parameters and deterministic test keys; no intermediate decryption.
use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    rotate_left_bounded_rns_ckks_with_prepared_execution as rotate_left_for_extraction,
    rotation_exponent_left, rotation_exponent_right, BoundedRnsGaloisKey, CkksCanonicalEmbedding,
    CkksChainState, PreparedBoundedRnsGaloisKey, RnsCkksCiphertext, RnsCkksEvaluationKeys,
    RnsCkksEvaluator,
};
use ccmm_rs::eblas::level1::{multiply_complex_slots_cp_prepared, PreparedComplexSlotsCp};
use ccmm_rs::eblas::mixed_radix_ckks::{
    packed_fft15_twiddled_radix3_cp_prepared, packed_radix5_dft_cp_prepared,
    PreparedFft15TwiddledRadix3Cp, PreparedRadix5Cp,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    BoundedRnsKeygenConfig,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, ModulusBasis,
    ModulusChain, PreparedRnsNttPlan, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::time::Instant;

const SIGMA: f64 = 3.19;
type Diagonals = BTreeMap<isize, Vec<Complex64>>;
fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * exponent as f64 / n as f64)
}
fn reverse_base4_4(mut i: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..4 {
        reversed = reversed * 4 + i % 4;
        i /= 4;
    }
    reversed
}

// Radix-4 DIT: spans 4, 16, 64, 256. Signed offsets prevent FFT256 block crossings.

fn decode(
    ct: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let plan = RnsNttPlan::new(ct.basis().moduli().to_vec(), embedding.degree());
    let decrypted = decrypt_rns_raw_with_ntt(ct.rlwe(), secret, &plan);
    let modulus = composite_modulus_big(decrypted.basis());
    let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("coefficient conversion")
                / ct.scale()
        })
        .collect();
    embedding.coefficients_to_slots(&coefficients)
}

use std::collections::BTreeSet;

use ccmm_rs::eblas::encrypted_fft3840::PreparedEncryptedFft3840;
use ccmm_rs::eblas::fused_fft256_handoff::{fuse_fft256_source_lane, PreparedFusedFft256Source};

use ccmm_rs::eblas::prepared_radix4_fft256::{pack_fft256_branch_diagonals, PreparedFft256Stage};

fn fft_root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(
        1.0,
        -2.0 * std::f64::consts::PI * (exponent % n) as f64 / n as f64,
    )
}

fn first_stage_diagonals(branch: usize) -> Diagonals {
    assert!(branch < 15);

    let mut diagonals = Diagonals::new();

    for i in 0..LANE_SLOTS {
        let p = i % 4;
        let group = i / 4;

        for q in 0..4 {
            let input_index = 4 * group + q;
            let offset = input_index as isize - i as isize;

            let coefficient =
                fft_root(4, p * q) * fft_root(3840, branch * reverse_base4_4(input_index % 256));

            diagonals
                .entry(offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); LANE_SLOTS])[i] = coefficient;
        }
    }

    diagonals
}

fn later_stage_diagonals(span: usize) -> Diagonals {
    assert!([16, 64, 256].contains(&span));

    let mut diagonals = Diagonals::new();
    let quarter = span / 4;

    for i in 0..LANE_SLOTS {
        let within = i % span;
        let p = within / quarter;
        let j = within % quarter;

        for q in 0..4 {
            let input_within = j + q * quarter;
            let offset = input_within as isize - within as isize;

            let coefficient = fft_root(span, q * j) * fft_root(4, p * q);

            diagonals
                .entry(offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); LANE_SLOTS])[i] = coefficient;
        }
    }

    diagonals
}

fn r27_load_pgm_rows(path: &str) -> Vec<Vec<Complex64>> {
    let bytes = std::fs::read(path).expect("cannot read R27 PGM image");

    // Binary PGM P5, 8-bit pixels, without comments.
    let mut tokens = Vec::new();
    let mut cursor = 0usize;

    while tokens.len() < 4 {
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }

        assert!(cursor < bytes.len(), "incomplete PGM header");

        let start = cursor;

        while cursor < bytes.len() && !bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }

        tokens.push(
            std::str::from_utf8(&bytes[start..cursor])
                .expect("invalid PGM header")
                .to_owned(),
        );
    }

    assert_eq!(tokens[0], "P5", "expected binary PGM P5");

    let width: usize = tokens[1].parse().expect("invalid PGM width");
    let height: usize = tokens[2].parse().expect("invalid PGM height");
    let maxval: usize = tokens[3].parse().expect("invalid PGM maxval");

    assert_eq!(width, 3840, "R27 requires 3840 image columns");
    assert!(height >= 16, "R27 requires at least 16 image rows");
    assert_eq!(maxval, 255, "R27 requires 8-bit grayscale");

    assert!(
        cursor < bytes.len() && bytes[cursor].is_ascii_whitespace(),
        "missing PGM pixel separator"
    );

    // PGM header ends with one whitespace separator.
    // Preserve subsequent pixel bytes exactly.
    if bytes[cursor] == b'\r' && bytes.get(cursor + 1) == Some(&b'\n') {
        cursor += 2;
    } else {
        cursor += 1;
    }

    let required = width * height;
    assert!(
        bytes.len() >= cursor + required,
        "PGM pixel data is incomplete"
    );

    let pixels = &bytes[cursor..cursor + required];

    let mut natural = vec![vec![Complex64::new(0.0, 0.0); 4096]; 15];

    for row in 0..16 {
        for column in 0..3840 {
            let tile = column / 256;
            let within = column % 256;

            // Centered, conservatively scaled grayscale samples.
            let pixel = pixels[row * width + column] as f64;
            let value = 0.001 * (pixel / 255.0 - 0.5);

            natural[tile][row * 256 + within] = Complex64::new(value, 0.0);
        }
    }

    println!("R27_IMAGE_WIDTH={width}");
    println!("R27_IMAGE_HEIGHT={height}");
    println!("R27_IMAGE_ROWS_PROCESSED=16");
    println!("R27_IMAGE_INPUT_STATUS=PASS");

    natural
}

// R28_4B1_REUSABLE_FFT15
#[allow(clippy::too_many_arguments)]
fn encrypt_and_fft15(
    natural: &[Vec<Complex64>],
    slots: usize,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    basis0: &ModulusBasis,
    scale: f64,
    secret: &[i8],
    plan0: &RnsNttPlan,
    prepared0: &PreparedRnsNttPlan,
    plan1: &RnsNttPlan,
    prepared1: &PreparedRnsNttPlan,
    plan2: &RnsNttPlan,
    prepared2: &PreparedRnsNttPlan,
    mut rng: &mut ChaCha20Rng,
    prepared_left: &PreparedBoundedRnsGaloisKey,
) -> Vec<RnsCkksCiphertext> {
    const HALF: usize = 4096;
    const TILES: usize = 15;

    // The physical FFT15 kernel has one ciphertext per pair of 4096-slot tiles.
    // Pack local FFT256 inputs in base-4 digit-reversed order so the FFT256
    // diagonal kernel can consume the actual FFT15 ciphertext directly.
    let physical: Vec<RnsCkksCiphertext> = (0..8)
        .map(|index| {
            let mut values = vec![Complex64::new(0.0, 0.0); slots];
            for half in 0..2 {
                let tile = 2 * index + half;
                if tile < TILES {
                    for slot in 0..HALF {
                        let block = (slot / 256) * 256;
                        let reordered = block + reverse_base4_4(slot % 256);
                        values[half * HALF + slot] = natural[tile][reordered];
                    }
                }
            }
            let plaintext = encode_rns(&values, embedding, basis0, scale);
            let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                secret,
                plan0,
                &mut rng,
            );
            RnsCkksCiphertext::new(rlwe, CkksChainState::top(chain, scale), chain)
        })
        .collect();

    let low_mask: Vec<Complex64> = (0..slots)
        .map(|s| Complex64::new(if s < HALF { 1.0 } else { 0.0 }, 0.0))
        .collect();
    let prepared_mask =
        PreparedComplexSlotsCp::new(&physical[0], &low_mask, embedding, chain, plan0);
    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(chain, &evaluation_keys);
    let aligned: Vec<RnsCkksCiphertext> = (0..TILES)
        .map(|tile| {
            let source = &physical[tile / 2];
            if tile % 2 == 0 {
                multiply_complex_slots_cp_prepared(source, &prepared_mask, chain, prepared0)
            } else {
                let shifted = rotate_left_for_extraction(
                    source,
                    HALF,
                    prepared_left,
                    chain,
                    plan0,
                    prepared0,
                );
                multiply_complex_slots_cp_prepared(&shifted, &prepared_mask, chain, prepared0)
            }
        })
        .collect();
    assert!(aligned.iter().all(|ct| ct.level() == 1));
    let radix5 = PreparedRadix5Cp::new(&aligned[0], embedding, chain, plan1);
    let branches: [[RnsCkksCiphertext; 5]; 3] = std::array::from_fn(|j| {
        packed_radix5_dft_cp_prepared(
            &evaluator,
            [
                &aligned[j],
                &aligned[j + 3],
                &aligned[j + 6],
                &aligned[j + 9],
                &aligned[j + 12],
            ],
            &radix5,
            chain,
            prepared1,
        )
    });
    let radix3 = PreparedFft15TwiddledRadix3Cp::new(&branches[0][0], embedding, chain, plan2);
    let groups: [[RnsCkksCiphertext; 3]; 5] = std::array::from_fn(|k| {
        packed_fft15_twiddled_radix3_cp_prepared(
            &evaluator,
            [&branches[0][k], &branches[1][k], &branches[2][k]],
            k,
            &radix3,
            chain,
            prepared2,
        )
    });

    let sources: Vec<RnsCkksCiphertext> = (0..15).map(|h| groups[h % 5][h / 5].clone()).collect();

    assert_eq!(sources.len(), 15);
    assert!(sources.iter().all(|ct| ct.level() == 3));

    println!("R26_H11_15C_FFT15_BRANCHES=15");
    println!("R26_H11_15C_FFT15_OUTPUT_LEVEL=3");
    sources
}

// R26_H11_15C_PACKED_PRODUCTION_INTEGRATION

type PackedRotationKeys = std::collections::BTreeMap<isize, PreparedHybridRnsGaloisKey>;

#[test]
#[ignore = "full encrypted FFT3840 integration is expensive"]
fn r26_h11_15c_integrated_encrypted_fft3840() {
    let total_started = Instant::now();

    let profile = research_profile_65536_for_levels(5, 5);
    let degree = profile.degree();
    assert_eq!(degree, 65536);

    let validated = profile.modulus_chain().top().clone();
    assert!(validated.moduli().len() >= 10);

    let q = ModulusBasis::new(validated.moduli()[..8].to_vec());
    let p = ModulusBasis::new(validated.moduli()[8..10].to_vec());
    let chain = ModulusChain::from_top_basis(q);

    assert_eq!(chain.max_level(), 7);

    let scale = profile.initial_scale();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    const HALF: usize = 4096;
    const TILES: usize = 15;

    let basis0 = chain.top().clone();
    let plan0 = RnsNttPlan::new(basis0.moduli().to_vec(), degree);
    let prepared0 = PreparedRnsNttPlan::new(&plan0);

    let plan1 = RnsNttPlan::new(chain.level(1).moduli().to_vec(), degree);
    let prepared1 = PreparedRnsNttPlan::new(&plan1);

    let plan2 = RnsNttPlan::new(chain.level(2).moduli().to_vec(), degree);
    let prepared2 = PreparedRnsNttPlan::new(&plan2);

    let plans: Vec<RnsNttPlan> = (3..7)
        .map(|level| RnsNttPlan::new(chain.level(level).moduli().to_vec(), degree))
        .collect();

    let prepared_plans: Vec<PreparedRnsNttPlan> =
        plans.iter().map(PreparedRnsNttPlan::new).collect();

    let qp_plans: Vec<RnsNttPlan> = (3..7)
        .map(|level| {
            let mut moduli = chain.level(level).moduli().to_vec();
            moduli.extend_from_slice(p.moduli());
            RnsNttPlan::new(moduli, degree)
        })
        .collect();

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x2611_150C);
    // One physical tile row: 15 horizontal tiles, 4096 samples per tile.
    // Each tile contains 16 independent 256-sample horizontal segments.
    // The encrypted FFT15 acts on the tile axis, preserving all 4096 slots.
    let natural: Vec<Vec<Complex64>> = if let Ok(path) = std::env::var("FHE_R27_IMAGE_PGM") {
        r27_load_pgm_rows(&path)
    } else {
        (0..TILES)
            .map(|tile| {
                (0..HALF)
                    .map(|slot| {
                        let x = (tile * HALF + slot) as f64;
                        Complex64::new(0.0003 * (0.013 * x).sin(), 0.0003 * (0.017 * x).cos())
                    })
                    .collect()
            })
            .collect()
    };
    // R28_4B2_SHARED_EXTRACTION_KEY
    let left_key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
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
    let prepared_left = PreparedBoundedRnsGaloisKey::new(&left_key, &plan0);

    let sources = encrypt_and_fft15(
        &natural,
        slots,
        &embedding,
        &chain,
        &basis0,
        scale,
        &secret,
        &plan0,
        &prepared0,
        &plan1,
        &prepared1,
        &plan2,
        &prepared2,
        &mut rng,
        &prepared_left,
    );

    // R28_4B3_SECOND_IMAGE
    // Opt-in two-image mode; preserve the existing single-image baseline.
    let second_image = if std::env::var("FHE_R28_TWO_IMAGES").as_deref() == Ok("1") {
        let natural_b: Vec<Vec<Complex64>> = natural
            .iter()
            .enumerate()
            .map(|(tile, row)| {
                row.iter()
                    .enumerate()
                    .map(|(slot, value)| {
                        let phase = 0.17 * (tile as f64) + 0.003 * (slot as f64);
                        *value * Complex64::new(phase.cos(), phase.sin())
                    })
                    .collect()
            })
            .collect();

        let mut image_rng = ChaCha20Rng::seed_from_u64(0x284B_3002);

        let sources_b = encrypt_and_fft15(
            &natural_b,
            slots,
            &embedding,
            &chain,
            &basis0,
            scale,
            &secret,
            &plan0,
            &prepared0,
            &plan1,
            &prepared1,
            &plan2,
            &prepared2,
            &mut image_rng,
            &prepared_left,
        );

        assert_eq!(sources_b.len(), 15);
        assert!(sources_b.iter().all(|ct| ct.level() == 3));

        println!("R28_4B3_SECOND_IMAGE_FFT15_STATUS=PASS");

        Some((natural_b, sources_b))
    } else {
        None
    };

    // R28_4B3_SECOND_IMAGE_CONSUMPTION
    // Temporary structural check until the second FFT256 call is integrated.
    if let Some((natural_b, sources_b)) = second_image.as_ref() {
        assert_eq!(natural_b.len(), 15);
        assert!(natural_b.iter().all(|tile| tile.len() == 4096));
        assert_eq!(sources_b.len(), 15);
        assert!(sources_b.iter().all(|ct| ct.level() == 3));
    }

    // Prepare fused placement and branch-specific first-stage
    // butterflies for both groups.
    let preparation_started = Instant::now();

    let fused: Vec<Diagonals> = (0..15)
        .map(|branch| {
            let destination_lane = if branch < 8 { branch } else { branch - 8 };

            fuse_fft256_source_lane(
                &first_stage_diagonals(branch),
                SLOTS,
                LANE_SLOTS,
                0,
                destination_lane,
            )
        })
        .collect();

    let prepared_sources: Vec<PreparedFusedFft256Source> = fused
        .iter()
        .map(|diagonals| {
            PreparedFusedFft256Source::new(3, diagonals, &embedding, &chain, &prepared_plans[0])
        })
        .collect();

    let later_diagonals: Vec<Diagonals> = [16, 64, 256]
        .iter()
        .map(|&span| {
            let local = later_stage_diagonals(span);
            let lanes: Vec<&Diagonals> = (0..8).map(|_| &local).collect();

            pack_fft256_branch_diagonals(&lanes, LANE_SLOTS)
        })
        .collect();

    let later_prepared: Vec<PreparedFft256Stage> = (0..3)
        .map(|index| {
            PreparedFft256Stage::new(
                index + 4,
                &later_diagonals[index],
                &embedding,
                &chain,
                &prepared_plans[index + 1],
            )
        })
        .collect();

    println!(
        "R26_H11_14F_PREPARATION_MS={:.3}",
        preparation_started.elapsed().as_secs_f64() * 1000.0
    );

    // Generate each required rotation key once per level.
    // The same level-specific keys are shared by groups A and B.
    let keygen_started = Instant::now();
    let mut stage_keys = Vec::new();

    for stage_index in 0..4 {
        let level = stage_index + 3;

        let offsets: BTreeSet<isize> = if stage_index == 0 {
            prepared_sources
                .iter()
                .flat_map(|prepared| prepared.rotation_offsets())
                .filter(|&offset| offset != 0)
                .collect()
        } else {
            later_diagonals[stage_index - 1]
                .keys()
                .copied()
                .filter(|&offset| offset != 0)
                .collect()
        };

        let stage_started = Instant::now();

        let workers = std::env::var("FHE_R27_KEYGEN_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);

        let offsets: Vec<isize> = offsets.into_iter().collect();

        // Derive independent per-key RNG seeds from the existing
        // deterministic test RNG before starting parallel work.
        let jobs: Vec<(isize, [u8; 32])> = offsets
            .into_iter()
            .map(|offset| {
                let mut seed = [0_u8; 32];
                rand::RngCore::fill_bytes(&mut rng, &mut seed);
                (offset, seed)
            })
            .collect();

        let generate = |(offset, seed): (isize, [u8; 32])| {
            let exponent = if offset > 0 {
                rotation_exponent_left(degree, offset as usize)
            } else {
                rotation_exponent_right(degree, (-offset) as usize)
            };

            let mut key_rng = ChaCha20Rng::from_seed(seed);

            let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(chain.level(level).clone(), 8),
                p.clone(),
                &secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                2,
                &qp_plans[stage_index],
                &mut key_rng,
            );

            (offset, key)
        };

        let generated: Vec<(isize, PreparedHybridRnsGaloisKey)> = if workers == 1 {
            jobs.into_iter().map(generate).collect()
        } else {
            let pool = ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .expect("R27 keygen thread pool creation failed");

            pool.install(|| jobs.into_par_iter().map(generate).collect())
        };

        let keys: PackedRotationKeys = generated.into_iter().collect();

        println!("R27_KEYGEN_THREADS={workers}");
        println!(
            "R26_H11_14F_STAGE_{stage_index}_ROTATION_KEYS={}",
            keys.len()
        );
        println!(
            "R26_H11_14F_STAGE_{stage_index}_KEYGEN_MS={:.3}",
            stage_started.elapsed().as_secs_f64() * 1000.0
        );

        stage_keys.push(keys);
    }

    println!(
        "R26_H11_14F_TOTAL_KEYGEN_MS={:.3}",
        keygen_started.elapsed().as_secs_f64() * 1000.0
    );

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    // R26_H11_15A_PRODUCTION_API_REGRESSION
    let evaluation_started = Instant::now();

    let contexts: Vec<PackedStageContext<'_>> = (0..4)
        .map(|stage| PackedStageContext {
            keys: &stage_keys[stage],
            evaluator: &evaluator,
            chain: &chain,
            qp_plan: &qp_plans[stage],
            prepared: &prepared_plans[stage],
        })
        .collect();

    let source_array: [RnsCkksCiphertext; 15] = sources
        .try_into()
        .unwrap_or_else(|_| panic!("expected 15 encrypted FFT256 sources"));

    let prepared_array: [PreparedFusedFft256Source; 15] = prepared_sources
        .try_into()
        .unwrap_or_else(|_| panic!("expected 15 prepared fused FFT256 sources"));

    let later_array: [PreparedFft256Stage; 3] = later_prepared
        .try_into()
        .unwrap_or_else(|_| panic!("expected three prepared FFT256 stages"));

    // R28_4A_INTEGRATION
    let prepared_fft_context =
        ccmm_rs::eblas::prepared_radix4_fft256::PreparedFft256ExecutionContext::new([
            &contexts[0],
            &contexts[1],
            &contexts[2],
            &contexts[3],
        ]);

    // R29_2A_REUSABLE_EFFT_SERVICE
    let service = PreparedEncryptedFft3840::new(
        &prepared_array,
        &later_array,
        [&contexts[0], &contexts[1], &contexts[2], &contexts[3]],
        &prepared_fft_context,
    );
    let results = service.execute(&source_array);

    // R28_4B3_TWO_IMAGE_FFT256
    // Reuse all FFT256 public operands, rotation keys, and QP context.
    let second_results = if let Some((_, sources_b)) = second_image.as_ref() {
        let started = Instant::now();

        let source_array_b: [RnsCkksCiphertext; 15] = sources_b
            .clone()
            .try_into()
            .unwrap_or_else(|_| panic!("expected 15 image-B FFT256 sources"));

        let results_b = service.execute(&source_array_b);

        assert_eq!(results_b.len(), 2);
        assert!(results_b.iter().all(|ct| ct.level() == 7));

        println!(
            "R28_4B3_IMAGE_1_FFT256_MS={:.3}",
            started.elapsed().as_secs_f64() * 1000.0
        );

        Some(results_b)
    } else {
        None
    };

    println!(
        "R26_H11_14F_TOTAL_EVALUATION_MS={:.3}",
        evaluation_started.elapsed().as_secs_f64() * 1000.0
    );

    // Independent cleartext oracle: 15-point DFT on tile axis, public
    // W3840 twiddle, then an independently coded iterative radix-2 FFT256.
    // In addition, selected bins are checked by direct O(3840) DFT.
    let mut reference = vec![Complex64::new(0.0, 0.0); 16 * 3840];
    for row in 0..16 {
        for k_cross in 0..TILES {
            let mut local = vec![Complex64::new(0.0, 0.0); 256];
            for (s, value) in local.iter_mut().enumerate() {
                let cross: Complex64 = (0..TILES)
                    .map(|tile| natural[tile][row * 256 + s] * root(TILES, k_cross * tile))
                    .sum();
                *value = cross * root(3840, k_cross * s);
            }
            // Iterative radix-2 DIT FFT256, bit-reversed input permutation.
            for i in 0usize..256 {
                let j = i.reverse_bits() >> (usize::BITS - 8);
                if i < j {
                    local.swap(i, j);
                }
            }
            let mut len = 2;
            while len <= 256 {
                for base in (0..256).step_by(len) {
                    for j in 0..len / 2 {
                        let a = local[base + j];
                        let b = local[base + j + len / 2] * root(len, j);
                        local[base + j] = a + b;
                        local[base + j + len / 2] = a - b;
                    }
                }
                len *= 2;
            }
            for (k_local, &value) in local.iter().enumerate() {
                reference[row * 3840 + k_cross + TILES * k_local] = value;
            }
        }
    }
    for &row in &[0usize, 7, 15] {
        for &frequency in &[0usize, 7, 255, 1024, 3839] {
            let direct: Complex64 = (0..3840)
                .map(|n| {
                    let tile = n / 256;
                    let s = n % 256;
                    natural[tile][row * 256 + s] * root(3840, frequency * n)
                })
                .sum();
            let staged = reference[row * 3840 + frequency];
            assert!(
                (direct - staged).norm() < 1e-8,
                "independent DFT disagreement: row={row} frequency={frequency}"
            );
        }
    }
    let validation_started = Instant::now();
    let mut error_sq = 0.0_f64;
    let mut reference_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;

    // R27.2b: optional decrypted-spectrum evidence export.
    let spectrum_path = std::env::var("FHE_R27_SPECTRUM_BIN").ok();
    let mut decrypted_spectrum = spectrum_path
        .as_ref()
        .map(|_| vec![Complex64::new(0.0, 0.0); 16 * 3840]);

    for (group, ciphertext) in results.iter().enumerate() {
        let actual = decode(ciphertext, &secret, &embedding);
        assert_eq!(ciphertext.level(), 7);

        let branches = if group == 0 { 0..8 } else { 8..15 };

        for (lane, branch) in branches.enumerate() {
            for row in 0..16 {
                for k in 0..256 {
                    let slot = lane * HALF + row * 256 + k;
                    let frequency = row * 3840 + branch + 15 * k;
                    if let Some(spectrum) = decrypted_spectrum.as_mut() {
                        spectrum[frequency] = actual[slot];
                    }

                    let expected = reference[frequency];
                    let error = (actual[slot] - expected).norm();

                    error_sq += error * error;
                    reference_sq += expected.norm_sqr();
                    max_abs = max_abs.max(error);
                }
            }
        }
    }

    let absolute_l2 = error_sq.sqrt();
    let reference_l2 = reference_sq.sqrt();
    let relative_l2 = (error_sq / reference_sq).sqrt();

    println!("R27_ABSOLUTE_L2={absolute_l2:.12e}");
    println!("R27_REFERENCE_L2={reference_l2:.12e}");
    println!("R27_ERROR_RMS={:.12e}", (error_sq / 61440.0).sqrt());

    if let (Some(path), Some(spectrum)) = (spectrum_path.as_ref(), decrypted_spectrum.as_ref()) {
        use std::io::Write;

        let mut file =
            std::fs::File::create(path).expect("failed to create decrypted FFT spectrum");

        // 61440 complex64 values: little-endian f64 real, f64 imaginary.
        for value in spectrum {
            file.write_all(&value.re.to_le_bytes())
                .expect("failed to write spectrum real component");
            file.write_all(&value.im.to_le_bytes())
                .expect("failed to write spectrum imaginary component");
        }

        file.flush().expect("failed to flush FFT spectrum");

        println!("R27_DECRYPTED_SPECTRUM_PATH={path}");
        println!("R27_DECRYPTED_SPECTRUM_VALUES={}", spectrum.len());
        println!("R27_DECRYPTED_SPECTRUM_STATUS=PASS");
    }

    println!("R26_H11_15C_OUTPUT_CIPHERTEXTS=2");
    println!("R26_H11_15C_OUTPUT_LEVEL=7");
    println!("R26_H11_15C_MAX_ABS={max_abs:.12e}");
    println!("R26_H11_15C_RELATIVE_L2={relative_l2:.12e}");
    println!(
        "R26_H11_15C_VALIDATION_MS={:.3}",
        validation_started.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "R26_H11_15C_TOTAL_MS={:.3}",
        total_started.elapsed().as_secs_f64() * 1000.0
    );

    assert!(
        relative_l2 < 1.0e-5,
        "encrypted FFT3840 numerical validation failed"
    );

    if let Some(results_b) = second_results.as_ref() {
        assert_eq!(results_b.len(), 2);
        assert!(results_b.iter().all(|ct| ct.level() == 7));
        println!("R28_4B3_SECOND_IMAGE_FFT256_STATUS=PASS");
    }

    // R28_4B4_INDEPENDENT_IMAGE_B_ORACLE
    if let (Some((natural_b, _)), Some(results_b)) =
        (second_image.as_ref(), second_results.as_ref())
    {
        let validation_started_b = Instant::now();

        let mut reference_b = vec![Complex64::new(0.0, 0.0); 16 * 3840];

        for row in 0..16 {
            for k_cross in 0..TILES {
                let mut local = vec![Complex64::new(0.0, 0.0); 256];

                for (s, value) in local.iter_mut().enumerate() {
                    let cross: Complex64 = (0..TILES)
                        .map(|tile| natural_b[tile][row * 256 + s] * root(TILES, k_cross * tile))
                        .sum();

                    *value = cross * root(3840, k_cross * s);
                }

                for i in 0usize..256 {
                    let j = i.reverse_bits() >> (usize::BITS - 8);
                    if i < j {
                        local.swap(i, j);
                    }
                }

                let mut len = 2;
                while len <= 256 {
                    for base in (0..256).step_by(len) {
                        for j in 0..len / 2 {
                            let a = local[base + j];
                            let b = local[base + j + len / 2] * root(len, j);

                            local[base + j] = a + b;
                            local[base + j + len / 2] = a - b;
                        }
                    }
                    len *= 2;
                }

                for (k_local, &value) in local.iter().enumerate() {
                    reference_b[row * 3840 + k_cross + TILES * k_local] = value;
                }
            }
        }

        for &row in &[0usize, 7, 15] {
            for &frequency in &[0usize, 7, 255, 1024, 3839] {
                let direct: Complex64 = (0..3840)
                    .map(|n| {
                        let tile = n / 256;
                        let slot = n % 256;
                        natural_b[tile][row * 256 + slot] * root(3840, frequency * n)
                    })
                    .sum();

                let staged = reference_b[row * 3840 + frequency];

                assert!(
                    (direct - staged).norm() < 1e-8,
                    "image B oracle disagreement: row={row} frequency={frequency}"
                );
            }
        }

        let mut error_sq_b = 0.0_f64;
        let mut reference_sq_b = 0.0_f64;
        let mut max_abs_b = 0.0_f64;
        let mut count_b = 0usize;

        for (group, ciphertext) in results_b.iter().enumerate() {
            let actual = decode(ciphertext, &secret, &embedding);

            assert_eq!(ciphertext.level(), 7);

            let branches = if group == 0 { 0..8 } else { 8..15 };

            for (lane, branch) in branches.enumerate() {
                for row in 0..16 {
                    for k in 0..256 {
                        let slot = lane * HALF + row * 256 + k;
                        let frequency = row * 3840 + branch + 15 * k;

                        let expected = reference_b[frequency];
                        let error = (actual[slot] - expected).norm();

                        error_sq_b += error * error;
                        reference_sq_b += expected.norm_sqr();
                        max_abs_b = max_abs_b.max(error);
                        count_b += 1;
                    }
                }
            }
        }

        assert_eq!(count_b, 61_440);
        assert!(reference_sq_b > 0.0);

        let relative_l2_b = (error_sq_b / reference_sq_b).sqrt();

        println!("R28_4B3_IMAGES=2");
        println!("R28_4B3_IMAGE_0_RELATIVE_L2={relative_l2:.12e}");
        println!("R28_4B3_IMAGE_1_RELATIVE_L2={relative_l2_b:.12e}");
        println!("R28_4B3_IMAGE_1_MAX_ABS={max_abs_b:.12e}");
        println!("R28_4B3_IMAGE_1_SPECTRAL_VALUES={count_b}");
        println!(
            "R28_4B3_IMAGE_1_VALIDATION_MS={:.3}",
            validation_started_b.elapsed().as_secs_f64() * 1000.0
        );

        assert!(
            relative_l2_b < 1.0e-5,
            "image B encrypted FFT3840 numerical validation failed"
        );

        println!("R28_4B3_TWO_IMAGE_STATUS=PASS");
    }

    println!("R26_H11_15C_ENCRYPTED_STATUS=PASS");
}
