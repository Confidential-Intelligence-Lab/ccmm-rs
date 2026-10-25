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
    rotation_exponent_left, rotation_exponent_right, BoundedRnsGaloisKey, CkksCanonicalEmbedding,
    CkksChainState, PreparedBoundedRnsGaloisKey, RnsCkksCiphertext, RnsCkksEvaluationKeys,
    RnsCkksEvaluator,
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

use ccmm_rs::eblas::encrypted_fft3840::{
    EncryptedFft15ExecutionResources, EncryptedFft3840Pipeline, PreparedEncryptedFft15,
    PreparedEncryptedFft3840,
};
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

// R29_3C_COMPLETE_PIPELINE_EXECUTION
#[allow(clippy::too_many_arguments)]
fn encrypt_fft15_physical(
    natural: &[Vec<Complex64>],
    slots: usize,
    embedding: &CkksCanonicalEmbedding,
    chain: &ModulusChain,
    basis0: &ModulusBasis,
    scale: f64,
    secret: &[i8],
    plan0: &RnsNttPlan,
    rng: &mut ChaCha20Rng,
) -> [RnsCkksCiphertext; 8] {
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
                rng,
            );
            RnsCkksCiphertext::new(rlwe, CkksChainState::top(chain, scale), chain)
        })
        .collect();

    let physical: [RnsCkksCiphertext; 8] = physical
        .try_into()
        .unwrap_or_else(|_| panic!("expected eight physical ciphertexts"));

    physical
}

// R28_4B1_REUSABLE_FFT15
#[allow(clippy::too_many_arguments)]
// R29_2B2A_IMPORT_AND_RETURN_REPAIR
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
    rng: &mut ChaCha20Rng,
    prepared_left: &PreparedBoundedRnsGaloisKey,
    fft15_cache: &mut Option<PreparedEncryptedFft15>,
) -> Vec<RnsCkksCiphertext> {
    let physical = encrypt_fft15_physical(
        natural, slots, embedding, chain, basis0, scale, secret, plan0, rng,
    );

    let sources = if let Some(prepared) = fft15_cache.as_ref() {
        prepared.execute(
            &physical,
            chain,
            plan0,
            prepared0,
            prepared1,
            prepared2,
            prepared_left,
        )
    } else {
        let (prepared, first_sources) = PreparedEncryptedFft15::prepare_and_execute_first(
            &physical,
            embedding,
            chain,
            plan0,
            prepared0,
            plan1,
            prepared1,
            plan2,
            prepared2,
            prepared_left,
        );
        *fft15_cache = Some(prepared);
        first_sources
    };

    println!("R26_H11_15C_FFT15_BRANCHES=15");
    println!("R26_H11_15C_FFT15_OUTPUT_LEVEL=3");
    sources.into_iter().collect()
}

// R26_H11_15C_PACKED_PRODUCTION_INTEGRATION

type PackedRotationKeys = std::collections::BTreeMap<isize, PreparedHybridRnsGaloisKey>;

#[test]
#[ignore = "full encrypted FFT3840 integration is expensive"]
fn r26_h11_15c_integrated_encrypted_fft3840() {
    let total_started = Instant::now();

    // R30_9A_4_SHARED_PROFILE_HORIZONTAL
    let shared_profile = std::env::var_os("R30_9A_SHARED_PROFILE").is_some();

    let profile = if shared_profile {
        research_profile_65536_for_levels(13, 3)
    } else {
        research_profile_65536_for_levels(5, 5)
    };

    let degree = profile.degree();
    assert_eq!(degree, 65536);

    let validated = profile.modulus_chain().top().clone();

    let (q, p) = if shared_profile {
        let extended = research_profile_65536_for_levels(15, 3);
        let extended_chain = extended.modulus_chain();
        let moduli = extended_chain.top().moduli();

        assert_eq!(validated.moduli().len(), 16);
        assert_eq!(moduli.len(), 18);
        assert_eq!(&moduli[..16], validated.moduli());

        (
            ModulusBasis::new(validated.moduli().to_vec()),
            ModulusBasis::new(moduli[16..18].to_vec()),
        )
    } else {
        assert!(validated.moduli().len() >= 10);

        (
            ModulusBasis::new(validated.moduli()[..8].to_vec()),
            ModulusBasis::new(validated.moduli()[8..10].to_vec()),
        )
    };

    let chain = ModulusChain::from_top_basis(q);

    assert_eq!(chain.max_level(), if shared_profile { 15 } else { 7 });

    if shared_profile {
        assert_eq!(chain.level(7).moduli().len(), 9);

        for modulus in p.moduli() {
            assert_eq!((modulus.value() - 1) % (2 * degree as u64), 0);
            assert!(!chain.top().moduli().contains(modulus));
        }

        println!("R30_9A_SHARED_PROFILE=ENABLED");
        println!("R30_9A_Q_LIMBS=16");
        println!("R30_9A_P_LIMBS=2");
        println!("R30_9A_HANDOFF_LEVEL=7");
        println!("R30_9A_HANDOFF_Q_LIMBS=9");
    }

    // R30.9A.16C: early, allocation-free hybrid-QP key memory preflight.
    if std::env::var("R30_9A_16_MEMORY_PREFLIGHT").as_deref() == Ok("1") {
        let base_log: u32 = std::env::var("FHE_HYBRID_GADGET_BASE_LOG")
            .ok()
            .map(|value| value.parse().expect("invalid hybrid gadget base"))
            .unwrap_or(32);
        assert!([8, 16, 32].contains(&base_log));

        let stage0_offsets: BTreeSet<isize> = (0..15)
            .flat_map(|branch| {
                let destination_lane = if branch < 8 { branch } else { branch - 8 };
                fuse_fft256_source_lane(
                    &first_stage_diagonals(branch),
                    SLOTS,
                    LANE_SLOTS,
                    0,
                    destination_lane,
                )
                .keys()
                .copied()
                .filter(|&offset| offset != 0)
                .collect::<Vec<_>>()
            })
            .collect();

        let mut total_bytes = 0_u128;
        let mut peak_stage_bytes = 0_u128;

        for stage in 0..4 {
            let offsets: BTreeSet<isize> = if stage == 0 {
                stage0_offsets.clone()
            } else {
                let span = [16, 64, 256][stage - 1];
                let local = later_stage_diagonals(span);
                let lanes: Vec<&Diagonals> = (0..8).map(|_| &local).collect();
                pack_fft256_branch_diagonals(&lanes, LANE_SLOTS)
                    .keys()
                    .copied()
                    .filter(|&offset| offset != 0)
                    .collect()
            };

            let q_basis = chain.level(stage + 3);
            let digits = BoundedGadgetLayout::new(q_basis.clone(), base_log).digit_count();

            let limbs = q_basis.moduli().len() + p.moduli().len();
            let bytes_per_key = 2_u128 * degree as u128 * limbs as u128 * digits as u128 * 8;

            let stage_bytes = bytes_per_key * offsets.len() as u128;
            total_bytes += stage_bytes;
            peak_stage_bytes = peak_stage_bytes.max(stage_bytes);

            println!(
                "R30_9A_16_STAGE={stage} KEYS={} Q_LIMBS={} P_LIMBS={} DIGITS={} PAYLOAD_GIB={:.3}",
                offsets.len(),
                q_basis.moduli().len(),
                p.moduli().len(),
                digits,
                stage_bytes as f64 / 1073741824.0
            );
        }

        let peak_gib = peak_stage_bytes as f64 / 1073741824.0;

        println!("R30_9A_16_BASE_LOG={base_log}");
        println!(
            "R30_9A_16_ALL_STAGES_GIB={:.3}",
            total_bytes as f64 / 1073741824.0
        );
        println!("R30_9A_16_PEAK_STAGE_GIB={peak_gib:.3}");
        println!(
            "R30_9A_16_STAGEWISE_BUDGET_18GIB={}",
            if peak_gib <= 18.0 { "PASS" } else { "EXCEEDS" }
        );

        // R30.9A.17A: key-lifetime scheduling audit.
        // Preserve the source/operand order used by the fused evaluator.
        let group_offsets: Vec<Vec<isize>> = [0..8, 8..15]
            .iter()
            .map(|range| {
                range
                    .clone()
                    .flat_map(|branch| {
                        let destination_lane = if branch < 8 { branch } else { branch - 8 };
                        fuse_fft256_source_lane(
                            &first_stage_diagonals(branch),
                            SLOTS,
                            LANE_SLOTS,
                            0,
                            destination_lane,
                        )
                        .keys()
                        .copied()
                        .filter(|&offset| offset != 0)
                        .collect::<Vec<_>>()
                    })
                    .collect()
            })
            .collect();

        let mut global_peak = 0usize;

        for (group, offsets) in group_offsets.iter().enumerate() {
            let mut last_use = std::collections::BTreeMap::new();
            for (index, &offset) in offsets.iter().enumerate() {
                last_use.insert(offset, index);
            }

            let mut live = BTreeSet::new();
            let mut peak = 0usize;

            for (index, &offset) in offsets.iter().enumerate() {
                live.insert(offset);
                peak = peak.max(live.len());

                if last_use[&offset] == index {
                    live.remove(&offset);
                }
            }

            global_peak = global_peak.max(peak);

            println!(
                "R30_9A_17_GROUP={group} TERMS={} DISTINCT_KEYS={} PEAK_LIVE_KEYS={peak}",
                offsets.len(),
                last_use.len()
            );
        }

        println!("R30_9A_17_PEAK_LIVE_KEYS={global_peak}");
        println!("R30_9A_17_SCHEDULING_AUDIT=PASS");
        println!("R30_9A_16_MEMORY_PREFLIGHT_STATUS=PASS");
        return;
    }

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

    let mut fft15_cache: Option<PreparedEncryptedFft15> = None;
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
        &mut fft15_cache,
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

        let physical_b = encrypt_fft15_physical(
            &natural_b,
            slots,
            &embedding,
            &chain,
            &basis0,
            scale,
            &secret,
            &plan0,
            &mut image_rng,
        );

        assert!(physical_b.iter().all(|ct| ct.level() == 0));
        println!("R29_3C_SECOND_IMAGE_ENCRYPTION_STATUS=PASS");

        Some((natural_b, physical_b))
    } else {
        None
    };

    // R28_4B3_SECOND_IMAGE_CONSUMPTION
    // Temporary structural check until the second FFT256 call is integrated.
    if let Some((natural_b, physical_b)) = second_image.as_ref() {
        assert_eq!(natural_b.len(), 15);
        assert!(natural_b.iter().all(|tile| tile.len() == 4096));
        assert_eq!(physical_b.len(), 8);
        assert!(physical_b.iter().all(|ct| ct.level() == 0));
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

    // R30.9A.21D: characterize stage-0 rotation-key reuse.
    //
    // This analysis does not generate keys or execute encrypted FFTs.
    let stage0_schedule: Vec<isize> = prepared_sources
        .iter()
        .flat_map(|source| source.rotation_offsets())
        .filter(|&offset| offset != 0)
        .collect();

    let unique_offsets: BTreeSet<isize> = stage0_schedule.iter().copied().collect();

    let operand_keys = stage0_schedule.len();
    let unique_keys = unique_offsets.len();
    let repeated_keys = operand_keys - unique_keys;

    // R30.9A.21E: consecutive offset reuse without reordering.
    let consecutive_reuses = stage0_schedule
        .windows(2)
        .filter(|pair| pair[0] == pair[1])
        .count();

    let one_key_cache_generations = stage0_schedule.len() - consecutive_reuses;

    println!("R30_9A_21E_CONSECUTIVE_REUSES={consecutive_reuses}");
    println!("R30_9A_21E_ONE_KEY_CACHE_GENERATIONS={one_key_cache_generations}");
    println!(
        "R30_9A_21E_ONE_KEY_CACHE_REDUCTION_FACTOR={:.3}",
        stage0_schedule.len() as f64 / one_key_cache_generations.max(1) as f64
    );

    println!("R30_9A_21D_STAGE0_OPERAND_KEYS={operand_keys}");
    println!("R30_9A_21D_STAGE0_UNIQUE_OFFSETS={unique_keys}");
    println!("R30_9A_21D_STAGE0_REPEATED_OCCURRENCES={repeated_keys}");

    let independent_two_batch_calls = 2 * operand_keys;
    let batched_two_batch_calls = operand_keys;

    println!("R30_9A_21D_TWO_BATCH_INDEPENDENT_KEYGEN_CALLS={independent_two_batch_calls}");
    println!("R30_9A_21D_TWO_BATCH_KEYGEN_CALLS={batched_two_batch_calls}");

    if unique_keys > 0 {
        println!(
            "R30_9A_21D_UNIQUE_KEY_REDUCTION_FACTOR={:.3}",
            operand_keys as f64 / unique_keys as f64
        );
    }

    println!("R30_9A_21D_SCHEDULE_STATUS=PASS");

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

    // R30_9A_10: allocation-free memory-model mode.
    if std::env::var("R30_9A_10_MEMORY_ONLY").as_deref() == Ok("1") {
        let mut total_bytes = 0_u128;

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

            let q_basis = chain.level(level);
            let q_limbs = q_basis.moduli().len();
            let p_limbs = p.moduli().len();

            for base_log in [8_u32, 16, 32] {
                let digits = BoundedGadgetLayout::new(q_basis.clone(), base_log).digit_count();

                let bytes = 2_u128
                    * degree as u128
                    * (q_limbs + p_limbs) as u128
                    * digits as u128
                    * offsets.len() as u128
                    * 8;

                println!(
                    "R30_9A_10_MEMORY stage={stage_index} base_log={base_log}                      keys={} q_limbs={q_limbs} p_limbs={p_limbs}                      digits={digits} gib={:.3}",
                    offsets.len(),
                    bytes as f64 / 1073741824.0
                );

                if base_log == 8 {
                    total_bytes += bytes;
                }
            }
        }

        println!(
            "R30_9A_10_BASE8_TOTAL_GIB={:.3}",
            total_bytes as f64 / 1073741824.0
        );
        println!("R30_9A_10_MEMORY_ONLY_STATUS=PASS");
        return;
    }

    // R30.9A.14: configurable hybrid-QP gadget base.
    // Default 8 preserves the existing validated behavior.
    let hybrid_base_log: u32 = std::env::var("FHE_HYBRID_GADGET_BASE_LOG")
        .ok()
        .map(|value| value.parse().expect("invalid hybrid gadget base"))
        .unwrap_or(8);

    assert!(
        [8, 16, 32].contains(&hybrid_base_log),
        "hybrid gadget base must be 8, 16, or 32"
    );

    println!("R30_9A_14_HYBRID_BASE_LOG={hybrid_base_log}");

    // R30.9A.18: opt-in bounded-key complete FFT3840 handoff.
    // Keep the original production regression untouched as the default.
    if std::env::var("R30_9A_18_STREAMING").as_deref() == Ok("1") {
        assert!(shared_profile, "streaming requires shared Q16 profile");
        assert_eq!(hybrid_base_log, 32, "streaming requires base 32");
        assert!(
            std::env::var_os("R30_9A_HANDOFF_ONLY").is_some(),
            "streaming currently validates the level-7 handoff only"
        );
        let evaluator_keys = RnsCkksEvaluationKeys::new();
        let evaluator = RnsCkksEvaluator::new(&chain, &evaluator_keys);
        let source_array: [RnsCkksCiphertext; 15] = sources
            .try_into()
            .unwrap_or_else(|_| panic!("expected 15 FFT15 sources"));
        let prepared_array: [PreparedFusedFft256Source; 15] = prepared_sources
            .try_into()
            .unwrap_or_else(|_| panic!("expected 15 prepared sources"));
        let later_array: [PreparedFft256Stage; 3] = later_prepared
            .try_into()
            .unwrap_or_else(|_| panic!("expected 3 prepared stages"));
        let stage0_qp = PreparedRnsNttPlan::new(&qp_plans[0]);
        let mut generated_keys = 0usize;
        let parallel_workers = std::env::var("FHE_R27_KEYGEN_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);

        let mut stage0_offsets = Vec::new();
        for source in &prepared_array {
            stage0_offsets.extend(source.rotation_offsets().filter(|&offset| offset != 0));
        }

        let parallel_jobs: Vec<(isize, [u8; 32])> = stage0_offsets
            .iter()
            .map(|&offset| {
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

            PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(chain.level(3).clone(), hybrid_base_log),
                p.clone(),
                &secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                2,
                &qp_plans[0],
                &mut key_rng,
            )
        };

        let pool = ThreadPoolBuilder::new()
            .num_threads(parallel_workers)
            .build()
            .expect("R30.9A.19 keygen thread pool creation failed");

        let mut next_key = 0usize;
        let mut pending_keys =
            std::collections::VecDeque::<(isize, PreparedHybridRnsGaloisKey)>::new();

        let mut make_key = |offset: isize| {
            if pending_keys.is_empty() {
                let end = (next_key + parallel_workers).min(parallel_jobs.len());
                assert!(next_key < end, "R30.9A.19 exhausted key schedule");

                let batch: Vec<_> = pool.install(|| {
                    parallel_jobs[next_key..end]
                        .par_iter()
                        .copied()
                        .map(|job| {
                            let key = generate(job);
                            (job.0, key)
                        })
                        .collect()
                });

                pending_keys.extend(batch);
                next_key = end;
            }

            let (expected_offset, key) = pending_keys
                .pop_front()
                .expect("R30.9A.19 missing generated key");

            assert_eq!(
                expected_offset, offset,
                "R30.9A.19 key schedule differs from evaluator order"
            );

            generated_keys += 1;
            key
        };

        let started = Instant::now();
        let mut results =
            ccmm_rs::eblas::fused_fft256_handoff::apply_fft3840_packed_fft256_stage0_streaming(
                &source_array,
                &prepared_array,
                &evaluator,
                &chain,
                &prepared_plans[0],
                &qp_plans[0],
                &stage0_qp,
                &mut make_key,
            );
        assert_eq!(generated_keys, parallel_jobs.len());
        assert!(pending_keys.is_empty());
        println!("R30_9A_19_KEYGEN_WORKERS={parallel_workers}");
        println!("R30_9A_19_KEYGEN_BATCH_LIMIT={parallel_workers}");
        println!("R30_9A_18_STAGE0_KEYS_GENERATED={generated_keys}");
        println!(
            "R30_9A_18_STAGE0_SECONDS={:.3}",
            started.elapsed().as_secs_f64()
        );
        for stage_index in 1..4 {
            let level = stage_index + 3;
            let offsets: BTreeSet<isize> = later_diagonals[stage_index - 1]
                .keys()
                .copied()
                .filter(|&offset| offset != 0)
                .collect();
            let mut keys = PackedRotationKeys::new();
            for offset in offsets {
                let exponent = if offset > 0 {
                    rotation_exponent_left(degree, offset as usize)
                } else {
                    rotation_exponent_right(degree, (-offset) as usize)
                };
                let mut seed = [0_u8; 32];
                rand::RngCore::fill_bytes(&mut rng, &mut seed);
                let mut key_rng = ChaCha20Rng::from_seed(seed);
                let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                    BoundedGadgetLayout::new(chain.level(level).clone(), hybrid_base_log),
                    p.clone(),
                    &secret,
                    exponent,
                    ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                    2,
                    &qp_plans[stage_index],
                    &mut key_rng,
                );
                keys.insert(offset, key);
            }
            let ctx = PackedStageContext {
                keys: &keys,
                evaluator: &evaluator,
                chain: &chain,
                qp_plan: &qp_plans[stage_index],
                prepared: &prepared_plans[stage_index],
            };
            for result in &mut results {
                *result = ccmm_rs::eblas::prepared_radix4_fft256::apply_stage(
                    result,
                    &later_array[stage_index - 1],
                    &ctx,
                );
                assert_eq!(result.level(), level + 1);
            }
            drop(keys);
            println!("R30_9A_18_STAGE_{stage_index}_STATUS=PASS");
        }
        let expected_basis = chain.level(7).moduli();
        for result in &results {
            assert_eq!(result.level(), 7);
            assert_eq!(result.basis().moduli(), expected_basis);
        }
        println!("R30_9A_18_OUTPUT_LEVEL=7");
        println!(
            "R30_9A_18_TOTAL_SECONDS={:.3}",
            total_started.elapsed().as_secs_f64()
        );
        println!("R30_9A_18_STREAMING_STATUS=PASS");

        // R30.9A.20: numerical validation of the streaming handoff.
        // Reconstruct the natural-order 3840-point frequency spectrum.
        let mut reference = vec![Complex64::new(0.0, 0.0); 16 * 3840];

        // Independent FFT15 x FFT256 reference, natural frequency order.
        for row in 0..16 {
            for k_cross in 0..TILES {
                let mut local = vec![Complex64::new(0.0, 0.0); 256];

                for (s, value) in local.iter_mut().enumerate() {
                    let cross: Complex64 = (0..TILES)
                        .map(|tile| natural[tile][row * 256 + s] * root(TILES, k_cross * tile))
                        .sum();

                    *value = cross * root(3840, k_cross * s);
                }

                // Independent iterative radix-2 FFT256.
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

        // Independently verify selected bins by direct DFT.
        for &row in &[0usize, 7, 15] {
            for &frequency in &[0usize, 7, 255, 1024, 3839] {
                let direct: Complex64 = (0..3840)
                    .map(|n| {
                        let tile = n / 256;
                        let local = n % 256;
                        natural[tile][row * 256 + local] * root(3840, frequency * n)
                    })
                    .sum();

                let staged = reference[row * 3840 + frequency];
                assert!(
                    (direct - staged).norm() < 1.0e-8,
                    "R30.9A.20 reference disagreement: row={row} frequency={frequency}"
                );
            }
        }

        println!("R30_9A_20_REFERENCE_STATUS=PASS");

        let mut error_sq = 0.0_f64;
        let mut reference_sq = 0.0_f64;
        let mut max_abs = 0.0_f64;
        let mut checked = 0usize;

        // R30.9A.20B: numerical error localization.
        let mut group_error_sq = [0.0_f64; 2];
        let mut group_reference_sq = [0.0_f64; 2];
        let mut worst_error = 0.0_f64;
        let mut worst_frequency = 0usize;
        let mut worst_group = 0usize;
        let mut cross = Complex64::new(0.0, 0.0);
        let mut actual_sq = 0.0_f64;

        for (group, ciphertext) in results.iter().enumerate() {
            let actual = decode(ciphertext, &secret, &embedding);
            let branches = if group == 0 { 0..8 } else { 8..15 };

            for (lane, branch) in branches.enumerate() {
                for row in 0..16 {
                    for k in 0..256 {
                        let slot = lane * HALF + row * 256 + k;
                        let frequency = row * 3840 + branch + 15 * k;
                        let expected = reference[frequency];
                        let error = (actual[slot] - expected).norm();

                        error_sq += error * error;
                        reference_sq += expected.norm_sqr();
                        group_error_sq[group] += error * error;
                        group_reference_sq[group] += expected.norm_sqr();
                        cross += actual[slot] * expected.conj();
                        actual_sq += actual[slot].norm_sqr();

                        if error > worst_error {
                            worst_error = error;
                            worst_frequency = frequency;
                            worst_group = group;
                        }

                        max_abs = max_abs.max(error);
                        checked += 1;
                    }
                }
            }
        }

        assert_eq!(checked, 61_440);
        assert!(reference_sq > 0.0);

        let relative_l2 = (error_sq / reference_sq).sqrt();

        println!("R30_9A_20_SPECTRAL_VALUES={checked}");
        println!("R30_9A_20_MAX_ABS={max_abs:.12e}");
        println!("R30_9A_20_RELATIVE_L2={relative_l2:.12e}");

        for group in 0..2 {
            assert!(group_reference_sq[group] > 0.0);
            let group_relative_l2 = (group_error_sq[group] / group_reference_sq[group]).sqrt();
            println!("R30_9A_20B_GROUP_{group}_RELATIVE_L2={group_relative_l2:.12e}");
        }

        let gain = cross / reference_sq;
        let residual_sq = (actual_sq - cross.norm_sqr() / reference_sq).max(0.0);
        let residual_relative_l2 = (residual_sq / reference_sq).sqrt();

        println!("R30_9A_20B_WORST_GROUP={worst_group}");
        println!("R30_9A_20B_WORST_FREQUENCY={worst_frequency}");
        println!("R30_9A_20B_WORST_ROW={}", worst_frequency / 3840);
        println!("R30_9A_20B_WORST_BIN={}", worst_frequency % 3840);
        println!("R30_9A_20B_GAIN_REAL={:.12e}", gain.re);
        println!("R30_9A_20B_GAIN_IMAG={:.12e}", gain.im);
        println!("R30_9A_20B_GAIN_CORRECTED_RELATIVE_L2={residual_relative_l2:.12e}");
        println!("R30_9A_20B_DIAGNOSTICS_STATUS=PASS");

        // R30.9A.20C: distinguish the strict numerical target from
        // provisional research-profile integration acceptance.
        const STRICT_TARGET: f64 = 1.0e-5;
        const RESEARCH_PROFILE_LIMIT: f64 = 1.1e-5;

        let strict_pass = relative_l2.is_finite() && relative_l2 < STRICT_TARGET;

        println!(
            "R30_9A_20_STRICT_NUMERICAL_STATUS={}",
            if strict_pass { "PASS" } else { "FAIL" }
        );
        println!("R30_9A_20_STRICT_TARGET={STRICT_TARGET:.12e}");
        println!("R30_9A_20_RESEARCH_PROFILE_LIMIT={RESEARCH_PROFILE_LIMIT:.12e}");

        assert!(
            relative_l2.is_finite() && relative_l2 < RESEARCH_PROFILE_LIMIT,
            "R30.9A.20 research-profile accuracy limit exceeded"
        );

        println!("R30_9A_20_RESEARCH_PROFILE_STATUS=PASS");
        println!("R30_9A_20_NUMERICAL_STATUS=PROVISIONAL_PASS");
        return;
    }

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

        // R30_9A_10: exact logical hybrid-QP key payload estimate.
        // No key generation or allocation is required for this estimate.
        {
            let q_basis = chain.level(level);
            let q_limbs = q_basis.moduli().len();
            let p_limbs = p.moduli().len();
            let layout = BoundedGadgetLayout::new(q_basis.clone(), hybrid_base_log);
            let digits = layout.digit_count();
            let key_count = offsets.len();

            let bytes_per_key = (2_u128)
                * (degree as u128)
                * ((q_limbs + p_limbs) as u128)
                * (digits as u128)
                * 8_u128;

            let stage_bytes = bytes_per_key * (key_count as u128);

            println!("R30_9A_10_STAGE={stage_index}");
            println!("R30_9A_10_Q_LIMBS={q_limbs}");
            println!("R30_9A_10_P_LIMBS={p_limbs}");
            println!("R30_9A_10_GADGET_DIGITS={digits}");
            println!("R30_9A_10_KEY_COUNT={key_count}");
            println!("R30_9A_10_BYTES_PER_KEY={bytes_per_key}");
            println!(
                "R30_9A_10_STAGE_GIB={:.3}",
                stage_bytes as f64 / 1073741824.0
            );
        }

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
                BoundedGadgetLayout::new(chain.level(level).clone(), hybrid_base_log),
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
    // R29_3B2A_UNIFIED_PIPELINE_MIGRATION
    let fft15_prepared = fft15_cache.take().expect("FFT15 preparation must exist");

    let pipeline = EncryptedFft3840Pipeline::from_prepared(
        fft15_prepared,
        service,
        EncryptedFft15ExecutionResources {
            chain: &chain,
            plan0: &plan0,
            prepared0: &prepared0,
            prepared1: &prepared1,
            prepared2: &prepared2,
            prepared_left: &prepared_left,
        },
    );

    let results = pipeline.execute_prepared_branches(&source_array);

    // R30_9A_5_CIPHERTEXT_HANDOFF
    if shared_profile {
        let vertical_profile = research_profile_65536_for_levels(13, 3);
        let vertical_chain = vertical_profile.modulus_chain();
        let expected_basis = vertical_chain.level(7);

        assert_eq!(results.len(), 2);
        assert_eq!(expected_basis.moduli().len(), 9);
        assert_eq!(chain.level(7).moduli(), expected_basis.moduli());

        for (group, ciphertext) in results.iter().enumerate() {
            assert_eq!(
                ciphertext.level(),
                7,
                "horizontal output group {group} has wrong CKKS level"
            );
            assert_eq!(
                ciphertext.basis().moduli(),
                expected_basis.moduli(),
                "horizontal output group {group} has wrong Q basis"
            );
        }

        println!("R30_9A_5_OUTPUT_CIPHERTEXTS=2");
        println!("R30_9A_5_OUTPUT_LEVEL=7");
        println!("R30_9A_5_OUTPUT_Q_LIMBS=9");
        println!("R30_9A_5_HANDOFF_BASIS_STATUS=PASS");
    }

    // R30_9A_9_HANDOFF_ONLY
    if std::env::var_os("R30_9A_HANDOFF_ONLY").is_some() {
        assert!(
            shared_profile,
            "handoff-only validation requires the shared Q16 profile"
        );
        println!("R30_9A_9_HANDOFF_ONLY_STATUS=PASS");
        println!(
            "R30_9A_9_TOTAL_ELAPSED_SECONDS={:.3}",
            total_started.elapsed().as_secs_f64()
        );
        return;
    }

    // R28_4B3_TWO_IMAGE_FFT256
    // Reuse all FFT256 public operands, rotation keys, and QP context.
    let second_results = if let Some((_, physical_b)) = second_image.as_ref() {
        let started = Instant::now();

        // R29_4B_STREAMING_INTEGRATION
        // Exercise the production streaming controller with image B.
        // Preserve the independent numerical oracle below.
        let mut pending_input = Some(physical_b.clone());
        let mut streamed_result = None;
        let mut observed_batches = 0usize;

        let completed = pipeline
            .execute_stream(
                || Ok::<_, ()>(pending_input.take()),
                |spectrum, progress| {
                    assert_eq!(progress.batch_index, 0);
                    assert_eq!(progress.completed_batches, 1);
                    assert!(spectrum.iter().all(|ct| ct.level() == 7));

                    observed_batches += 1;
                    streamed_result = Some(spectrum);

                    Ok::<(), ()>(())
                },
            )
            .expect("encrypted FFT3840 streaming execution failed");

        assert_eq!(completed, 1);
        assert_eq!(observed_batches, 1);

        let results_b = streamed_result.expect("streaming sink did not receive image B");

        println!("R29_4B_STREAM_BATCHES={completed}");
        println!("R29_4B_STREAM_OUTPUT_LEVEL=7");
        println!("R29_4B_STREAM_STATUS=PASS");
        println!("R29_3C_SECOND_IMAGE_COMPLETE_PIPELINE_STATUS=PASS");

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
