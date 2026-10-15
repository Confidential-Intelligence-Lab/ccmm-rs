// R16.5.13b: real ciphertext continuity FFT15 -> fused twiddle FFT256.
// One cross-frequency branch, 16 local FFT256 transforms; not full FFT3840.
// Research-only parameters and deterministic test keys; no intermediate decryption.
use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next, research_profile_16384,
    rotate_left_bounded_rns_ckks_with_prepared_execution as rotate_left_for_extraction,
    rotate_left_bounded_rns_ckks_with_prepared_execution,
    rotate_right_bounded_rns_ckks_with_prepared_execution, rotation_exponent_left,
    rotation_exponent_right, BoundedRnsGaloisKey, CkksCanonicalEmbedding, CkksChainState,
    PreparedBoundedRnsGaloisKey, PreparedRnsPlaintextNtt, RnsCkksCiphertext, RnsCkksEvaluationKeys,
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
type RotationKeys = BTreeMap<isize, PreparedBoundedRnsGaloisKey>;
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
fn stage_diagonals(slots: usize, span: usize, k_cross: usize) -> Diagonals {
    assert!([4, 16, 64, 256].contains(&span));
    let quarter = span / 4;
    let mut diagonals = Diagonals::new();
    for i in 0..slots {
        let within = i % span;
        let p = within / quarter;
        let j = within % quarter;
        for q in 0..4 {
            let input_within = j + q * quarter;
            let offset = input_within as isize - within as isize;
            let mut coefficient = root(span, q * j) * root(4, p * q);
            if span == 4 {
                // The input is base-4 digit-reversed. Twiddle exponents refer
                // to natural local sample positions, not packed slot indices.
                let packed_input = i as isize + offset;
                assert!(packed_input >= 0 && (packed_input as usize) < slots);
                let natural_local = reverse_base4_4((packed_input as usize) % 256);
                coefficient *= root(3840, k_cross * natural_local);
            }
            diagonals
                .entry(offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); slots])[i] = coefficient;
        }
    }
    diagonals
}

struct StageContext<'a> {
    keys: &'a RotationKeys,
    evaluator: &'a RnsCkksEvaluator<'a>,
    embedding: &'a CkksCanonicalEmbedding,
    chain: &'a ModulusChain,
    plan: &'a RnsNttPlan,
    prepared: &'a PreparedRnsNttPlan,
}
fn apply_stage(
    input: &RnsCkksCiphertext,
    diagonals: &Diagonals,
    ctx: &StageContext<'_>,
) -> RnsCkksCiphertext {
    let plain_scale = ctx
        .chain
        .dropped_modulus(input.level())
        .expect("rescale level available")
        .value() as f64;
    let mut accumulator: Option<RnsCkksCiphertext> = None;
    for (&offset, diagonal) in diagonals {
        let rotated = if offset > 0 {
            Some(rotate_left_bounded_rns_ckks_with_prepared_execution(
                input,
                offset as usize,
                ctx.keys.get(&offset).expect("left rotation key"),
                ctx.chain,
                ctx.plan,
                ctx.prepared,
            ))
        } else if offset < 0 {
            Some(rotate_right_bounded_rns_ckks_with_prepared_execution(
                input,
                (-offset) as usize,
                ctx.keys.get(&offset).expect("right rotation key"),
                ctx.chain,
                ctx.plan,
                ctx.prepared,
            ))
        } else {
            None
        };
        let source = rotated.as_ref().unwrap_or(input);
        let plain = encode_rns(diagonal, ctx.embedding, input.basis(), plain_scale);
        let prepared_plain = PreparedRnsPlaintextNtt::new(&plain, ctx.prepared);
        let term = multiply_plain_rns_ckks_with_prepared_ntt(
            source,
            &prepared_plain,
            plain_scale,
            ctx.chain,
            ctx.prepared,
        );
        accumulator = Some(match accumulator {
            Some(previous) => ctx.evaluator.add(&previous, &term),
            None => term,
        });
    }
    rescale_rns_ckks_to_next(&accumulator.expect("nonempty stage"), ctx.chain)
}

fn generate_keys(
    diagonals: &Diagonals,
    basis: &ModulusBasis,
    plan: &RnsNttPlan,
    degree: usize,
    secret: &[i8],
    rng: &mut ChaCha20Rng,
) -> RotationKeys {
    diagonals
        .keys()
        .filter(|&&offset| offset != 0)
        .map(|&offset| {
            let exponent = if offset > 0 {
                rotation_exponent_left(degree, offset as usize)
            } else {
                rotation_exponent_right(degree, (-offset) as usize)
            };
            let key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedRnsKeygenConfig {
                    plaintext_modulus: 2,
                    layout: BoundedGadgetLayout::new(basis.clone(), 8),
                    plan,
                },
                secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                rng,
            );
            (offset, PreparedBoundedRnsGaloisKey::new(&key, plan))
        })
        .collect()
}

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

#[test]
fn r16_5_13b_actual_fft15_to_fused_fft256() {
    let total = Instant::now();
    let profile = research_profile_16384();
    let degree = profile.degree();
    let chain = profile.modulus_chain();
    assert!(chain.max_level() >= 7);
    let scale = profile.initial_scale();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();
    assert_eq!(slots % 256, 0);
    const HALF: usize = 4096;
    const TILES: usize = 15;
    let basis0 = chain.top().clone();
    let plan0 = RnsNttPlan::new(basis0.moduli().to_vec(), degree);
    let prepared0 = PreparedRnsNttPlan::new(&plan0);
    let plan1 = RnsNttPlan::new(chain.level(1).moduli().to_vec(), degree);
    let prepared1 = PreparedRnsNttPlan::new(&plan1);
    let plan2 = RnsNttPlan::new(chain.level(2).moduli().to_vec(), degree);
    let prepared2 = PreparedRnsNttPlan::new(&plan2);
    let bases: Vec<ModulusBasis> = (3..7).map(|level| chain.level(level).clone()).collect();
    let plans: Vec<RnsNttPlan> = bases
        .iter()
        .map(|basis| RnsNttPlan::new(basis.moduli().to_vec(), degree))
        .collect();
    let prepared_plans: Vec<PreparedRnsNttPlan> =
        plans.iter().map(PreparedRnsNttPlan::new).collect();
    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();
    let mut rng = ChaCha20Rng::seed_from_u64(0x1651_130B);
    // One physical tile row: 15 horizontal tiles, 4096 samples per tile.
    // Each tile contains 16 independent 256-sample horizontal segments.
    // The encrypted FFT15 acts on the tile axis, preserving all 4096 slots.
    let natural: Vec<Vec<Complex64>> = (0..TILES)
        .map(|tile| {
            (0..HALF)
                .map(|slot| {
                    let x = (tile * HALF + slot) as f64;
                    Complex64::new(0.0003 * (0.013 * x).sin(), 0.0003 * (0.017 * x).cos())
                })
                .collect()
        })
        .collect();
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
            let plaintext = encode_rns(&values, &embedding, &basis0, scale);
            let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                &secret,
                &plan0,
                &mut rng,
            );
            RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain)
        })
        .collect();
    let keygen_start = Instant::now();
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
    let low_mask: Vec<Complex64> = (0..slots)
        .map(|s| Complex64::new(if s < HALF { 1.0 } else { 0.0 }, 0.0))
        .collect();
    let prepared_mask =
        PreparedComplexSlotsCp::new(&physical[0], &low_mask, &embedding, &chain, &plan0);
    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);
    let aligned: Vec<RnsCkksCiphertext> = (0..TILES)
        .map(|tile| {
            let source = &physical[tile / 2];
            if tile % 2 == 0 {
                multiply_complex_slots_cp_prepared(source, &prepared_mask, &chain, &prepared0)
            } else {
                let shifted = rotate_left_for_extraction(
                    source,
                    HALF,
                    &prepared_left,
                    &chain,
                    &plan0,
                    &prepared0,
                );
                multiply_complex_slots_cp_prepared(&shifted, &prepared_mask, &chain, &prepared0)
            }
        })
        .collect();
    assert!(aligned.iter().all(|ct| ct.level() == 1));
    let radix5 = PreparedRadix5Cp::new(&aligned[0], &embedding, &chain, &plan1);
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
            &chain,
            &prepared1,
        )
    });
    let radix3 = PreparedFft15TwiddledRadix3Cp::new(&branches[0][0], &embedding, &chain, &plan2);
    let groups: [[RnsCkksCiphertext; 3]; 5] = std::array::from_fn(|k| {
        packed_fft15_twiddled_radix3_cp_prepared(
            &evaluator,
            [&branches[0][k], &branches[1][k], &branches[2][k]],
            k,
            &radix3,
            &chain,
            &prepared2,
        )
    });
    let k_cross = 7usize;
    // Actual FFT15 output, no intermediate decrypt, no mod-switch simulation.
    let ct = &groups[k_cross % 5][k_cross / 5];
    assert_eq!(ct.level(), 3);
    let fft15_ms = keygen_start.elapsed().as_secs_f64() * 1000.0;
    // Independent 3840-point DFT reference for this cross-tile frequency:
    // output frequency = k_cross + 15 * k_local, for each local 256-sample row.
    let expected: Vec<Complex64> = (0..HALF)
        .map(|i| {
            let block = (i / 256) * 256;
            let k_local = i % 256;
            (0..TILES)
                .flat_map(|tile| (0..256).map(move |s| (tile, s)))
                .map(|(tile, s)| {
                    let input = natural[tile][block + s];
                    let n = tile * 256 + s;
                    input * root(3840, (k_cross + 15 * k_local) * n)
                })
                .sum()
        })
        .collect();
    let spans = [4usize, 16, 64, 256];
    let diagonals: Vec<Diagonals> = spans
        .iter()
        .map(|&span| stage_diagonals(slots, span, k_cross))
        .collect();
    let keygen_start = Instant::now();
    let keys: Vec<RotationKeys> = (0..4)
        .map(|level| {
            generate_keys(
                &diagonals[level],
                &bases[level],
                &plans[level],
                degree,
                &secret,
                &mut rng,
            )
        })
        .collect();
    let keygen_ms = keygen_start.elapsed().as_secs_f64() * 1000.0;
    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);
    let start = Instant::now();
    let mut current = ct.clone();
    let mut stage_times_ms = Vec::new();
    for level in 0..4 {
        let stage_start = Instant::now();
        current = apply_stage(
            &current,
            &diagonals[level],
            &StageContext {
                keys: &keys[level],
                evaluator: &evaluator,
                embedding: &embedding,
                chain: &chain,
                plan: &plans[level],
                prepared: &prepared_plans[level],
            },
        );
        assert_eq!(current.level(), level + 4, "unexpected stage level");
        stage_times_ms.push(stage_start.elapsed().as_secs_f64() * 1000.0);
    }
    let execution_ms = start.elapsed().as_secs_f64() * 1000.0;
    let actual = decode(&current, &secret, &embedding);

    let mut error_sq = 0.0;
    let mut reference_sq = 0.0;
    let mut max_abs = 0.0_f64;
    for (&a, &e) in actual[..HALF].iter().zip(expected.iter()) {
        let delta = (a - e).norm();
        error_sq += delta * delta;
        reference_sq += e.norm_sqr();
        max_abs = max_abs.max(delta);
    }
    let relative = (error_sq / reference_sq).sqrt();
    println!("R16_COMPOSED_K_CROSS={k_cross}");
    println!("R16_COMPOSED_FFT15_MS={fft15_ms:.3}");
    println!("R16_COMPOSED_REAL_CIPHERTEXT_CONTINUITY=true");
    println!("R16_COMPOSED_INPUT_LEVEL=3");
    println!("R16_HORIZ_FUSED_TWIDDLE_EXTRA_LEVELS=0");
    println!("R16_HORIZ_FUSED_DEGREE={degree}");
    println!("R16_HORIZ_FUSED_SLOTS={slots}");
    println!("R16_HORIZ_FUSED_TRANSFORMS={}", HALF / 256);
    println!("R16_HORIZ_FUSED_STAGE_SPANS={spans:?}");
    println!(
        "R16_HORIZ_FUSED_DIAGONALS={:?}",
        diagonals.iter().map(|d| d.len()).collect::<Vec<_>>()
    );
    println!(
        "R16_HORIZ_FUSED_ROTATION_KEYS={}",
        keys.iter().map(|k| k.len()).sum::<usize>()
    );
    println!("R16_HORIZ_FUSED_FINAL_LEVEL={}", current.level());
    println!("R16_HORIZ_FUSED_FINAL_SCALE={:.12e}", current.scale());
    println!("R16_HORIZ_FUSED_REL_L2={relative:.12e}");
    println!("R16_HORIZ_FUSED_MAX_ABS={max_abs:.12e}");
    println!("R16_HORIZ_FUSED_KEYGEN_MS={keygen_ms:.3}");
    println!("R16_HORIZ_FUSED_STAGE_TIMES_MS={stage_times_ms:?}");
    println!("R16_HORIZ_FUSED_EXECUTION_MS={execution_ms:.3}");
    println!(
        "R16_HORIZ_FUSED_TOTAL_MS={:.3}",
        total.elapsed().as_secs_f64() * 1000.0
    );
    let pass = current.level() == 7 && relative.is_finite() && relative < 5e-2 && max_abs < 5e-2;
    println!("R16_COMPOSED_STATUS={}", if pass { "PASS" } else { "FAIL" });
    assert!(pass, "encrypted FFT256 numerical validation failed");
}
