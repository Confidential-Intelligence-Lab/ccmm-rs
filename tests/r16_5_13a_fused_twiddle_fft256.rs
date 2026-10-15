// R16.5.13a: fused horizontal W3840 twiddle in the first radix-4 stage.
// Input starts at level 3 to model the level budget after physical FFT15.
// This is NOT yet an encrypted FFT15-to-FFT256 composition.
// Research-only parameters and deterministic test keys; no intermediate decryption.
use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    mod_switch_rns_ckks_to_next, multiply_plain_rns_ckks_with_prepared_ntt,
    rescale_rns_ckks_to_next, research_profile_16384,
    rotate_left_bounded_rns_ckks_with_prepared_execution,
    rotate_right_bounded_rns_ckks_with_prepared_execution, rotation_exponent_left,
    rotation_exponent_right, BoundedRnsGaloisKey, CkksCanonicalEmbedding, CkksChainState,
    PreparedBoundedRnsGaloisKey, PreparedRnsPlaintextNtt, RnsCkksCiphertext, RnsCkksEvaluationKeys,
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
fn r16_5_13a_fused_horizontal_twiddle_level_budget() {
    let total = Instant::now();
    let profile = research_profile_16384();
    let degree = profile.degree();
    let chain = profile.modulus_chain();
    assert!(chain.max_level() >= 7);
    let scale = profile.initial_scale();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();
    assert_eq!(slots % 256, 0);
    let basis0 = chain.top().clone();
    let encryption_plan = RnsNttPlan::new(basis0.moduli().to_vec(), degree);
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
    let mut rng = ChaCha20Rng::seed_from_u64(0x1651_130A);
    let natural: Vec<Complex64> = (0..slots)
        .map(|i| {
            let x = i as f64;
            Complex64::new(0.001 * (0.013 * x).sin(), 0.001 * (0.017 * x).cos())
        })
        .collect();
    let packed: Vec<Complex64> = (0..slots)
        .map(|i| natural[(i / 256) * 256 + reverse_base4_4(i % 256)])
        .collect();
    // Representative nontrivial cross-tile frequency branch (0 < k_cross < 15).
    let k_cross = 7usize;
    let expected: Vec<Complex64> = (0..slots)
        .map(|i| {
            let block = (i / 256) * 256;
            let k = i % 256;
            (0..256)
                .map(|j| natural[block + j] * root(3840, k_cross * j) * root(256, (k * j) % 256))
                .sum()
        })
        .collect();
    let plaintext = encode_rns(&packed, &embedding, &basis0, scale);
    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &encryption_plan,
        &mut rng,
    );
    let mut ct = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);
    // Explicitly simulate arrival at level 3. This does NOT reproduce
    // accumulated FFT15 noise; full composition is a separate checkpoint.
    for _ in 0..3 {
        ct = mod_switch_rns_ckks_to_next(&ct, &chain);
    }
    assert_eq!(ct.level(), 3);
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
    let mut current = ct;
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
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        let delta = (a - e).norm();
        error_sq += delta * delta;
        reference_sq += e.norm_sqr();
        max_abs = max_abs.max(delta);
    }
    let relative = (error_sq / reference_sq).sqrt();
    println!("R16_HORIZ_FUSED_K_CROSS={k_cross}");
    println!("R16_HORIZ_FUSED_INPUT_LEVEL=3");
    println!("R16_HORIZ_FUSED_TWIDDLE_EXTRA_LEVELS=0");
    println!("R16_HORIZ_FUSED_DEGREE={degree}");
    println!("R16_HORIZ_FUSED_SLOTS={slots}");
    println!("R16_HORIZ_FUSED_TRANSFORMS={}", slots / 256);
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
    let pass = current.level() == 7 && relative.is_finite() && relative < 1e-3 && max_abs < 1e-3;
    println!(
        "R16_HORIZ_FUSED_STATUS={}",
        if pass { "PASS" } else { "FAIL" }
    );
    assert!(pass, "encrypted FFT256 numerical validation failed");
}
