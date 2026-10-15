// R16.5.15c: four tile-local stride-256 encrypted FFT16 transforms.
// Research-only, standalone integration test. No intermediate decryption.
// R16.5.14d: encrypted FFT2160 = FFT135 DIF + fused-twiddle FFT16 DIT.
// 135 ciphertexts, each carrying 16 local rows in CKKS slots 0..15.
// Research-only deterministic keys. No intermediate decryption.
use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next,
    research_profile_32768_for_levels, rotate_left_bounded_rns_ckks_with_prepared_execution,
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
fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (exponent % n) as f64 / n as f64)
}

type Diagonals = BTreeMap<isize, Vec<Complex64>>;
type RotationKeys = BTreeMap<isize, PreparedBoundedRnsGaloisKey>;
fn reverse_base4_2(i: usize) -> usize {
    (i % 4) * 4 + i / 4
}

// Radix-4 DIT over 16 rows, each row separated by 256 CKKS slots.
// The diagonal masks restrict each rotation to its own 4096-slot tile.
const ROW_STRIDE: usize = 256;
const TILE_SLOTS: usize = 4096;
fn stage_diagonals(slots: usize, span: usize, k_cross: usize) -> Diagonals {
    assert!(span == 4 || span == 16);
    assert_eq!(slots % TILE_SLOTS, 0);
    let quarter = span / 4;
    let mut diagonals = Diagonals::new();
    for i in 0..slots {
        let local_row = (i % TILE_SLOTS) / ROW_STRIDE;
        let within = local_row % span;
        let p = within / quarter;
        let j = within % quarter;
        for q in 0..4 {
            let input_row = (local_row / span) * span + j + q * quarter;
            let offset = (input_row as isize - local_row as isize) * ROW_STRIDE as isize;
            let source = i as isize + offset;
            assert!(source >= 0 && source < slots as isize);
            assert_eq!(source as usize / TILE_SLOTS, i / TILE_SLOTS);
            let mut coefficient = root(span, q * j) * root(4, p * q);
            if span == 4 {
                coefficient *= root(2160, k_cross * reverse_base4_2(input_row));
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
                .expect("CKKS coefficient must fit f64")
                / ct.scale()
        })
        .collect();
    embedding.coefficients_to_slots(&coefficients)
}

#[test]
#[ignore = "degree-32768 CKKS rotations and key generation; run explicitly"]
fn r16_5_15c_stride256_encrypted_fft16() {
    const K_CROSS: usize = 7;
    const TILES: usize = 4;
    let started = Instant::now();
    let profile = research_profile_32768_for_levels(13, 3);
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    assert_eq!(chain.top().moduli().len(), 16);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();
    assert_eq!(slots, TILES * TILE_SLOTS);
    let basis0 = chain.top().clone();
    let plan0 = RnsNttPlan::new(basis0.moduli().to_vec(), degree);
    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();
    let mut rng = ChaCha20Rng::seed_from_u64(0x1651_15C0);
    let mut values = vec![Complex64::new(0.0, 0.0); slots];
    let mut natural = vec![Complex64::new(0.0, 0.0); slots];
    for tile in 0..TILES {
        for row in 0..16 {
            for frequency in 0..ROW_STRIDE {
                let x = (tile * TILE_SLOTS + row * ROW_STRIDE + frequency) as f64;
                natural[tile * TILE_SLOTS + row * ROW_STRIDE + frequency] =
                    Complex64::new(0.0001 * (0.013 * x).sin(), 0.0001 * (0.017 * x).cos());
            }
        }
        for packed_row in 0..16 {
            let row = reverse_base4_2(packed_row);
            for frequency in 0..ROW_STRIDE {
                values[tile * TILE_SLOTS + packed_row * ROW_STRIDE + frequency] =
                    natural[tile * TILE_SLOTS + row * ROW_STRIDE + frequency];
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
    let mut ct = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);
    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);
    for (stage, span) in [4usize, 16].into_iter().enumerate() {
        let stage_started = Instant::now();
        let basis = chain.level(stage).clone();
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let prepared = PreparedRnsNttPlan::new(&plan);
        let diagonals = stage_diagonals(slots, span, K_CROSS);
        let rotation_keys = generate_keys(&diagonals, &basis, &plan, degree, &secret, &mut rng);
        let ctx = StageContext {
            keys: &rotation_keys,
            evaluator: &evaluator,
            embedding: &embedding,
            chain: &chain,
            plan: &plan,
            prepared: &prepared,
        };
        ct = apply_stage(&ct, &diagonals, &ctx);
        assert_eq!(ct.level(), stage + 1);
        assert_eq!(ct.scale(), scale);
        println!("R16_STRIDE256_STAGE_{}_LEVEL={}", stage + 1, ct.level());
        println!(
            "R16_STRIDE256_STAGE_{}_MS={:.3}",
            stage + 1,
            stage_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    let actual = decode(&ct, &secret, &embedding);
    let mut error_sq = 0.0;
    let mut reference_sq = 0.0;
    let mut max_abs = 0.0_f64;
    for tile in 0..TILES {
        for frequency in 0..ROW_STRIDE {
            for k_local in 0..16 {
                let expected: Complex64 = (0..16)
                    .map(|row| {
                        let input = natural[tile * TILE_SLOTS + row * ROW_STRIDE + frequency];
                        input * root(2160, K_CROSS * row) * root(16, k_local * row)
                    })
                    .sum();
                let got = actual[tile * TILE_SLOTS + k_local * ROW_STRIDE + frequency];
                let delta = (got - expected).norm();
                error_sq += delta * delta;
                reference_sq += expected.norm_sqr();
                max_abs = max_abs.max(delta);
            }
        }
    }
    let rel_l2 = (error_sq / reference_sq).sqrt();
    println!("R16_STRIDE256_FINAL_LEVEL={}", ct.level());
    println!("R16_STRIDE256_REL_L2={rel_l2:.12e}");
    println!("R16_STRIDE256_MAX_ABS={max_abs:.12e}");
    println!(
        "R16_STRIDE256_TOTAL_MS={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    let pass = ct.level() == 2
        && rel_l2.is_finite()
        && max_abs.is_finite()
        && rel_l2 < 1e-2
        && max_abs < 1e-3;
    println!(
        "R16_STRIDE256_ENCRYPTED_STATUS={}",
        if pass { "PASS" } else { "FAIL" }
    );
    assert!(
        pass,
        "stride-256 encrypted FFT16 accuracy outside research thresholds"
    );
}
