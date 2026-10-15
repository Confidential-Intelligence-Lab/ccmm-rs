// R16.5.12c: encrypted FFT4, baseline two-level versus fused one-level.
// Research-only validation. Input is bit-reversed within each four-slot block.
// No intermediate decryption. Deterministic secret/RNG are test-only.
use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next, research_profile_16384,
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
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
    PreparedRnsNttPlan, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::f64::consts::PI;
use std::time::Instant;

const SIGMA: f64 = 3.19;
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
fn zero() -> Complex64 {
    Complex64::new(0.0, 0.0)
}
fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}
fn bitrev2(j: usize) -> usize {
    [0, 2, 1, 3][j]
}
fn omega(k: usize, j: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (k * j) as f64 / 4.0)
}

// Seven signed diagonals, offsets -3..=3. No FFT4 block boundary crossing.
// For output i, offset d reads input i+d within the same four-slot block.
fn diagonals(slots: usize, kind: usize) -> [Vec<Complex64>; 7] {
    let mut result = std::array::from_fn(|_| vec![zero(); slots]);
    for i in 0..slots {
        let r = i % 4;
        for j in 0..4 {
            let coefficient = match kind {
                1 if r % 2 == 0 && (j == r || j == r + 1) => one(),
                1 if r % 2 == 1 && j == r - 1 => one(),
                1 if r % 2 == 1 && j == r => -one(),
                2 if r < 2 && j == r => one(),
                2 if r < 2 && j == r + 2 => omega(r, 1),
                2 if r >= 2 && j == r - 2 => one(),
                2 if r >= 2 && j == r => -omega(r - 2, 1),
                3 => omega(r, bitrev2(j)),
                _ => zero(),
            };
            result[(j as isize - r as isize + 3) as usize][i] = coefficient;
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn apply_stage(
    input: &RnsCkksCiphertext,
    diagonals: &[Vec<Complex64>; 7],
    left_keys: &[Option<PreparedBoundedRnsGaloisKey>; 3],
    right_keys: &[Option<PreparedBoundedRnsGaloisKey>; 3],
    evaluator: &RnsCkksEvaluator<'_>,
    embedding: &CkksCanonicalEmbedding,
    chain: &ccmm_rs::ring::ModulusChain,
    plan: &RnsNttPlan,
    prepared_plan: &PreparedRnsNttPlan,
) -> RnsCkksCiphertext {
    let dropped = chain
        .dropped_modulus(input.level())
        .expect("rescale level available");
    let plain_scale = dropped.value() as f64;
    let mut accumulator: Option<RnsCkksCiphertext> = None;
    for (index, diagonal) in diagonals.iter().enumerate() {
        if diagonal.iter().all(|&v| v == zero()) {
            continue;
        }
        let offset = index as isize - 3;
        let rotated = if offset > 0 {
            Some(rotate_left_bounded_rns_ckks_with_prepared_execution(
                input,
                offset as usize,
                left_keys[offset as usize - 1].as_ref().expect("left key"),
                chain,
                plan,
                prepared_plan,
            ))
        } else if offset < 0 {
            let magnitude = (-offset) as usize;
            Some(rotate_right_bounded_rns_ckks_with_prepared_execution(
                input,
                magnitude,
                right_keys[magnitude - 1].as_ref().expect("right key"),
                chain,
                plan,
                prepared_plan,
            ))
        } else {
            None
        };
        let source = rotated.as_ref().unwrap_or(input);
        let plain = encode_rns(diagonal, embedding, input.basis(), plain_scale);
        let prepared_plain = PreparedRnsPlaintextNtt::new(&plain, prepared_plan);
        let term = multiply_plain_rns_ckks_with_prepared_ntt(
            source,
            &prepared_plain,
            plain_scale,
            chain,
            prepared_plan,
        );
        accumulator = Some(match accumulator {
            Some(previous) => evaluator.add(&previous, &term),
            None => term,
        });
    }
    rescale_rns_ckks_to_next(&accumulator.expect("nonempty diagonals"), chain)
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
fn metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    let mut err = 0.0;
    let mut reference = 0.0;
    let mut max_abs = 0.0_f64;
    for (&a, &e) in actual.iter().zip(expected) {
        let delta = (a - e).norm();
        err += delta * delta;
        reference += e.norm_sqr();
        max_abs = max_abs.max(delta);
    }
    ((err / reference).sqrt(), max_abs)
}

#[test]
fn r16_5_12c_encrypted_fft4_fused_one_level() {
    let total = Instant::now();
    let profile = research_profile_16384();
    let degree = profile.degree();
    let chain = profile.modulus_chain();
    assert!(chain.max_level() >= 2);
    let scale = profile.initial_scale();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();
    assert_eq!(slots % 4, 0);
    let basis0 = chain.top().clone();
    let basis1 = chain.level(1).clone();
    let plan0 = RnsNttPlan::new(basis0.moduli().to_vec(), degree);
    let plan1 = RnsNttPlan::new(basis1.moduli().to_vec(), degree);
    let prepared0 = PreparedRnsNttPlan::new(&plan0);
    let prepared1 = PreparedRnsNttPlan::new(&plan1);
    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();
    let mut rng = ChaCha20Rng::seed_from_u64(0x1651_120C);
    let natural: Vec<Complex64> = (0..slots)
        .map(|i| {
            let x = i as f64;
            Complex64::new(0.003 * (0.013 * x).sin(), 0.003 * (0.017 * x).cos())
        })
        .collect();
    let packed: Vec<Complex64> = (0..slots)
        .map(|i| natural[(i / 4) * 4 + bitrev2(i % 4)])
        .collect();
    let expected: Vec<Complex64> = (0..slots)
        .map(|i| {
            let block = (i / 4) * 4;
            let k = i % 4;
            (0..4).map(|j| natural[block + j] * omega(k, j)).sum()
        })
        .collect();
    let plaintext = encode_rns(&packed, &embedding, &basis0, scale);
    let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &plan0,
        &mut rng,
    );
    let ct = RnsCkksCiphertext::new(rlwe, CkksChainState::top(&chain, scale), &chain);

    let keygen_start = Instant::now();
    let make_key = |basis: &ccmm_rs::ring::ModulusBasis,
                    plan: &RnsNttPlan,
                    exponent: usize,
                    rng: &mut ChaCha20Rng| {
        let key = BoundedRnsGaloisKey::generate_with_distribution_ntt_rng(
            BoundedRnsKeygenConfig {
                plaintext_modulus: 2,
                layout: BoundedGadgetLayout::new(basis.clone(), 8),
                plan,
            },
            &secret,
            exponent,
            ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
            rng,
        );
        PreparedBoundedRnsGaloisKey::new(&key, plan)
    };
    let left0 = std::array::from_fn(|i| {
        Some(make_key(
            &basis0,
            &plan0,
            rotation_exponent_left(degree, i + 1),
            &mut rng,
        ))
    });
    let right0 = std::array::from_fn(|i| {
        Some(make_key(
            &basis0,
            &plan0,
            rotation_exponent_right(degree, i + 1),
            &mut rng,
        ))
    });
    let left1 = [
        None,
        Some(make_key(
            &basis1,
            &plan1,
            rotation_exponent_left(degree, 2),
            &mut rng,
        )),
        None,
    ];
    let right1 = [
        None,
        Some(make_key(
            &basis1,
            &plan1,
            rotation_exponent_right(degree, 2),
            &mut rng,
        )),
        None,
    ];
    let keygen_ms = ms(keygen_start);
    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);
    let stage1 = diagonals(slots, 1);
    let stage2 = diagonals(slots, 2);
    let fused = diagonals(slots, 3);

    let baseline_start = Instant::now();
    let intermediate = apply_stage(
        &ct, &stage1, &left0, &right0, &evaluator, &embedding, &chain, &plan0, &prepared0,
    );
    let baseline = apply_stage(
        &intermediate,
        &stage2,
        &left1,
        &right1,
        &evaluator,
        &embedding,
        &chain,
        &plan1,
        &prepared1,
    );
    let baseline_ms = ms(baseline_start);
    let fused_start = Instant::now();
    let fused_ct = apply_stage(
        &ct, &fused, &left0, &right0, &evaluator, &embedding, &chain, &plan0, &prepared0,
    );
    let fused_ms = ms(fused_start);

    assert_eq!(baseline.level(), 2);
    assert_eq!(fused_ct.level(), 1);
    let (baseline_rel, baseline_max) = metrics(&decode(&baseline, &secret, &embedding), &expected);
    let (fused_rel, fused_max) = metrics(&decode(&fused_ct, &secret, &embedding), &expected);
    println!("R16_FFT4_DEGREE={degree}");
    println!("R16_FFT4_SLOTS={slots}");
    println!("R16_FFT4_TRANSFORMS={}", slots / 4);
    println!("R16_FFT4_KEYGEN_MS={keygen_ms:.3}");
    println!("R16_FFT4_BASELINE_LEVEL={}", baseline.level());
    println!("R16_FFT4_FUSED_LEVEL={}", fused_ct.level());
    println!("R16_FFT4_BASELINE_SCALE={:.12e}", baseline.scale());
    println!("R16_FFT4_FUSED_SCALE={:.12e}", fused_ct.scale());
    println!("R16_FFT4_BASELINE_REL_L2={baseline_rel:.12e}");
    println!("R16_FFT4_BASELINE_MAX_ABS={baseline_max:.12e}");
    println!("R16_FFT4_FUSED_REL_L2={fused_rel:.12e}");
    println!("R16_FFT4_FUSED_MAX_ABS={fused_max:.12e}");
    println!("R16_FFT4_BASELINE_MS={baseline_ms:.3}");
    println!("R16_FFT4_FUSED_MS={fused_ms:.3}");
    println!("R16_FFT4_TOTAL_MS={:.3}", ms(total));
    let pass = baseline_rel.is_finite()
        && fused_rel.is_finite()
        && baseline_rel < 1e-3
        && fused_rel < 1e-3
        && baseline_max < 1e-3
        && fused_max < 1e-3;
    println!("R16_FFT4_STATUS={}", if pass { "PASS" } else { "FAIL" });
    assert!(pass, "encrypted FFT4 numerical validation failed");
}
