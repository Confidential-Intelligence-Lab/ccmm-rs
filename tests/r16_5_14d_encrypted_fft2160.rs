// R16.5.14d: encrypted FFT2160 = FFT135 DIF + fused-twiddle FFT16 DIT.
// 135 ciphertexts, each carrying 16 local rows in CKKS slots 0..15.
// Research-only deterministic keys. No intermediate decryption.
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
use ccmm_rs::eblas::level1::{multiply_complex_slots_cp_prepared, PreparedComplexSlotsCp};
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

const N: usize = 135;
const RADICES: [usize; 4] = [5, 3, 3, 3];
const SIGMA: f64 = 3.19;
fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (exponent % n) as f64 / n as f64)
}

type Diagonals = BTreeMap<isize, Vec<Complex64>>;
type RotationKeys = BTreeMap<isize, PreparedBoundedRnsGaloisKey>;
fn reverse_base4_2(i: usize) -> usize {
    (i % 4) * 4 + i / 4
}

// Radix-4 DIT: span 4 then span 16. Signed offsets prevent FFT16 block crossings.
fn stage_diagonals(slots: usize, span: usize, k_cross: usize) -> Diagonals {
    assert!(span == 4 || span == 16);
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
                // Twiddle refers to the natural local-row index before DIT permutation.
                let packed_input = (i as isize + offset) as usize;
                let natural_local = reverse_base4_2(packed_input % 16);
                coefficient *= root(2160, k_cross * natural_local);
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

// Matches the mixed-radix DIF permutation validated in R16.5.14c.1.
fn natural_index(index: usize) -> usize {
    let mut remaining = index;
    let mut digits = [0usize; 4];
    for (stage, &radix) in RADICES.iter().rev().enumerate() {
        digits[3 - stage] = remaining % radix;
        remaining /= radix;
    }
    assert_eq!(remaining, 0);
    digits[0] + RADICES[0] * (digits[1] + RADICES[1] * (digits[2] + RADICES[2] * digits[3]))
}

#[test]
#[ignore = "135 ciphertexts and rotation keys; run explicitly"]
fn r16_5_14d_encrypted_vertical_fft2160_six_levels() {
    let started = Instant::now();
    let profile = research_profile_16384();
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    assert!(chain.max_level() >= 6);
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();
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
    let mut rng = ChaCha20Rng::seed_from_u64(0x1651_4C02);
    let sample: Vec<Complex64> = (0..2160)
        .map(|t| {
            let x = t as f64;
            Complex64::new(0.0001 * (0.17 * x).sin(), 0.0001 * (0.11 * x).cos())
        })
        .collect();
    let encrypt_started = Instant::now();
    let mut ciphertexts: Vec<RnsCkksCiphertext> = (0..N)
        .map(|index| {
            let mut values = vec![Complex64::new(0.0, 0.0); slots];
            let tile_row = index;
            for (packed_local, value) in values.iter_mut().enumerate().take(16) {
                let local_row = reverse_base4_2(packed_local);
                *value = sample[tile_row * 16 + local_row];
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
    println!(
        "R16_FFT135_ENCRYPT_MS={:.3}",
        encrypt_started.elapsed().as_secs_f64() * 1000.0
    );
    let keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &keys);
    let mut span = N;
    for (stage, &radix) in RADICES.iter().enumerate() {
        let stage_started = Instant::now();
        let m = span / radix;
        let level = ciphertexts[0].level();
        assert_eq!(level, stage);
        let plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), degree);
        let prepared_plan = PreparedRnsNttPlan::new(&plan);
        let mut next: Vec<Option<RnsCkksCiphertext>> = vec![None; N];
        for base in (0..N).step_by(span) {
            for j in 0..m {
                for k in 0..radix {
                    let mut output: Option<RnsCkksCiphertext> = None;
                    for q in 0..radix {
                        let input = &ciphertexts[base + j + q * m];
                        let coefficient = root(radix, q * k) * root(span, j * k);
                        let term = if (coefficient - Complex64::new(1.0, 0.0)).norm() < 1e-14 {
                            mod_switch_rns_ckks_to_next(input, &chain)
                        } else {
                            let coefficients = vec![coefficient; slots];
                            let prepared = PreparedComplexSlotsCp::new(
                                input,
                                &coefficients,
                                &embedding,
                                &chain,
                                &plan,
                            );
                            multiply_complex_slots_cp_prepared(
                                input,
                                &prepared,
                                &chain,
                                &prepared_plan,
                            )
                        };
                        output = Some(match output {
                            Some(acc) => evaluator.add(&acc, &term),
                            None => term,
                        });
                    }
                    let destination = base + j + k * m;
                    assert!(next[destination].is_none());
                    next[destination] = output;
                }
            }
        }
        ciphertexts = next
            .into_iter()
            .map(|ct| ct.expect("missing butterfly output"))
            .collect();
        span = m;
        for ct in &ciphertexts {
            assert_eq!(ct.level(), stage + 1);
            assert_eq!(ct.scale(), scale);
        }
        println!("R16_FFT135_STAGE_{}_RADIX={radix}", stage + 1);
        println!("R16_FFT135_STAGE_{}_LEVEL={}", stage + 1, stage + 1);
        println!(
            "R16_FFT135_STAGE_{}_MS={:.3}",
            stage + 1,
            stage_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    assert_eq!(span, 1);
    // FFT135 output remains in mixed-radix digit-reversed ciphertext order.
    // Twiddle-fused radix-4 DIT FFT16 acts on the 16 packed local rows.
    let evaluation_started = Instant::now();
    let mut keygen_ms = 0.0;
    for (stage, span16) in [4usize, 16].into_iter().enumerate() {
        let level = 4 + stage;
        let basis = chain.level(level).clone();
        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
        let prepared_plan = PreparedRnsNttPlan::new(&plan);
        // Different k_cross values change coefficients, not rotation offsets.
        let representative = stage_diagonals(slots, span16, 0);
        let keygen_started = Instant::now();
        let rotation_keys =
            generate_keys(&representative, &basis, &plan, degree, &secret, &mut rng);
        keygen_ms += keygen_started.elapsed().as_secs_f64() * 1000.0;
        let stage_started = Instant::now();
        let mut next = Vec::with_capacity(N);
        for (index, ct) in ciphertexts.iter().enumerate() {
            let k_cross = natural_index(index);
            let custom_diagonals = (stage == 0).then(|| stage_diagonals(slots, span16, k_cross));
            let diagonals = custom_diagonals.as_ref().unwrap_or(&representative);
            let ctx = StageContext {
                keys: &rotation_keys,
                evaluator: &evaluator,
                embedding: &embedding,
                chain: &chain,
                plan: &plan,
                prepared: &prepared_plan,
            };
            let result = apply_stage(ct, diagonals, &ctx);
            assert_eq!(result.level(), level + 1);
            assert_eq!(result.scale(), scale);
            next.push(result);
        }
        ciphertexts = next;
        println!("R16_VERTICAL_FFT16_STAGE_{}_LEVEL={}", stage + 1, level + 1);
        println!(
            "R16_VERTICAL_FFT16_STAGE_{}_MS={:.3}",
            stage + 1,
            stage_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    println!("R16_VERTICAL_FFT16_KEYGEN_MS={keygen_ms:.3}");
    println!(
        "R16_VERTICAL_FFT16_EVAL_MS={:.3}",
        evaluation_started.elapsed().as_secs_f64() * 1000.0
    );
    // Independent direct DFT2160 oracle, no intermediate ciphertext decryption.
    let reference_started = Instant::now();
    let reference: Vec<Complex64> = (0..2160)
        .map(|k| (0..2160).map(|t| sample[t] * root(2160, k * t)).sum())
        .collect();
    println!(
        "R16_VERTICAL_REFERENCE_MS={:.3}",
        reference_started.elapsed().as_secs_f64() * 1000.0
    );
    let validation_started = Instant::now();
    let mut error_sq = 0.0;
    let mut reference_sq = 0.0;
    let mut max_abs = 0.0_f64;
    let mut seen = vec![false; 2160];
    for (index, ct) in ciphertexts.iter().enumerate() {
        let k_cross = natural_index(index);
        let actual = decode(ct, &secret, &embedding);
        #[allow(clippy::needless_range_loop)]
        for k_local in 0..16 {
            let frequency = k_cross + 135 * k_local;
            assert!(!seen[frequency]);
            seen[frequency] = true;
            let expected = reference[frequency];
            let delta = (actual[k_local] - expected).norm();
            error_sq += delta * delta;
            reference_sq += expected.norm_sqr();
            max_abs = max_abs.max(delta);
        }
    }
    assert!(seen.iter().all(|&v| v));
    let rel_l2 = (error_sq / reference_sq).sqrt();
    println!("R16_VERTICAL_FINAL_LEVEL={}", ciphertexts[0].level());
    println!("R16_VERTICAL_REL_L2={rel_l2:.12e}");
    println!("R16_VERTICAL_MAX_ABS={max_abs:.12e}");
    println!(
        "R16_VERTICAL_VALIDATION_MS={:.3}",
        validation_started.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "R16_VERTICAL_TOTAL_MS={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    let pass = ciphertexts[0].level() == 6
        && rel_l2.is_finite()
        && max_abs.is_finite()
        && rel_l2 < 1e-2
        && max_abs < 1e-3;
    println!(
        "R16_VERTICAL_ENCRYPTED_STATUS={}",
        if pass { "PASS" } else { "FAIL" }
    );
    assert!(
        pass,
        "encrypted FFT2160 accuracy outside research thresholds"
    );
}
