//! R16.6c.2: four-stage prepared FFT256 exact equivalence.
//! Reduced research parameters; no cryptographic security claim.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next,
    rotate_left_hybrid_rns_ckks_with_prepared_ntt, rotate_right_hybrid_rns_ckks_with_prepared_ntt,
    rotation_exponent_left, rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState,
    PreparedHybridRnsGaloisKey, PreparedRnsPlaintextNtt, RnsCkksCiphertext, RnsCkksEvaluationKeys,
    RnsCkksEvaluator,
};
use ccmm_rs::eblas::prepared_radix4_fft256::{
    apply_stage, apply_stage_canonical, PreparedFft256Stage, StageContext,
};
use ccmm_rs::grafting::{encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout};
use ccmm_rs::ring::{Modulus, ModulusBasis, ModulusChain, PreparedRnsNttPlan, RnsNttPlan};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::time::Instant;

type Diagonals = BTreeMap<isize, Vec<Complex64>>;
type RotationKeys = BTreeMap<isize, PreparedHybridRnsGaloisKey>;

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

// R16.6c.2: baseline hybrid FFT256 stage for exact equivalence.
fn apply_stage_baseline(
    input: &RnsCkksCiphertext,
    diagonals: &Diagonals,
    ctx: &StageContext<'_>,
    embedding: &CkksCanonicalEmbedding,
) -> RnsCkksCiphertext {
    let plain_scale = ctx
        .chain
        .dropped_modulus(input.level())
        .expect("rescale level available")
        .value() as f64;

    let mut accumulator: Option<RnsCkksCiphertext> = None;

    for (&offset, diagonal) in diagonals {
        let rotated = if offset > 0 {
            Some(rotate_left_hybrid_rns_ckks_with_prepared_ntt(
                input,
                offset as usize,
                ctx.keys.get(&offset).expect("left rotation key"),
                ctx.chain,
                ctx.qp_plan,
            ))
        } else if offset < 0 {
            Some(rotate_right_hybrid_rns_ckks_with_prepared_ntt(
                input,
                (-offset) as usize,
                ctx.keys.get(&offset).expect("right rotation key"),
                ctx.chain,
                ctx.qp_plan,
            ))
        } else {
            None
        };

        let source = rotated.as_ref().unwrap_or(input);
        let plain = encode_rns(diagonal, embedding, input.basis(), plain_scale);
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

#[test]
fn r16_6c2_production_fft256() {
    let degree = 1024usize;
    let scale = (1u64 << 25) as f64;

    let profile = ccmm_rs::ckks::research_profile_32768_for_levels(13, 3);
    let validated_q = profile.modulus_chain().top().clone();

    let q = ModulusBasis::new(validated_q.moduli()[..5].to_vec());

    let p = ModulusBasis::new(vec![
        Modulus::new(1_125_899_908_022_273),
        Modulus::new(1_125_899_908_612_097),
    ]);

    for modulus in q.moduli().iter().chain(p.moduli()) {
        assert_eq!(modulus.value() % (2 * degree as u64), 1);
    }

    let chain = ModulusChain::from_top_basis(q.clone());
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    assert_eq!(slots, 512);
    assert_eq!(slots % 256, 0);

    assert_eq!(chain.max_level(), 4);

    let stage_bases: Vec<ModulusBasis> = (0..4).map(|level| chain.level(level).clone()).collect();

    let stage_plans: Vec<RnsNttPlan> = stage_bases
        .iter()
        .map(|basis| RnsNttPlan::new(basis.moduli().to_vec(), degree))
        .collect();

    let prepared_plans: Vec<PreparedRnsNttPlan> =
        stage_plans.iter().map(PreparedRnsNttPlan::new).collect();

    let qp_plans: Vec<RnsNttPlan> = stage_bases
        .iter()
        .map(|basis| {
            let mut moduli = basis.moduli().to_vec();
            moduli.extend_from_slice(p.moduli());
            RnsNttPlan::new(moduli, degree)
        })
        .collect();

    let input: Vec<Complex64> = (0..slots)
        .map(|i| {
            Complex64::new(
                ((i % 17) as f64 - 8.0) / 128.0,
                ((i % 11) as f64 - 5.0) / 256.0,
            )
        })
        .collect();

    let plaintext = encode_rns(&input, &embedding, &q, scale);

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x166b_2001);

    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        &secret,
        &stage_plans[0],
        &mut rng,
    );

    let ciphertext = RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, scale), &chain);

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let spans = [4usize, 16, 64, 256];
    let mut current = ciphertext;

    let mut total_keygen_ms = 0.0;
    let mut total_preparation_ms = 0.0;
    let mut total_baseline_ms = 0.0;
    let mut total_optimized_ms = 0.0;

    for (stage_index, &span) in spans.iter().enumerate() {
        assert_eq!(current.level(), stage_index);

        let diagonals = stage_diagonals(slots, span, 0);
        let basis = &stage_bases[stage_index];

        let mut keys = RotationKeys::new();
        let keygen_started = Instant::now();

        for &offset in diagonals.keys().filter(|&&offset| offset != 0) {
            let exponent = if offset > 0 {
                rotation_exponent_left(degree, offset as usize)
            } else {
                rotation_exponent_right(degree, (-offset) as usize)
            };

            let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(basis.clone(), 8),
                p.clone(),
                &secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
                2,
                &qp_plans[stage_index],
                &mut rng,
            );

            keys.insert(offset, key);
        }

        // R16.7a.5: exact hybrid-QP rotation equivalence.
        // Reuse one canonical CRT reconstruction across stage rotations.
        // R16_7C3B3_PREPARED_ROTATION_BENCH
        let qp_prepare_started = Instant::now();
        let prepared_qp = ccmm_rs::ring::PreparedRnsNttPlan::new(&qp_plans[stage_index]);
        let qp_preparation_ms = qp_prepare_started.elapsed().as_secs_f64() * 1000.0;

        let mut canonical_samples = Vec::new();
        let mut prepared_samples = Vec::new();
        let mut prepared_equality_checks = 0usize;

        for (&offset, key) in &keys {
            let canonical = ccmm_rs::ckks::apply_hybrid_rns_galois_automorphism_with_prepared_ntt(
                current.rlwe(),
                key,
                &qp_plans[stage_index],
            );

            let prepared = ccmm_rs::ckks::apply_hybrid_rns_galois_automorphism_prepared_qp(
                current.rlwe(),
                key,
                &qp_plans[stage_index],
                &prepared_qp,
            );

            assert_eq!(
                canonical,
                prepared,
                "R16.7c.3b prepared-QP mismatch at stage {}, offset {}",
                stage_index + 1,
                offset
            );

            prepared_equality_checks += 1;

            for repeat in 0..5 {
                if repeat % 2 == 0 {
                    let started = Instant::now();
                    let result =
                        ccmm_rs::ckks::apply_hybrid_rns_galois_automorphism_with_prepared_ntt(
                            current.rlwe(),
                            key,
                            &qp_plans[stage_index],
                        );
                    canonical_samples.push(started.elapsed().as_secs_f64() * 1000.0);
                    std::hint::black_box(result);

                    let started = Instant::now();
                    let result = ccmm_rs::ckks::apply_hybrid_rns_galois_automorphism_prepared_qp(
                        current.rlwe(),
                        key,
                        &qp_plans[stage_index],
                        &prepared_qp,
                    );
                    prepared_samples.push(started.elapsed().as_secs_f64() * 1000.0);
                    std::hint::black_box(result);
                } else {
                    let started = Instant::now();
                    let result = ccmm_rs::ckks::apply_hybrid_rns_galois_automorphism_prepared_qp(
                        current.rlwe(),
                        key,
                        &qp_plans[stage_index],
                        &prepared_qp,
                    );
                    prepared_samples.push(started.elapsed().as_secs_f64() * 1000.0);
                    std::hint::black_box(result);

                    let started = Instant::now();
                    let result =
                        ccmm_rs::ckks::apply_hybrid_rns_galois_automorphism_with_prepared_ntt(
                            current.rlwe(),
                            key,
                            &qp_plans[stage_index],
                        );
                    canonical_samples.push(started.elapsed().as_secs_f64() * 1000.0);
                    std::hint::black_box(result);
                }
            }
        }

        assert_eq!(prepared_equality_checks, 6);
        assert_eq!(canonical_samples.len(), 30);
        assert_eq!(prepared_samples.len(), 30);

        canonical_samples.sort_by(f64::total_cmp);
        prepared_samples.sort_by(f64::total_cmp);

        let median = |samples: &[f64]| (samples[14] + samples[15]) / 2.0;

        let canonical_ms = median(&canonical_samples);
        let prepared_ms = median(&prepared_samples);
        let execution_speedup = canonical_ms / prepared_ms;

        let amortized_canonical_ms = 32.0 * canonical_ms;
        let amortized_prepared_ms = qp_preparation_ms + 32.0 * prepared_ms;
        let amortized_speedup = amortized_canonical_ms / amortized_prepared_ms;

        println!(
            "R16_7C3B3_STAGE_{}_QP_PREPARATION_MS={qp_preparation_ms:.3}",
            stage_index + 1
        );
        println!(
            "R16_7C3B3_STAGE_{}_CANONICAL_MEDIAN_MS={canonical_ms:.3}",
            stage_index + 1
        );
        println!(
            "R16_7C3B3_STAGE_{}_PREPARED_MEDIAN_MS={prepared_ms:.3}",
            stage_index + 1
        );
        println!(
            "R16_7C3B3_STAGE_{}_EXECUTION_SPEEDUP={execution_speedup:.3}",
            stage_index + 1
        );
        println!(
            "R16_7C3B3_STAGE_{}_AMORTIZED_32_SPEEDUP={amortized_speedup:.3}",
            stage_index + 1
        );
        println!(
            "R16_7C3B3_STAGE_{}_EXACT_RLWE_EQUALITY=PASS",
            stage_index + 1
        );

        // R16_7C2B_ROTATION_ATTRIBUTION
        // Reuse the current stage's ciphertext, keys and QP plan.
        // Key generation is excluded from these measurements.
        // R16_7C3A_SUBKERNEL_REPORT
        let mut subkernel_totals = [0.0_f64; 5];
        let mut profile_totals = [0.0_f64; 5];
        let mut profile_checks = 0usize;

        for (&offset, key) in &keys {
            let (profiled, timing) = ccmm_rs::ckks::profile_hybrid_rns_galois_rotation(
                current.rlwe(),
                key,
                &qp_plans[stage_index],
            );

            let reference = ccmm_rs::ckks::apply_hybrid_rns_galois_automorphism_with_prepared_ntt(
                current.rlwe(),
                key,
                &qp_plans[stage_index],
            );

            assert_eq!(
                profiled,
                reference,
                "R16.7c.2b profiled rotation mismatch: stage {}, offset {}",
                stage_index + 1,
                offset
            );

            let samples = [
                timing.automorphism_ms,
                timing.decomposition_ms,
                timing.base_term_ms,
                timing.key_switch_ntt_ms,
                timing.modulus_down_ms,
            ];

            for (total, sample) in profile_totals.iter_mut().zip(samples) {
                assert!(sample.is_finite() && sample >= 0.0);
                *total += sample;
            }

            let subkernels = [
                timing.ntt_initialization_ms,
                timing.digit_construction_ms,
                timing.digit_forward_ntt_ms,
                timing.pointwise_mac_ms,
                timing.inverse_ntt_ms,
            ];

            for (total, sample) in subkernel_totals.iter_mut().zip(subkernels) {
                assert!(sample.is_finite() && sample >= 0.0);
                *total += sample;
            }

            profile_checks += 1;
        }

        assert_eq!(profile_checks, 6);

        let total_profiled_ms: f64 = profile_totals.iter().sum();
        let names = [
            "AUTOMORPHISM",
            "DECOMPOSITION",
            "BASE_TERM",
            "KEY_SWITCH_NTT",
            "MODULUS_DOWN",
        ];

        println!(
            "R16_7C2B_STAGE_{}_ROTATIONS={profile_checks}",
            stage_index + 1
        );

        for (name, total) in names.iter().zip(profile_totals) {
            println!(
                "R16_7C2B_STAGE_{}_{}_TOTAL_MS={total:.3}",
                stage_index + 1,
                name
            );
            println!(
                "R16_7C2B_STAGE_{}_{}_SHARE_PCT={:.2}",
                stage_index + 1,
                name,
                if total_profiled_ms > 0.0 {
                    100.0 * total / total_profiled_ms
                } else {
                    0.0
                }
            );
        }

        println!(
            "R16_7C2B_STAGE_{}_PROFILED_TOTAL_MS={total_profiled_ms:.3}",
            stage_index + 1
        );
        let subkernel_names = [
            "INITIALIZATION",
            "DIGIT_CONSTRUCTION",
            "DIGIT_FORWARD_NTT",
            "POINTWISE_MAC",
            "INVERSE_NTT",
        ];

        let subkernel_total_ms: f64 = subkernel_totals.iter().sum();
        let key_switch_total_ms = profile_totals[3];

        for (name, total) in subkernel_names.iter().zip(subkernel_totals) {
            println!("R16_7C3A_STAGE_{}_{}_MS={total:.3}", stage_index + 1, name);
            println!(
                "R16_7C3A_STAGE_{}_{}_SHARE_PCT={:.2}",
                stage_index + 1,
                name,
                if key_switch_total_ms > 0.0 {
                    100.0 * total / key_switch_total_ms
                } else {
                    0.0
                }
            );
        }

        println!(
            "R16_7C3A_STAGE_{}_SUBKERNEL_TOTAL_MS={subkernel_total_ms:.3}",
            stage_index + 1
        );
        println!(
            "R16_7C3A_STAGE_{}_KEY_SWITCH_TOTAL_MS={key_switch_total_ms:.3}",
            stage_index + 1
        );
        println!(
            "R16_7C3A_STAGE_{}_UNATTRIBUTED_MS={:.3}",
            stage_index + 1,
            key_switch_total_ms - subkernel_total_ms
        );
        println!(
            "R16_7C3A_STAGE_{}_EXACT_RLWE_EQUALITY=PASS",
            stage_index + 1
        );

        println!(
            "R16_7C2B_STAGE_{}_EXACT_RLWE_EQUALITY=PASS",
            stage_index + 1
        );

        let shared_crt = ccmm_rs::ckks::PreparedHybridRotationInput::new(current.rlwe());

        let mut rotation_checks = 0usize;

        for (&offset, key) in &keys {
            let baseline_rotation =
                ccmm_rs::ckks::apply_hybrid_rns_galois_automorphism_with_prepared_ntt(
                    current.rlwe(),
                    key,
                    &qp_plans[stage_index],
                );

            let shared_rotation =
                ccmm_rs::ckks::apply_hybrid_rns_galois_automorphism_with_shared_crt(
                    &shared_crt,
                    key,
                    &qp_plans[stage_index],
                );

            let crt_reference = ccmm_rs::ckks::apply_hybrid_rns_galois_automorphism_crt_reference(
                current.rlwe(),
                key,
                &qp_plans[stage_index],
            );

            assert_eq!(
                baseline_rotation,
                crt_reference,
                "R16.7b.4 CRT/direct-RNS RLWE mismatch: stage {}, offset {}",
                stage_index + 1,
                offset
            );

            assert_eq!(
                baseline_rotation,
                shared_rotation,
                "R16.7a.5 RLWE mismatch: stage {}, offset {}",
                stage_index + 1,
                offset
            );

            rotation_checks += 1;
        }

        assert_eq!(rotation_checks, 6);

        println!(
            "R16_7A5_STAGE_{}_ROTATION_CHECKS={rotation_checks}",
            stage_index + 1
        );
        println!("R16_7A5_STAGE_{}_EXACT_RLWE_EQUALITY=PASS", stage_index + 1);

        println!(
            "R16_7B4_STAGE_{}_CRT_DIRECT_RNS_EXACT_EQUALITY=PASS",
            stage_index + 1
        );

        // R16_7B5_CRT_DIRECT_RNS_BENCHMARK
        // Paired measurements with identical ciphertexts, keys and QP plans.
        // Key generation and correctness checks are outside the timed region.
        let mut crt_samples_ms = Vec::new();
        let mut direct_samples_ms = Vec::new();
        const REPEATS: usize = 5;

        for (&offset, key) in &keys {
            for repeat in 0..REPEATS {
                let run_crt = || {
                    ccmm_rs::ckks::apply_hybrid_rns_galois_automorphism_crt_reference(
                        current.rlwe(),
                        key,
                        &qp_plans[stage_index],
                    )
                };
                let run_direct = || {
                    ccmm_rs::ckks::apply_hybrid_rns_galois_automorphism_with_prepared_ntt(
                        current.rlwe(),
                        key,
                        &qp_plans[stage_index],
                    )
                };

                let (crt, direct, crt_ms, direct_ms) = if repeat % 2 == 0 {
                    let start = Instant::now();
                    let crt = run_crt();
                    let crt_ms = start.elapsed().as_secs_f64() * 1000.0;

                    let start = Instant::now();
                    let direct = run_direct();
                    let direct_ms = start.elapsed().as_secs_f64() * 1000.0;

                    (crt, direct, crt_ms, direct_ms)
                } else {
                    let start = Instant::now();
                    let direct = run_direct();
                    let direct_ms = start.elapsed().as_secs_f64() * 1000.0;

                    let start = Instant::now();
                    let crt = run_crt();
                    let crt_ms = start.elapsed().as_secs_f64() * 1000.0;

                    (crt, direct, crt_ms, direct_ms)
                };

                assert_eq!(
                    crt,
                    direct,
                    "R16.7b.5 mismatch: stage {}, offset {}, repeat {}",
                    stage_index + 1,
                    offset,
                    repeat
                );

                crt_samples_ms.push(crt_ms);
                direct_samples_ms.push(direct_ms);
            }
        }

        assert_eq!(crt_samples_ms.len(), 6 * REPEATS);
        assert_eq!(direct_samples_ms.len(), 6 * REPEATS);

        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(|a, b| a.total_cmp(b));
            let middle = samples.len() / 2;
            (samples[middle - 1] + samples[middle]) / 2.0
        };

        let crt_median_ms = median(&mut crt_samples_ms);
        let direct_median_ms = median(&mut direct_samples_ms);

        println!(
            "R16_7B5_STAGE_{}_CRT_MEDIAN_MS={crt_median_ms:.3}",
            stage_index + 1
        );
        println!(
            "R16_7B5_STAGE_{}_DIRECT_RNS_MEDIAN_MS={direct_median_ms:.3}",
            stage_index + 1
        );
        println!(
            "R16_7B5_STAGE_{}_SPEEDUP={:.3}",
            stage_index + 1,
            crt_median_ms / direct_median_ms
        );
        println!("R16_7B5_STAGE_{}_EXACT_RLWE_EQUALITY=PASS", stage_index + 1);

        let keygen_ms = keygen_started.elapsed().as_secs_f64() * 1000.0;
        total_keygen_ms += keygen_ms;

        let ctx = StageContext {
            keys: &keys,
            evaluator: &evaluator,
            chain: &chain,
            qp_plan: &qp_plans[stage_index],
            prepared: &prepared_plans[stage_index],
        };

        let preparation_started = Instant::now();
        let prepared_stage = PreparedFft256Stage::new(
            stage_index,
            &diagonals,
            &embedding,
            &chain,
            &prepared_plans[stage_index],
        );
        let preparation_ms = preparation_started.elapsed().as_secs_f64() * 1000.0;
        total_preparation_ms += preparation_ms;

        // R16_7C3C2_PRODUCTION_FFT256_VALIDATION
        // Paired complete-stage execution, including prepared-QP construction.
        let canonical_started = Instant::now();
        let canonical_stage = apply_stage_canonical(&current, &prepared_stage, &ctx);
        let canonical_stage_ms = canonical_started.elapsed().as_secs_f64() * 1000.0;

        let prepared_started = Instant::now();
        let prepared_stage_output = apply_stage(&current, &prepared_stage, &ctx);
        let prepared_stage_ms = prepared_started.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(
            canonical_stage.rlwe(),
            prepared_stage_output.rlwe(),
            "prepared-QP FFT stage RLWE mismatch at stage {}",
            stage_index + 1
        );
        assert_eq!(
            canonical_stage.state(),
            prepared_stage_output.state(),
            "prepared-QP FFT stage state mismatch at stage {}",
            stage_index + 1
        );

        println!(
            "R16_7C3C2_STAGE_{}_CANONICAL_MS={canonical_stage_ms:.3}",
            stage_index + 1
        );
        println!(
            "R16_7C3C2_STAGE_{}_PREPARED_MS={prepared_stage_ms:.3}",
            stage_index + 1
        );
        println!(
            "R16_7C3C2_STAGE_{}_SPEEDUP={:.3}",
            stage_index + 1,
            canonical_stage_ms / prepared_stage_ms
        );
        println!(
            "R16_7C3C2_STAGE_{}_EXACT_RLWE_EQUALITY=PASS",
            stage_index + 1
        );

        let baseline_started = Instant::now();
        let baseline = apply_stage_baseline(&current, &diagonals, &ctx, &embedding);
        let baseline_ms = baseline_started.elapsed().as_secs_f64() * 1000.0;
        total_baseline_ms += baseline_ms;

        let optimized_started = Instant::now();
        let optimized = apply_stage(&current, &prepared_stage, &ctx);
        let optimized_ms = optimized_started.elapsed().as_secs_f64() * 1000.0;
        total_optimized_ms += optimized_ms;

        assert_eq!(
            baseline,
            optimized,
            "R16.6c.2 ciphertext mismatch at stage {}",
            stage_index + 1
        );
        assert_eq!(optimized.level(), stage_index + 1);
        assert_eq!(optimized.basis(), chain.level(stage_index + 1));

        println!("R16_6C2_STAGE_{}_SPAN={span}", stage_index + 1);
        println!(
            "R16_6C2_STAGE_{}_DIAGONALS={}",
            stage_index + 1,
            diagonals.len()
        );
        println!("R16_6C2_STAGE_{}_KEYGEN_MS={keygen_ms:.3}", stage_index + 1);
        println!(
            "R16_6C2_STAGE_{}_PREPARATION_MS={preparation_ms:.3}",
            stage_index + 1
        );
        println!(
            "R16_6C2_STAGE_{}_BASELINE_MS={baseline_ms:.3}",
            stage_index + 1
        );
        println!(
            "R16_6C2_STAGE_{}_OPTIMIZED_MS={optimized_ms:.3}",
            stage_index + 1
        );
        println!("R16_6C2_STAGE_{}_EXACT_EQUALITY=PASS", stage_index + 1);

        current = optimized;
    }

    println!("R16_6C2_DEGREE={degree}");
    println!("R16_6C2_SLOTS={slots}");
    println!("R16_6C2_TOTAL_KEYGEN_MS={total_keygen_ms:.3}");
    println!("R16_6C2_TOTAL_PREPARATION_MS={total_preparation_ms:.3}");
    println!("R16_6C2_TOTAL_BASELINE_MS={total_baseline_ms:.3}");
    println!("R16_6C2_TOTAL_OPTIMIZED_MS={total_optimized_ms:.3}");
    println!("R16_6C2_FINAL_LEVEL={}", current.level());
    println!("R16_6C2_EXACT_EQUIVALENCE_STATUS=PASS");
}
