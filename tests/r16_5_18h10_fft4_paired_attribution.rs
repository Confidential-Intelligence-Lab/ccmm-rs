// R16.5.18h.4: complete Gaussian QP hybrid radix-4 FFT256.
// Research parameters only. Independent plaintext diagonal oracle.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    multiply_plain_rns_ckks_with_prepared_ntt, rescale_rns_ckks_to_next,
    research_profile_32768_for_levels, rotate_left_hybrid_rns_ckks_with_prepared_ntt,
    rotate_right_hybrid_rns_ckks_with_prepared_ntt, rotation_exponent_left,
    rotation_exponent_right, CkksCanonicalEmbedding, CkksChainState, PreparedHybridRnsGaloisKey,
    PreparedRnsPlaintextNtt, RnsCkksCiphertext, RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
};
use ccmm_rs::ring::{
    centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Modulus,
    ModulusBasis, ModulusChain, PreparedRnsNttPlan, RnsNttPlan,
};
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Diagonals = BTreeMap<isize, Vec<Complex64>>;
type HybridKeys = BTreeMap<isize, PreparedHybridRnsGaloisKey>;

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * exponent as f64 / n as f64)
}

fn stage_diagonals(slots: usize, span: usize) -> Diagonals {
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
            let coefficient = root(span, q * j) * root(4, p * q);

            diagonals
                .entry(offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); slots])[i] = coefficient;
        }
    }

    diagonals
}

fn plaintext_stage(input: &[Complex64], diagonals: &Diagonals) -> Vec<Complex64> {
    let slots = input.len();
    let mut output = vec![Complex64::new(0.0, 0.0); slots];

    for (&offset, diagonal) in diagonals {
        for i in 0..slots {
            let source = (i as isize + offset).rem_euclid(slots as isize) as usize;
            output[i] += diagonal[i] * input[source];
        }
    }

    output
}

fn error_metrics(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());

    let mut error_sq = 0.0;
    let mut reference_sq = 0.0;
    let mut max_abs = 0.0_f64;

    for (&a, &e) in actual.iter().zip(expected) {
        let delta = (a - e).norm();
        error_sq += delta * delta;
        reference_sq += e.norm_sqr();
        max_abs = max_abs.max(delta);
    }

    assert!(reference_sq > 0.0);
    ((error_sq / reference_sq).sqrt(), max_abs)
}

fn decode(
    ciphertext: &RnsCkksCiphertext,
    secret: &[i8],
    embedding: &CkksCanonicalEmbedding,
) -> Vec<Complex64> {
    let plan = RnsNttPlan::new(ciphertext.basis().moduli().to_vec(), embedding.degree());
    let decrypted = decrypt_rns_raw_with_ntt(ciphertext.rlwe(), secret, &plan);
    let modulus = composite_modulus_big(decrypted.basis());

    let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
        .into_iter()
        .map(|value| {
            centered_representative_big(&value, &modulus)
                .to_f64()
                .expect("coefficient conversion")
                / ciphertext.scale()
        })
        .collect();

    embedding.coefficients_to_slots(&coefficients)
}

#[test]
fn r16_5_18h10_fft4_paired_attribution() {
    const DEGREE: usize = 1024;
    const SCALE: f64 = 33_554_432.0;
    const SIGMA: f64 = 3.19;

    let validated = research_profile_32768_for_levels(13, 3);
    let validated_q = validated.modulus_chain().top().clone();

    let q = ModulusBasis::new(validated_q.moduli()[..5].to_vec());
    let p = ModulusBasis::new(vec![
        Modulus::new(1_125_899_908_022_273),
        Modulus::new(1_125_899_908_612_097),
    ]);

    for modulus in q.moduli().iter().chain(p.moduli()) {
        assert_eq!(modulus.value() % (2 * DEGREE as u64), 1);
    }

    let chain = ModulusChain::from_top_basis(q.clone());
    let embedding = CkksCanonicalEmbedding::new(DEGREE);
    let slots = embedding.slot_count();

    let q_plan = RnsNttPlan::new(q.moduli().to_vec(), DEGREE);

    assert_eq!(slots % 256, 0);

    let natural: Vec<Complex64> = (0..slots)
        .map(|i| {
            let x = i as f64;
            Complex64::new(0.001 * (0.013 * x).sin(), 0.001 * (0.017 * x).cos())
        })
        .collect();

    let reverse_base4_4 = |mut i: usize| {
        let mut reversed = 0;
        for _ in 0..4 {
            reversed = reversed * 4 + i % 4;
            i /= 4;
        }
        reversed
    };

    let packed: Vec<Complex64> = (0..slots)
        .map(|i| natural[(i / 256) * 256 + reverse_base4_4(i % 256)])
        .collect();

    let expected: Vec<Complex64> = (0..slots)
        .map(|i| {
            let block = (i / 256) * 256;
            let k = i % 256;
            (0..256)
                .map(|j| natural[block + j] * root(256, (k * j) % 256))
                .sum()
        })
        .collect();

    let spans = [4usize, 16, 64, 256];

    let diagonals: Vec<Diagonals> = spans
        .iter()
        .map(|&span| stage_diagonals(slots, span))
        .collect();

    for (&span, stage) in spans.iter().zip(&diagonals) {
        assert!(!stage.is_empty(), "empty radix-4 stage: {span}");
    }

    let mut stage_reference = packed.clone();
    let mut stage_references = Vec::new();

    for stage in &diagonals {
        stage_reference = plaintext_stage(&stage_reference, stage);
        stage_references.push(stage_reference.clone());
    }

    let (oracle_rel, oracle_max) = error_metrics(stage_references.last().unwrap(), &expected);

    println!("R16_5_18H5_PLAINTEXT_ORACLE_REL_L2={oracle_rel:.12e}");
    println!("R16_5_18H5_PLAINTEXT_ORACLE_MAX_ABS={oracle_max:.12e}");

    assert!(
        oracle_rel < 1.0e-10,
        "plaintext radix-4 schedule disagrees with DFT256"
    );

    let secret: Vec<i8> = (0..DEGREE)
        .map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x1803_000d);

    let plaintext = encode_rns(&packed, &embedding, &q, SCALE);
    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &q_plan,
        &mut rng,
    );

    let initial_ciphertext =
        RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, SCALE), &chain);

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    // G0: Initial CKKS encoding and Gaussian encryption.
    let decoded_input = decode(&initial_ciphertext, &secret, &embedding);

    let (g0_rel, g0_max) = error_metrics(&decoded_input, &packed);

    println!("R16_5_18H6_G0_INPUT_REL_L2={g0_rel:.12e}");
    println!("R16_5_18H6_G0_INPUT_MAX_ABS={g0_max:.12e}");

    // R16.5.18h.10: paired-seed P2/P3 FFT4 attribution.
    const SEEDS: [u64; 5] = [
        0x1810_0001,
        0x1810_0002,
        0x1810_0003,
        0x1810_0004,
        0x1810_0005,
    ];

    let p3_candidate = Modulus::new(1_125_899_908_630_529);

    assert_eq!(p3_candidate.value() % (2 * DEGREE as u64), 1);
    assert!(!q.moduli().contains(&p3_candidate));
    assert!(!p.moduli().contains(&p3_candidate));

    let stage_index = 0usize;
    let span = 4usize;
    assert_eq!(spans[stage_index], span);

    let first_diagonals = &diagonals[stage_index];
    let stage_q = chain.level(stage_index).clone();
    let stage_q_plan = RnsNttPlan::new(stage_q.moduli().to_vec(), DEGREE);
    let stage_prepared = PreparedRnsNttPlan::new(&stage_q_plan);

    let ideal_reference = &stage_references[0];
    let decrypted_reference = plaintext_stage(&decoded_input, first_diagonals);

    let mut p2_total_errors = Vec::new();
    let mut p3_total_errors = Vec::new();
    let mut p2_incremental_errors = Vec::new();
    let mut p3_incremental_errors = Vec::new();

    for (seed_index, seed) in SEEDS.iter().copied().enumerate() {
        for p_count in [2usize, 3usize] {
            let auxiliary_basis = if p_count == 2 {
                p.clone()
            } else {
                let mut primes = p.moduli().to_vec();
                primes.push(p3_candidate);
                ModulusBasis::new(primes)
            };

            let mut qp_moduli = stage_q.moduli().to_vec();
            qp_moduli.extend_from_slice(auxiliary_basis.moduli());

            let qp_plan = RnsNttPlan::new(qp_moduli, DEGREE);
            let mut key_rng = ChaCha20Rng::seed_from_u64(seed);

            let key_start = std::time::Instant::now();

            let keys: HybridKeys = first_diagonals
                .keys()
                .filter(|&&offset| offset != 0)
                .map(|&offset| {
                    let exponent = if offset > 0 {
                        rotation_exponent_left(DEGREE, offset as usize)
                    } else {
                        rotation_exponent_right(DEGREE, (-offset) as usize)
                    };

                    let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                        BoundedGadgetLayout::new(stage_q.clone(), 8),
                        auxiliary_basis.clone(),
                        &secret,
                        exponent,
                        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                        2,
                        &qp_plan,
                        &mut key_rng,
                    );

                    (offset, key)
                })
                .collect();

            let keygen_s = key_start.elapsed().as_secs_f64();

            assert_eq!(keys.len(), 6);

            let rotate = |offset: isize| {
                if offset > 0 {
                    rotate_left_hybrid_rns_ckks_with_prepared_ntt(
                        &initial_ciphertext,
                        offset as usize,
                        keys.get(&offset).expect("left key"),
                        &chain,
                        &qp_plan,
                    )
                } else {
                    rotate_right_hybrid_rns_ckks_with_prepared_ntt(
                        &initial_ciphertext,
                        (-offset) as usize,
                        keys.get(&offset).expect("right key"),
                        &chain,
                        &qp_plan,
                    )
                }
            };

            let mut rotation_errors = Vec::new();
            let mut rotation_max_abs = 0.0_f64;

            for &offset in keys.keys() {
                let rotated = rotate(offset);

                assert_eq!(rotated.level(), 0);
                assert_eq!(rotated.state(), initial_ciphertext.state());

                let actual = decode(&rotated, &secret, &embedding);

                let reference: Vec<Complex64> = (0..slots)
                    .map(|i| {
                        let source = (i as isize + offset).rem_euclid(slots as isize) as usize;
                        decoded_input[source]
                    })
                    .collect();

                let (rel, max_abs) = error_metrics(&actual, &reference);

                assert!(rel.is_finite());
                assert!(max_abs.is_finite());

                rotation_errors.push(rel);
                rotation_max_abs = rotation_max_abs.max(max_abs);

                println!(
                    "R16_5_18H10_SEED={} P={} OFFSET={} ROT_INCREMENTAL_REL_L2={:.12e} ROT_INCREMENTAL_MAX_ABS={:.12e}",
                    seed_index + 1,
                    p_count,
                    offset,
                    rel,
                    max_abs,
                );
            }

            let mean_rotation_error =
                rotation_errors.iter().sum::<f64>() / rotation_errors.len() as f64;

            let execution_start = std::time::Instant::now();

            let plain_scale = chain
                .dropped_modulus(initial_ciphertext.level())
                .expect("first rescale modulus")
                .value() as f64;

            let mut accumulator: Option<RnsCkksCiphertext> = None;

            for (&offset, diagonal) in first_diagonals {
                let rotated = if offset == 0 {
                    None
                } else {
                    Some(rotate(offset))
                };

                let source = rotated.as_ref().unwrap_or(&initial_ciphertext);
                let encoded = encode_rns(diagonal, &embedding, source.basis(), plain_scale);

                let prepared_plain = PreparedRnsPlaintextNtt::new(&encoded, &stage_prepared);

                let term = multiply_plain_rns_ckks_with_prepared_ntt(
                    source,
                    &prepared_plain,
                    plain_scale,
                    &chain,
                    &stage_prepared,
                );

                accumulator = Some(match accumulator {
                    Some(previous) => evaluator.add(&previous, &term),
                    None => term,
                });
            }

            let result =
                rescale_rns_ckks_to_next(&accumulator.expect("nonempty FFT4 stage"), &chain);

            let execution_s = execution_start.elapsed().as_secs_f64();

            assert_eq!(result.level(), 1);
            assert_eq!(result.basis(), chain.level(1));

            let actual = decode(&result, &secret, &embedding);

            let (total_rel, total_max) = error_metrics(&actual, ideal_reference);

            let (incremental_rel, incremental_max) = error_metrics(&actual, &decrypted_reference);

            for value in [
                total_rel,
                total_max,
                incremental_rel,
                incremental_max,
                mean_rotation_error,
                rotation_max_abs,
            ] {
                assert!(value.is_finite());
            }

            println!(
                "R16_5_18H10_SEED={} P={} MEAN_ROT_REL_L2={:.12e} MAX_ROT_ABS={:.12e} FFT4_TOTAL_REL_L2={:.12e} FFT4_TOTAL_MAX_ABS={:.12e} FFT4_INCREMENTAL_REL_L2={:.12e} FFT4_INCREMENTAL_MAX_ABS={:.12e} KEYGEN_S={:.6} EXECUTION_S={:.6}",
                seed_index + 1,
                p_count,
                mean_rotation_error,
                rotation_max_abs,
                total_rel,
                total_max,
                incremental_rel,
                incremental_max,
                keygen_s,
                execution_s,
            );

            if p_count == 2 {
                p2_total_errors.push(total_rel);
                p2_incremental_errors.push(incremental_rel);
            } else {
                p3_total_errors.push(total_rel);
                p3_incremental_errors.push(incremental_rel);
            }
        }
    }

    fn summarize(values: &[f64]) -> (f64, f64, f64, f64) {
        assert!(values.len() >= 2);

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let median = sorted[sorted.len() / 2];
        let worst = *sorted.last().expect("nonempty");

        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);

        (mean, median, variance.sqrt(), worst)
    }

    for (label, values) in [
        ("P2_TOTAL", &p2_total_errors),
        ("P3_TOTAL", &p3_total_errors),
        ("P2_INCREMENTAL", &p2_incremental_errors),
        ("P3_INCREMENTAL", &p3_incremental_errors),
    ] {
        let (mean, median, stddev, worst) = summarize(values);

        println!(
            "R16_5_18H10_SUMMARY={} N={} MEAN={:.12e} MEDIAN={:.12e} STDDEV={:.12e} WORST={:.12e}",
            label,
            values.len(),
            mean,
            median,
            stddev,
            worst,
        );
    }

    let p3_better_total = p2_total_errors
        .iter()
        .zip(&p3_total_errors)
        .filter(|(p2, p3)| p3 < p2)
        .count();

    let p3_better_incremental = p2_incremental_errors
        .iter()
        .zip(&p3_incremental_errors)
        .filter(|(p2, p3)| p3 < p2)
        .count();

    println!(
        "R16_5_18H10_P3_BETTER_TOTAL={}/{}",
        p3_better_total,
        SEEDS.len(),
    );

    println!(
        "R16_5_18H10_P3_BETTER_INCREMENTAL={}/{}",
        p3_better_incremental,
        SEEDS.len(),
    );

    println!("R16_5_18H10_FFT4_PAIRED_ATTRIBUTION=PASS");
}
