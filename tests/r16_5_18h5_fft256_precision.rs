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
fn r16_5_18h5_fft256_precision_characterization() {
    let hybrid_base_log: u32 = std::env::var("FHE_HYBRID_GADGET_BASE_LOG")
        .ok()
        .map(|value| value.parse().expect("invalid hybrid gadget base"))
        .unwrap_or(8);
    assert!([8, 16, 32].contains(&hybrid_base_log));
    println!("R30_9A_15_HYBRID_BASE_LOG={hybrid_base_log}");

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

    for (diagnostic_name, key_distribution) in [
        (
            "GAUSSIAN",
            ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        ),
        ("ZERO_NOISE", ErrorDistribution::BoundedUniform { bound: 0 }),
    ] {
        let mut current = initial_ciphertext.clone();

        for (stage_index, &span) in spans.iter().enumerate() {
            assert_eq!(current.level(), stage_index);

            let stage_q = chain.level(stage_index).clone();

            for modulus in stage_q.moduli() {
                assert!(
                    !p.moduli().contains(modulus),
                    "Q/P modulus overlap at stage {stage_index}"
                );
            }

            let stage_q_plan = RnsNttPlan::new(stage_q.moduli().to_vec(), DEGREE);

            let stage_prepared = PreparedRnsNttPlan::new(&stage_q_plan);

            let mut stage_qp_moduli = stage_q.moduli().to_vec();
            stage_qp_moduli.extend_from_slice(p.moduli());

            let stage_qp_plan = RnsNttPlan::new(stage_qp_moduli, DEGREE);

            let stage_diagonals = &diagonals[stage_index];

            let keys: HybridKeys = stage_diagonals
                .keys()
                .filter(|&&offset| offset != 0)
                .map(|&offset| {
                    let exponent = if offset > 0 {
                        rotation_exponent_left(DEGREE, offset as usize)
                    } else {
                        rotation_exponent_right(DEGREE, (-offset) as usize)
                    };

                    let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                        BoundedGadgetLayout::new(stage_q.clone(), hybrid_base_log),
                        p.clone(),
                        &secret,
                        exponent,
                        key_distribution,
                        2,
                        &stage_qp_plan,
                        &mut rng,
                    );

                    (offset, key)
                })
                .collect();

            let plain_scale = chain
                .dropped_modulus(current.level())
                .expect("rescale level")
                .value() as f64;

            let mut accumulator: Option<RnsCkksCiphertext> = None;

            for (&offset, diagonal) in stage_diagonals {
                let rotated = if offset > 0 {
                    Some(rotate_left_hybrid_rns_ckks_with_prepared_ntt(
                        &current,
                        offset as usize,
                        keys.get(&offset).expect("left hybrid key"),
                        &chain,
                        &stage_qp_plan,
                    ))
                } else if offset < 0 {
                    Some(rotate_right_hybrid_rns_ckks_with_prepared_ntt(
                        &current,
                        (-offset) as usize,
                        keys.get(&offset).expect("right hybrid key"),
                        &chain,
                        &stage_qp_plan,
                    ))
                } else {
                    None
                };

                let source = rotated.as_ref().unwrap_or(&current);
                assert_eq!(source.state(), current.state());

                let encoded = encode_rns(diagonal, &embedding, current.basis(), plain_scale);

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

            current =
                rescale_rns_ckks_to_next(&accumulator.expect("nonempty radix-4 stage"), &chain);

            assert_eq!(current.level(), stage_index + 1);
            assert_eq!(current.basis(), chain.level(stage_index + 1));

            let decoded_stage = decode(&current, &secret, &embedding);
            let (stage_rel, stage_max) =
                error_metrics(&decoded_stage, &stage_references[stage_index]);

            println!(
            "R16_5_18H5_MODE={diagnostic_name} STAGE={} SPAN={} REL_L2={stage_rel:.12e} MAX_ABS={stage_max:.12e}",
            stage_index + 1,
            span,
        );

            println!(
                "R16_5_18H4_STAGE={} SPAN={} KEYS={} LEVEL={} SCALE={:.12e}",
                stage_index + 1,
                span,
                keys.len(),
                current.level(),
                current.scale(),
            );
        }

        let actual = decode(&current, &secret, &embedding);
        let (relative, max_abs) = error_metrics(&actual, &expected);

        println!(
            "R16_5_18H5_MODE={diagnostic_name} FINAL_REL_L2={relative:.12e} FINAL_MAX_ABS={max_abs:.12e}"
        );

        assert!(relative.is_finite() && max_abs.is_finite());

        if diagnostic_name == "GAUSSIAN" {
            assert!(
                relative < 1.0e-2,
                "Gaussian FFT256 exceeded the existing h.4 tolerance"
            );
        }
    }

    println!("R16_5_18H5_PRECISION_CHARACTERIZATION=PASS");
}
