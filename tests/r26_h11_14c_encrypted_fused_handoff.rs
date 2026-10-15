use ccmm_rs::eblas::prepared_radix4_fft256::{
    apply_stage, pack_fft256_branch_diagonals, PreparedFft256Stage,
};
// R26_H11_14C_ENCRYPTED_FUSED_HANDOFF
// Research-only parameters. Expensive key generation: run with --ignored.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_65536_for_levels, rotation_exponent_left, rotation_exponent_right,
    CkksCanonicalEmbedding, CkksChainState, PreparedHybridRnsGaloisKey, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::fused_fft256_handoff::{
    apply_fft3840_packed_fft256_from_fft15, apply_fused_fft256_sources,
    apply_fused_source_diagonals_plain, fuse_fft256_source_lane, PreparedFusedFft256Source,
};
use ccmm_rs::eblas::prepared_radix4_fft256::{Diagonals, RotationKeys, StageContext};
use ccmm_rs::grafting::{
    decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
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
use std::collections::BTreeSet;
use std::time::Instant;

const DEGREE: usize = 65536;
const SLOTS: usize = DEGREE / 2;
const LANE_SLOTS: usize = 4096;
const SIGMA: f64 = 3.19;

fn decrypt_slots(
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

struct EncryptionContext<'a> {
    embedding: &'a CkksCanonicalEmbedding,
    basis: &'a ModulusBasis,
    plan: &'a RnsNttPlan,
    chain: &'a ModulusChain,
    secret: &'a [i8],
    scale: f64,
}

fn encrypt_slots(
    values: &[Complex64],
    ctx: &EncryptionContext<'_>,
    rng: &mut ChaCha20Rng,
) -> RnsCkksCiphertext {
    let embedding = ctx.embedding;
    let basis = ctx.basis;
    let plan = ctx.plan;
    let chain = ctx.chain;
    let secret = ctx.secret;
    let scale = ctx.scale;

    let plaintext = encode_rns(values, embedding, basis, scale);

    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        secret,
        plan,
        rng,
    );

    RnsCkksCiphertext::new(encrypted, CkksChainState::new(chain, 3, scale), chain)
}

#[test]
#[ignore = "65K-ring hybrid key generation is expensive"]
fn r26_h11_14c_encrypted_fused_handoff() {
    let started = Instant::now();

    let profile = research_profile_65536_for_levels(2, 5);
    let degree = profile.degree();

    assert_eq!(degree, DEGREE);

    let validated_q = profile.modulus_chain().top().clone();
    let q = ModulusBasis::new(validated_q.moduli()[..5].to_vec());
    let chain = ModulusChain::from_top_basis(q);
    let scale = profile.initial_scale();

    assert_eq!(chain.max_level(), 4);

    let embedding = CkksCanonicalEmbedding::new(degree);
    assert_eq!(embedding.slot_count(), SLOTS);

    let p = ModulusBasis::new(validated_q.moduli()[5..7].to_vec());

    let basis = chain.level(3).clone();

    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let prepared_plan = PreparedRnsNttPlan::new(&plan);

    let mut qp_moduli = basis.moduli().to_vec();
    qp_moduli.extend_from_slice(p.moduli());

    let qp_plan = RnsNttPlan::new(qp_moduli, degree);

    // Identity diagonals restricted to one 4096-slot source lane.
    let mut identity = Diagonals::new();
    identity.insert(0, vec![Complex64::new(1.0, 0.0); LANE_SLOTS]);

    let fused_a = fuse_fft256_source_lane(&identity, SLOTS, LANE_SLOTS, 0, 0);
    let fused_b = fuse_fft256_source_lane(&identity, SLOTS, LANE_SLOTS, 0, 1);

    let prepared_a =
        PreparedFusedFft256Source::new(3, &fused_a, &embedding, &chain, &prepared_plan);
    let prepared_b =
        PreparedFusedFft256Source::new(3, &fused_b, &embedding, &chain, &prepared_plan);

    let source_a: Vec<Complex64> = (0..SLOTS)
        .map(|index| {
            Complex64::new(
                ((index % 19) as f64 - 9.0) / 4096.0,
                ((index % 7) as f64 - 3.0) / 8192.0,
            )
        })
        .collect();

    let source_b: Vec<Complex64> = (0..SLOTS)
        .map(|index| {
            Complex64::new(
                ((index % 13) as f64 - 6.0) / 4096.0,
                ((index % 11) as f64 - 5.0) / 8192.0,
            )
        })
        .collect();

    let expected_a = apply_fused_source_diagonals_plain(&source_a, &fused_a);
    let expected_b = apply_fused_source_diagonals_plain(&source_b, &fused_b);

    let expected: Vec<Complex64> = expected_a
        .iter()
        .zip(&expected_b)
        .map(|(a, b)| *a + *b)
        .collect();

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x5232_4841_4E44);

    let encryption_started = Instant::now();

    let encryption_ctx = EncryptionContext {
        embedding: &embedding,
        basis: &basis,
        plan: &plan,
        chain: &chain,
        secret: &secret,
        scale,
    };

    let encrypted_a = encrypt_slots(&source_a, &encryption_ctx, &mut rng);
    let encrypted_b = encrypt_slots(&source_b, &encryption_ctx, &mut rng);

    println!(
        "R26_H11_14C_ENCRYPTION_MS={:.3}",
        encryption_started.elapsed().as_secs_f64() * 1000.0
    );

    assert_eq!(encrypted_a.level(), 3);
    assert_eq!(encrypted_b.level(), 3);

    let offsets: BTreeSet<isize> = prepared_a
        .rotation_offsets()
        .chain(prepared_b.rotation_offsets())
        .filter(|&offset| offset != 0)
        .collect();

    let keygen_started = Instant::now();
    let mut keys = RotationKeys::new();

    for offset in offsets {
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
            ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
            2,
            &qp_plan,
            &mut rng,
        );

        keys.insert(offset, key);
    }

    println!("R26_H11_14C_ROTATION_KEYS={}", keys.len());
    println!(
        "R26_H11_14C_KEYGEN_MS={:.3}",
        keygen_started.elapsed().as_secs_f64() * 1000.0
    );

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let ctx = StageContext {
        keys: &keys,
        evaluator: &evaluator,
        chain: &chain,
        qp_plan: &qp_plan,
        prepared: &prepared_plan,
    };

    let evaluation_started = Instant::now();

    let output = apply_fused_fft256_sources(
        &[(&encrypted_a, &prepared_a), (&encrypted_b, &prepared_b)],
        &ctx,
    );

    println!(
        "R26_H11_14C_EVALUATION_MS={:.3}",
        evaluation_started.elapsed().as_secs_f64() * 1000.0
    );

    assert_eq!(output.level(), 4);

    let actual = decrypt_slots(&output, &secret, &embedding);

    let mut error_sq = 0.0_f64;
    let mut reference_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut inactive_max = 0.0_f64;

    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        let error = (got - want).norm();
        error_sq += error * error;
        reference_sq += want.norm_sqr();
        max_abs = max_abs.max(error);

        if index >= 2 * LANE_SLOTS {
            inactive_max = inactive_max.max(got.norm());
        }
    }

    let relative_l2 = (error_sq / reference_sq).sqrt();

    println!("R26_H11_14C_RELATIVE_L2={relative_l2:.12e}");
    println!("R26_H11_14C_MAX_ABS_ERROR={max_abs:.12e}");
    println!("R26_H11_14C_INACTIVE_MAX={inactive_max:.12e}");
    println!("R26_H11_14C_INPUT_LEVEL=3");
    println!("R26_H11_14C_OUTPUT_LEVEL={}", output.level());
    println!("R26_H11_14C_SOURCE_CIPHERTEXTS=2");
    println!(
        "R26_H11_14C_TOTAL_MS={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );

    assert!(
        relative_l2 < 5.0e-3,
        "fused encrypted handoff exceeds relative error bound"
    );

    assert!(
        inactive_max < 5.0e-3,
        "inactive output lanes are not sufficiently suppressed"
    );

    println!("R26_H11_14C_ENCRYPTED_STATUS=PASS");
}

// R26_H11_14D_REAL_FFT256_STAGE

fn fft_root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(
        1.0,
        -2.0 * std::f64::consts::PI * (exponent % n) as f64 / n as f64,
    )
}

fn reverse_base4_4(mut index: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..4 {
        reversed = reversed * 4 + index % 4;
        index /= 4;
    }
    reversed
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

// Independent mathematical oracle, not a diagonal evaluation.
fn first_stage_reference(input: &[Complex64], branch: usize) -> Vec<Complex64> {
    assert_eq!(input.len(), LANE_SLOTS);

    let mut output = vec![Complex64::new(0.0, 0.0); LANE_SLOTS];

    for group in 0..LANE_SLOTS / 4 {
        for p in 0..4 {
            let mut sum = Complex64::new(0.0, 0.0);

            for q in 0..4 {
                let index = 4 * group + q;

                sum += input[index]
                    * fft_root(3840, branch * reverse_base4_4(index % 256))
                    * fft_root(4, p * q);
            }

            output[4 * group + p] = sum;
        }
    }

    output
}

#[test]
#[ignore = "65K-ring hybrid key generation is expensive"]
fn r26_h11_14d_encrypted_fused_fft256_first_stage() {
    const BRANCH_A: usize = 1;
    const BRANCH_B: usize = 3;

    let started = Instant::now();

    let profile = research_profile_65536_for_levels(2, 5);
    let degree = profile.degree();
    assert_eq!(degree, DEGREE);

    let validated_q = profile.modulus_chain().top().clone();
    let q = ModulusBasis::new(validated_q.moduli()[..5].to_vec());
    let chain = ModulusChain::from_top_basis(q);
    let scale = profile.initial_scale();

    assert_eq!(chain.max_level(), 4);

    let embedding = CkksCanonicalEmbedding::new(degree);
    let p = ModulusBasis::new(validated_q.moduli()[5..7].to_vec());

    let basis = chain.level(3).clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);
    let prepared_plan = PreparedRnsNttPlan::new(&plan);

    let mut qp_moduli = basis.moduli().to_vec();
    qp_moduli.extend_from_slice(p.moduli());
    let qp_plan = RnsNttPlan::new(qp_moduli, degree);

    let diagonals_a = first_stage_diagonals(BRANCH_A);
    let diagonals_b = first_stage_diagonals(BRANCH_B);

    let fused_a = fuse_fft256_source_lane(&diagonals_a, SLOTS, LANE_SLOTS, 0, 0);
    let fused_b = fuse_fft256_source_lane(&diagonals_b, SLOTS, LANE_SLOTS, 0, 1);

    let source_a: Vec<Complex64> = (0..SLOTS)
        .map(|index| {
            Complex64::new(
                ((index % 19) as f64 - 9.0) / 4096.0,
                ((index % 7) as f64 - 3.0) / 8192.0,
            )
        })
        .collect();

    let source_b: Vec<Complex64> = (0..SLOTS)
        .map(|index| {
            Complex64::new(
                ((index % 13) as f64 - 6.0) / 4096.0,
                ((index % 11) as f64 - 5.0) / 8192.0,
            )
        })
        .collect();

    let mut expected = vec![Complex64::new(0.0, 0.0); SLOTS];

    let reference_a = first_stage_reference(&source_a[..LANE_SLOTS], BRANCH_A);
    let reference_b = first_stage_reference(&source_b[..LANE_SLOTS], BRANCH_B);

    expected[..LANE_SLOTS].copy_from_slice(&reference_a);
    expected[LANE_SLOTS..2 * LANE_SLOTS].copy_from_slice(&reference_b);

    let oracle_a = apply_fused_source_diagonals_plain(&source_a, &fused_a);
    let oracle_b = apply_fused_source_diagonals_plain(&source_b, &fused_b);

    let mut oracle_max = 0.0_f64;

    for i in 0..SLOTS {
        oracle_max = oracle_max.max((oracle_a[i] + oracle_b[i] - expected[i]).norm());
    }

    println!("R26_H11_14D_PLAINTEXT_ORACLE_MAX={oracle_max:.12e}");

    assert!(
        oracle_max < 1.0e-12,
        "fused diagonals disagree with independent butterfly"
    );

    let prepared_a =
        PreparedFusedFft256Source::new(3, &fused_a, &embedding, &chain, &prepared_plan);
    let prepared_b =
        PreparedFusedFft256Source::new(3, &fused_b, &embedding, &chain, &prepared_plan);

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x5232_4841_4E45);

    let encryption_ctx = EncryptionContext {
        embedding: &embedding,
        basis: &basis,
        plan: &plan,
        chain: &chain,
        secret: &secret,
        scale,
    };

    let encryption_started = Instant::now();

    let encrypted_a = encrypt_slots(&source_a, &encryption_ctx, &mut rng);
    let encrypted_b = encrypt_slots(&source_b, &encryption_ctx, &mut rng);

    println!(
        "R26_H11_14D_ENCRYPTION_MS={:.3}",
        encryption_started.elapsed().as_secs_f64() * 1000.0
    );

    assert_eq!(encrypted_a.level(), 3);
    assert_eq!(encrypted_b.level(), 3);

    let offsets: BTreeSet<isize> = prepared_a
        .rotation_offsets()
        .chain(prepared_b.rotation_offsets())
        .filter(|&offset| offset != 0)
        .collect();

    println!("R26_H11_14D_REQUIRED_ROTATIONS={}", offsets.len());

    let keygen_started = Instant::now();
    let mut keys = RotationKeys::new();

    for offset in offsets {
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
            ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
            2,
            &qp_plan,
            &mut rng,
        );

        keys.insert(offset, key);
    }

    println!(
        "R26_H11_14D_KEYGEN_MS={:.3}",
        keygen_started.elapsed().as_secs_f64() * 1000.0
    );
    println!("R26_H11_14D_ROTATION_KEYS={}", keys.len());

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let ctx = StageContext {
        keys: &keys,
        evaluator: &evaluator,
        chain: &chain,
        qp_plan: &qp_plan,
        prepared: &prepared_plan,
    };

    let evaluation_started = Instant::now();

    let output = apply_fused_fft256_sources(
        &[(&encrypted_a, &prepared_a), (&encrypted_b, &prepared_b)],
        &ctx,
    );

    println!(
        "R26_H11_14D_EVALUATION_MS={:.3}",
        evaluation_started.elapsed().as_secs_f64() * 1000.0
    );

    assert_eq!(output.level(), 4);

    let actual = decrypt_slots(&output, &secret, &embedding);

    let mut error_sq = 0.0_f64;
    let mut reference_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut inactive_max = 0.0_f64;

    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        let error = (got - want).norm();

        error_sq += error * error;
        reference_sq += want.norm_sqr();
        max_abs = max_abs.max(error);

        if index >= 2 * LANE_SLOTS {
            inactive_max = inactive_max.max(got.norm());
        }
    }

    let relative_l2 = (error_sq / reference_sq).sqrt();

    println!("R26_H11_14D_RELATIVE_L2={relative_l2:.12e}");
    println!("R26_H11_14D_MAX_ABS_ERROR={max_abs:.12e}");
    println!("R26_H11_14D_INACTIVE_MAX={inactive_max:.12e}");
    println!("R26_H11_14D_INPUT_LEVEL=3");
    println!("R26_H11_14D_OUTPUT_LEVEL={}", output.level());
    println!("R26_H11_14D_BRANCHES=1,3");
    println!(
        "R26_H11_14D_TOTAL_MS={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );

    assert!(
        relative_l2 < 5.0e-3,
        "encrypted fused radix-4 stage exceeds error bound"
    );

    assert!(
        inactive_max < 5.0e-3,
        "inactive FFT256 lanes exceed error bound"
    );

    println!("R26_H11_14D_ENCRYPTED_STATUS=PASS");
}

// R26_H11_14E_COMPLETE_FFT256

fn complete_fft256_reference(input: &[Complex64], branch: usize) -> Vec<Complex64> {
    assert_eq!(input.len(), LANE_SLOTS);

    let mut output = vec![Complex64::new(0.0, 0.0); LANE_SLOTS];

    for block in 0..LANE_SLOTS / 256 {
        let start = block * 256;

        for k in 0..256 {
            let mut sum = Complex64::new(0.0, 0.0);

            for physical in 0..256 {
                let natural = reverse_base4_4(physical);

                sum += input[start + physical]
                    * fft_root(3840, branch * natural)
                    * fft_root(256, k * natural);
            }

            output[start + k] = sum;
        }
    }

    output
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

#[test]
#[ignore = "complete 65K-ring encrypted FFT256 is expensive"]
fn r26_h11_14e_complete_encrypted_fused_fft256() {
    const BRANCH_A: usize = 1;
    const BRANCH_B: usize = 3;

    let started = Instant::now();

    let profile = research_profile_65536_for_levels(5, 5);
    let degree = profile.degree();
    assert_eq!(degree, DEGREE);

    let validated = profile.modulus_chain().top().clone();
    assert!(validated.moduli().len() >= 10);

    let q = ModulusBasis::new(validated.moduli()[..8].to_vec());
    let p = ModulusBasis::new(validated.moduli()[8..10].to_vec());

    let chain = ModulusChain::from_top_basis(q);
    assert_eq!(chain.max_level(), 7);

    let scale = profile.initial_scale();
    let embedding = CkksCanonicalEmbedding::new(degree);

    let mut plans = Vec::new();
    let mut prepared_plans = Vec::new();
    let mut qp_plans = Vec::new();

    for level in 3..7 {
        let basis = chain.level(level);

        let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

        let mut qp = basis.moduli().to_vec();
        qp.extend_from_slice(p.moduli());

        qp_plans.push(RnsNttPlan::new(qp, degree));
        prepared_plans.push(PreparedRnsNttPlan::new(&plan));
        plans.push(plan);
    }

    let first_a = first_stage_diagonals(BRANCH_A);
    let first_b = first_stage_diagonals(BRANCH_B);

    let fused_a = fuse_fft256_source_lane(&first_a, SLOTS, LANE_SLOTS, 0, 0);
    let fused_b = fuse_fft256_source_lane(&first_b, SLOTS, LANE_SLOTS, 0, 1);

    let prepared_a =
        PreparedFusedFft256Source::new(3, &fused_a, &embedding, &chain, &prepared_plans[0]);
    let prepared_b =
        PreparedFusedFft256Source::new(3, &fused_b, &embedding, &chain, &prepared_plans[0]);

    let source_a: Vec<Complex64> = (0..SLOTS)
        .map(|i| {
            Complex64::new(
                ((i % 19) as f64 - 9.0) / 4096.0,
                ((i % 7) as f64 - 3.0) / 8192.0,
            )
        })
        .collect();

    let source_b: Vec<Complex64> = (0..SLOTS)
        .map(|i| {
            Complex64::new(
                ((i % 13) as f64 - 6.0) / 4096.0,
                ((i % 11) as f64 - 5.0) / 8192.0,
            )
        })
        .collect();

    let mut expected = vec![Complex64::new(0.0, 0.0); SLOTS];

    let reference_a = complete_fft256_reference(&source_a[..LANE_SLOTS], BRANCH_A);
    let reference_b = complete_fft256_reference(&source_b[..LANE_SLOTS], BRANCH_B);

    expected[..LANE_SLOTS].copy_from_slice(&reference_a);
    expected[LANE_SLOTS..2 * LANE_SLOTS].copy_from_slice(&reference_b);

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x5232_4841_4E46);

    let encryption_ctx = EncryptionContext {
        embedding: &embedding,
        basis: chain.level(3),
        plan: &plans[0],
        chain: &chain,
        secret: &secret,
        scale,
    };

    let encryption_started = Instant::now();

    let encrypted_a = encrypt_slots(&source_a, &encryption_ctx, &mut rng);
    let encrypted_b = encrypt_slots(&source_b, &encryption_ctx, &mut rng);

    println!(
        "R26_H11_14E_ENCRYPTION_MS={:.3}",
        encryption_started.elapsed().as_secs_f64() * 1000.0
    );

    let evaluator_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluator_keys);

    let later_diagonals: Vec<Diagonals> = [16, 64, 256]
        .iter()
        .map(|&span| {
            let local = later_stage_diagonals(span);

            // Both packed lanes use identical post-twiddle FFT stages.
            // Build eight full lanes so the diagonal vectors match SLOTS.
            let lanes: Vec<&Diagonals> = (0..8).map(|_| &local).collect();

            pack_fft256_branch_diagonals(&lanes, LANE_SLOTS)
        })
        .collect();

    let mut current: Option<RnsCkksCiphertext> = None;

    for stage_index in 0..4 {
        let level = stage_index + 3;

        let offsets: BTreeSet<isize> = if stage_index == 0 {
            prepared_a
                .rotation_offsets()
                .chain(prepared_b.rotation_offsets())
                .filter(|&offset| offset != 0)
                .collect()
        } else {
            later_diagonals[stage_index - 1]
                .keys()
                .copied()
                .filter(|&offset| offset != 0)
                .collect()
        };

        let keygen_started = Instant::now();
        let mut keys = RotationKeys::new();

        for offset in offsets {
            let exponent = if offset > 0 {
                rotation_exponent_left(degree, offset as usize)
            } else {
                rotation_exponent_right(degree, (-offset) as usize)
            };

            let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(chain.level(level).clone(), 8),
                p.clone(),
                &secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                2,
                &qp_plans[stage_index],
                &mut rng,
            );

            keys.insert(offset, key);
        }

        println!(
            "R26_H11_14E_STAGE_{stage_index}_KEYGEN_MS={:.3}",
            keygen_started.elapsed().as_secs_f64() * 1000.0
        );
        println!(
            "R26_H11_14E_STAGE_{stage_index}_ROTATION_KEYS={}",
            keys.len()
        );

        let ctx = StageContext {
            keys: &keys,
            evaluator: &evaluator,
            chain: &chain,
            qp_plan: &qp_plans[stage_index],
            prepared: &prepared_plans[stage_index],
        };

        let evaluation_started = Instant::now();

        let output = if stage_index == 0 {
            apply_fused_fft256_sources(
                &[(&encrypted_a, &prepared_a), (&encrypted_b, &prepared_b)],
                &ctx,
            )
        } else {
            let prepared = PreparedFft256Stage::new(
                level,
                &later_diagonals[stage_index - 1],
                &embedding,
                &chain,
                &prepared_plans[stage_index],
            );

            apply_stage(
                current.as_ref().expect("prior FFT stage missing"),
                &prepared,
                &ctx,
            )
        };

        println!(
            "R26_H11_14E_STAGE_{stage_index}_EVALUATION_MS={:.3}",
            evaluation_started.elapsed().as_secs_f64() * 1000.0
        );

        assert_eq!(output.level(), level + 1);
        current = Some(output);
    }

    let output = current.expect("FFT256 produced no output");
    assert_eq!(output.level(), 7);

    let actual = decrypt_slots(&output, &secret, &embedding);

    let mut error_sq = 0.0_f64;
    let mut reference_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut inactive_max = 0.0_f64;

    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        let error = (got - want).norm();

        error_sq += error * error;
        reference_sq += want.norm_sqr();
        max_abs = max_abs.max(error);

        if index >= 2 * LANE_SLOTS {
            inactive_max = inactive_max.max(got.norm());
        }
    }

    let relative_l2 = (error_sq / reference_sq).sqrt();

    println!("R26_H11_14E_RELATIVE_L2={relative_l2:.12e}");
    println!("R26_H11_14E_MAX_ABS_ERROR={max_abs:.12e}");
    println!("R26_H11_14E_INACTIVE_MAX={inactive_max:.12e}");
    println!("R26_H11_14E_OUTPUT_LEVEL={}", output.level());
    println!("R26_H11_14E_BRANCHES=1,3");
    println!(
        "R26_H11_14E_TOTAL_MS={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );

    assert!(
        relative_l2 < 5.0e-3,
        "complete encrypted FFT256 exceeds error bound"
    );
    assert!(
        inactive_max < 5.0e-3,
        "inactive FFT256 lanes exceed error bound"
    );

    println!("R26_H11_14E_ENCRYPTED_STATUS=PASS");
}

// R26_H11_14F_FULL_BRANCH_HANDOFF

fn r26_h11_14f_branch_input(branch: usize) -> Vec<Complex64> {
    assert!(branch < 15);

    (0..SLOTS)
        .map(|index| {
            Complex64::new(
                ((index * 7 + branch * 11) % 97) as f64 / 8192.0,
                ((index * 3 + branch * 13) % 89) as f64 / 16384.0,
            )
        })
        .collect()
}

fn r26_h11_14f_plain_group(branches: &[usize]) -> Vec<Complex64> {
    assert!(!branches.is_empty());
    assert!(branches.len() <= SLOTS / LANE_SLOTS);

    let mut result = vec![Complex64::new(0.0, 0.0); SLOTS];

    for (lane, &branch) in branches.iter().enumerate() {
        let source = r26_h11_14f_branch_input(branch);
        let diagonals = first_stage_diagonals(branch);

        let fused = fuse_fft256_source_lane(&diagonals, SLOTS, LANE_SLOTS, 0, lane);

        let actual = apply_fused_source_diagonals_plain(&source, &fused);

        let reference = first_stage_reference(&source[..LANE_SLOTS], branch);

        let start = lane * LANE_SLOTS;
        let end = start + LANE_SLOTS;

        let max_error = actual[start..end]
            .iter()
            .zip(&reference)
            .map(|(&a, &b)| (a - b).norm())
            .fold(0.0_f64, f64::max);

        assert!(
            max_error < 1.0e-12,
            "branch {branch} fused butterfly mismatch: {max_error:e}"
        );

        for (destination, &value) in result.iter_mut().zip(&actual) {
            *destination += value;
        }
    }

    result
}

#[test]
fn r26_h11_14f_full_15_branch_plaintext_handoff() {
    const GROUP_A: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    const GROUP_B: [usize; 7] = [8, 9, 10, 11, 12, 13, 14];

    let output_a = r26_h11_14f_plain_group(&GROUP_A);
    let output_b = r26_h11_14f_plain_group(&GROUP_B);

    for (group, output, active_lanes) in [
        ("A", &output_a, GROUP_A.len()),
        ("B", &output_b, GROUP_B.len()),
    ] {
        let inactive_max = output
            .iter()
            .skip(active_lanes * LANE_SLOTS)
            .map(|value| value.norm())
            .fold(0.0_f64, f64::max);

        assert_eq!(
            inactive_max, 0.0,
            "group {group} has nonzero inactive lanes"
        );

        println!("R26_H11_14F_GROUP_{group}_ACTIVE_LANES={active_lanes}");
        println!("R26_H11_14F_GROUP_{group}_INACTIVE_MAX={inactive_max:.12e}");
    }

    println!("R26_H11_14F_BRANCHES=15");
    println!("R26_H11_14F_PACKED_CIPHERTEXTS=2");
    println!("R26_H11_14F_PLAINTEXT_STATUS=PASS");
}

// R26_H11_14F_FULL_ENCRYPTED_PIPELINE

#[test]
#[ignore = "15-source 65K-ring encrypted FFT256 is expensive"]
fn r26_h11_14f_full_15_branch_encrypted_fft256() {
    const GROUP_A: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    const GROUP_B: [usize; 7] = [8, 9, 10, 11, 12, 13, 14];

    let total_started = Instant::now();

    let profile = research_profile_65536_for_levels(5, 5);
    let degree = profile.degree();
    assert_eq!(degree, DEGREE);

    let validated = profile.modulus_chain().top().clone();
    assert!(validated.moduli().len() >= 10);

    let q = ModulusBasis::new(validated.moduli()[..8].to_vec());
    let p = ModulusBasis::new(validated.moduli()[8..10].to_vec());
    let chain = ModulusChain::from_top_basis(q);

    assert_eq!(chain.max_level(), 7);

    let embedding = CkksCanonicalEmbedding::new(degree);
    let scale = profile.initial_scale();

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

    let mut rng = ChaCha20Rng::seed_from_u64(0x5232_4841_4E47);

    let encryption_ctx = EncryptionContext {
        embedding: &embedding,
        basis: chain.level(3),
        plan: &plans[0],
        chain: &chain,
        secret: &secret,
        scale,
    };

    // Fifteen independent ciphertexts, each with its source data
    // in lane zero and nonzero data in the remaining source lanes.
    let encryption_started = Instant::now();

    let sources: Vec<RnsCkksCiphertext> = (0..15)
        .map(|branch| {
            let input = r26_h11_14f_branch_input(branch);
            encrypt_slots(&input, &encryption_ctx, &mut rng)
        })
        .collect();

    println!(
        "R26_H11_14F_ENCRYPTION_MS={:.3}",
        encryption_started.elapsed().as_secs_f64() * 1000.0
    );
    println!("R26_H11_14F_SOURCE_CIPHERTEXTS={}", sources.len());

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

        let mut keys = RotationKeys::new();
        let stage_started = Instant::now();

        for offset in offsets {
            let exponent = if offset > 0 {
                rotation_exponent_left(degree, offset as usize)
            } else {
                rotation_exponent_right(degree, (-offset) as usize)
            };

            let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(chain.level(level).clone(), 8),
                p.clone(),
                &secret,
                exponent,
                ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                2,
                &qp_plans[stage_index],
                &mut rng,
            );

            keys.insert(offset, key);
        }

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

    let contexts: Vec<StageContext<'_>> = (0..4)
        .map(|stage| StageContext {
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

    let results = apply_fft3840_packed_fft256_from_fft15(
        &source_array,
        &prepared_array,
        &later_array,
        [&contexts[0], &contexts[1], &contexts[2], &contexts[3]],
    );

    println!(
        "R26_H11_14F_TOTAL_EVALUATION_MS={:.3}",
        evaluation_started.elapsed().as_secs_f64() * 1000.0
    );

    // Independent 256-point DFT reference for every branch.
    let validation_started = Instant::now();

    for (group_index, (group_name, branches)) in
        [("A", GROUP_A.as_slice()), ("B", GROUP_B.as_slice())]
            .iter()
            .enumerate()
    {
        let actual = decrypt_slots(&results[group_index], &secret, &embedding);

        let mut expected = vec![Complex64::new(0.0, 0.0); SLOTS];

        for (lane, &branch) in branches.iter().enumerate() {
            let input = r26_h11_14f_branch_input(branch);
            let reference = complete_fft256_reference(&input[..LANE_SLOTS], branch);

            let start = lane * LANE_SLOTS;
            expected[start..start + LANE_SLOTS].copy_from_slice(&reference);
        }

        let mut error_sq = 0.0_f64;
        let mut reference_sq = 0.0_f64;
        let mut max_abs = 0.0_f64;
        let mut inactive_max = 0.0_f64;

        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            let error = (got - want).norm();

            error_sq += error * error;
            reference_sq += want.norm_sqr();
            max_abs = max_abs.max(error);

            if index >= branches.len() * LANE_SLOTS {
                inactive_max = inactive_max.max(got.norm());
            }
        }

        let relative_l2 = (error_sq / reference_sq).sqrt();

        println!("R26_H11_14F_GROUP_{group_name}_RELATIVE_L2={relative_l2:.12e}");
        println!("R26_H11_14F_GROUP_{group_name}_MAX_ABS={max_abs:.12e}");
        println!("R26_H11_14F_GROUP_{group_name}_INACTIVE_MAX={inactive_max:.12e}");
        println!(
            "R26_H11_14F_GROUP_{group_name}_ACTIVE_LANES={}",
            branches.len()
        );

        assert!(
            relative_l2 < 5.0e-3,
            "group {group_name} encrypted FFT256 correctness failure"
        );
        assert!(
            inactive_max < 5.0e-3,
            "group {group_name} inactive-lane contamination"
        );
    }

    println!(
        "R26_H11_14F_VALIDATION_MS={:.3}",
        validation_started.elapsed().as_secs_f64() * 1000.0
    );
    println!("R26_H11_14F_BRANCHES=15");
    println!("R26_H11_14F_OUTPUT_CIPHERTEXTS={}", results.len());
    println!("R26_H11_14F_OUTPUT_LEVEL=7");
    println!(
        "R26_H11_14F_TOTAL_MS={:.3}",
        total_started.elapsed().as_secs_f64() * 1000.0
    );
    println!("R26_H11_14F_ENCRYPTED_STATUS=PASS");
}

// R26_H11_15B_PLAINTEXT_COMPOSITION

#[test]
fn r26_h11_15b_plaintext_fft3840_composition() {
    const N: usize = 3840;
    const CROSS: usize = 15;
    const LOCAL: usize = 256;

    let input: Vec<Complex64> = (0..N)
        .map(|i| {
            Complex64::new(
                ((i * 7 + 3) % 97) as f64 / 8192.0,
                ((i * 11 + 5) % 89) as f64 / 16384.0,
            )
        })
        .collect();

    let mut composed = vec![Complex64::new(0.0, 0.0); N];

    for h in 0..CROSS {
        let mut branch = vec![Complex64::new(0.0, 0.0); LOCAL];

        for s in 0..LOCAL {
            let cross: Complex64 = (0..CROSS)
                .map(|tile| input[tile * LOCAL + s] * fft_root(CROSS, h * tile))
                .sum();

            branch[s] = cross * fft_root(N, h * s);
        }

        // The reference kernel consumes 16 base-4 digit-reversed
        // FFT256 blocks in a 4096-slot lane.
        let mut packed = vec![Complex64::new(0.0, 0.0); LANE_SLOTS];

        for physical in 0..LOCAL {
            packed[physical] = branch[reverse_base4_4(physical)];
        }

        // The FFT3840 twiddle is already included in `branch`.
        // Branch 0 therefore avoids applying it a second time.
        let transformed = complete_fft256_reference(&packed, 0);

        for k in 0..LOCAL {
            composed[h + CROSS * k] = transformed[k];
        }
    }

    let mut max_error = 0.0_f64;
    let mut error_sq = 0.0_f64;
    let mut reference_sq = 0.0_f64;

    for (frequency, &actual) in composed.iter().enumerate() {
        let reference: Complex64 = (0..N).map(|n| input[n] * fft_root(N, frequency * n)).sum();

        let error = (actual - reference).norm();

        max_error = max_error.max(error);
        error_sq += error * error;
        reference_sq += reference.norm_sqr();
    }

    let relative_l2 = (error_sq / reference_sq).sqrt();

    println!("R26_H11_15B_LENGTH={N}");
    println!("R26_H11_15B_BRANCHES={CROSS}");
    println!("R26_H11_15B_LOCAL_FFT={LOCAL}");
    println!("R26_H11_15B_MAX_ABS={max_error:.12e}");
    println!("R26_H11_15B_RELATIVE_L2={relative_l2:.12e}");

    assert!(
        relative_l2 < 1.0e-10,
        "FFT3840 composition disagrees with direct DFT"
    );

    println!("R26_H11_15B_PLAINTEXT_STATUS=PASS");
}
