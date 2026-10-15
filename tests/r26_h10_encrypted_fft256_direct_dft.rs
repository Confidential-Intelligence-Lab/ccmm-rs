// R26 H9: complete four-stage encrypted eight-lane FFT256.
// Research-only parameters and deterministic test keys.
// Run explicitly with --ignored; compilation alone does not generate keys.

use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_65536_for_levels, rotation_exponent_left, rotation_exponent_right,
    CkksCanonicalEmbedding, CkksChainState, PreparedHybridRnsGaloisKey, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::prepared_radix4_fft256::{
    apply_stage, pack_fft256_branch_diagonals, PreparedFft256Stage, StageContext,
};
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
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::time::Instant;

const DEGREE: usize = 65_536;
const LANES: usize = 8;
const LANE_SLOTS: usize = 4096;
const SLOTS: usize = LANES * LANE_SLOTS;
const BRANCHES: [usize; LANES] = [0, 1, 3, 5, 7, 9, 11, 14];
const SIGMA: f64 = 3.19;
const SPANS: [usize; 4] = [4, 16, 64, 256];

type Diagonals = BTreeMap<isize, Vec<Complex64>>;
type RotationKeys = BTreeMap<isize, PreparedHybridRnsGaloisKey>;

fn root(n: usize, exponent: usize) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * (exponent % n) as f64 / n as f64)
}

fn reverse_base4_4(mut index: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..4 {
        reversed = reversed * 4 + index % 4;
        index /= 4;
    }
    reversed
}

fn stage_diagonals(branch: usize, span: usize) -> Diagonals {
    assert!(branch < 15);
    assert!(SPANS.contains(&span));
    let mut diagonals = Diagonals::new();
    let quarter = span / 4;

    for i in 0..LANE_SLOTS {
        let within = i % span;
        let p = within / quarter;
        let j = within % quarter;

        for q in 0..4 {
            let input_within = j + q * quarter;
            let offset = input_within as isize - within as isize;
            let input_index = (i as isize + offset) as usize;

            let mut coefficient = root(span, q * j) * root(4, p * q);

            if span == 4 {
                coefficient *= root(3840, branch * reverse_base4_4(input_index % 256));
            }

            let diagonal = diagonals
                .entry(offset)
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); LANE_SLOTS]);
            diagonal[i] = coefficient;
        }
    }

    diagonals
}

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

fn independent_direct_dft(physical_input: &[Complex64], branch: usize) -> Vec<Complex64> {
    const N: usize = 256;

    assert_eq!(physical_input.len(), N);

    let mut natural = vec![Complex64::new(0.0, 0.0); N];

    for (physical_index, &value) in physical_input.iter().enumerate() {
        natural[reverse_base4_4(physical_index)] = value;
    }

    let mut output = vec![Complex64::new(0.0, 0.0); N];

    for (k, result) in output.iter_mut().enumerate() {
        let mut sum = Complex64::new(0.0, 0.0);

        for (n, &value) in natural.iter().enumerate() {
            sum += value * root(3840, branch * n) * root(N, k * n);
        }

        *result = sum;
    }

    output
}

fn independent_packed_reference(input: &[Complex64]) -> Vec<Complex64> {
    const N: usize = 256;

    assert_eq!(input.len(), SLOTS);

    let mut expected = vec![Complex64::new(0.0, 0.0); SLOTS];

    for (lane, &branch) in BRANCHES.iter().enumerate() {
        let lane_start = lane * LANE_SLOTS;

        for block in 0..LANE_SLOTS / N {
            let start = lane_start + block * N;
            let end = start + N;

            let reference = independent_direct_dft(&input[start..end], branch);

            expected[start..end].copy_from_slice(&reference);
        }
    }

    expected
}

#[test]
#[ignore = "64K-ring hybrid-Q/P key generation is expensive; run explicitly"]
fn r26_h10_encrypted_fft256_direct_dft() {
    let total_started = Instant::now();

    let profile = research_profile_65536_for_levels(2, 5);
    let degree = profile.degree();
    let validated_q = profile.modulus_chain().top().clone();
    let q = ModulusBasis::new(validated_q.moduli()[..5].to_vec());
    let chain = ModulusChain::from_top_basis(q.clone());
    let scale = profile.initial_scale();

    assert_eq!(degree, DEGREE);
    assert_eq!(chain.max_level(), 4);

    let embedding = CkksCanonicalEmbedding::new(degree);
    assert_eq!(embedding.slot_count(), SLOTS);

    let p = ModulusBasis::new(validated_q.moduli()[5..7].to_vec());

    for modulus in q.moduli().iter().chain(p.moduli()) {
        assert_eq!(modulus.value() % (2 * DEGREE as u64), 1);
    }
    for modulus in p.moduli() {
        assert!(!q.moduli().contains(modulus));
    }

    let bases: Vec<ModulusBasis> = (0..4).map(|level| chain.level(level).clone()).collect();
    let plans: Vec<RnsNttPlan> = bases
        .iter()
        .map(|basis| RnsNttPlan::new(basis.moduli().to_vec(), degree))
        .collect();
    let prepared_plans: Vec<PreparedRnsNttPlan> =
        plans.iter().map(PreparedRnsNttPlan::new).collect();

    let qp_plans: Vec<RnsNttPlan> = bases
        .iter()
        .map(|basis| {
            let mut moduli = basis.moduli().to_vec();
            moduli.extend_from_slice(p.moduli());
            RnsNttPlan::new(moduli, degree)
        })
        .collect();

    let diagonals: Vec<Diagonals> = SPANS
        .iter()
        .map(|&span| {
            let lanes: Vec<Diagonals> = BRANCHES
                .iter()
                .map(|&branch| stage_diagonals(branch, span))
                .collect();
            let references: Vec<&Diagonals> = lanes.iter().collect();
            pack_fft256_branch_diagonals(&references, LANE_SLOTS)
        })
        .collect();

    let input: Vec<Complex64> = (0..SLOTS)
        .map(|index| {
            let lane = index / LANE_SLOTS;
            let local = index % LANE_SLOTS;
            Complex64::new(
                ((local % 17) as f64 - 8.0) / 4096.0 * (lane + 1) as f64,
                ((local % 11) as f64 - 5.0) / 8192.0,
            )
        })
        .collect();

    let expected = independent_packed_reference(&input);

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x5232_3648_3800);

    let encryption_started = Instant::now();
    let plaintext = encode_rns(&input, &embedding, &q, scale);
    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
        &secret,
        &plans[0],
        &mut rng,
    );
    let ciphertext = RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, scale), &chain);
    println!(
        "R26_H10_ENCRYPTED_ENCRYPTION_MS={:.3}",
        encryption_started.elapsed().as_secs_f64() * 1000.0
    );

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let mut current = ciphertext;

    for stage_index in 0..4 {
        let span = SPANS[stage_index];
        let stage_started = Instant::now();

        let keygen_started = Instant::now();
        let mut keys = RotationKeys::new();

        for &offset in diagonals[stage_index].keys().filter(|&&offset| offset != 0) {
            let exponent = if offset > 0 {
                rotation_exponent_left(degree, offset as usize)
            } else {
                rotation_exponent_right(degree, (-offset) as usize)
            };

            let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(bases[stage_index].clone(), 8),
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

        println!("R26_H10_ENCRYPTED_STAGE_{stage_index}_SPAN={span}");
        println!("R26_H10_ENCRYPTED_STAGE_{stage_index}_KEYS={}", keys.len());
        println!(
            "R26_H10_ENCRYPTED_STAGE_{stage_index}_KEYGEN_MS={:.3}",
            keygen_started.elapsed().as_secs_f64() * 1000.0
        );

        let preparation_started = Instant::now();
        let prepared_stage = PreparedFft256Stage::new(
            stage_index,
            &diagonals[stage_index],
            &embedding,
            &chain,
            &prepared_plans[stage_index],
        );

        println!(
            "R26_H10_ENCRYPTED_STAGE_{stage_index}_PREPARATION_MS={:.3}",
            preparation_started.elapsed().as_secs_f64() * 1000.0
        );

        let ctx = StageContext {
            keys: &keys,
            evaluator: &evaluator,
            chain: &chain,
            qp_plan: &qp_plans[stage_index],
            prepared: &prepared_plans[stage_index],
        };

        let evaluation_started = Instant::now();
        current = apply_stage(&current, &prepared_stage, &ctx);

        println!(
            "R26_H10_ENCRYPTED_STAGE_{stage_index}_EVALUATION_MS={:.3}",
            evaluation_started.elapsed().as_secs_f64() * 1000.0
        );

        assert_eq!(current.level(), stage_index + 1);

        println!(
            "R26_H10_ENCRYPTED_STAGE_{stage_index}_TOTAL_MS={:.3}",
            stage_started.elapsed().as_secs_f64() * 1000.0
        );
    }

    let result = current;

    let decryption_started = Instant::now();
    let actual = decrypt_slots(&result, &secret, &embedding);
    println!(
        "R26_H10_ENCRYPTED_DECRYPTION_MS={:.3}",
        decryption_started.elapsed().as_secs_f64() * 1000.0
    );

    let mut error_sq = 0.0_f64;
    let mut reference_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (&got, &want) in actual.iter().zip(&expected) {
        let error = (got - want).norm();
        error_sq += error * error;
        reference_sq += want.norm_sqr();
        max_abs = max_abs.max(error);
    }

    assert!(reference_sq > 0.0);
    let relative_l2 = (error_sq / reference_sq).sqrt();

    println!("R26_H10_ENCRYPTED_RELATIVE_L2={relative_l2:.12e}");
    println!("R26_H10_ENCRYPTED_MAX_ABS_ERROR={max_abs:.12e}");
    println!("R26_H10_ENCRYPTED_INPUT_LEVEL=0");
    println!("R26_H10_ENCRYPTED_OUTPUT_LEVEL={}", result.level());
    println!("R26_H10_ENCRYPTED_LANES={LANES}");
    println!("R26_H10_ENCRYPTED_SLOTS={SLOTS}");
    println!(
        "R26_H10_ENCRYPTED_TOTAL_MS={:.3}",
        total_started.elapsed().as_secs_f64() * 1000.0
    );

    assert!(
        relative_l2 < 5.0e-3,
        "encrypted eight-lane FFT256 exceeds relative error bound"
    );

    println!("R26_H10_ENCRYPTED_STATUS=PASS");
}
